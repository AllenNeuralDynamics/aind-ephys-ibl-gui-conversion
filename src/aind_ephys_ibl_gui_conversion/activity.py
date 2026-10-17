"""Sampled AP activity using the detector from conversion PR #46.

The detector and robust block reduction preserve PR head 7731283. Sampling
uses bounded snippets of the existing reads. Stream output keeps physical
contacts instead of constructing the PR's normalized display grid.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from functools import partial

import numpy as np
import scipy.ndimage
import scipy.signal

from aind_ephys_ibl_gui_conversion.metric_context import (
    BlockContext,
    SampleBudget,
    SnippetPlan,
    SourceContext,
)
from aind_ephys_ibl_gui_conversion.metric_extensions import (
    ChannelOutput,
    Metric,
    MetricBlockResult,
    StreamContext,
    block_provenance,
    channel_metric,
)
from aind_ephys_ibl_gui_conversion.window_pipeline import (
    MetricDefinition,
    Product,
)


@dataclass(frozen=True)
class ActivityParams:
    """Ben's bandpass, MAD threshold, and neighboring-support thresholds."""

    highpass_hz: float = 300.0
    lowpass_hz: float = 6000.0
    threshold_sigma: float = 4.5
    support_sigma: float = 3.0

    def filter(self, sampling_rate: float) -> np.ndarray:
        """Validate detector parameters and build the original SOS filter."""
        values = np.array(list(asdict(self).values()))
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("Activity parameters must be finite and positive")
        nyquist = sampling_rate / 2.0
        lowpass = min(self.lowpass_hz, nyquist * 0.95)
        if self.highpass_hz >= lowpass:
            raise ValueError(
                "AP high-pass must be below the effective low-pass"
            )
        return np.asarray(
            scipy.signal.butter(
                3,
                (self.highpass_hz / nyquist, lowpass / nyquist),
                btype="bandpass",
                output="sos",
            )
        )


def _supported_candidates(
    filtered: np.ndarray,
    sigma: np.ndarray,
    depths_um: np.ndarray,
    *,
    sampling_rate: float,
    threshold_sigma: float,
    support_sigma: float,
) -> np.ndarray:
    """Preserve Ben's peak, artifact-veto, and channel-support rules.

    Support is an index neighborhood after depth/lateral sorting. Its size
    is inferred from central depths; it is not a physical-distance graph.
    This known geometry limitation is retained for detector parity.
    """
    z = filtered / sigma[None, :]
    refractory = max(round(0.0005 * sampling_rate), 3)
    support_time = max(round(0.00025 * sampling_rate), 3)
    middle_depth = float(np.median(depths_um))
    support_channels = max(
        int(np.sum(np.abs(depths_um - middle_depth) <= 40.0)), 7
    )
    support_channels = min(support_channels, filtered.shape[1])
    if support_channels % 2 == 0 and support_channels > 1:
        support_channels -= 1
    artifact = np.mean(np.abs(z) > 8.0, axis=1) > 0.08
    artifact = scipy.ndimage.maximum_filter1d(
        artifact.astype(np.uint8), size=refractory * 2 + 1
    ).astype(bool)
    loose = scipy.ndimage.maximum_filter1d(
        (z < -support_sigma).astype(np.uint8),
        size=support_time,
        axis=0,
    )
    support = scipy.ndimage.convolve1d(
        loose,
        np.ones(support_channels, dtype=np.uint16),
        axis=1,
        mode="constant",
        cval=0,
    )
    local_minimum = filtered <= scipy.ndimage.minimum_filter1d(
        filtered, size=refractory, axis=0, mode="nearest"
    )
    candidates = local_minimum & (z < -threshold_sigma) & (support >= 2)
    candidates[artifact] = False
    return np.asarray(candidates, dtype=bool)


def _health_mask(
    noises: np.ndarray, groups: tuple[np.ndarray, ...]
) -> np.ndarray:
    """Apply the original local-noise and temporal-stability channel gates."""
    noise_uv = np.median(noises, axis=0)
    healthy = np.zeros(noise_uv.size, dtype=bool)
    for indices in groups:
        local_noise = scipy.ndimage.median_filter(
            noise_uv[indices], size=21, mode="nearest"
        )
        ratio = noise_uv[indices] / np.maximum(
            local_noise, np.finfo(float).eps
        )
        temporal_ratio = np.percentile(noises[:, indices], 90, axis=0)
        temporal_ratio /= np.maximum(
            np.percentile(noises[:, indices], 10, axis=0), np.finfo(float).eps
        )
        healthy[indices] = (
            np.isfinite(noise_uv[indices])
            & (ratio >= 0.25)
            & (ratio <= 4.0)
            & (temporal_ratio <= 5.0)
        )
    return healthy


@dataclass(frozen=True)
class ActivityResult:
    """Native-contact trimmed rates (Hz), MAD noise (µV), and coverage.

    Coverage counts retained snippet time from healthy channels. Artifact
    vetoes do not subtract time, and trimming rate tails does not change the
    per-window denominator. Coverage is not a pooled-count denominator.
    """

    rate_hz: np.ndarray
    noise_uv: np.ndarray
    healthy: np.ndarray
    sampled_seconds: np.ndarray
    sampling: SnippetPlan


class ActivityAccumulator:
    """Keep only small, indexed per-window rate/noise summaries per block."""

    def __init__(
        self,
        source: SourceContext,
        sampling: SnippetPlan,
        params: ActivityParams,
        sos: np.ndarray,
    ) -> None:
        """Precompute shank channel order and allocate summary arrays."""
        self.source = source
        self.sampling = sampling
        self.params = params
        self.sos = sos
        self.slots = {
            index: slot for slot, index in enumerate(sampling.window_indices)
        }
        self.groups = tuple(
            (indices := np.flatnonzero(source.shank_indices == shank))[
                np.lexsort(
                    (
                        source.locations[indices, 0],
                        source.locations[indices, 1],
                    )
                )
            ]
            for shank in np.unique(source.shank_indices)
        )
        self.health_groups = tuple(
            (indices := np.flatnonzero(source.shank_indices == shank))[
                np.argsort(source.locations[indices, 1], kind="stable")
            ]
            for shank in np.unique(source.shank_indices)
        )
        shape = (len(self.slots), source.n_channels)
        self.rates = np.zeros(shape, dtype=float)
        self.noises = np.zeros(shape, dtype=float)
        self.seen: set[int] = set()

    def update(self, window_index: int, inputs: tuple[Product, ...]) -> None:
        """Filter independent snippets without retaining raw inputs."""
        if window_index in self.seen:
            raise ValueError("Activity window was processed twice")
        (raw,) = inputs
        assert isinstance(raw, np.ndarray)
        if raw.shape != (
            self.sampling.request.n_samples,
            self.source.n_channels,
        ):
            raise ValueError("Activity input shape does not match its plan")
        slot = self.slots[window_index]
        edge = self.sampling.edge_samples
        for indices in self.groups:
            shank_raw = raw[:, indices].copy()
            shank_raw -= np.median(shank_raw, axis=1, keepdims=True)
            filtered = scipy.signal.sosfiltfilt(self.sos, shank_raw, axis=0)
            filtered = filtered.astype(np.float32)[edge:-edge]
            sigma = np.median(np.abs(filtered), axis=0) / 0.67448975
            positive = sigma[np.isfinite(sigma) & (sigma > 0)]
            floor = (
                np.percentile(positive, 10) * 0.5
                if positive.size
                else np.finfo(np.float32).eps
            )
            sigma = np.maximum(sigma, floor)
            candidates = _supported_candidates(
                filtered,
                sigma,
                self.source.locations[indices, 1],
                sampling_rate=self.source.sampling_frequency,
                threshold_sigma=self.params.threshold_sigma,
                support_sigma=self.params.support_sigma,
            )
            self.rates[slot, indices] = (
                candidates.sum(axis=0) / self.sampling.seconds_per_window
            )
            self.noises[slot, indices] = sigma
        self.seen.add(window_index)

    def finish(self) -> ActivityResult:
        """Apply the original trimmed mean and channel-health gates."""
        if len(self.seen) != len(self.slots):
            raise ValueError("Activity sampling did not complete")
        ordered = np.sort(self.rates, axis=0)
        trim = math.floor(0.1 * len(ordered))
        rates = np.mean(ordered[trim:-trim] if trim else ordered, axis=0)
        healthy = _health_mask(self.noises, self.health_groups)
        rates[~healthy] = np.nan
        seconds = np.where(
            healthy, len(ordered) * self.sampling.seconds_per_window, 0.0
        )
        return ActivityResult(
            rates,
            np.median(self.noises, axis=0),
            healthy,
            seconds,
            self.sampling,
        )


def raw_activity(
    *,
    sampling: SampleBudget = SampleBudget(),
    params: ActivityParams = ActivityParams(),
    name: str = "rawActivity",
) -> Metric:
    """Estimate AP event rates on physical contacts from bounded snippets.

    Use independent per-block accumulators and the standard contact writer.
    Repeated contact observations combine by the median of healthy block
    rates; sampled duration describes coverage, not a pooled-count denominator.
    """

    def define(context: BlockContext) -> MetricDefinition:
        """Resolve a small AP budget and private filter before trace reads."""
        plan = sampling.resolve(context.ap)
        sos = params.filter(context.ap.sampling_frequency)
        if plan.edge_samples < 1:
            raise ValueError(
                "Activity requires at least one trimmed edge sample"
            )
        if plan.request.n_samples <= 3 * (2 * len(sos) + 1):
            raise ValueError(
                "Activity snippet is too short for filter padding"
            )
        logging.info(
            "[Metrics] %s %s: %d snippets, %.3f analyzed seconds",
            name,
            context.block_id,
            len(plan.window_indices),
            len(plan.window_indices) * plan.seconds_per_window,
        )
        return MetricDefinition(
            name,
            (plan.request,),
            partial(ActivityAccumulator, context.ap, plan, params, sos),
            window_indices=plan.window_indices,
        )

    def reduce(
        context: StreamContext,
        results: tuple[MetricBlockResult[ActivityResult], ...],
    ) -> ChannelOutput:
        """Take the median of healthy block rates at each physical contact."""
        n_rows = len(context.channel_table.raw_ind)
        values: list[list[float]] = [[] for _ in range(n_rows)]
        exposure = np.zeros(n_rows, dtype=float)
        provenance = []
        for result in results:
            rows = context.channel_rows(result.context.ap)
            value = result.value
            arrays = (
                value.rate_hz,
                value.noise_uv,
                value.healthy,
                value.sampled_seconds,
            )
            if any(array.shape != rows.shape for array in arrays):
                raise ValueError(
                    "Activity result shape does not match source channels"
                )
            valid = value.healthy & np.isfinite(value.rate_hz)
            for row, rate in zip(
                rows[valid], value.rate_hz[valid], strict=True
            ):
                values[row].append(float(rate))
            np.add.at(exposure, rows[valid], value.sampled_seconds[valid])
            details = block_provenance(result.context, "ap")
            details["sampling"] = value.sampling.metadata()
            details["healthy_channels"] = int(valid.sum())
            provenance.append(details)
        rates = np.array(
            [np.median(rates) if rates else np.nan for rates in values]
        )
        output = ChannelOutput(
            rates,
            exposure,
            "Hz",
            {
                "parameters": asdict(params),
                "pooling": "median of healthy per-block trimmed-mean rates",
                "artifact_time_subtracted": False,
                "coverage_is_rate_denominator": False,
                "spatial_support": "legacy depth-sorted index neighborhood",
                "reference": "per-shank common median",
                "blocks": provenance,
            },
        )
        output.validate(n_rows)
        return output

    return channel_metric(name=name, define=define, reduce=reduce)
