"""Normalized activity-density profiles estimated directly from raw AP data.

The estimator detects spatially supported negative threshold crossings in a
short AP sample.  It is a coarse activity proxy, not an estimate of an
absolute number of units.  Output profiles retain a complete Neuropixels bank
grid so that a removed contact cannot shorten a bank or move a following
surface-finding bank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.ndimage
import scipy.signal

from aind_ephys_ibl_gui_conversion.channel_metadata import (
    normalized_channel_groups,
)
from aind_ephys_ibl_gui_conversion.metrics import block_source
from aind_ephys_ibl_gui_conversion.types import ExperimentBlock


@dataclass(frozen=True)
class UnitDensityParams:
    """Parameters for raw-AP activity-density estimation.

    ``bank_span_um`` is the physical length represented by one acquired
    Neuropixels bank.  With the usual 20-um row pitch this produces 192 rows
    per bank, independent of removed contacts at either edge.
    """

    chunks: int = 1
    chunk_seconds: float = 1.0
    highpass_hz: float = 300.0
    lowpass_hz: float = 6000.0
    threshold_sigma: float = 4.5
    support_sigma: float = 3.0
    smooth_um: float = 80.0
    bank_span_um: float = 3840.0


def _trimmed_mean(values: np.ndarray, proportion: float = 0.1) -> np.ndarray:
    """Return a mean with equal tails removed along the first axis."""
    ordered = np.sort(values, axis=0)
    trim = math.floor(proportion * ordered.shape[0])
    if trim == 0:
        return np.mean(ordered, axis=0)
    return np.mean(ordered[trim:-trim], axis=0)


def _normalize_density(
    values: np.ndarray, depths_um: np.ndarray
) -> np.ndarray:
    """Normalize a non-negative profile to unit area in millimetres."""
    values = np.maximum(np.asarray(values, dtype=float), 0.0)
    area = np.trapezoid(values, depths_um / 1000.0)
    if not np.isfinite(area) or area <= 0:
        return np.zeros_like(values)
    return values / area


def _supported_candidates(
    filtered: np.ndarray,
    sigma: np.ndarray,
    depths_um: np.ndarray,
    *,
    sampling_rate: float,
    threshold_sigma: float,
    support_sigma: float,
) -> np.ndarray:
    """Detect negative peaks with nearby spatial threshold support."""
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
    return candidates


def estimate_block_unit_density(
    block: ExperimentBlock,
    params: UnitDensityParams | None = None,
) -> np.ndarray:
    """Estimate per-channel crossing rate and noise from one AP block."""
    params = params or UnitDensityParams()
    if params.chunks < 1:
        raise ValueError("chunks must be at least 1")
    if params.chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")

    recording = block.recording
    sampling_rate = float(recording.get_sampling_frequency())
    samples_per_chunk = round(params.chunk_seconds * sampling_rate)
    available = recording.get_num_samples() // samples_per_chunk
    if available < 1:
        raise RuntimeError("Recording is shorter than one unit-density chunk")
    chunks = min(params.chunks, available)
    if chunks == 1:
        block_indexes = np.asarray([available // 2], dtype=int)
    else:
        block_indexes = np.unique(
            np.linspace(0, available - 1, chunks).round().astype(int)
        )

    nyquist = sampling_rate / 2.0
    lowpass_hz = min(params.lowpass_hz, nyquist * 0.95)
    if params.highpass_hz >= lowpass_hz:
        raise ValueError("AP high-pass must be below the effective low-pass")
    sos = scipy.signal.butter(
        3,
        (params.highpass_hz / nyquist, lowpass_hz / nyquist),
        btype="bandpass",
        output="sos",
    )
    edge = max(round(0.025 * sampling_rate), 1)
    retained_samples = samples_per_chunk - 2 * edge
    if retained_samples <= 0:
        raise RuntimeError(
            "Unit-density chunk is too short after filter edges"
        )

    locations = np.asarray(recording.get_channel_locations(), dtype=float)
    groups = normalized_channel_groups(recording)
    rates: list[np.ndarray] = []
    noises: list[np.ndarray] = []
    for block_index in block_indexes:
        start = int(block_index * samples_per_chunk)
        raw = recording.get_traces(
            start_frame=start,
            end_frame=start + samples_per_chunk,
            return_in_uV=True,
        ).astype(np.float32)
        chunk_rate = np.zeros(recording.get_num_channels(), dtype=float)
        chunk_noise = np.zeros(recording.get_num_channels(), dtype=float)
        for shank in np.unique(groups):
            indices = np.flatnonzero(groups == shank)
            shank_locations = locations[indices]
            order = np.lexsort((shank_locations[:, 0], shank_locations[:, 1]))
            sorted_indices = indices[order]
            shank_raw = raw[:, sorted_indices].copy()
            shank_raw -= np.median(shank_raw, axis=1, keepdims=True)
            filtered = scipy.signal.sosfiltfilt(sos, shank_raw, axis=0).astype(
                np.float32
            )[edge:-edge]
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
                locations[sorted_indices, 1],
                sampling_rate=sampling_rate,
                threshold_sigma=params.threshold_sigma,
                support_sigma=params.support_sigma,
            )
            chunk_rate[sorted_indices] = candidates.sum(axis=0) / (
                retained_samples / sampling_rate
            )
            chunk_noise[sorted_indices] = sigma
        rates.append(chunk_rate)
        noises.append(chunk_noise)

    chunk_rates = np.asarray(rates)
    chunk_noise = np.asarray(noises)
    rate_hz = _trimmed_mean(chunk_rates)
    noise_uv = np.median(chunk_noise, axis=0)
    healthy = np.zeros(recording.get_num_channels(), dtype=bool)
    for shank in np.unique(groups):
        indices = np.flatnonzero(groups == shank)
        order = np.argsort(locations[indices, 1], kind="stable")
        sorted_indices = indices[order]
        local_noise = scipy.ndimage.median_filter(
            noise_uv[sorted_indices], size=21, mode="nearest"
        )
        ratio = noise_uv[sorted_indices] / np.maximum(
            local_noise, np.finfo(float).eps
        )
        temporal_ratio = np.percentile(
            chunk_noise[:, sorted_indices], 90, axis=0
        )
        temporal_ratio /= np.maximum(
            np.percentile(chunk_noise[:, sorted_indices], 10, axis=0),
            np.finfo(float).eps,
        )
        healthy[sorted_indices] = (
            np.isfinite(noise_uv[sorted_indices])
            & (ratio >= 0.25)
            & (ratio <= 4.0)
            & (temporal_ratio <= 5.0)
        )

    rate_hz[~healthy] = np.nan
    return rate_hz


def _row_pitch(blocks: list[ExperimentBlock], shank: int) -> float:
    """Infer contact-row pitch from all blocks recording one shank."""
    spacings: list[np.ndarray] = []
    for block in blocks:
        locations = np.asarray(block.recording.get_channel_locations())
        groups = normalized_channel_groups(block.recording)
        depths = np.unique(locations[groups == shank, 1])
        if depths.size > 1:
            positive = np.diff(depths)
            spacings.append(positive[positive > 0])
    available = [spacing for spacing in spacings if spacing.size]
    if not available:
        raise RuntimeError(f"Cannot infer row pitch for shank {shank}")
    return float(np.min(np.concatenate(available)))


def _source_rows(
    blocks: list[ExperimentBlock],
    estimates: list[np.ndarray],
    shank: int,
    source: str,
    pitch_um: float,
    bank_span_um: float,
) -> np.ndarray:
    """Pool blocks onto one complete source-bank grid."""
    selected = [
        (block, estimate)
        for block, estimate in zip(blocks, estimates, strict=True)
        if block_source(block) == source
    ]
    observed_depths = np.concatenate(
        [
            np.asarray(block.recording.get_channel_locations())[
                groups == shank, 1
            ]
            for block, _ in selected
            if np.any(
                (groups := normalized_channel_groups(block.recording)) == shank
            )
        ]
    )
    anchor = (
        math.floor(float(np.min(observed_depths)) / bank_span_um)
        * bank_span_um
    )
    relative_max = float(np.max(observed_depths) - anchor)
    n_banks = max(1, math.ceil((relative_max + pitch_um) / bank_span_um))
    n_rows = round(n_banks * bank_span_um / pitch_um)
    rates: list[list[float]] = [[] for _ in range(n_rows)]

    for block, estimate in selected:
        locations = np.asarray(block.recording.get_channel_locations())
        groups = normalized_channel_groups(block.recording)
        on_shank = np.flatnonzero(groups == shank)
        for channel in on_shank:
            row = int(round((locations[channel, 1] - anchor) / pitch_um))
            if row < 0 or row >= n_rows:
                continue
            if np.isfinite(estimate[channel]):
                rates[row].append(float(estimate[channel]))

    return np.asarray(
        [np.median(values) if values else np.nan for values in rates],
        dtype=float,
    )


def compose_raw_unit_density(
    blocks: list[ExperimentBlock],
    estimates: list[np.ndarray],
    params: UnitDensityParams | None = None,
) -> np.ndarray:
    """Pool blocks and stack surface banks immediately above main banks."""
    params = params or UnitDensityParams()
    if not blocks or len(blocks) != len(estimates):
        raise ValueError("At least one block estimate is required")
    source_order = ("main", "surface")
    profiles: list[np.ndarray] = []
    shanks = sorted(
        {
            int(shank)
            for block in blocks
            for shank in normalized_channel_groups(block.recording)
        }
    )
    for shank in shanks:
        pitch_um = _row_pitch(blocks, shank)
        rates: list[np.ndarray] = []
        for source in source_order:
            if not any(
                np.any(normalized_channel_groups(block.recording) == shank)
                for block in blocks
                if block_source(block) == source
            ):
                continue
            rates.append(
                _source_rows(
                    blocks,
                    estimates,
                    shank,
                    source,
                    pitch_um,
                    params.bank_span_um,
                )
            )
        row_rate = np.concatenate(rates)
        depths = np.arange(row_rate.size, dtype=float) * pitch_um

        finite = np.isfinite(row_rate)
        if np.any(finite):
            filled = np.interp(depths, depths[finite], row_rate[finite])
            baseline_hz = float(np.percentile(row_rate[finite], 20))
            activity = np.log1p(np.maximum(filled - baseline_hz, 0.0))
            activity = scipy.ndimage.gaussian_filter1d(
                activity,
                sigma=max(params.smooth_um / pitch_um, 0.5),
                mode="nearest",
            )
            activity -= np.percentile(activity, 5)
            density = _normalize_density(activity, depths)
        else:
            density = np.zeros_like(depths)
        profiles.append(density)

    output = np.full(
        (len(profiles), max(profile.size for profile in profiles)),
        np.nan,
        dtype=np.float32,
    )
    for row, profile in enumerate(profiles):
        output[row, : profile.size] = profile
    return output
