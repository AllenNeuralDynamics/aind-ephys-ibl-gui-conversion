"""Brain-surface line detection through the sampled metric API."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import scipy.fft

from aind_ephys_ibl_gui_conversion.lines import (
    BlockLines,
    LineAccumulator,
    LineParams,
    fit_transitions,
    plateau_onset,
    pool_lines,
    usable_channels,
)
from aind_ephys_ibl_gui_conversion.metric_context import BlockContext
from aind_ephys_ibl_gui_conversion.metric_extensions import (
    Metric,
    MetricBlockResult,
    StreamContext,
    block_provenance,
)
from aind_ephys_ibl_gui_conversion.spectral_accumulators import RmsAccumulator
from aind_ephys_ibl_gui_conversion.types import ChannelTable
from aind_ephys_ibl_gui_conversion.window_pipeline import (
    FftRequest,
    MetricDefinition,
    Product,
    Spectrum,
)

_FILES = (
    "_iblqc_ephysLineAmplitude.amps.npy",
    "_iblqc_ephysLineAmplitude.freqs.npy",
    "brain_surface.json",
    "_iblqc_ephysLineAmplitude.metadata.json",
)


@dataclass
class _LineBlock:
    """Owned spectral power and AP RMS for contact rejection."""

    power: LineAccumulator
    rms_ap: np.ndarray


@dataclass(frozen=True)
class SurfaceLineOutput:
    """Pure stream result, with one amplitude row per line and block."""

    amplitudes: np.ndarray
    frequencies: np.ndarray
    surface: dict
    metadata: dict


class _SurfaceAccumulator:
    """Recover unreferenced line power without modifying borrowed spectra."""

    def __init__(self, context: BlockContext, params: LineParams) -> None:
        """Allocate independent state for one block's native AP channels."""
        self.lines = LineAccumulator(
            context.ap.read_samples,
            context.ap.sampling_frequency,
            context.ap.n_channels,
            params,
        )
        request = context.ap_fft
        correction = (
            np.sqrt(request.n_samples / context.lfp_window_power)
            if request.taper == "hann"
            else 1.0
        )
        self.rms = RmsAccumulator(
            context.n_windows,
            context.ap.n_channels,
            context.frequencies(request) > 300.0,
            request.n_samples,
            correction,
        )

    def update(self, window_index: int, inputs: tuple[Product, ...]) -> None:
        """Add owned band power and RMS from shared read-only products."""
        spectrum, ap_spectrum = inputs
        assert isinstance(spectrum, Spectrum)
        reference = spectrum.reference
        restored = None
        if reference is not None:
            restored = scipy.fft.rfft(
                self.lines.hann * (reference - reference.mean())
            )[self.lines.band]
        band = spectrum.coefficients[self.lines.band]
        for start in range(0, band.shape[1], self.lines.channel_chunk):
            cols = slice(start, start + self.lines.channel_chunk)
            owned = band[:, cols].copy()
            if restored is not None:
                owned += restored[:, None]
            self.lines.power[:, cols] += owned.real**2 + owned.imag**2
        self.lines.n_windows += 1
        self.rms.update(window_index, (ap_spectrum,))

    def finish(self) -> _LineBlock:
        """Return sufficient statistics for pooling and surface fitting."""
        return _LineBlock(self.lines, self.rms.finish())


def _define_lines(
    context: BlockContext, *, params: LineParams
) -> MetricDefinition:
    """Share NP2's Hann FFT, or request a full AP Hann FFT on NP1."""
    request = (
        FftRequest("ap", context.ap.read_samples, demean=True, taper="hann")
        if context.separate_lfp
        else context.lfp_fft
    )
    return MetricDefinition(
        "brainSurfaceLines",
        (request, context.ap_fft),
        partial(_SurfaceAccumulator, context, params),
    )


def _on_rows(values: np.ndarray, rows: np.ndarray, n_rows: int) -> np.ndarray:
    """Map native amplitudes to canonical contact rows with missing NaNs."""
    output = np.full((values.shape[0], n_rows), np.nan, dtype=np.float32)
    output[:, rows] = values
    return output


def _line_groups(found: list[BlockLines | None]) -> list[list[int]]:
    """Blocks measured at the same lines, which are fitted together."""
    groups: list[list[int]] = []
    for i, block_lines in enumerate(found):
        if block_lines is None:
            continue
        for group in groups:
            ref = found[group[0]]
            assert ref is not None
            if np.array_equal(ref.freqs_hz, block_lines.freqs_hz):
                group.append(i)
                break
        else:
            groups.append([i])
    return groups


def _shank_profile(
    results: tuple[MetricBlockResult[_LineBlock], ...],
    block_maps: list[np.ndarray],
    found: list[BlockLines | None],
    group: list[int],
    channel_table: ChannelTable,
    shank: int,
    line_params: LineParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Usable channels of *shank* over the blocks in *group*: line
    amplitudes, probe-local locations and block of each channel."""
    amps, locations, blocks = [], [], []
    for block_idx in group:
        block_lines = found[block_idx]
        assert block_lines is not None
        rows = block_maps[block_idx][: block_lines.amps_v.shape[1]]
        on_shank = np.flatnonzero(channel_table.shank_ind[rows] == shank)
        if not on_shank.size:
            continue
        rms_ap = np.sqrt(
            np.mean(
                np.square(results[block_idx].value.rms_ap[:, on_shank]), axis=0
            )
        )
        keep = on_shank[usable_channels(rms_ap, line_params)]
        amps.append(block_lines.amps_v[:, keep])
        locations.append(channel_table.local_coordinates[rows[keep]])
        blocks.append(np.full(keep.size, block_idx))
    if not amps or not block_lines.freqs_hz.size:
        return None
    return (
        np.concatenate(amps, axis=1),
        np.concatenate(locations),
        np.concatenate(blocks),
    )


def _shank_fits(
    amps_v: np.ndarray,
    locations: np.ndarray,
    blocks: np.ndarray,
    driven: int | None,
    line_params: LineParams,
) -> list[tuple[str, dict, np.ndarray]]:
    """Each candidate on one shank with its method and the lines it was
    fitted to, as indices into ``amps_v``."""
    every = np.arange(amps_v.shape[0])
    fits = [
        ("lines", t, every)
        for t in fit_transitions(amps_v, locations, line_params, blocks=blocks)
    ]
    air = [t for _, t, _ in fits if t["boundary"] == "air"]
    if air:
        onset = plateau_onset(
            amps_v, locations, air[0], line_params, blocks=blocks
        )
        if onset is not None:
            fits.append(("plateau", onset, every))
    if driven is not None:
        fits += [
            ("driven", t, np.array([driven]))
            for t in fit_transitions(
                amps_v[[driven]],
                locations,
                line_params,
                blocks=blocks,
                max_transitions=1,
                min_agreeing_lines=1,
            )
        ]
    return fits


def _reduce_lines(
    context: StreamContext,
    results: tuple[MetricBlockResult[_LineBlock], ...],
    line_params: LineParams,
    driven_tone_hz: float | None = None,
) -> SurfaceLineOutput:
    """Reduce interference-line amplitudes and brain-surface candidates.

    Lines are found across all blocks and measured in each, so every block
    reports the same lines. Each amplitude row is one line in one block,
    ``NaN`` on rows that block did not record; rows are not stitched across
    blocks because a line's level drifts between recordings. Each shank is
    fitted once over every block that recorded it, with a baseline per
    block, and every passing transition is kept as a candidate. Below one
    labelled the air boundary, where the lines leave their bath level is
    added as a ``plateau`` candidate. None is chosen as the surface;
    candidates are listed strongest first, the air boundary last.
    ``brain_surface.json`` ``lines`` is aligned with
    the amplitude rows. Input block results remain available for inspection.

    Parameters
    ----------
    context : StreamContext
        Canonical channel geometry.
    results : tuple[MetricBlockResult[_LineBlock], ...]
        Owned results in stream block order.
    line_params : LineParams
        Fit settings.
    driven_tone_hz : float or None
        Frequency of a tone driven into the animal. The line within
        ``line_params.background_hz[0]`` of it is also fitted alone and its
        transition kept as a ``driven`` candidate.
    """
    found = pool_lines([r.value.power for r in results], line_params)
    channel_table = context.channel_table
    block_maps = [context.channel_rows(r.context.ap) for r in results]
    n_rows = len(channel_table.raw_ind)
    amps: list[np.ndarray] = []
    lines: list[dict] = []
    # Amplitude row of each block's first line.
    first_row: dict[int, int] = {}
    for block_idx, (r, row_map, block_lines) in enumerate(
        zip(results, block_maps, found, strict=True)
    ):
        if block_lines is None:
            continue
        first_row[block_idx] = len(lines)
        amps.append(_on_rows(block_lines.amps_v, row_map, n_rows))
        lines.extend(
            {
                "freq_hz": float(f),
                "line_index": j,
                "block_index": block_idx,
                "label": r.context.block_source,
                "excess_db": None if np.isnan(excess) else float(excess),
            }
            for j, (f, excess) in enumerate(
                zip(block_lines.freqs_hz, block_lines.excess_db, strict=True)
            )
        )

    candidates: dict[int, list[dict]] = {}
    for group in _line_groups(found):
        freqs = found[group[0]].freqs_hz  # type: ignore[union-attr]
        driven = None
        if driven_tone_hz is not None and freqs.size:
            j = int(np.argmin(np.abs(freqs - driven_tone_hz)))
            if abs(freqs[j] - driven_tone_hz) <= line_params.background_hz[0]:
                driven = j
        for shank in np.unique(channel_table.shank_ind):
            pooled = _shank_profile(
                results,
                block_maps,
                found,
                group,
                channel_table,
                int(shank),
                line_params,
            )
            if pooled is None:
                continue
            shank_amps, locations, blocks = pooled
            fits = _shank_fits(
                shank_amps, locations, blocks, driven, line_params
            )
            block_indices = sorted({int(b) for b in blocks})
            for method, t, line_indices in fits:
                for line in t["lines"]:
                    j = int(line_indices[line.pop("row")])
                    line["line_index"] = j
                    line["freq_hz"] = float(freqs[j])
                    line["amplitude_rows"] = {
                        b: first_row[b] + j for b in block_indices
                    }
                t.pop("passes", None)
                candidates.setdefault(int(shank), []).append(
                    {"method": method, "block_indices": block_indices} | t
                )

    shanks = {
        str(shank): {
            # The air boundary is never the surface, so it goes last.
            "candidates": sorted(
                found_on_shank,
                key=lambda c: (c["boundary"] == "air", -c["delta_bic"]),
            )
        }
        for shank, found_on_shank in sorted(candidates.items())
    }
    for shank in np.unique(channel_table.shank_ind):
        shanks.setdefault(str(int(shank)), {"candidates": []})

    return SurfaceLineOutput(
        np.concatenate(amps).astype(np.float32),
        np.array([line["freq_hz"] for line in lines], dtype=np.float64),
        {
            "version": "1.0",
            "depth_frame": "channels.localCoordinates y, um",
            "shanks": dict(sorted(shanks.items(), key=lambda kv: int(kv[0]))),
            "lines": lines,
        },
        {
            "method": "unreferenced_spectral_excess",
            "cmr": False,
            "parameters": asdict(line_params),
            "driven_tone_hz": driven_tone_hz,
            "amplitude_units": "V rms",
            "row_order": "channels.localCoordinates",
            "ap_rejection_band_min_hz": 300.0,
            "blocks": [block_provenance(r.context, "ap") for r in results],
        },
    )


def _write_lines(output_folder: Path, result: SurfaceLineOutput) -> None:
    """Save the declared line arrays, candidate JSON, and method metadata."""
    surface = json.dumps(result.surface, indent=2) + "\n"
    metadata = json.dumps(result.metadata, indent=2) + "\n"
    np.save(output_folder / _FILES[0], result.amplitudes)
    np.save(output_folder / _FILES[1], result.frequencies)
    (output_folder / _FILES[2]).write_text(surface)
    (output_folder / _FILES[3]).write_text(metadata)


def brain_surface_lines(
    *,
    params: LineParams | None = None,
    driven_tone_hz: float | None = None,
) -> Metric:
    """Configure opt-in interference amplitudes and surface candidates.

    Pass this configuration in ``extract_continuous(extra_metrics=(... ,))``.
    Lines are measured on unreferenced AP data and pooled across compatible
    frequency grids. Candidates include shared line transitions, the air
    boundary, plateau onset, and optionally the declared driven tone.
    No candidate is automatically selected as the brain surface.
    """
    if driven_tone_hz is not None and (
        not np.isfinite(driven_tone_hz) or driven_tone_hz <= 0
    ):
        raise ValueError("Driven tone frequency must be finite and positive")
    params = LineParams() if params is None else params
    return Metric(
        name="brainSurfaceLines",
        define=partial(_define_lines, params=params),
        reduce=partial(
            _reduce_lines,
            line_params=params,
            driven_tone_hz=driven_tone_hz,
        ),
        write=_write_lines,
        output_files=_FILES,
    )
