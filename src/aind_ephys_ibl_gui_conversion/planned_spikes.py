"""Extract only selected main analyzers into one consistent ALF index space."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aind_ephys_ibl_gui_conversion.planned_extraction import (
    PlannedExtraction,
    analyzer_rows,
)
from aind_ephys_ibl_gui_conversion.spikes import _quality_metrics_with_labels


def _analyzer_arrays(
    analyzer: Any, rows: np.ndarray, shanks: np.ndarray
) -> dict[str, np.ndarray]:
    """Keep extension arrays aligned to the analyzer's dense spike vector."""
    vector = analyzer.sorting.to_spike_vector()
    samples, units = vector["sample_index"], vector["unit_index"]
    if (
        np.any(vector["segment_index"] != 0)
        or np.any(samples < 0)
        or np.any(samples >= analyzer.get_num_samples(0))
        or np.any(units < 0)
        or np.any(units >= len(analyzer.unit_ids))
    ):
        raise ValueError("Spike vector is outside planned sample/unit space")
    amps = np.asarray(analyzer.get_extension("spike_amplitudes").get_data())
    depths = np.asarray(
        analyzer.get_extension("spike_locations").get_data()["y"]
    )
    if amps.shape != samples.shape or depths.shape != samples.shape:
        raise ValueError("Spike extensions do not align with spike vector")
    template_ext = analyzer.get_extension("templates")
    templates = template_ext.get_templates(save=False)
    if (
        templates.ndim != 3
        or templates.shape[0] != len(analyzer.unit_ids)
        or templates.shape[2] != len(rows)
    ):
        raise ValueError("Templates do not use dense analyzer channel order")
    peaks = np.argmax(np.ptp(templates, axis=1), axis=1)
    waveforms = templates[np.arange(len(peaks)), :, peaks]
    peak_rows = rows[peaks]
    unit_shanks = shanks[peak_rows]
    trough_interval = (
        np.argmax(waveforms, axis=1) - np.argmin(waveforms, axis=1)
    ) / analyzer.sampling_frequency
    return dict(
        times=samples / analyzer.sampling_frequency,
        clusters=units,
        amps=-amps.astype(float),
        depths=depths,
        spike_shanks=unit_shanks[units],
        channels=peak_rows,
        unit_shanks=unit_shanks,
        waveforms=waveforms[:, :, None],
        peak_to_trough=trough_interval,
    )


def _merge_waveforms(
    chunks: list[dict[str, np.ndarray]],
    analyzers: tuple[tuple[dict[str, Any], Any, Path], ...],
) -> np.ndarray:
    """Align varying template windows at sample zero and pad absent samples."""
    before = [
        int(a.get_extension("templates").nbefore) for _, a, _ in analyzers
    ]
    common_before = max(before)
    after = max(c["waveforms"].shape[1] - b for c, b in zip(chunks, before))
    for chunk, original_before in zip(chunks, before):
        waveform = chunk["waveforms"]
        aligned = np.full((len(waveform), common_before + after, 1), np.nan)
        start = common_before - original_before
        aligned[:, start : start + waveform.shape[1], :] = waveform
        chunk["waveforms"] = aligned
    rate = float(analyzers[0][1].sampling_frequency)
    return (np.arange(common_before + after) - common_before) / rate


def extract_planned_spikes(execution: PlannedExtraction) -> None:
    """Write global unit rows and chronologically aligned spike arrays.

    Waveforms retain the peak channel as their single channel axis, with its
    canonical row saved explicitly. This avoids combining unrelated group-local
    waveform channel positions across analyzers.
    """
    chunks, metrics = [], []
    offset = 0
    for reference, analyzer, path in execution.analyzers:
        rows = analyzer_rows(analyzer, execution.channel_table)
        chunk = _analyzer_arrays(
            analyzer, rows, execution.channel_table.shank_ind
        )
        chunk["clusters"] = chunk["clusters"] + offset
        frame = _quality_metrics_with_labels(
            analyzer, path.parent.parent / "curated" / path.name
        )
        if not frame.index.is_unique or set(frame.index) != set(
            analyzer.unit_ids
        ):
            raise ValueError("Quality metrics do not match analyzer unit IDs")
        frame = frame.reindex(analyzer.unit_ids)
        frame["analyzer_id"] = reference["id"]
        frame["analyzer_group"] = reference["group"]
        frame["source_unit_id"] = analyzer.unit_ids
        frame.index = pd.RangeIndex(
            offset, offset + len(frame), name="cluster_id"
        )
        metrics.append(frame)
        offset += len(analyzer.unit_ids)
        chunks.append(chunk)

    waveform_times = _merge_waveforms(chunks, execution.analyzers)
    # One permutation applies to every per-spike measurement and label.
    merged = {
        key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]
    }
    order = np.argsort(merged["times"], kind="stable")
    output = execution.stream.output_folder
    output.mkdir(parents=True, exist_ok=True)
    for name in ("times", "clusters", "amps", "depths"):
        values = merged[name][order]
        if name == "clusters":
            values = values.astype(np.uint32)
        np.save(output / f"spikes.{name}.npy", values)
    np.save(output / "spike_shank_indices.npy", merged["spike_shanks"][order])
    np.save(output / "unit_shank_indices.npy", merged["unit_shanks"])
    np.save(output / "clusters.channels.npy", merged["channels"])
    np.save(
        output / "clusters.waveformChannels.npy", merged["channels"][:, None]
    )
    np.save(output / "clusters.waveforms.npy", merged["waveforms"])
    np.save(output / "clusters.waveformTimes.npy", waveform_times)
    np.save(output / "clusters.peakToTrough.npy", merged["peak_to_trough"])
    pd.concat(metrics).to_csv(output / "clusters.metrics.csv")
