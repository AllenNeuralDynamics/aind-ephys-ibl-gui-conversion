"""Validate the ALF and QC joins consumed by the alignment GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def audit_collection(folder: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Check the consumer-facing joins and preserve quantitative evidence."""
    load = lambda name: np.load(folder / name, mmap_mode="r")  # noqa: E731
    times = load("spikes.times.npy")
    clusters = load("spikes.clusters.npy")
    channels = load("clusters.channels.npy")
    coordinates = load("channels.localCoordinates.npy")
    np.testing.assert_array_equal(
        load("channels.rawInd.npy"), np.arange(len(coordinates))
    )
    assert load("channels.contactId.npy").shape == (len(coordinates),)
    unit_shanks = load("unit_shank_indices.npy")
    channel_shanks = load("channels.shankInd.npy")
    spike_shanks = load("spike_shank_indices.npy")
    metrics = pd.read_csv(folder / "clusters.metrics.csv")
    duration = plan["main_segment"]["duration_s"]
    assert times.ndim == 1 and np.isfinite(times).all()
    assert np.all(np.diff(times) >= 0)
    assert not len(times) or (times[0] >= 0 and times[-1] < duration)
    assert len(times) == len(clusters) == len(spike_shanks)
    for name in ("amps", "depths"):
        assert load(f"spikes.{name}.npy").shape == times.shape
    assert len(channels) == len(unit_shanks) == len(metrics)
    assert load("clusters.peakToTrough.npy").shape == (len(channels),)
    waveforms = load("clusters.waveforms.npy")
    assert waveforms.ndim == 3 and waveforms.shape[0] == len(channels)
    assert load("clusters.waveformTimes.npy").shape == (waveforms.shape[1],)
    np.testing.assert_array_equal(
        load("clusters.waveformChannels.npy"), channels[:, None]
    )
    assert metrics.cluster_id.tolist() == list(range(len(channels)))
    assert metrics[["analyzer_id", "source_unit_id"]].duplicated().sum() == 0
    assert not len(clusters) or (
        clusters.min() >= 0 and clusters.max() < len(channels)
    )
    assert np.all((channels >= 0) & (channels < len(coordinates)))
    _, local_channel_shanks = np.unique(channel_shanks, return_inverse=True)
    np.testing.assert_array_equal(unit_shanks, local_channel_shanks[channels])
    np.testing.assert_array_equal(spike_shanks, unit_shanks[clusters])
    coverage = plan["shank_coverage"]
    physical_unit_shanks = channel_shanks[channels]
    assert set(physical_unit_shanks) <= set(coverage["sorted"])
    assert not set(physical_unit_shanks) & set(coverage["missing"])
    expected = sum(a["unit_count"] for a in plan["analyzers"])
    assert len(channels) == expected
    rows = json.loads((folder / "band_corr/row_channels.json").read_text())
    assert rows["matrix_rows"] == list(range(len(coordinates)))
    for shank, entry in rows["shanks"].items():
        assert np.all(channel_shanks[entry["rows"]] == int(shank))
    for path in (folder / "band_corr").glob("*_mean_corr.npy"):
        if "_shank" not in path.name:
            assert (
                np.load(path, mmap_mode="r").shape == (len(coordinates),) * 2
            )
    receipt = json.loads((folder / "ephys_extraction.json").read_text())
    assert receipt["status"] == "complete"
    assert sum(b["selected_main"] for b in receipt["blocks"]) == 1
    return dict(
        status="passed",
        spike_count=len(times),
        unit_count=len(channels),
        channel_rows=len(coordinates),
        unit_shanks=np.unique(unit_shanks).tolist(),
        physical_unit_shanks=np.unique(physical_unit_shanks).tolist(),
        channel_shanks=np.unique(channel_shanks).tolist(),
        shank_coverage=coverage,
        main_segment=plan["main_segment"],
        first_spike_s=float(times[0]) if len(times) else None,
        last_spike_s=float(times[-1]) if len(times) else None,
        qc_blocks=receipt["blocks"],
        skipped_blocks=receipt["skipped_blocks"],
    )
