"""Combined activity and surface-line registration through stream output."""

import json

import numpy as np
import pytest

from aind_ephys_ibl_gui_conversion.activity import raw_activity
from aind_ephys_ibl_gui_conversion.io import _assemble_and_save_stream
from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from aind_ephys_ibl_gui_conversion.surface_lines import brain_surface_lines
from aind_ephys_ibl_gui_conversion.types import ExperimentBlock, ProbeStream
from tests.test_lines import (
    N_CHANNELS,
    _recording,
    _stepped_rms,
    _traces,
)


@pytest.mark.parametrize("separate_lfp", [False, True])
def test_activity_and_surface_match_separate_runs(
    tmp_path, monkeypatch, separate_lfp
):
    """Both consumers share reads and preserve independent saved results."""
    ap = _recording(
        _traces(8.0, _stepped_rms(), second_rms_uv=_stepped_rms() * 0.6)
    )
    lfp = None
    if separate_lfp:
        noise = np.random.default_rng(5).normal(0, 10, (20000, N_CHANNELS))
        lfp = _recording(noise.astype(np.float32), fs=2500.0)
    block = ExperimentBlock(ap, lfp, 0, source="main")
    reads = []

    def count_reader(reader, source):
        """Wrap a recording read without changing its samples."""

        def counted(**kwargs):
            """Record each requested frame interval before reading."""
            reads.append((source, kwargs["start_frame"], kwargs["end_frame"]))
            return reader(**kwargs)

        return counted

    for source, recording in (("ap", ap), ("lfp", lfp)):
        if recording is not None:
            monkeypatch.setattr(
                recording,
                "get_traces",
                count_reader(recording.get_traces, source),
            )

    surface = brain_surface_lines()
    activity = raw_activity()
    results, schedules = {}, {}
    for name, metrics in (
        ("surface", (surface,)),
        ("activity", (activity,)),
        ("combined", (activity, surface)),
    ):
        reads.clear()
        result = _compute_all_metrics(
            block, extra_metrics=metrics, block_id="combined-test"
        )
        results[name] = result
        schedules[name] = sorted(reads)
        stream = ProbeStream("s", "ProbeA", [block], tmp_path / name)
        _assemble_and_save_stream(
            stream, [result], 30.0, 4.0, extra_metrics=metrics
        )

    assert schedules["combined"] == schedules["surface"]
    assert schedules["combined"] == schedules["activity"]
    assert (
        results["combined"].metric_stats.fft_calls
        == results["surface"].metric_stats.fft_calls
    )
    for attribute in ("rms_ap", "rms_lfp", "psd_power", "psd_freqs"):
        np.testing.assert_array_equal(
            getattr(results["combined"], attribute),
            getattr(results["surface"], attribute),
        )
    for band in results["surface"].coherency:
        np.testing.assert_array_equal(
            results["combined"].coherency[band],
            results["surface"].coherency[band],
        )
    for name, metric in (("surface", surface), ("activity", activity)):
        for filename in metric.output_files:
            standalone = tmp_path / name / filename
            combined = tmp_path / "combined" / filename
            if filename.endswith(".npy"):
                np.testing.assert_array_equal(
                    np.load(standalone), np.load(combined)
                )
            else:
                assert json.loads(standalone.read_text()) == json.loads(
                    combined.read_text()
                )
