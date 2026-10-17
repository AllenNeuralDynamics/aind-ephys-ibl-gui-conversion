"""Check inferred sampling, channel identity, and preparation provenance."""

import json
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from aind_ephys_ibl_gui_conversion.io import _save_extra_metrics
from aind_ephys_ibl_gui_conversion.metric_context import SampleBudget
from aind_ephys_ibl_gui_conversion.metric_extensions import (
    ChannelComputation,
    MetricBlockResult,
    StreamContext,
    channel_mean,
    lfp_band_power,
    reduce_channel_mean,
)
from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from aind_ephys_ibl_gui_conversion.types import ChannelTable
from aind_ephys_ibl_gui_conversion.window_pipeline import RawRequest
from tests.test_window_pipeline import make_block


def mean_absolute(traces):
    """Calculate without retaining the supplied native traces."""
    return np.mean(np.abs(traces), axis=0, dtype=np.float64)


def save_mean(tmp_path, result, metric, source):
    """Write through the real stream adapter using the source's geometry."""
    table = ChannelTable(
        np.arange(source.n_channels),
        source.contact_ids,
        source.locations,
        source.shank_indices,
    )
    _save_extra_metrics(tmp_path, table, [result], (metric,))
    return (
        np.load(tmp_path / metric.output_files[0]),
        np.load(tmp_path / metric.output_files[1]),
        json.loads((tmp_path / metric.output_files[2]).read_text()),
    )


def test_prepared_mean_infers_lf_geometry_duration_and_clamped_offsets(
    tmp_path,
):
    """A short reordered LF input uses its own columns, rate, and intervals."""
    block = make_block("np1")
    block.lfp_recording = block.lfp_recording.select_channels([3, 1, 5])
    block.lfp_recording = block.lfp_recording.frame_slice(0, 3000)
    preparations = []

    def prepare(context):
        preparations.append(context.block_id)
        return ChannelComputation(
            RawRequest("lfp", n_samples=500, offset=250),
            mean_absolute,
            window_indices=(1,),
            metadata={
                "method": "absolute voltage",
                "sample_intervals": [[-1, -1]],
                "sampling_frequency": -1,
            },
        )

    metric = channel_mean(name="lfOffset", prepare=prepare, units="uV")
    result = _compute_all_metrics(
        block, window_duration=2, extra_metrics=(metric,)
    )
    source = result.metric_context.lfp
    values, seconds, details = save_mean(tmp_path, result, metric, source)
    traces = block.lfp_recording.get_traces(
        start_frame=1250, end_frame=1750, return_in_uV=True
    )
    np.testing.assert_allclose(values, mean_absolute(traces))
    np.testing.assert_array_equal(seconds, 0.5)
    assert len(preparations) == 1
    assert details["input_source"] == "lfp"
    provenance = details["blocks"][0]
    assert provenance["method"] == "absolute voltage"
    assert provenance["sampling_frequency"] == 1000
    assert provenance["sample_intervals"] == [[1250, 1750]]
    assert provenance["analyzed_intervals"] == [[1250, 1750]]
    assert provenance["analyzed_seconds"] == 0.5
    assert provenance["window_indices"] == [1]
    assert provenance["read_start_samples"] == [0, 1000]
    assert provenance["request"]["offset"] == 250


def test_snippet_plan_derives_core_duration_and_provenance(tmp_path):
    """Padding is excluded from exposure while the callback trims its core."""
    budget = SampleBudget(max_windows=3, seconds=1, edge_seconds=0.1)

    def compute(traces, *, edge):
        return mean_absolute(traces[edge:-edge])

    def prepare(context):
        plan = budget.resolve(context.ap)
        return ChannelComputation(
            plan, partial(compute, edge=plan.edge_samples)
        )

    block = make_block("np2")
    metric = channel_mean(name="core", prepare=prepare, units="uV")
    result = _compute_all_metrics(
        block, window_duration=2, extra_metrics=(metric,)
    )
    source = result.metric_context.ap
    values, seconds, details = save_mean(tmp_path, result, metric, source)
    core_traces = np.concatenate(
        [
            block.recording.get_traces(
                start_frame=start, end_frame=end, return_in_uV=True
            )
            for start, end in ((1200, 2800), (5200, 6800))
        ]
    )
    np.testing.assert_allclose(values, mean_absolute(core_traces))
    np.testing.assert_allclose(seconds, 1.6)
    provenance = details["blocks"][0]
    assert provenance["sample_intervals"] == [[1000, 3000], [5000, 7000]]
    assert provenance["analyzed_intervals"] == [[1200, 2800], [5200, 6800]]
    assert provenance["analyzed_seconds"] == 1.6


def test_ap_fft_duration_uses_transform_length_not_parent_read(tmp_path):
    """An NP1 AP transform covers less time than its surrounding raw read."""
    metric = channel_mean(
        name="apSpectrum",
        units="arbitrary",
        prepare=lambda context: ChannelComputation(
            context.ap_fft,
            lambda spectrum: np.zeros(spectrum.coefficients.shape[1]),
        ),
    )
    result = _compute_all_metrics(
        make_block("np1"), window_duration=2, extra_metrics=(metric,)
    )
    source = result.metric_context.ap
    _, seconds, details = save_mean(tmp_path, result, metric, source)
    np.testing.assert_array_equal(seconds, 2)
    provenance = details["blocks"][0]
    assert provenance["read_samples"] == 4000
    assert provenance["request"]["n_samples"] == 2000
    assert provenance["sample_intervals"] == [[0, 2000], [4000, 6000]]
    assert provenance["analyzed_seconds"] == 2


@pytest.mark.parametrize("kind", ["np1", "np2"])
def test_fft_preparation_records_transform_and_support_without_extra_work(
    kind, tmp_path
):
    """Shared FFTs supply their native source, duration, and preprocessing."""
    metric = lfp_band_power(window_indices=(1,))
    block = make_block(kind)
    baseline = _compute_all_metrics(block, window_duration=2)
    result = _compute_all_metrics(
        block, window_duration=2, extra_metrics=(metric,)
    )
    source = result.metric_context.lfp
    _, seconds, details = save_mean(tmp_path, result, metric, source)
    np.testing.assert_array_equal(seconds, 2)
    assert result.metric_stats.fft_calls == baseline.metric_stats.fft_calls
    assert details["band_hz"] == [30, 100]
    provenance = details["blocks"][0]
    assert provenance["input_source"] == "lfp"
    assert provenance["request"]["reference"] == "median"
    assert provenance["request"]["taper"] == "hann"
    assert provenance["window_indices"] == [1]
    assert provenance["analyzed_seconds"] == 2


@pytest.mark.parametrize(
    "bad_input,message",
    [
        ("outside", "exceeds"),
        ("duplicate", "window index"),
        ("stale_start", "does not match"),
        ("stale_rate", "does not match"),
        ("ambiguous", "already selects"),
        ("empty_core", "analyzed core"),
    ],
)
def test_preparation_mistakes_fail_before_reads(
    monkeypatch, bad_input, message
):
    """Input declarations validate their own intervals and block identity."""
    block = make_block("np2")

    def forbidden(*args, **kwargs):
        raise AssertionError("Trace read started")

    monkeypatch.setattr(block.recording, "get_traces", forbidden)

    def prepare(context):
        if bad_input == "outside":
            return ChannelComputation(
                RawRequest("ap", context.ap.read_samples, offset=1),
                mean_absolute,
            )
        if bad_input == "duplicate":
            return ChannelComputation(
                RawRequest("ap", 100), mean_absolute, window_indices=(0, 0)
            )
        plan = SampleBudget(seconds=1, edge_seconds=0).resolve(context.ap)
        if bad_input == "stale_start":
            plan = replace(
                plan, starts=tuple(start + 1 for start in plan.starts)
            )
        elif bad_input == "stale_rate":
            plan = replace(plan, sampling_frequency=100)
        elif bad_input == "empty_core":
            plan = replace(plan, edge_samples=plan.request.n_samples)
        return ChannelComputation(
            plan,
            mean_absolute,
            window_indices=(0,) if bad_input == "ambiguous" else None,
        )

    metric = channel_mean(name="badInput", prepare=prepare, units="uV")
    with pytest.raises(ValueError, match=message):
        _compute_all_metrics(block, window_duration=2, extra_metrics=(metric,))


@pytest.mark.parametrize("conflict", ["source", "compute", "window_indices"])
def test_raw_and_prepared_forms_are_explicit(conflict):
    """Preparation owns its input selection without conflicting arguments."""
    arguments = {
        "source": "ap",
        "compute": mean_absolute,
        "window_indices": (0,),
    }
    with pytest.raises(ValueError, match="preparation or raw"):
        channel_mean(
            name="ambiguous",
            units="uV",
            prepare=lambda _: None,
            **{conflict: arguments[conflict]},
        )


def test_reduction_rejects_source_override_and_preserves_empty_rows():
    """An explicit reducer source cannot contradict a prepared result."""
    metric = lfp_band_power()
    result = _compute_all_metrics(
        make_block("np1"), window_duration=2, extra_metrics=(metric,)
    )
    context = result.metric_context
    source = context.lfp
    table = ChannelTable(
        np.arange(source.n_channels),
        source.contact_ids,
        source.locations,
        source.shank_indices,
    )
    results = (MetricBlockResult(context, result.metric_results[metric.name]),)
    with pytest.raises(ValueError, match="does not match prepared"):
        reduce_channel_mean(
            StreamContext(table), results, source="ap", units="uV"
        )
    empty = metric.reduce(StreamContext(table), ())
    assert np.isnan(empty.values).all()
    assert not empty.sampled_seconds.any()
