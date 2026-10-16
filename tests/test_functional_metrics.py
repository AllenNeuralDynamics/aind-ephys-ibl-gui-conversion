"""Exercise function composition, block-local preparation, and shared I/O."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import pytest

from aind_ephys_ibl_gui_conversion.io import _save_extra_metrics
from aind_ephys_ibl_gui_conversion.metric_extensions import (
    ChannelMeanAccumulator,
    Metric,
    channel_mean,
    channel_metric,
    reduce_channel_mean,
)
from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from aind_ephys_ibl_gui_conversion.types import ChannelTable
from aind_ephys_ibl_gui_conversion.window_pipeline import (
    MetricDefinition,
    RawRequest,
)
from tests.test_window_pipeline import make_block


def test_preparation_is_block_local_and_multi_input_functions_borrow():
    """Each block binds its own computation and calls it once per window."""
    definitions = []
    calls = []

    def prepare(context):
        offset = context.block_index + 1

        def compute(first, second):
            assert not first.flags.writeable
            assert not second.flags.writeable
            calls.append((offset, first.shape))
            return np.mean(np.abs(first), axis=0) * offset

        source = context.ap
        definition = MetricDefinition(
            name="prepared",
            inputs=(
                RawRequest("ap", source.read_samples),
                RawRequest("ap", source.read_samples, dtype="float64"),
            ),
            create=partial(
                ChannelMeanAccumulator,
                n_channels=source.n_channels,
                seconds=source.read_samples / source.sampling_frequency,
                compute=compute,
            ),
        )
        definitions.append(definition)
        return definition

    metric = channel_metric(
        name="prepared",
        define=prepare,
        reduce=partial(reduce_channel_mean, source="ap", units="uV"),
    )
    first, second = make_block("np2"), make_block("np2")
    second.block_index = 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda block: _compute_all_metrics(
                    block, window_duration=2, extra_metrics=(metric,)
                ),
                (first, second),
            )
        )
    assert len(definitions) == 2
    assert sorted(offset for offset, _ in calls) == [1, 1, 2, 2]
    assert definitions[0].create() is not definitions[0].create()
    np.testing.assert_allclose(
        results[1].metric_results[metric.name].weighted_sum,
        results[0].metric_results[metric.name].weighted_sum * 2,
    )
    np.testing.assert_array_equal(
        results[1].metric_results[metric.name].sampled_seconds,
        results[0].metric_results[metric.name].sampled_seconds,
    )


def test_custom_accumulator_reduction_and_writer_need_no_base_class(tmp_path):
    """A stateful estimator can supply all protocol operations directly."""

    class PeakAccumulator:
        def __init__(self, n_channels):
            self.peak = np.zeros(n_channels)

        def update(self, window_index, inputs):
            self.peak = np.maximum(self.peak, np.abs(inputs[0]).max(axis=0))

        def finish(self):
            return self.peak

    def prepare(context):
        return MetricDefinition(
            "peak",
            (RawRequest("ap", context.ap.read_samples),),
            partial(PeakAccumulator, context.ap.n_channels),
        )

    def reduce(context, results):
        peaks = np.zeros(len(context.channel_table.raw_ind))
        for result in results:
            rows = context.channel_rows(result.context.ap)
            np.maximum.at(peaks, rows, result.value)
        return peaks

    def write(folder, result):
        np.save(folder / "peak.npy", result)

    metric = Metric("peak", prepare, reduce, write, ("peak.npy",))
    result = _compute_all_metrics(
        make_block("np2"), window_duration=2, extra_metrics=(metric,)
    )
    source = result.metric_context.ap
    table = ChannelTable(
        np.arange(source.n_channels),
        source.contact_ids,
        source.locations,
        source.shank_indices,
    )
    _save_extra_metrics(tmp_path, table, [result], (metric,))
    np.testing.assert_array_equal(
        np.load(tmp_path / "peak.npy"), result.metric_results[metric.name]
    )


def test_raw_callable_shape_and_source_fail_early():
    """Invalid input sources and output shapes fail with useful errors."""
    with pytest.raises(ValueError, match="source"):
        channel_mean(name="bad", source="unknown", compute=np.mean, units="uV")

    def compute(traces):
        return np.array(1.0)

    metric = channel_mean(
        name="badShape", source="ap", compute=compute, units="uV"
    )
    with pytest.raises(ValueError, match="shape"):
        _compute_all_metrics(
            make_block("np2"), window_duration=2, extra_metrics=(metric,)
        )


def test_raw_function_metric_has_no_additional_recording_reads(monkeypatch):
    """Enabling a function adds arithmetic without adding recording access."""
    block = make_block("np2")
    original = block.recording.get_traces
    reads = []

    def read(*args, **kwargs):
        reads.append((kwargs.get("start_frame"), kwargs.get("end_frame")))
        return original(*args, **kwargs)

    monkeypatch.setattr(block.recording, "get_traces", read)
    baseline = _compute_all_metrics(block, window_duration=2)
    baseline_reads = sorted(reads)
    reads.clear()
    metric = channel_mean(
        name="meanAbsolute",
        source="ap",
        compute=lambda traces: np.mean(np.abs(traces), axis=0),
        units="uV",
    )
    result = _compute_all_metrics(
        block, window_duration=2, extra_metrics=(metric,)
    )
    assert sorted(reads) == baseline_reads
    assert result.metric_stats.fft_calls == baseline.metric_stats.fft_calls
