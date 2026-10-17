"""Authoritative read plans, simple identities, and pure channel reductions."""

import numpy as np
import pytest

from aind_ephys_ibl_gui_conversion.io import _save_extra_metrics
from aind_ephys_ibl_gui_conversion.metric_extensions import (
    ChannelMeanResult,
    ChannelOutput,
    MetricBlockResult,
    StreamContext,
    channel_mean,
    lfp_band_power,
    validate_metrics,
)
from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from aind_ephys_ibl_gui_conversion.types import ChannelTable
from tests.test_window_pipeline import make_block


def table_for(context):
    """Build the existing channel-table shape from native test metadata."""
    return ChannelTable(
        np.arange(6),
        np.arange(6).astype(str),
        context.ap.locations,
        context.ap.shank_indices,
    )


def test_context_records_actual_clamped_lfp_reads(monkeypatch):
    """Metadata and trace access share one integer interval plan."""
    block = make_block("np1")
    block.source = None
    block.lfp_recording = block.lfp_recording.frame_slice(0, 3000)
    calls = []
    original = block.lfp_recording.get_traces

    def counted(**kwargs):
        calls.append((kwargs["start_frame"], kwargs["end_frame"]))
        return original(**kwargs)

    monkeypatch.setattr(block.lfp_recording, "get_traces", counted)
    result = _compute_all_metrics(
        block, window_duration=2, main_recording_min_secs=5
    )
    context = result.metric_context
    assert context.block_source == "main"
    assert sorted(calls) == [(0, 2000), (1000, 3000)]
    np.testing.assert_array_equal(context.lfp.window_starts, [0, 1000])
    assert (
        context.block_id
        != _compute_all_metrics(
            block, window_duration=2
        ).metric_context.block_id
    )


def test_named_instances_share_fft_and_have_independent_outputs(tmp_path):
    """Two configurations need only explicit names, without schemas."""
    metrics = (
        lfp_band_power(30, 100, name="betaPower"),
        lfp_band_power(100, 200, name="gammaPower"),
    )
    result = _compute_all_metrics(
        make_block("np2"), window_duration=2, extra_metrics=metrics
    )
    assert sum(result.metric_stats.fft_calls.values()) == 2
    _save_extra_metrics(
        tmp_path, table_for(result.metric_context), [result], metrics
    )
    for metric in metrics:
        assert all((tmp_path / file).exists() for file in metric.output_files)
    assert not np.array_equal(
        np.load(tmp_path / metrics[0].output_files[0]),
        np.load(tmp_path / metrics[1].output_files[0]),
    )


@pytest.mark.parametrize(
    "metrics, message",
    [
        (
            (
                lfp_band_power(name="sharedPower"),
                channel_mean(
                    name="anotherMean",
                    attribute="sharedPower",
                    source="ap",
                    compute=lambda traces: np.mean(traces, axis=0),
                    units="uV",
                ),
            ),
            "collision",
        ),
        ((lfp_band_power(name="localCoordinates"),), "reserved"),
        ((lfp_band_power(name="../escape"),), "identifier"),
    ],
)
def test_collisions_and_reserved_names_fail_before_reads(
    monkeypatch, metrics, message
):
    """Plan errors cannot overwrite another metric or channel metadata."""
    block = make_block("np2")

    def forbidden(**kwargs):
        raise AssertionError("Trace read started")

    monkeypatch.setattr(block.recording, "get_traces", forbidden)
    with pytest.raises(ValueError, match=message):
        _compute_all_metrics(block, window_duration=2, extra_metrics=metrics)


def test_pure_reduction_checks_shape_without_destination():
    """Reduction needs no paths and rejects malformed channel results."""
    result = _compute_all_metrics(make_block("np2"), window_duration=2)
    context = result.metric_context
    malformed = ChannelMeanResult(np.zeros(5), np.ones(5), input_source="lfp")
    with pytest.raises(ValueError, match="shape"):
        lfp_band_power().reduce(
            StreamContext(table_for(context)),
            (MetricBlockResult(context, malformed),),
        )
    with pytest.raises(ValueError, match="shape"):
        ChannelOutput(np.zeros(2), np.ones(3), "Hz", {}).validate(2)
    with pytest.raises(ValueError, match="duration"):
        ChannelOutput(np.zeros(2), -np.ones(2), "Hz", {}).validate(2)


@pytest.mark.parametrize(
    "filename",
    [
        "band_corr",
        "_iblqc_ephysTimeRmsAPMain.rms.npy",
        "channels.rawInd.npy",
    ],
)
def test_existing_outputs_are_reserved(filename):
    """Protect existing artifacts beyond the common channel writer."""
    from types import SimpleNamespace

    metric = SimpleNamespace(name="custom", output_files=(filename,))
    with pytest.raises(ValueError, match="reserved"):
        validate_metrics((metric,))


def test_new_ephys_objects_are_allowed():
    """Reservation protects owned objects without claiming the namespace."""
    from types import SimpleNamespace

    metric = SimpleNamespace(
        name="custom", output_files=("_iblqc_ephysCustom.value.npy",)
    )
    validate_metrics((metric,))
