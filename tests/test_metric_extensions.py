"""Test resolved registration and physical-contact pooling through outputs."""

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
import spikeinterface as si

from aind_ephys_ibl_gui_conversion.ephys import extract_continuous
from aind_ephys_ibl_gui_conversion.io import _save_extra_metrics
from aind_ephys_ibl_gui_conversion.metric_extensions import (
    ChannelMeanAccumulator,
    ChannelMeanResult,
    StreamContext,
    band_power,
    channel_mean,
    lfp_band_power,
)
from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from aind_ephys_ibl_gui_conversion.types import ChannelTable, ProbeStream
from aind_ephys_ibl_gui_conversion.window_pipeline import Spectrum
from tests.test_window_pipeline import make_block


@pytest.mark.parametrize("kind", ["np1", "np2"])
def test_context_snapshots_resolved_schedule_and_native_geometry(kind):
    """Registration exposes the actual schedule with independent metadata."""
    block = make_block(kind)
    result = _compute_all_metrics(block, window_duration=2)
    context = result.metric_context
    assert context.n_windows == len(result.timestamps) == 2
    assert context.separate_lfp == (kind == "np1")
    np.testing.assert_array_equal(context.ap.window_starts, [0, 4000])
    np.testing.assert_array_equal(
        context.lfp.window_starts, [0, 2000 if kind == "np1" else 4000]
    )
    assert context.ap_fft.n_samples == (2000 if kind == "np1" else 4000)
    assert not hasattr(context.ap, "recording")
    np.testing.assert_array_equal(
        context.ap.locations, block.recording.get_channel_locations()
    )
    with pytest.raises(FrozenInstanceError):
        context.block_index = 5
    for array in (
        context.ap.locations,
        context.timestamps,
        context.ap.channel_ids,
    ):
        assert not array.flags.writeable
    block.recording.set_property("location", np.zeros((6, 2)))
    assert context.ap.locations.max() > 0


@pytest.mark.parametrize("kind", ["np1", "np2"])
def test_opt_in_band_power_matches_existing_psd_without_extra_fft(kind):
    """The statistic integrates the same PSD and shares exact transforms."""
    result = _compute_all_metrics(
        make_block(kind), window_duration=2, extra_metrics=(lfp_band_power(),)
    )
    value = result.metric_results[lfp_band_power().name]
    mask = (result.psd_freqs >= 30) & (result.psd_freqs < 100)
    expected = (
        result.psd_power[mask].sum(axis=0) * np.diff(result.psd_freqs)[0]
    )
    actual = value.weighted_sum / value.sampled_seconds * 1e-12
    np.testing.assert_allclose(actual, expected, rtol=3e-6)
    np.testing.assert_array_equal(value.sampled_seconds, 4)
    assert sum(result.metric_stats.fft_calls.values()) == (
        4 if kind == "np1" else 2
    )


def test_band_power_matches_known_sinusoid_power():
    """Hann-corrected band power recovers A²/2 for a well-resolved sinusoid."""
    fs = 1000
    time = np.arange(6000) / fs
    gains = np.array([-2, -1, 1, 2])
    traces = np.sin(2 * np.pi * 50 * time[:, None]) * gains
    traces += np.sin(2 * np.pi * 200 * time[:, None]) * gains * 3
    recording = si.NumpyRecording(traces.astype(np.float32), fs)
    recording.set_dummy_probe_from_locations(
        np.c_[np.zeros(4), np.arange(4) * 20]
    )
    recording.set_channel_gains(1)
    recording.set_channel_offsets(0)
    block = replace(make_block("np2"), recording=recording)
    result = _compute_all_metrics(
        block, window_duration=2, extra_metrics=(lfp_band_power(40, 60),)
    )
    value = result.metric_results[lfp_band_power().name]
    np.testing.assert_allclose(
        value.weighted_sum / value.sampled_seconds, gains**2 / 2, rtol=2e-6
    )


def test_nonfinite_channel_has_no_exposure_and_subset_is_respected():
    """Undefined power never contributes a valid zero or a denominator."""
    from functools import partial

    accumulator = ChannelMeanAccumulator(
        n_channels=2,
        seconds=1,
        compute=partial(band_power, mask=np.array([False, True]), scale=0.5),
    )
    coefficients = np.array([[0, 0], [2, np.nan]], dtype=np.complex64)
    accumulator.update(0, (Spectrum(coefficients),))
    value = accumulator.finish()
    np.testing.assert_array_equal(value.weighted_sum, [2, 0])
    np.testing.assert_array_equal(value.sampled_seconds, [1, 0])
    result = _compute_all_metrics(
        make_block("np2"),
        window_duration=2,
        extra_metrics=(lfp_band_power(window_indices=(1,)),),
    )
    np.testing.assert_array_equal(
        result.metric_results[lfp_band_power().name].sampled_seconds, 2
    )


def test_pooling_uses_sampled_time_and_source_geometry(tmp_path):
    """Reordered LF contacts, overlap, and absent coverage keep row order."""
    first = _compute_all_metrics(make_block("np1"), window_duration=2)
    second = _compute_all_metrics(make_block("np1"), window_duration=2)
    # Native LF order is [20, 0] then [40, 20], independent of AP order.
    coordinates = np.array([[0, 20], [0, 0], [0, 40], [0, 60]])
    table = ChannelTable(
        np.arange(4),
        np.array(["a", "b", "c", "d"]),
        coordinates,
        np.zeros(4, dtype=int),
    )
    for result, locations, power, seconds in (
        (first, coordinates[[0, 1]], [20, 4], [2, 2]),
        (second, coordinates[[2, 0]], [120, 120], [4, 4]),
    ):
        source = replace(result.metric_context.lfp, locations=locations)
        result.metric_context = replace(result.metric_context, lfp=source)
        result.metric_results[lfp_band_power().name] = ChannelMeanResult(
            np.array(power), np.array(seconds), input_source="lfp"
        )
    _save_extra_metrics(tmp_path, table, [first, second], (lfp_band_power(),))
    np.testing.assert_allclose(
        np.load(tmp_path / "_iblqc_channels.lfpBandPower.npy"),
        np.array([140 / 6, 2, 30, np.nan]) * 1e-12,
    )
    np.testing.assert_array_equal(
        np.load(tmp_path / "_iblqc_channels.lfpBandPowerSampledDuration.npy"),
        [6, 2, 4, 0],
    )
    metadata = json.loads(
        (tmp_path / "_iblqc_lfpBandPower.metadata.json").read_text()
    )
    assert metadata["units"] == "V^2"
    assert len(metadata["blocks"]) == 2
    with pytest.raises(ValueError, match="absent"):
        StreamContext(table).channel_rows(
            replace(first.metric_context.lfp, locations=np.array([[0, 999]]))
        )


def test_invalid_registration_fails_before_recording_reads(monkeypatch):
    """Bad frequency bands and duplicate registrations fail during planning."""
    block = make_block("np2")

    def forbidden(*args, **kwargs):
        raise AssertionError("Trace read started")

    monkeypatch.setattr(block.recording, "get_traces", forbidden)
    for metrics, message in (
        ((lfp_band_power(0, 100),), "DC"),
        ((lfp_band_power(), lfp_band_power()), "unique"),
    ):
        with pytest.raises(ValueError, match=message):
            _compute_all_metrics(
                block, window_duration=2, extra_metrics=metrics
            )


def test_concurrent_extraction_registers_and_saves_optional_metric(
    monkeypatch, tmp_path
):
    """Public extraction carries extensions through workers and assembly.

    Omitting ``extra_metrics`` runs the defaults; ``()`` runs none.
    """
    import aind_ephys_ibl_gui_conversion.ephys as ephys

    blocks = [make_block("np2"), make_block("np2")]
    blocks[1].block_index = 0
    blocks[1].source = "surface"
    monkeypatch.setattr(
        ephys, "get_ecephys_stream_names", lambda _: (["probe"], tmp_path, 2)
    )

    def load_streams(*args, **kwargs):
        return [ProbeStream("probe", "probe", blocks, Path(args[3]) / "probe")]

    monkeypatch.setattr(ephys, "load_probe_streams", load_streams)
    monkeypatch.setattr(ephys.os, "cpu_count", lambda: 2)
    for name, metrics in (
        ("old", ()),
        ("default", None),
        (
            "new",
            (
                lfp_band_power(),
                channel_mean(
                    name="meanAbsolute",
                    source="ap",
                    compute=lambda traces: np.mean(np.abs(traces), axis=0),
                    units="uV",
                ),
            ),
        ),
    ):
        destination = tmp_path / name
        destination.mkdir()
        extract_continuous(
            tmp_path,
            destination,
            session_folder=tmp_path,
            rms_window_duration=2,
            num_parallel_jobs=2,
            extra_metrics=metrics,
        )
    old, new = tmp_path / "old" / "probe", tmp_path / "new" / "probe"
    assert not (old / "_iblqc_channels.lfpBandPower.npy").exists()
    assert not (old / "_iblqc_channels.rawActivity.npy").exists()
    default = tmp_path / "default" / "probe"
    assert np.load(default / "_iblqc_channels.rawActivity.npy").shape == (6,)
    assert not (new / "_iblqc_channels.rawActivity.npy").exists()
    for path in old.rglob("*.npy"):
        before, after = np.load(path), np.load(new / path.relative_to(old))
        if before.dtype.kind in "OUS":
            np.testing.assert_array_equal(before, after)
        else:
            np.testing.assert_allclose(before, after, rtol=3e-6, atol=2e-7)
    assert np.load(new / "_iblqc_channels.lfpBandPower.npy").shape == (6,)
    np.testing.assert_array_equal(
        np.load(new / "_iblqc_channels.lfpBandPowerSampledDuration.npy"), 8
    )
    metadata = json.loads(
        (new / "_iblqc_meanAbsolute.metadata.json").read_text()
    )
    assert metadata["units"] == "uV"
    assert [block["block_id"] for block in metadata["blocks"]] == [
        "probe/block-0",
        "probe/block-1",
    ]
    assert [block["source"] for block in metadata["blocks"]] == [
        "main",
        "surface",
    ]
    assert np.load(new / "_iblqc_channels.meanAbsolute.npy").shape == (6,)
