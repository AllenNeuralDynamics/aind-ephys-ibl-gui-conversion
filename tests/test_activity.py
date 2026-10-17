"""Activity events, physical geometry, bounded sampling, and reductions."""

from dataclasses import replace

import numpy as np
import pytest
import spikeinterface as si

from aind_ephys_ibl_gui_conversion.activity import (
    ActivityAccumulator,
    ActivityParams,
    ActivityResult,
    _health_mask,
    _neighbors,
    _supported_candidates,
    raw_activity,
)
from aind_ephys_ibl_gui_conversion.channel_metadata import build_channel_table
from aind_ephys_ibl_gui_conversion.io import _save_extra_metrics
from aind_ephys_ibl_gui_conversion.metric_context import (
    SampleBudget,
    SourceContext,
)
from aind_ephys_ibl_gui_conversion.metric_extensions import (
    MetricBlockResult,
    StreamContext,
)
from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from aind_ephys_ibl_gui_conversion.types import ChannelTable
from tests.test_window_pipeline import make_block


def reference_recording():
    """Seeded two-shank input with paired sites, bad noise, and 12 chunks."""
    fs, samples, chunks, channels = 2500, 625, 12, 32
    rng = np.random.default_rng(122)
    traces = rng.normal(0, 8, (samples * chunks, channels)).astype(np.float32)
    for chunk in range(chunks):
        for event in range(80, samples - 80, 90):
            start = chunk * samples + event
            traces[start, [4, 20]] -= 50
            traces[start, [5, 21]] -= 32
    traces[:, 8] *= 20
    groups = np.repeat([0, 1], 16)
    locations = np.c_[
        np.repeat([0, 250], 16) + np.tile([0, 32], 16),
        np.tile(np.repeat(np.arange(8) * 20, 2), 2),
    ]
    order = rng.permutation(channels)
    recording = si.NumpyRecording(traces[:, order], fs)
    recording.set_channel_gains(1)
    recording.set_channel_offsets(0)
    recording.set_dummy_probe_from_locations(locations[order])
    recording.set_property("group", groups[order])
    return recording


def test_robust_reduction_with_out_of_order_windows():
    """Out-of-order updates preserve rates, health masks, and coverage."""
    recording = reference_recording()
    starts = np.arange(12) * 0.25
    source = SourceContext.from_recording(recording, 625, starts)
    plan = SampleBudget(12, 0.25).resolve(source)
    params = ActivityParams(lowpass_hz=1000)
    accumulator = ActivityAccumulator(
        source, plan, params, params.filter(2500)
    )
    for index in reversed(plan.window_indices):
        raw = recording.get_traces(
            start_frame=index * 625,
            end_frame=(index + 1) * 625,
            return_in_uV=True,
        )
        raw.flags.writeable = False
        accumulator.update(index, (raw,))
    result = accumulator.finish()
    ordered = ActivityAccumulator(source, plan, params, params.filter(2500))
    for index in plan.window_indices:
        ordered.update(
            index,
            (
                recording.get_traces(
                    start_frame=index * 625,
                    end_frame=(index + 1) * 625,
                    return_in_uV=True,
                ),
            ),
        )
    np.testing.assert_array_equal(result.rate_hz, ordered.finish().rate_hz)
    assert np.any(result.rate_hz > 0)
    assert np.any(~result.healthy)
    np.testing.assert_allclose(
        result.sampled_seconds[result.healthy], 12 * (625 - 124) / 2500
    )
    assert len(accumulator.rates) == 12
    assert not hasattr(accumulator, "raw")


def test_known_supported_peak_and_artifact_veto():
    """Count a supported peak, reject an isolated peak and global artifact."""
    filtered = np.zeros((201, 32), dtype=np.float32)
    filtered[50, 10:12] = [-6, -4]
    filtered[100, 20] = -6
    filtered[150, :4] = -9
    candidates = _supported_candidates(
        filtered,
        np.ones(32),
        _neighbors(np.c_[np.zeros(32), np.arange(32) * 20], 40),
        sampling_rate=30000,
        threshold_sigma=4.5,
        support_sigma=3,
    )
    assert candidates.sum() == 1
    assert candidates[50, 10]


def test_activity_reuses_reads_and_fft_with_small_budget(monkeypatch):
    """A raw-time consumer adds neither recording reads nor transforms."""
    block = make_block("np2")
    reads = []
    original = block.recording.get_traces

    def counted(**kwargs):
        reads.append((kwargs["start_frame"], kwargs["end_frame"]))
        return original(**kwargs)

    monkeypatch.setattr(block.recording, "get_traces", counted)
    result = _compute_all_metrics(
        block,
        window_duration=2,
        extra_metrics=(raw_activity(sampling=SampleBudget(max_windows=1)),),
    )
    activity = result.metric_results["rawActivity"]
    assert sorted(reads) == [(0, 4000), (4000, 8000)]
    assert sum(result.metric_stats.fft_calls.values()) == 2
    assert activity.sampling.window_indices == (1,)
    assert activity.sampling.starts == (5000,)
    assert activity.sampling.seconds_per_window == 0.95


def test_activity_pooling_preserves_median_of_block_rates_and_missing_rows():
    """Coverage is descriptive; unequal exposure does not change the median."""
    result = _compute_all_metrics(make_block("np2"), window_duration=2)
    context = result.metric_context
    table = ChannelTable(
        np.arange(6),
        np.arange(6).astype(str),
        context.ap.locations,
        context.ap.shank_indices,
    )
    sample = SampleBudget(max_windows=1).resolve(context.ap)
    blocks = []
    for rate, seconds in ((2.0, 1.0), (20.0, 100.0)):
        rates = np.full(6, rate)
        rates[-1] = np.nan
        healthy = np.isfinite(rates)
        value = ActivityResult(
            rates, np.ones(6), healthy, np.where(healthy, seconds, 0), sample
        )
        blocks.append(MetricBlockResult(context, value))
    output = raw_activity().reduce(StreamContext(table), tuple(blocks))
    np.testing.assert_array_equal(output.values[:5], 11)
    np.testing.assert_array_equal(output.sampled_seconds[:5], 101)
    assert np.isnan(output.values[-1]) and output.sampled_seconds[-1] == 0
    assert output.metadata["coverage_is_rate_denominator"] is False


@pytest.mark.parametrize(
    "metric",
    [
        raw_activity(sampling=SampleBudget(seconds=5)),
        raw_activity(sampling=SampleBudget(edge_seconds=0)),
        raw_activity(params=ActivityParams(highpass_hz=2000)),
    ],
)
def test_bad_activity_configuration_fails_before_reads(metric, monkeypatch):
    """Reject unsupported sampling/filter requests during registration."""
    block = make_block("np2")

    def forbidden(**kwargs):
        raise AssertionError("Trace read started")

    monkeypatch.setattr(block.recording, "get_traces", forbidden)
    with pytest.raises(ValueError):
        _compute_all_metrics(block, window_duration=2, extra_metrics=(metric,))


@pytest.mark.parametrize("separation", [(0, 700), (100, 0)])
def test_distant_contacts_cannot_support_one_another(separation):
    """Split-bank gaps and lateral distances cannot become array neighbors."""
    locations = np.c_[np.zeros(32), np.arange(32) * 20.0]
    locations[10] = (0, 0)
    locations[11] = separation
    filtered = np.zeros((201, 32), dtype=np.float32)
    filtered[100, 10:12] = [-6, -4]
    detected = _supported_candidates(
        filtered,
        np.ones(32),
        _neighbors(locations, 40),
        sampling_rate=30000,
        threshold_sigma=4.5,
        support_sigma=3,
    )
    assert not detected.any()


def test_flat_peaks_count_once_and_refractory_uses_full_interval():
    """A four-sample plateau is one event; peaks 0.4 ms apart compete."""
    locations = np.c_[np.zeros(32), np.arange(32) * 20.0]
    filtered = np.zeros((201, 32), dtype=np.float32)
    filtered[50:54, 10:12] = [-6, -4]
    filtered[100, 10:12] = [-6, -4]
    filtered[112, 10:12] = [-7, -4]
    detected = _supported_candidates(
        filtered,
        np.ones(32),
        _neighbors(locations, 40),
        sampling_rate=30000,
        threshold_sigma=4.5,
        support_sigma=3,
    )
    assert detected.sum() == 2
    assert detected[112, 10] and not detected[100, 10]


def test_detector_is_invariant_to_channel_order():
    """Native trace order does not define spatial adjacency."""
    locations = np.c_[np.zeros(32), np.arange(32) * 20.0]
    filtered = np.zeros((201, 32), dtype=np.float32)
    filtered[100, 10:12] = [-6, -4]
    order = np.random.default_rng(4).permutation(32)

    def detect(data, coords):
        return _supported_candidates(
            data,
            np.ones(32),
            _neighbors(coords, 40),
            sampling_rate=30000,
            threshold_sigma=4.5,
            support_sigma=3,
        )

    np.testing.assert_array_equal(
        detect(filtered, locations)[:, order],
        detect(filtered[:, order], locations[order]),
    )


def test_health_noise_baseline_does_not_bridge_a_bank_gap():
    """Uniform noise on a distant bank cannot reject a quiet bank."""
    locations = np.c_[np.zeros(8), [0, 20, 40, 60, 1000, 1020, 1040, 1060]]
    noise = np.tile([1, 1, 1, 1, 20, 20, 20, 20], (3, 1))
    assert _health_mask(noise, _neighbors(locations, 200)).all()


def test_trimmed_block_mean_keeps_coverage_as_descriptive_support():
    """Remove one rate from each tail without reducing analyzed duration."""
    source = SourceContext.from_recording(
        reference_recording(), 625, np.arange(12) * 0.25
    )
    plan = SampleBudget(12, 0.25).resolve(source)
    params = ActivityParams(lowpass_hz=1000)
    accumulator = ActivityAccumulator(
        source, plan, params, params.filter(2500)
    )
    accumulator.rates[:] = np.array([0, *range(1, 11), 100])[:, None]
    accumulator.noises[:] = 1
    accumulator.seen.update(plan.window_indices)
    result = accumulator.finish()
    np.testing.assert_allclose(result.rate_hz, 5.5)
    np.testing.assert_allclose(
        result.sampled_seconds, 12 * plan.seconds_per_window
    )


def test_support_neighbors_never_cross_shanks():
    """Coincident probe-local coordinates on different shanks stay separate."""
    source = SourceContext.from_recording(
        reference_recording(), 625, np.array([0.0])
    )
    source = replace(source, locations=np.zeros_like(source.locations))
    plan = SampleBudget(max_windows=1, seconds=0.25).resolve(source)
    params = ActivityParams(lowpass_hz=1000)
    accumulator = ActivityAccumulator(
        source, plan, params, params.filter(2500)
    )
    for indices, neighbors in zip(
        accumulator.groups, accumulator.support_neighbors, strict=True
    ):
        assert len(np.unique(source.shank_indices[indices])) == 1
        assert all(len(nearby) == len(indices) for nearby in neighbors)


def test_output_retains_overlap_pitch_columns_depth_and_missingness(tmp_path):
    """Save native contact rows without grids, interpolation, or stacking."""
    first = np.array([[0, 500], [32, 515], [0, 530], [32, 545]])
    second = np.array([[32, 545], [0, 560], [32, 580], [0, 600]])
    recordings = []
    for coords in (first, second):
        recording = si.NumpyRecording(np.zeros((1000, 4), np.float32), 2000)
        recording.set_property("location", coords)
        recording.set_property("group", np.zeros(4, dtype=int))
        recordings.append(recording)
    table, maps = build_channel_table(recordings)
    template = _compute_all_metrics(make_block("np2"), window_duration=2)
    blocks = []
    for index, (recording, rates) in enumerate(
        zip(recordings, ([1, 2, 3, 4], [8, 10, np.nan, 12]), strict=True)
    ):
        source = SourceContext.from_recording(recording, 1000, np.array([0.0]))
        context = replace(
            template.metric_context,
            ap=source,
            block_index=index,
            block_source="main" if index == 0 else "surface",
            block_id=f"block-{index}",
        )
        plan = SampleBudget(1, 0.5).resolve(source)
        healthy = np.isfinite(rates)
        value = ActivityResult(
            np.asarray(rates),
            np.ones(4),
            healthy,
            np.where(healthy, plan.seconds_per_window, 0.0),
            plan,
        )
        blocks.append(
            replace(
                template,
                metric_context=context,
                metric_results={"rawActivity": value},
            )
        )
    _save_extra_metrics(tmp_path, table, blocks, (raw_activity(),))
    saved = np.load(tmp_path / "_iblqc_channels.rawActivity.npy")
    duration = np.load(
        tmp_path / "_iblqc_channels.rawActivitySampledDuration.npy"
    )
    assert len(saved) == 7  # The overlapping physical contact has one row.
    np.testing.assert_array_equal(saved[maps[0]], [1, 2, 3, 6])
    np.testing.assert_array_equal(saved[maps[1]], [6, 10, np.nan, 12])
    assert duration[maps[1][2]] == 0
    assert duration[maps[0][-1]] == pytest.approx(0.9)
    np.testing.assert_array_equal(table.local_coordinates[maps[0]], first)
    np.testing.assert_array_equal(table.local_coordinates[maps[1]], second)
