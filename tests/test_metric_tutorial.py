"""Execute authoring examples from the tutorial against the real pipeline."""

import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from aind_ephys_ibl_gui_conversion.io import _save_extra_metrics
from aind_ephys_ibl_gui_conversion.metric_extensions import (
    ChannelMeanAccumulator,
    ChannelMeanResult,
    MetricBlockResult,
    StreamContext,
    lfp_band_power,
)
from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from aind_ephys_ibl_gui_conversion.types import ChannelTable
from aind_ephys_ibl_gui_conversion.window_pipeline import Spectrum
from tests.test_window_pipeline import make_block


@pytest.fixture
def tutorial(monkeypatch):
    """Load complete definitions directly from the published code blocks."""
    path = Path(__file__).resolve().parents[1] / "docs/metric_extensions.md"
    blocks = re.findall(r"```python\n(.*?)\n```", path.read_text(), re.S)
    module = ModuleType("metric_tutorial")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    for block in blocks:
        if "from my_metrics import" in block:
            continue
        exec(compile(block, str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("kind", ["np1", "np2"])
@pytest.mark.parametrize("source_name", ["ap", "lfp"])
def test_tutorial_metric_runs_and_saves_without_extra_ffts(
    tutorial, kind, source_name, tmp_path
):
    """A documented metric consumes worker inputs and writes contact means."""
    block = make_block(kind)
    metric = getattr(tutorial, f"{source_name}_mean_absolute")
    baseline = _compute_all_metrics(block, window_duration=2)
    result = _compute_all_metrics(
        block, window_duration=2, extra_metrics=(metric,)
    )
    context = result.metric_context
    source = context.ap if source_name == "ap" else context.lfp
    recording = (
        block.recording
        if source_name == "ap" or kind == "np2"
        else block.lfp_recording
    )
    traces = np.concatenate(
        [
            recording.get_traces(
                start_frame=int(start),
                end_frame=int(start + source.read_samples),
                return_in_uV=True,
            )
            for start in source.window_starts
        ]
    )
    expected = np.mean(np.abs(traces), axis=0, dtype=np.float64)
    table = ChannelTable(
        np.arange(source.n_channels),
        source.contact_ids,
        source.locations,
        source.shank_indices,
    )
    _save_extra_metrics(tmp_path, table, [result], (metric,))
    values, durations, metadata = metric.output_files
    np.testing.assert_allclose(np.load(tmp_path / values), expected)
    np.testing.assert_allclose(
        np.load(tmp_path / durations),
        len(source.window_starts)
        * source.read_samples
        / source.sampling_frequency,
    )
    details = json.loads((tmp_path / metadata).read_text())
    assert details["units"] == "uV"
    assert details["input_source"] == source_name
    assert result.metric_stats.fft_calls == baseline.metric_stats.fft_calls
    np.testing.assert_array_equal(result.psd_power, baseline.psd_power)
    np.testing.assert_array_equal(result.rms_ap, baseline.rms_ap)


def test_tutorial_pooling_weights_duration_and_maps_contacts(tutorial):
    """Pool overlap and permutations, preserving absent coverage."""
    context = _compute_all_metrics(make_block("np1"), window_duration=2)
    context = context.metric_context
    locations = context.ap.locations
    table = ChannelTable(
        np.arange(4),
        np.array(["a", "b", "c", "d"]),
        locations[:4],
        np.zeros(4, dtype=int),
    )
    first = replace(context.ap, locations=locations[[1, 0]])
    second = replace(context.ap, locations=locations[[2, 1]])
    results = (
        MetricBlockResult(
            replace(context, ap=first),
            ChannelMeanResult(np.array([20.0, 4.0]), np.array([2.0, 2.0])),
        ),
        MetricBlockResult(
            replace(context, ap=second),
            ChannelMeanResult(np.array([120.0, 120.0]), np.array([4.0, 4.0])),
        ),
    )
    metric = tutorial.ap_mean_absolute
    output = metric.reduce(StreamContext(table), results)
    np.testing.assert_allclose(output.values, [2, 140 / 6, 30, np.nan])
    np.testing.assert_array_equal(output.sampled_seconds, [2, 6, 4, 0])
    empty = metric.reduce(StreamContext(table), ())
    assert np.isnan(empty.values).all()
    assert not empty.sampled_seconds.any()


def test_tutorial_accumulator_handles_missing_values_and_out_of_order(
    tutorial,
):
    """Invalid channel windows contribute neither values nor duration."""
    accumulator = ChannelMeanAccumulator(2, 1.0, tutorial.mean_absolute)
    accumulator.update(1, (np.array([[-2.0, 4.0], [2.0, -4.0]]),))
    accumulator.update(0, (np.array([[-4.0, np.nan], [4.0, -8.0]]),))
    value = accumulator.finish()
    np.testing.assert_allclose(
        value.weighted_sum / value.sampled_seconds, [3, 4]
    )
    np.testing.assert_array_equal(value.sampled_seconds, [2, 1])
    assert value.weighted_sum.shape == (2,)
    with pytest.raises(ValueError, match="shape"):
        accumulator.update(2, (np.zeros((5, 3)),))
    with pytest.raises(ValueError, match="seconds"):
        ChannelMeanAccumulator(2, 0, tutorial.mean_absolute)


def test_documented_fft_factory_matches_library_and_reuses_request(tutorial):
    """The worked FFT implementation constructs the production computation."""
    result = _compute_all_metrics(make_block("np2"), window_duration=2)
    context = result.metric_context
    metric = lfp_band_power()
    definition = tutorial.spectral_metric.define(context)
    assert definition.inputs == (context.lfp_fft,)
    documented = definition.create()
    actual = metric.define(context).create()
    rng = np.random.default_rng(17)
    spectrum = Spectrum(
        rng.normal(size=(context.lfp_fft.n_samples // 2 + 1, 6))
        + 1j * rng.normal(size=(context.lfp_fft.n_samples // 2 + 1, 6))
    )
    for accumulator in (documented, actual):
        accumulator.update(0, (spectrum,))
    np.testing.assert_array_equal(
        documented.finish().weighted_sum, actual.finish().weighted_sum
    )


@pytest.mark.parametrize("kind", ["np1", "np2"])
def test_documented_spectral_metric_matches_production(
    tutorial, kind, tmp_path
):
    """The composed FFT example runs through the actual worker and writer."""
    documented = tutorial.spectral_metric
    production = lfp_band_power(name="reference")
    result = _compute_all_metrics(
        make_block(kind),
        window_duration=2,
        extra_metrics=(documented, production),
    )
    source = result.metric_context.lfp
    table = ChannelTable(
        np.arange(source.n_channels),
        source.contact_ids,
        source.locations,
        source.shank_indices,
    )
    _save_extra_metrics(tmp_path, table, [result], (documented, production))
    for index in (0, 1):
        np.testing.assert_array_equal(
            np.load(tmp_path / documented.output_files[index]),
            np.load(tmp_path / production.output_files[index]),
        )
    assert sum(result.metric_stats.fft_calls.values()) == (
        4 if kind == "np1" else 2
    )


def test_documented_snippet_metric_derives_its_sampling(tutorial):
    """The budget example runs with derived selection, exposure, and source."""
    metric = tutorial.short_mean
    result = _compute_all_metrics(
        make_block("np2"), window_duration=2, extra_metrics=(metric,)
    )
    value = result.metric_results[metric.name]
    np.testing.assert_array_equal(value.sampled_seconds, 2)
    assert value.input_source == "ap"
    assert value.provenance["window_indices"] == (0, 1)
    assert value.provenance["sample_intervals"] == [[1000, 3000], [5000, 7000]]


@pytest.mark.parametrize(
    "weighted_sum,seconds",
    [([np.inf], [1]), ([1], [-1]), ([1], [np.nan]), ([1], [0])],
)
def test_mean_helper_rejects_invalid_block_statistics(
    tutorial, weighted_sum, seconds
):
    """Malformed native totals cannot silently contaminate a contact mean."""
    context = _compute_all_metrics(make_block("np1"), window_duration=2)
    context = context.metric_context
    source = replace(context.ap, locations=context.ap.locations[:1])
    table = ChannelTable(
        np.array([0]), np.array(["a"]), source.locations, np.array([0])
    )
    result = MetricBlockResult(
        replace(context, ap=source),
        ChannelMeanResult(np.array(weighted_sum), np.array(seconds)),
    )
    with pytest.raises(ValueError):
        tutorial.ap_mean_absolute.reduce(StreamContext(table), (result,))
