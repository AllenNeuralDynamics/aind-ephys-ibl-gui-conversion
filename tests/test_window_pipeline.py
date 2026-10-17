"""Regression tests for shared work, ownership, and metric extension."""

import weakref
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import spikeinterface as si

from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from aind_ephys_ibl_gui_conversion.spectral_accumulators import RmsAccumulator
from aind_ephys_ibl_gui_conversion.types import ExperimentBlock
from aind_ephys_ibl_gui_conversion.window_pipeline import (
    FftRequest,
    MetricDefinition,
    RawRequest,
    Spectrum,
    WindowPlan,
)


class Capture:
    """Reduce inputs while observing ownership without retaining buffers."""

    def __init__(self, *, destructive=False):
        """Keep reduced scalar results and weak references only."""
        self.values = []
        self.owners = []
        self.destructive = destructive

    def update(self, window_index, inputs):
        """Read borrowed inputs or test their write protection."""
        (product,) = inputs
        array = (
            product.coefficients if isinstance(product, Spectrum) else product
        )
        self.values.append((window_index, float(np.abs(array).sum())))
        self.owners.append(
            weakref.ref(array.base if array.base is not None else array)
        )
        if self.destructive:
            array[:] = 0
        else:
            assert not array.flags.writeable

    def finish(self):
        """Return only reduced scalar values."""
        return self.values


def make_block(kind):
    """Build deterministic NP1/NP2 signals with two physical shanks."""
    recordings = []
    for fs, seed in ((2000.0, 41), (1000.0, 42)):
        time = np.arange(int(6 * fs)) / fs
        traces = np.random.default_rng(seed).normal(0, 0.2, (time.size, 6))
        traces += np.sin(2 * np.pi * time[:, None] * 50)
        traces += np.sin(2 * np.pi * time[:, None] * 420) * np.arange(1, 7)
        recording = si.NumpyRecording([traces.astype(np.float32)], fs)
        locations = np.c_[np.repeat([0, 250], 3), np.tile([0, 20, 40], 2)]
        recording.set_dummy_probe_from_locations(locations)
        recording.set_property("group", np.repeat([0, 1], 3))
        recording.set_channel_gains(1)
        recording.set_channel_offsets(0)
        recordings.append(recording)
    return ExperimentBlock(
        recordings[0], recordings[1] if kind == "np1" else None, 0, "main"
    )


def make_plan(definitions, n_windows=1, separate_lfp=False):
    """Bind a tiny, fully materialized source window for protocol tests."""
    return WindowPlan(
        tuple(definitions),
        n_windows=n_windows,
        source_samples={"ap": 128, "lfp": 128},
        separate_lfp=separate_lfp,
    )


def test_independent_consumers_share_fft_and_np2_source_alias():
    """Separate, equal requests for AP/LFP resolve to one physical FFT."""
    request = FftRequest("ap", 128)
    first, second = Capture(), Capture()
    plan = make_plan(
        [
            MetricDefinition("first", (request,), lambda: first),
            MetricDefinition(
                "second", (replace(request, source="lfp"),), lambda: second
            ),
        ]
    )
    raw = np.random.default_rng(0).normal(size=(128, 4)).astype(np.float32)
    plan.consume(0, raw, None)
    assert sum(plan.stats.fft_calls.values()) == 1
    assert first.values == second.values
    assert all(owner() is None for owner in first.owners + second.owners)


@pytest.mark.parametrize(
    "change",
    [
        {"source": "lfp"},
        {"n_samples": 32},
        {"offset": 1},
        {"dtype": "float64"},
        {"reference": "median"},
        {"demean": True},
        {"taper": "hann"},
        {"norm": "ortho"},
    ],
)
def test_different_signal_semantics_never_share_fft(change):
    """A changed source, interval, processing, or norm requires its own FFT."""
    request = FftRequest("ap", 64)
    plan = make_plan(
        [
            MetricDefinition("first", (request,), Capture),
            MetricDefinition("second", (replace(request, **change),), Capture),
        ],
        separate_lfp=True,
    )
    raw = np.ones((128, 4), dtype=np.float32)
    plan.consume(0, raw, raw.copy())
    assert sum(plan.stats.fft_calls.values()) == 2


def test_selection_computes_only_requested_windows():
    """A sparse metric neither runs nor materializes products elsewhere."""
    capture = Capture()
    plan = make_plan(
        [
            MetricDefinition(
                "sample",
                (FftRequest("ap", 128),),
                lambda: capture,
                window_indices=(1, 3),
            )
        ],
        n_windows=5,
    )
    for index in range(5):
        plan.consume(index, np.ones((128, 2), dtype=np.float32), None)
    assert [index for index, _ in capture.values] == [1, 3]
    assert sum(plan.stats.fft_calls.values()) == 2


def test_raw_consumers_cannot_modify_reader_buffers():
    """A plugin gets a read-only view even when it consumes raw samples."""
    plan = make_plan(
        [
            MetricDefinition(
                "bad",
                (RawRequest("ap", 128),),
                partial(Capture, destructive=True),
            )
        ]
    )
    raw = np.ones((128, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="read-only"):
        plan.consume(0, raw, None)
    np.testing.assert_array_equal(raw, 1)


@pytest.mark.parametrize("selection", [(1, 1), (-1,), (2,)])
def test_invalid_selection_fails_before_accumulator_creation(selection):
    """Bad selection cannot allocate metric state or start any reads."""

    def forbidden():
        raise AssertionError("Accumulator was created")

    with pytest.raises(ValueError, match="window index"):
        make_plan(
            [
                MetricDefinition(
                    "bad",
                    (RawRequest("ap", 128),),
                    forbidden,
                    window_indices=selection,
                )
            ],
            n_windows=2,
        )


def test_outside_read_fails_before_work():
    """A metric cannot extend the underlying read through this interface."""
    with pytest.raises(ValueError, match="outside the read"):
        make_plan(
            [MetricDefinition("long", (RawRequest("ap", 129),), Capture)]
        )


def test_raw_reducers_run_before_fft_allocation(monkeypatch):
    """Raw scratch does not overlap an unnecessarily live shared FFT."""
    import scipy.fft

    order = []
    original = scipy.fft.rfft

    def transform(*args, **kwargs):
        order.append("fft")
        return original(*args, **kwargs)

    class RawCapture(Capture):
        def update(self, window_index, inputs):
            order.append("raw")
            super().update(window_index, inputs)

    monkeypatch.setattr(scipy.fft, "rfft", transform)
    plan = make_plan(
        [
            MetricDefinition("spectral", (FftRequest("ap", 128),), Capture),
            MetricDefinition("raw", (RawRequest("ap", 128),), RawCapture),
        ]
    )
    plan.consume(0, np.ones((128, 4), dtype=np.float32), None)
    assert order == ["raw", "fft"]


@pytest.mark.parametrize("kind", ["np1", "np2"])
def test_frozen_baseline_outputs(kind):
    """Preserve outputs captured from main before the refactor."""
    result = _compute_all_metrics(make_block(kind), window_duration=2)
    path = Path(__file__).parent / "data" / "metric_pipeline_baseline.npz"
    with np.load(path) as baseline:
        for field in (
            "rms_ap",
            "rms_lfp",
            "psd_power",
            "psd_freqs",
            "timestamps",
        ):
            np.testing.assert_allclose(
                getattr(result, field),
                baseline[f"{kind}_{field}"],
                rtol=2e-6,
                atol=1e-15,
            )
        for field in ("correlation", "coherency"):
            for band, values in getattr(result, field).items():
                np.testing.assert_allclose(
                    values,
                    baseline[f"{kind}_{field}_{band}"],
                    rtol=2e-6,
                    atol=2e-7,
                )


def test_extra_spectral_metric_uses_existing_fft():
    """An extension reuses the default LF transform and normalization."""
    block = make_block("np2")

    class ExtraRms:
        """Declare an extra consumer using only scheduler-owned metadata."""

        name = "extra_rms"
        output_files = ()

        def define(self, context):
            """Resolve the LF band without reconstructing the schedule."""
            request = context.lfp_fft
            frequencies = context.frequencies(request)
            mask = (frequencies >= 1) & (frequencies <= 300)
            return MetricDefinition(
                self.name,
                (request,),
                partial(
                    RmsAccumulator,
                    context.n_windows,
                    context.lfp.n_channels,
                    mask,
                    request.n_samples,
                    np.sqrt(request.n_samples / context.lfp_window_power),
                ),
            )

    result = _compute_all_metrics(
        block, window_duration=2, extra_metrics=(ExtraRms(),)
    )
    np.testing.assert_allclose(
        result.metric_results["extra_rms"], result.rms_lfp / 1e-6, rtol=2e-7
    )
    assert sum(result.metric_stats.fft_calls.values()) == 2
