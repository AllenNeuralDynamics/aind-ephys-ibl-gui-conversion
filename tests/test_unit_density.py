"""Tests for raw-AP normalized activity-density profiles."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from aind_ephys_ibl_gui_conversion.io import _save_raw_unit_density
from aind_ephys_ibl_gui_conversion.types import ExperimentBlock
from aind_ephys_ibl_gui_conversion.unit_density import (
    UnitDensityParams,
    compose_raw_unit_density,
    estimate_block_unit_density,
)


def _estimate(
    source: str,
    depths: list[float],
    rates: list[float],
    healthy: list[bool] | None = None,
) -> tuple[ExperimentBlock, np.ndarray]:
    """Build a one-shank channel estimate for profile-assembly tests."""
    import spikeinterface as si

    n_channels = len(depths)
    recording = si.NumpyRecording(
        [np.zeros((1, n_channels), dtype=np.float32)], 30000.0
    )
    recording.set_property("location", np.c_[np.zeros(n_channels), depths])
    recording.set_property("group", np.zeros(n_channels, dtype=int))
    estimate = np.asarray(rates, dtype=float)
    estimate[~np.asarray(healthy or [True] * n_channels)] = np.nan
    return ExperimentBlock(recording, None, 0, source=source), estimate


def test_surface_bank_is_concatenated_above_full_main_bank() -> None:
    """A surface bank starts after the nominal bank, not the last contact."""
    params = UnitDensityParams(bank_span_um=40.0, smooth_um=10.0)
    main_block, main = _estimate("main", [0.0, 10.0, 20.0], [1.0, 4.0, 2.0])
    surface_block, surface = _estimate("surface", [0.0, 10.0], [2.0, 8.0])

    profile = compose_raw_unit_density(
        [main_block, surface_block], [main, surface], params
    )

    assert profile.shape == (1, 8)
    area = np.trapezoid(profile[0], np.arange(8) * 10.0 / 1000)
    assert area == pytest.approx(1)


def test_removed_contact_keeps_its_row_inside_bank() -> None:
    """Missing internal rows are retained rather than collapsing the axis."""
    params = UnitDensityParams(bank_span_um=40.0, smooth_um=10.0)
    block, estimate = _estimate("main", [0.0, 20.0, 30.0], [1.0, 4.0, 2.0])

    profile = compose_raw_unit_density([block], [estimate], params)

    assert profile.shape == (1, 4)


def test_removed_contact_at_bank_edge_does_not_shorten_bank() -> None:
    """A missing first/last row remains represented by the nominal span."""
    params = UnitDensityParams(bank_span_um=40.0, smooth_um=10.0)
    block, estimate = _estimate("main", [10.0, 20.0], [4.0, 2.0])

    profile = compose_raw_unit_density([block], [estimate], params)

    assert profile.shape == (1, 4)


def test_save_writes_complete_alf_object() -> None:
    """The MVP emits one two-dimensional density array and nothing else."""
    from types import SimpleNamespace

    params = UnitDensityParams(bank_span_um=40.0, smooth_um=10.0)
    block, estimate = _estimate("main", [0.0, 10.0], [1.0, 4.0])
    result = SimpleNamespace(block=block, unit_density=estimate)
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        _save_raw_unit_density(output, [result], params)
        (path,) = output.iterdir()
        assert path.name == "_iblqc_ephysRawUnitDensity.density.npy"
        assert np.load(path).shape == (1, 4)


def test_estimator_returns_one_value_per_input_channel() -> None:
    """Threshold processing never collapses paired sites at equal depths."""
    import spikeinterface as si

    sampling_rate = 2500.0
    rng = np.random.default_rng(2)
    traces = rng.normal(0, 8, (int(1.2 * sampling_rate), 8)).astype(np.float32)
    recording = si.NumpyRecording([traces], sampling_rate)
    recording.set_property(
        "location",
        np.c_[np.tile([0.0, 32.0], 4), np.repeat(np.arange(4) * 20.0, 2)],
    )
    recording.set_property("group", np.zeros(8, dtype=int))
    estimate = estimate_block_unit_density(
        ExperimentBlock(recording, None, 0, source="main"),
        UnitDensityParams(lowpass_hz=1000.0),
    )

    assert estimate.shape == (8,)
