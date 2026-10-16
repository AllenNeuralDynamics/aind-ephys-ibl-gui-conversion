"""Sampling budgets stay inside existing reads without silent extra work."""

from dataclasses import replace

import numpy as np
import pytest

from aind_ephys_ibl_gui_conversion.metric_context import SampleBudget
from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from tests.test_window_pipeline import make_block


def source_context():
    """Use a resolved source with twenty illustrative existing read windows."""
    result = _compute_all_metrics(make_block("np2"), window_duration=2)
    return replace(
        result.metric_context.ap,
        window_starts=np.arange(20) * 60000,
        read_samples=8000,
    )


def test_distributed_budget_exposes_actual_analyzed_intervals():
    """Seconds become integer samples and trimming affects exposure."""
    source = source_context()
    plan = SampleBudget().resolve(source)
    assert plan.window_indices == (0, 10, 19)
    assert plan.request.n_samples == 2000
    assert plan.request.offset == 3000
    assert plan.edge_samples == 50
    assert plan.seconds_per_window == 0.95
    assert plan.metadata()["analyzed_seconds"] == pytest.approx(2.85)
    assert plan.metadata()["sample_intervals"][0] == [3000, 5000]
    assert plan.metadata()["analyzed_intervals"][0] == [3050, 4950]
    assert SampleBudget(max_windows=1).resolve(source).window_indices == (10,)


@pytest.mark.parametrize(
    "budget, message",
    [
        (SampleBudget(max_windows=0), "max_windows"),
        (SampleBudget(max_windows=1.5), "max_windows"),
        (SampleBudget(seconds=float("nan")), "finite"),
        (SampleBudget(seconds=5), "fit"),
        (SampleBudget(seconds=0.01), "edges"),
        (SampleBudget(edge_seconds=-0.1), "edges"),
    ],
)
def test_invalid_budget_is_explicit(budget, message):
    """Reject requests instead of adding reads or changing durations."""
    with pytest.raises(ValueError, match=message):
        budget.resolve(source_context())


def test_overlapping_cores_are_rejected():
    """Clamped source windows cannot silently count the same time twice."""
    source = replace(source_context(), window_starts=np.array([0, 1000]))
    with pytest.raises(ValueError, match="overlap"):
        SampleBudget(max_windows=2).resolve(source)
