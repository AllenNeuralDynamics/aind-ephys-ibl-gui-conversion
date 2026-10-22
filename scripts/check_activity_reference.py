"""Compare activity implementations on identical real-data snippets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from types import ModuleType

import numpy as np
import spikeinterface as si
from benchmark_activity import original_module
from threadpoolctl import threadpool_limits

from aind_ephys_ibl_gui_conversion.activity import (
    ActivityAccumulator,
    ActivityParams,
    _neighbors,
    _supported_candidates,
)
from aind_ephys_ibl_gui_conversion.metric_context import (
    SampleBudget,
    SourceContext,
)
from aind_ephys_ibl_gui_conversion.types import ExperimentBlock


def event_examples(original: ModuleType) -> dict[str, list[int]]:
    """Reproduce the plateau, refractory, and physical-support defects."""
    counts = {}
    for case in ("plateau", "refractory", "bank_gap"):
        data = np.zeros((201, 32), dtype=np.float32)
        coords = np.c_[np.zeros(32), np.arange(32) * 20.0]
        if case == "plateau":
            data[50:54, 10:12] = [-6, -4]
        elif case == "refractory":
            data[100, 10:12] = [-6, -4]
            data[112, 10:12] = [-7, -4]
        else:
            coords[16:, 1] += 680
            data[100, 15:17] = [-6, -4]
        options = dict(
            sampling_rate=30000, threshold_sigma=4.5, support_sigma=3
        )
        old = original._supported_candidates(
            data, np.ones(32), coords[:, 1], **options
        )
        new = _supported_candidates(
            data, np.ones(32), _neighbors(coords, 40), **options
        )
        counts[case] = [int(old.sum()), int(new.sum())]
    assert counts == {
        "plateau": [4, 1],
        "refractory": [2, 1],
        "bank_gap": [1, 0],
    }
    return counts


def main() -> None:
    """Assert integration parity and quantify intentional detector changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    args = parser.parse_args()
    original = original_module()
    ref = "ea025eb"
    source = subprocess.check_output(
        [
            "rtk",
            "proxy",
            "git",
            "show",
            f"{ref}:src/aind_ephys_ibl_gui_conversion/activity.py",
        ],
        text=True,
    )
    legacy = ModuleType("_integration_activity")
    sys.modules[legacy.__name__] = legacy
    exec(compile(source, ref, "exec"), legacy.__dict__)
    recording = si.read_zarr(args.recording)
    fs = recording.get_sampling_frequency()
    starts = np.linspace(0, recording.get_num_samples() / fs - 4, 3)
    context = SourceContext.from_recording(recording, round(4 * fs), starts)
    plan = SampleBudget(max_windows=3).resolve(context)

    class ExactSnippets:
        """Present selected real snippets as the original chunk schedule."""

        def __getattr__(self, name):
            return getattr(recording, name)

        def get_num_samples(self):
            """Expose exactly the selected snippets to chunk scheduling."""
            return len(plan.starts) * plan.request.n_samples

        def get_traces(self, *, start_frame, end_frame, **kwargs):
            """Map a reference chunk read onto its real recording interval."""
            index, offset = divmod(start_frame, plan.request.n_samples)
            assert offset == 0
            start = plan.starts[index]
            return recording.get_traces(
                start_frame=start,
                end_frame=start + end_frame - start_frame,
                **kwargs,
            )

    params = ActivityParams()
    old_params = legacy.ActivityParams()
    integrated = legacy.ActivityAccumulator(
        context, plan, old_params, old_params.filter(fs)
    )
    corrected = ActivityAccumulator(context, plan, params, params.filter(fs))
    with threadpool_limits(limits=1):
        reference = original.estimate_block_unit_density(
            ExperimentBlock(ExactSnippets(), None, 0, "main"),
            original.UnitDensityParams(chunks=3),
        )
        for index, start in reversed(
            list(zip(plan.window_indices, plan.starts, strict=True))
        ):
            raw = recording.get_traces(
                start_frame=start,
                end_frame=start + plan.request.n_samples,
                return_in_uV=True,
            )
            raw.flags.writeable = False
            integrated.update(index, (raw,))
            corrected.update(index, (raw,))
    before = integrated.finish().rate_hz
    after = corrected.finish().rate_hz
    np.testing.assert_array_equal(before, reference)
    valid = np.isfinite(before) & np.isfinite(after)
    print(
        json.dumps(
            {
                "integration_matches_original_exactly": True,
                "synthetic_event_counts_original_rewrite": event_examples(
                    original
                ),
                "sampling": plan.metadata(),
                "compared_contacts": int(valid.sum()),
                "changed_contacts": int(
                    np.count_nonzero(before[valid] != after[valid])
                ),
                "changed_validity": int(
                    np.count_nonzero(np.isfinite(before) != np.isfinite(after))
                ),
                "median_absolute_change_hz": float(
                    np.median(np.abs(before[valid] - after[valid]))
                ),
                "max_absolute_change_hz": float(
                    np.max(np.abs(before[valid] - after[valid]))
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
