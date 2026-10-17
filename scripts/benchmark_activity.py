"""Compare original and rewritten activity on the same compressed recording."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import numpy as np
import scipy.fft
import spikeinterface as si
import zarr
from benchmark_metric_pipeline import compare_arrays, result_arrays
from threadpoolctl import threadpool_limits

from aind_ephys_ibl_gui_conversion.activity import raw_activity
from aind_ephys_ibl_gui_conversion.channel_metadata import build_channel_table
from aind_ephys_ibl_gui_conversion.metric_context import SampleBudget
from aind_ephys_ibl_gui_conversion.metric_extensions import (
    MetricBlockResult,
    StreamContext,
)
from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from aind_ephys_ibl_gui_conversion.types import ExperimentBlock

ORIGINAL = "7731283a10ed8b9fd7aa1a0ef9a49b8a289682fa"


def original_module() -> ModuleType:
    """Load the reviewed PR snapshot without adding it to the package."""
    source = subprocess.check_output(
        [
            "rtk",
            "proxy",
            "git",
            "show",
            f"{ORIGINAL}:src/aind_ephys_ibl_gui_conversion/unit_density.py",
        ],
        text=True,
    )
    module = ModuleType("_original_activity")
    sys.modules[module.__name__] = module
    exec(compile(source, ORIGINAL, "exec"), module.__dict__)
    return module


def worker(args: argparse.Namespace) -> None:
    """Measure one implementation in a fresh process with bounded threads."""
    original = original_module() if args.worker == "original" else None
    recording = si.read_zarr(args.recording)
    boundaries = np.linspace(
        0, recording.get_num_samples(), args.blocks + 1, dtype=int
    )
    blocks = [
        ExperimentBlock(recording.frame_slice(int(a), int(b)), None, i, "main")
        for i, (a, b) in enumerate(
            zip(boundaries[:-1], boundaries[1:], strict=True)
        )
    ]
    lock = threading.Lock()
    reads: list[list[tuple[int, int]]] = [[] for _ in blocks]
    for i, block in enumerate(blocks):
        get_traces = block.recording.get_traces

        def counted(*positional, _get=get_traces, _i=i, **kwargs):
            """Count the exact intervals passed to the real recording."""
            with lock:
                reads[_i].append((kwargs["start_frame"], kwargs["end_frame"]))
            return _get(*positional, **kwargs)

        block.recording.get_traces = counted
    fft_calls: Counter[str] = Counter()
    chunks: Counter[str] = Counter()
    original_fft = scipy.fft.rfft
    original_chunk = zarr.DirectoryStore.__getitem__

    def counted_fft(array, *positional, **kwargs):
        """Count transform shapes without substituting their calculation."""
        with lock:
            fft_calls[str(array.shape)] += 1
        return original_fft(array, *positional, **kwargs)

    def counted_chunk(store, key):
        """Count actual compressed trace-chunk fetches."""
        if key.startswith("traces_seg") and not key.split("/")[-1].startswith(
            "."
        ):
            with lock:
                chunks[key] += 1
        return original_chunk(store, key)

    scipy.fft.rfft = counted_fft
    zarr.DirectoryStore.__getitem__ = counted_chunk
    metrics = (
        (raw_activity(sampling=SampleBudget(max_windows=args.snippets)),)
        if args.worker == "rewrite"
        else ()
    )

    def compute(block):
        """Use independent states and FFT policy for concurrent blocks."""
        with scipy.fft.set_workers(1):
            result = _compute_all_metrics(block, extra_metrics=metrics)
            rates = (
                original.estimate_block_unit_density(
                    block, original.UnitDensityParams(chunks=args.snippets)
                )
                if original
                else None
            )
        return result, rates

    with threadpool_limits(limits=1):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.blocks) as pool:
            results = list(pool.map(compute, blocks))
        if original:
            original.compose_raw_unit_density(
                blocks, [rates for _, rates in results]
            )
        if metrics:
            table, _ = build_channel_table(
                [block.recording for block in blocks]
            )
            metrics[0].reduce(
                StreamContext(table),
                tuple(
                    MetricBlockResult(
                        result.metric_context,
                        result.metric_results[metrics[0].name],
                    )
                    for result, _ in results
                ),
            )
        elapsed = time.perf_counter() - started
    arrays = {
        f"block_{i}_{name}": value
        for i, (result, _) in enumerate(results)
        for name, value in result_arrays(result).items()
    }
    np.savez(args.arrays, **arrays)
    report = {
        "seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "reads": [sorted(calls) for calls in reads],
        "fft_calls": dict(fft_calls),
        "chunk_reads": dict(chunks),
        "duration_seconds": recording.get_duration(),
        "channels": recording.get_num_channels(),
        "blocks": args.blocks,
        "rate_summaries": [
            {
                "healthy_channels": int(np.isfinite(rates).sum()),
                "median_hz": float(np.nanmedian(rates)),
                "max_hz": float(np.nanmax(rates)),
            }
            for result, old_rates in results
            if (
                rates := (
                    old_rates
                    if original
                    else result.metric_results[metrics[0].name].rate_hz
                    if metrics
                    else None
                )
            )
            is not None
        ],
    }
    Path(args.output).write_text(json.dumps(report))


def main() -> None:
    """Interleave fresh-process trials and verify existing outputs and work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--worker", choices=("baseline", "original", "rewrite")
    )
    parser.add_argument("--snippets", type=int, default=1)
    parser.add_argument("--arrays")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.blocks < 1 or args.repeats < 1:
        parser.error("blocks and repeats must be positive")
    cases = [("baseline", 1), ("original", 1), ("rewrite", 1), ("rewrite", 3)]
    reports = []
    with tempfile.TemporaryDirectory(prefix="activity-benchmark-") as folder:
        root = Path(folder)
        for repeat in range(args.repeats):
            baseline_arrays = None
            baseline = None
            trials = []
            for index in np.random.default_rng(repeat).permutation(len(cases)):
                mode, snippets = cases[index]
                report_file = root / f"{repeat}_{index}.json"
                arrays_file = report_file.with_suffix(".npz")
                command = [
                    sys.executable,
                    __file__,
                    "--worker",
                    mode,
                    "--recording",
                    args.recording,
                    "--blocks",
                    str(args.blocks),
                    "--snippets",
                    str(snippets),
                    "--output",
                    str(report_file),
                    "--arrays",
                    str(arrays_file),
                ]
                subprocess.run(command, check=True)
                report = json.loads(report_file.read_text())
                report.update(mode=mode, snippets=snippets, repeat=repeat)
                if mode == "baseline":
                    baseline, baseline_arrays = report, arrays_file
                trials.append((report, arrays_file))
                print(
                    json.dumps(
                        {
                            k: report[k]
                            for k in (
                                "mode",
                                "snippets",
                                "repeat",
                                "seconds",
                                "peak_rss_mib",
                            )
                        }
                    ),
                    flush=True,
                )
            assert baseline is not None and baseline_arrays is not None
            for report, arrays in trials:
                compare_arrays(baseline_arrays, arrays)
                assert report["fft_calls"] == baseline["fft_calls"]
                if report["mode"] == "rewrite":
                    assert report["reads"] == baseline["reads"]
                    assert report["chunk_reads"] == baseline["chunk_reads"]
                report["existing_outputs_match"] = True
                reports.append(report)
            Path(args.output).write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
