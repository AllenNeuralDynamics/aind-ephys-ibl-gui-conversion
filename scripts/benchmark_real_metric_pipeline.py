"""Compare original and planned metrics on local recordings, read-only."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.fft
import spikeinterface as si
import zarr
from benchmark_metric_pipeline import (
    compare_arrays,
    load_baseline,
    result_arrays,
)
from threadpoolctl import threadpool_limits

from aind_ephys_ibl_gui_conversion.channel_metadata import build_channel_table
from aind_ephys_ibl_gui_conversion.io import _save_extra_metrics
from aind_ephys_ibl_gui_conversion.metric_extensions import lfp_band_power
from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from aind_ephys_ibl_gui_conversion.types import ExperimentBlock

CASES = {
    "raw_786867": {
        "kind": "binary",
        "path": "/mnt/Data/ephys/raw_chunk_786867_50213-1_10min.dat",
    },
    "openephys_696242": {
        "kind": "openephys",
        "path": "/mnt/Data/ephys/696242_2023-11-29_16-04-55",
        "stream_name": "Record Node 104#Neuropix-PXI-100.46109",
    },
    "wavpack_786867": {
        "kind": "zarr",
        "path": "/tmp/ibl-real-metric-benchmark/real_786867.zarr",
    },
}


def load_recording(case):
    """Open native SI extractors and retain the supplied physical metadata."""
    if case["kind"] == "zarr":
        recording = si.read_zarr(case["path"])
    elif case["kind"] == "openephys":
        from spikeinterface.extractors import read_openephys

        recording = read_openephys(
            case["path"], stream_name=case["stream_name"]
        )
    else:
        path = Path(case["path"])
        metadata = json.loads(path.with_suffix(".json").read_text())
        recording = si.read_binary(
            path,
            sampling_frequency=metadata["sampling_frequency"],
            dtype=metadata["dtype"],
            num_channels=metadata["num_channels"],
            channel_ids=metadata["channel_ids"],
            gain_to_uV=metadata["gain_to_uV"],
            offset_to_uV=metadata["offset_to_uV"],
        )
        for name in ("location", "group", "inter_sample_shift"):
            recording.set_property(name, metadata[name])
        assert recording.get_num_samples() == metadata["num_samples"]
    return recording


def worker(args):
    """Measure sparse metrics across the actual duration in a fresh process."""
    chunk_reads = Counter()
    lock = threading.Lock()
    original_getitem = zarr.DirectoryStore.__getitem__

    def counted_chunk(store, key):
        """Count actual trace-chunk access without replacing the SI reader."""
        if key.startswith("traces_seg") and not key.split("/")[-1].startswith(
            "."
        ):
            with lock:
                chunk_reads[key] += 1
        return original_getitem(store, key)

    zarr.DirectoryStore.__getitem__ = counted_chunk
    recording = load_recording(CASES[args.case])
    boundaries = np.linspace(
        0, recording.get_num_samples(), args.concurrent_blocks + 1, dtype=int
    )
    recordings = [
        recording.frame_slice(int(start), int(stop))
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    compute = (
        load_baseline(args.baseline_file)
        if args.worker == "baseline" and not args.baseline_current
        else _compute_all_metrics
    )
    reads = [[] for _ in recordings]
    for index, source in enumerate(recordings):
        source.get_traces = instrument_reads(source.get_traces, reads[index])
    fft_calls = Counter()
    original_fft = scipy.fft.rfft

    def counted_fft(array, *positional, **kwargs):
        """Count actual FFT calls, for all spectral consumers."""
        key = json.dumps(
            [
                list(array.shape),
                str(array.dtype),
                kwargs.get("norm") or "backward",
            ]
        )
        with lock:
            fft_calls[key] += 1
        return original_fft(array, *positional, **kwargs)

    scipy.fft.rfft = counted_fft
    extra_metrics = ()
    if args.worker == "planned":
        if args.with_extra:
            extra_metrics += (lfp_band_power(),)

    def compute_block(item):
        """Use a separate accumulator and thread-local FFT policy per block."""
        index, source = item
        kwargs = {"extra_metrics": extra_metrics} if extra_metrics else {}
        with scipy.fft.set_workers(1):
            return compute(
                ExperimentBlock(source, None, index, "main"), **kwargs
            )

    with threadpool_limits(limits=1), scipy.fft.set_workers(1):
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrent_blocks) as pool:
            results = list(pool.map(compute_block, enumerate(recordings)))
        seconds = time.perf_counter() - start
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report = {
        "seconds": seconds,
        "peak_rss_mib": peak_mib,
        "duration_seconds": float(recording.get_duration()),
        "channels": recording.get_num_channels(),
        "sampling_frequency": float(recording.get_sampling_frequency()),
        "windows": [len(result.timestamps) for result in results],
        "reads": [sorted(calls) for calls in reads],
        "fft_calls": dict(fft_calls),
        "chunk_reads": dict(chunk_reads),
        "with_extra": bool(extra_metrics),
    }
    if args.arrays_file:
        arrays = {
            f"b{index}_{name}": value
            for index, result in enumerate(results)
            for name, value in result_arrays(result).items()
        }
        np.savez(args.arrays_file, **arrays)
    if extra_metrics:
        verify_extra_output(args, results, recordings, extra_metrics)
    print(json.dumps(report))


def instrument_reads(get_traces, calls):
    """Bind one extractor and its log without capturing mutable loop state."""

    def counted(*positional, **kwargs):
        """Observe native trace calls while returning unmodified arrays."""
        calls.append((kwargs["start_frame"], kwargs["end_frame"]))
        return get_traces(*positional, **kwargs)

    return counted


def verify_extra_output(args, results, recordings, extra_metrics):
    """Check pooled output against independently normalized existing PSD."""
    output = Path(args.output_dir) / f"{args.case}_extra"
    output.mkdir(exist_ok=True, parents=True)
    table, maps = build_channel_table(recordings)
    _save_extra_metrics(output, table, results, extra_metrics)
    if not args.with_extra:
        return
    total = np.zeros(len(table.raw_ind))
    exposure = np.zeros_like(total)
    for rows, result in zip(maps, results, strict=True):
        mask = (result.psd_freqs >= 30) & (result.psd_freqs < 100)
        df = np.diff(result.psd_freqs)[0]
        seconds = len(result.timestamps) * 4
        np.add.at(
            total, rows, result.psd_power[mask].sum(axis=0) * df * seconds
        )
        np.add.at(exposure, rows, seconds)
    actual = np.load(output / "_iblqc_channels.lfpBandPower.npy")
    np.testing.assert_allclose(actual, total / exposure, rtol=3e-6)
    np.testing.assert_array_equal(
        np.load(output / "_iblqc_channels.lfpBandPowerSampledDuration.npy"),
        exposure,
    )


def run_case(args, case, directory):
    """Alternate process order and compare scientific outputs and work."""
    trials = {"baseline": [], "planned": []}
    arrays = {name: directory / f"{case}_{name}.npz" for name in trials}
    for repeat in range(args.repeats):
        order = ("baseline", "planned")
        if repeat % 2:
            order = order[::-1]
        for implementation in order:
            command = [
                sys.executable,
                __file__,
                "--worker",
                implementation,
                "--case",
                case,
                "--baseline-file",
                str(directory / "baseline_metrics.py"),
                "--concurrent-blocks",
                str(args.concurrent_blocks),
                "--output-dir",
                str(directory),
            ]
            if args.with_extra:
                command += ["--with-extra"]
            if args.baseline_current:
                command += ["--baseline-current"]
            if repeat == 0:
                command += ["--arrays-file", str(arrays[implementation])]
            try:
                completed = subprocess.run(
                    command, check=True, capture_output=True, text=True
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"Benchmark worker failed:\n{exc.stderr}"
                ) from exc
            trial = json.loads(completed.stdout)
            trials[implementation].append(trial)
            print(
                json.dumps(
                    {
                        "case": case,
                        "implementation": implementation,
                        "trial": repeat + 1,
                        "seconds": trial["seconds"],
                    }
                ),
                flush=True,
            )
    compare_arrays(arrays["baseline"], arrays["planned"])
    expected = trials["baseline"][0]
    for values in trials.values():
        for trial in values:
            for name in ("reads", "fft_calls", "windows", "chunk_reads"):
                assert trial[name] == expected[name], (case, name)
    summary = {
        name: {
            field: float(np.median([trial[field] for trial in values]))
            for field in ("seconds", "peak_rss_mib")
        }
        for name, values in trials.items()
    }
    summary["time_ratio"] = (
        summary["planned"]["seconds"] / summary["baseline"]["seconds"]
    )
    summary["rss_ratio"] = (
        summary["planned"]["peak_rss_mib"]
        / summary["baseline"]["peak_rss_mib"]
    )
    summary["trials"] = trials
    return summary


def main(args):
    """Save reproducible benchmark evidence without writing source data."""
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    source = subprocess.run(
        [
            "rtk",
            "proxy",
            "git",
            "show",
            "202625c:src/aind_ephys_ibl_gui_conversion/metrics.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (directory / "baseline_metrics.py").write_text(source)
    report = {
        "baseline_ref": "working-tree" if args.baseline_current else "202625c",
        "sources": CASES,
        "cases": {},
        "concurrent_blocks": args.concurrent_blocks,
        "with_extra": args.with_extra,
    }
    for case in args.cases:
        report["cases"][case] = run_case(args, case, directory)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["baseline", "planned"])
    parser.add_argument("--case", choices=list(CASES))
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=list(CASES),
        default=["raw_786867", "openephys_696242"],
    )
    parser.add_argument("--concurrent-blocks", type=int, default=1)
    parser.add_argument("--with-extra", action="store_true")
    parser.add_argument("--baseline-current", action="store_true")
    parser.add_argument("--baseline-file")
    parser.add_argument("--arrays-file")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output-dir", default="/tmp/ibl-real-metric-benchmark"
    )
    parser.add_argument(
        "--output", default="metric_pipeline_real_results.json"
    )
    args = parser.parse_args()
    if args.worker:
        worker(args)
    else:
        main(args)
