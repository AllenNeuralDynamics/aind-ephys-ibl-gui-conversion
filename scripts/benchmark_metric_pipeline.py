"""Compare the prototype with its frozen Git baseline in fresh processes.

Uses repeated synthetic four-second samples, in memory or in local
WavPack-compressed Zarr. This measures computation and local decompression,
not S3 latency or complete recording conversion.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.fft
import zarr
from threadpoolctl import threadpool_limits
from wavpack_numcodecs import WavPack, set_num_decoding_threads

from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from aind_ephys_ibl_gui_conversion.types import ExperimentBlock


class CountingStore(zarr.DirectoryStore):
    """Count compressed chunk reads independently of logical trace reads."""

    def __init__(self, path):
        """Open a local directory store with a thread-safe read counter."""
        super().__init__(path)
        self.chunk_reads = 0
        self._lock = threading.Lock()

    def __getitem__(self, key):
        """Count data chunks while excluding array metadata."""
        value = super().__getitem__(key)
        if all(part.isdigit() for part in key.split(".")):
            with self._lock:
                self.chunk_reads += 1
        return value


class SampledRecording:
    """Expose virtual time windows backed by one repeated sample."""

    def __init__(self, directory, source, storage, channels, windows):
        """Open sample storage without allocating a full recording."""
        self.fs = 30000 if source == "ap" else 2500
        self.channels = channels
        self.windows = windows
        self.calls = []
        self.store = None
        if storage == "memory":
            self.data = np.load(directory / f"{source}.npy", mmap_mode="r")
        else:
            self.store = CountingStore(str(directory / f"{source}.zarr"))
            self.data = zarr.open_array(self.store, mode="r")

    def get_sampling_frequency(self):
        """Return this source's sampling frequency."""
        return self.fs

    def get_num_samples(self):
        """Expose enough virtual duration for the requested window count."""
        return (self.windows + 1) * 4 * self.fs

    def get_duration(self):
        """Return virtual duration in seconds."""
        return self.get_num_samples() / self.fs

    def get_num_channels(self):
        """Return the native channel count."""
        return self.channels

    def get_channel_locations(self):
        """Return regular physical contact coordinates."""
        return np.c_[np.zeros(self.channels), np.arange(self.channels) * 20]

    def get_property(self, name):
        """Supply one shank without optional contact metadata."""
        return np.zeros(self.channels, dtype=int) if name == "group" else None

    def get_probe(self):
        """Leave optional probe/contact metadata absent."""
        raise ValueError("No probe attached")

    def get_channel_ids(self):
        """Return native channel positions."""
        return np.arange(self.channels)

    def get_traces(self, start_frame, end_frame, return_in_uV):
        """Read one repeated sample as scaled float32 data."""
        assert return_in_uV
        length = end_frame - start_frame
        assert length <= self.data.shape[0]
        self.calls.append((start_frame, end_frame))
        return np.asarray(self.data[:length]).astype(np.float32)


def prepare(directory, channels):
    """Write small deterministic samples outside the timed child processes."""
    directory.mkdir(parents=True, exist_ok=True)
    for source, fs in (("ap", 30000), ("lfp", 2500)):
        path = directory / f"{source}.npy"
        if path.exists():
            existing = np.load(path, mmap_mode="r")
            if existing.shape != (4 * fs, channels):
                raise ValueError("Use a new data directory for this shape")
            if not (directory / f"{source}.zarr").exists():
                raise ValueError("Incomplete fixture; use a new directory")
            continue
        data = np.random.default_rng(42).integers(
            -128, 129, (4 * fs, channels), dtype=np.int16
        )
        np.save(path, data)
        array = zarr.open_array(
            str(directory / f"{source}.zarr"),
            mode="w",
            shape=data.shape,
            dtype=data.dtype,
            chunks=(fs, min(64, channels)),
            compressor=WavPack(num_decoding_threads=1),
        )
        array[:] = data


def load_baseline(path):
    """Load the pre-refactor implementation with the same library versions."""
    spec = importlib.util.spec_from_file_location("baseline_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._compute_all_metrics


def result_arrays(result):
    """Expose existing scientific outputs for full numerical comparison."""
    arrays = {
        field: getattr(result, field)
        for field in (
            "rms_ap",
            "rms_lfp",
            "psd_power",
            "psd_freqs",
            "timestamps",
        )
    }
    for field in ("correlation", "coherency"):
        arrays.update(
            {
                f"{field}_{band}": value
                for band, value in getattr(result, field).items()
            }
        )
    return arrays


def worker(args):
    """Measure one implementation, including imports in process peak RSS."""
    directory = Path(args.data_dir)
    ap = SampledRecording(
        directory, "ap", args.storage, args.channels, args.windows
    )
    lfp = (
        SampledRecording(
            directory, "lfp", args.storage, args.channels, args.windows
        )
        if args.kind == "np1"
        else None
    )
    compute = (
        load_baseline(args.baseline_file)
        if args.worker == "baseline"
        else _compute_all_metrics
    )
    fft_calls = Counter()
    original_fft = scipy.fft.rfft

    def counted_fft(array, *positional, **kwargs):
        """Record actual transform sizes, for the existing metrics."""
        key = json.dumps(
            [
                list(array.shape),
                str(array.dtype),
                kwargs.get("norm") or "backward",
            ]
        )
        fft_calls[key] += 1
        return original_fft(array, *positional, **kwargs)

    scipy.fft.rfft = counted_fft
    set_num_decoding_threads(1)
    with threadpool_limits(limits=1), scipy.fft.set_workers(1):
        start = time.perf_counter()
        result = compute(
            ExperimentBlock(ap, lfp, 0, "main"),
            window_interval=4,
            min_coherency_windows=1,
        )
        elapsed = time.perf_counter() - start
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    sources = [ap] if lfp is None else [ap, lfp]
    payload = {
        "seconds": elapsed,
        "peak_rss_mib": peak_mib,
        "reads": [sorted(source.calls) for source in sources],
        "chunk_reads": sum(
            source.store.chunk_reads
            for source in sources
            if source.store is not None
        ),
        "fft_calls": dict(fft_calls),
    }
    if args.arrays_file:
        np.savez(args.arrays_file, **result_arrays(result))
    print(json.dumps(payload))


def compare_arrays(baseline, planned):
    """Compare complete arrays with the existing numerical precision."""
    with np.load(baseline) as old, np.load(planned) as new:
        assert set(old.files) == set(new.files)
        for name in old.files:
            atol = (
                2e-7 if "correlation" in name or "coherency" in name else 1e-15
            )
            np.testing.assert_allclose(
                new[name], old[name], rtol=3e-6, atol=atol
            )


def main(args):
    """Alternate isolated baseline/prototype trials and report work parity."""
    directory = Path(args.data_dir)
    prepare(directory, args.channels)
    baseline = directory / "baseline_metrics.py"
    source = subprocess.run(
        [
            "rtk",
            "proxy",
            "git",
            "show",
            f"{args.baseline_ref}:src/aind_ephys_ibl_gui_conversion/metrics.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    baseline.write_text(source)
    report = {
        "baseline_ref": args.baseline_ref,
        "channels": args.channels,
        "windows": args.windows,
        "repeats": args.repeats,
        "cases": {},
    }
    for storage in ("memory", "wavpack"):
        for kind in ("np1", "np2"):
            trials = {"baseline": [], "planned": []}
            arrays = {
                name: directory / f"{storage}_{kind}_{name}.npz"
                for name in trials
            }
            for repeat in range(args.repeats):
                order = (
                    ("baseline", "planned")
                    if repeat % 2 == 0
                    else ("planned", "baseline")
                )
                for implementation in order:
                    command = [
                        sys.executable,
                        __file__,
                        "--worker",
                        implementation,
                        "--baseline-file",
                        str(baseline),
                        "--data-dir",
                        str(directory),
                        "--storage",
                        storage,
                        "--kind",
                        kind,
                        "--channels",
                        str(args.channels),
                        "--windows",
                        str(args.windows),
                    ]
                    if repeat == 0:
                        command += [
                            "--arrays-file",
                            str(arrays[implementation]),
                        ]
                    completed = subprocess.run(
                        command, check=True, capture_output=True, text=True
                    )
                    trials[implementation].append(json.loads(completed.stdout))
            compare_arrays(arrays["baseline"], arrays["planned"])
            first, second = trials["baseline"][0], trials["planned"][0]
            for name in ("reads", "chunk_reads", "fft_calls"):
                assert first[name] == second[name], (storage, kind, name)
            summary = {
                implementation: {
                    field: float(np.median([trial[field] for trial in values]))
                    for field in ("seconds", "peak_rss_mib")
                }
                for implementation, values in trials.items()
            }
            summary["time_ratio"] = (
                summary["planned"]["seconds"] / summary["baseline"]["seconds"]
            )
            summary["rss_ratio"] = (
                summary["planned"]["peak_rss_mib"]
                / summary["baseline"]["peak_rss_mib"]
            )
            summary["trials"] = trials
            report["cases"][f"{storage}_{kind}"] = summary
            print(
                json.dumps(
                    {
                        "case": f"{storage}_{kind}",
                        "time_ratio": summary["time_ratio"],
                        "rss_ratio": summary["rss_ratio"],
                    }
                ),
                flush=True,
            )
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["baseline", "planned"])
    parser.add_argument("--baseline-ref", default="202625c")
    parser.add_argument("--baseline-file")
    parser.add_argument("--arrays-file")
    parser.add_argument(
        "--data-dir", default="/tmp/ibl-metric-pipeline-benchmark"
    )
    parser.add_argument(
        "--output", default="/tmp/ibl-metric-pipeline-results.json"
    )
    parser.add_argument("--storage", choices=["memory", "wavpack"])
    parser.add_argument("--kind", choices=["np1", "np2"])
    parser.add_argument("--channels", type=int, default=384)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.worker:
        worker(arguments)
    else:
        main(arguments)
