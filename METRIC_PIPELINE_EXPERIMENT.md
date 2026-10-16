# Sampled metric API validation

This branch starts from local main commit `6b184fa`. It adds a shared window execution
plan and function-based metric extensions. RMS, PSD, and coherence use the
plan, with their existing normalization, sampling schedule, read prefetch,
block workers, output assembly, and filenames.

The [authoring tutorial](docs/metric_extensions.md) describes the API. The
optional LFP band-power metric exercises the public preparation, accumulation,
contact reduction, and output path. Raw AP examples exercise bounded snippets
and ordinary computation functions.

## Correctness

The frozen fixture in `tests/data/metric_pipeline_baseline.npz` contains 30
arrays captured directly from commit `202625c` for synthetic NP1 and NP2 recordings:
RMS, PSD, timestamps, and real/complex coherency in all five bands.
Tests compare against that implementation's numerical results. Local main adds
only geometry documentation to that baseline. All 139 tests pass.

Additional tests check shared transforms, read-only borrowed inputs, release
after the last consumer, raw computation before FFT allocation, out-of-order
updates, and named metrics with independent outputs. Preparation derives
geometry, analyzed duration, and exact integer intervals from requests and
snippet plans. Tests cover clamped/reordered LFP inputs, short AP transforms,
filter-edge accounting, and invalid plans before reads.

The tutorial's code runs through the real block worker and writer. Concurrent
extraction checks all existing NPY outputs with extensions enabled against
the same extraction without extensions. Other tests exercise custom
accumulation, reduction, and saving without library base classes.

## Performance checks

The benchmark scripts compare the current implementation with main in fresh
processes. They verify every numerical array, recording read intervals, FFT
sizes/counts, and actual compressed trace-chunk accesses. With band power
enabled, its saved result is independently checked against the existing PSD
integrated over the same band.

Synthetic cases use 384 channels and four windows with memory and WavPack
inputs for NP1 and NP2. Real compressed cases use a lossless copy of the
ten-minute recording excerpt under `/mnt/Data/ephys`, retaining source gains,
offsets, and probe geometry. The four-worker case splits that recording into
four contiguous 150-second slices, with twenty windows per slice. These
artificial boundaries exercise concurrency rather than independent experiments.

FFT, BLAS, and decoder threads are limited to one for these checks. They do
not establish remote-storage throughput, cold-read performance, or the best
production thread settings. Timing and peak RSS from single paired runs are
smoke measurements, not evidence of a general speedup.

The real-data serial run took 38.90 seconds for the baseline and 20.20 seconds
with the API and band power. Four concurrent blocks took 32.80 and 34.01 seconds,
respectively. All numerical, read, FFT, and compressed-chunk checks passed.
The large serial timing difference warrants repeated measurements before
drawing a performance conclusion.

## Reproduce

Use `uv sync --locked` to prepare an environment, then:

```sh
uv run pytest -q
uv run python scripts/benchmark_metric_pipeline.py --channels 384 --windows 4 --repeats 1 --output /tmp/ibl-metric-api-main-synthetic.json
uv run python scripts/prepare_real_metric_zarr.py
uv run python scripts/benchmark_real_metric_pipeline.py --cases wavpack_786867 --with-extra --repeats 1 --output-dir /tmp/ibl-metric-api-main-serial --output /tmp/ibl-metric-api-main-serial/report.json
uv run python scripts/benchmark_real_metric_pipeline.py --cases wavpack_786867 --with-extra --repeats 1 --concurrent-blocks 4 --output-dir /tmp/ibl-metric-api-main-concurrent --output /tmp/ibl-metric-api-main-concurrent/report.json
```

Recording data is read without modification. Benchmark arrays and the lossless
compressed copy go under `/tmp`. JSON reports record individual measurements
and work counters. Keep memory and runtime claims limited to the measured case;
there is no global memory admission scheduler or general task graph.
