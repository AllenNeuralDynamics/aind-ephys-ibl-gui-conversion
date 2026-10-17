# Continuous metric sampling and execution

This guide describes the current continuous-metric pipeline in
`extract_continuous()`, `_compute_all_metrics()`, and stream assembly.

## A block represents one experiment for one probe stream

`load_probe_streams()` loads each `experiment<N>_<stream>.zarr` into an
`ExperimentBlock`. Neuropixels 1.0 blocks also carry a paired LFP recording.
If an experiment contains multiple recording segments, the loader retains the
longest segment and resets its time origin. `block_index` enumerates the
experiments found on disk; it is not necessarily the experiment number minus
one, because uploaded experiment numbers can have gaps.

A `ProbeStream` collects these blocks, including blocks from a separate
surface-finding asset when supplied. Main/surface provenance describes where
a block came from. Its contact coordinates determine spatial coverage.

## Workers sample time windows within each block

`extract_continuous()` schedules one task per experiment block across all
selected streams. Each block worker calls `_compute_all_metrics()`, which
samples time windows throughout that block's recording. The default windows
are four seconds long, spaced at most 30 seconds apart; short recordings use
a shorter interval, subject to the window length.

These are three different units: an experiment block is a recording task, a
time window is a sampled interval within that recording, and a storage chunk
is the underlying Zarr/WavPack read unit. A short requested interval may still
require decompressing a larger storage chunk.

## Metrics share reads and FFT results

Each scheduled read returns AP/wideband data and, for NP1, paired LFP data.
NP1 AP RMS uses a short FFT of the first second of the AP window. Its LFP
metrics use the separate LFP recording. NP2 AP and LFP metrics share the FFT
of the wideband window.

The processing loop converts traces to float32. LFP/wideband processing then
subtracts a common median reference, demeans, and applies a Hann window in
place before computing its FFT. RMS, PSD, and band-limited cross-spectral
calculations share these transforms. Raw traces and transformed buffers
therefore represent different processing stages.

`WindowPlan` resolves each metric's input requests once per block and shares
identical FFT requests within each window. Raw-only consumers run before FFT
allocation. Consumers borrow read-only inputs for an update; the plan releases
intermediate products after their last consumer. Accumulators keep their own
summaries rather than retaining borrowed traces or spectra.

RMS values are retained per time window. PSD and cross-spectral statistics
accumulate across windows; coherence and correlation outputs are derived from
those accumulated statistics.

## Reads and FFT workers have explicit bounds

The outer block pool uses `num_parallel_jobs`. Each block prefetches up to
eight raw windows, processing completed reads while replenishing the bounded
queue. It does not queue the entire recording's windows into memory.

Reads can finish out of time order. Each accumulator receives sequential
updates within its block, with a window index identifying the sampled interval.
Different block workers use independent accumulators.

FFT workers per block are `max(1, cpu_count // max(1, num_parallel_jobs))`.
WavPack decoding threads are also configured from `num_parallel_jobs` when
the codec exposes that setting. Multiple block workers can hold sample
buffers and per-channel-pair accumulators concurrently. Band accumulators
have square channel axes, so their memory use also depends on channel count.

Logs report elapsed time per block and for all streams together. These totals
include the current metric calculations; they do not isolate individual
metrics' costs.

## Stream assembly uses the shared channel table

After block workers finish, results are restored to each stream's block order
and passed to `_assemble_and_save_stream()`. Block-local channel maps place
measurements onto the shared contact table. RMS output retains missing
contacts as `NaN`; pairwise support has separate limitations documented in the
[channel geometry contract](shank_channel_metadata_spec.md).

Optional metrics enter through `extract_continuous(extra_metrics=...)`.
Each metric defines its inputs and creates an accumulator for each block.
`BlockMetrics` carries the finished results and resolved block context into
stream assembly. Each metric then reduces its block results before writing
its output; per-channel helpers map results onto the shared contact table.

The [authoring tutorial](metric_extensions.md) builds metrics from ordinary
functions using these hooks. It covers AP traces, LFP traces, shared FFTs, and
bounded snippets, with an [API reference](metric_extension_reference.md) for
custom accumulation, reduction, and saving.
