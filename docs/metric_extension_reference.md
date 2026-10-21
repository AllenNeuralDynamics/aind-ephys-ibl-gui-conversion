# Metric extension API reference

Start with the [tutorial](metric_extensions.md) to enable a metric and follow
its calculation. This page describes the contracts for implementing metrics.

Continuous-metric extensions run inside `extract_continuous()`. The pipeline
reads sparse windows, shares requested FFTs, and creates a separate accumulator
for each metric in each block. A block contains one retained recording segment
from an Open Ephys experiment within a probe stream.

This guide describes the extension API implemented in this checkout. Existing
QC metrics run by default; callers enable additional metrics with
`extra_metrics`.

## Callers select metrics through `extra_metrics`

Set the paths to your raw session, sorted data, and result folder. The raw
session must contain `ecephys_compressed`; see the
[input layout](../README.md#input-data-expectations) for details.

```python
from pathlib import Path

from aind_ephys_ibl_gui_conversion import extract_continuous
from aind_ephys_ibl_gui_conversion.metric_extensions import lfp_band_power

results = Path("/results")
results.mkdir(parents=True, exist_ok=True)

extract_continuous(
    sorting_folder=Path("/data/session_sorted"),
    session_folder=Path("/data/session"),
    results_folder=results,
    extra_metrics=(lfp_band_power(low_hz=30, high_hz=100),),
)
```

Each metric writes into the usual per-probe output folder. Both supplied
extensions produce one scalar per existing channel-table row:

| Metric | Default output attribute | Units | Reduction across overlapping blocks |
| --- | --- | --- | --- |
| `lfp_band_power` | `lfpBandPower` | V² | Mean weighted by valid sampled time |

For attribute `lfpBandPower`, the common writer produces:

```text
_iblqc_channels.lfpBandPower.npy
_iblqc_channels.lfpBandPowerSampledDuration.npy
_iblqc_lfpBandPower.metadata.json
```

The duration array records valid analyzed time in seconds. Band power pools
values by this duration. Unmeasured contacts retain NaN values and zero
coverage. Metadata records units, parameters, provenance, and the reduction
policy.

Give distinct names to distinct configurations:

```python
metrics = (
    lfp_band_power(30, 100, name="betaPower"),
    lfp_band_power(100, 200, name="gammaPower"),
)
```

These instances write attributes `betaPower` and `gammaPower`. Names must
start with a letter and contain only letters, digits, or underscores. The
pipeline rejects duplicate instance names, colliding output filenames, and
reserved existing artifacts before trace reads begin.

## Sampling budgets bound processing within existing reads

`SampleBudget` selects snippets from the pipeline's scheduled windows. It
centers each snippet within its read and distributes multiple selections
between the first and last available windows. A one-window budget selects
the middle window. If fewer windows exist than requested, it uses those
available.

`seconds` includes the filter edges. At 30 kHz, a one-second snippet with
25 ms removed from each edge retains 28,500 samples, or 0.95 seconds. Three
snippets therefore analyze 2.85 seconds. Requests that exceed existing reads
or produce overlapping analyzed cores fail during planning.

A raw metric resolves its budget in `define()`:

```python
sampling = self.sampling.resolve(context.ap)
request = sampling.request
indices = sampling.window_indices
```

The returned `SnippetPlan` supplies a `RawRequest`, selected window indices,
edge length in samples, and exact source intervals through `metadata()`.
The accumulator must apply its filter and trim those edges; the helper plans
the core without altering the input array.

More snippets increase processing cost even when recording reads stay the
same. Keep sampling explicit and measure the computation's added cost.

## A metric defines work, reduces results, and writes outputs

Compose functions with `Metric`, `channel_metric()`, or `channel_mean()`.
Protocol-based classes can also implement `WindowMetric`. Keep configuration
immutable. Callers register the instance through `extra_metrics`; no global
registry is required.

| Method | Pipeline calls it | Author supplies |
| --- | --- | --- |
| `define(block_context)` | Once per block, before trace reads | A `MetricDefinition` with input requests, a factory, and optional window selection |
| `reduce(stream_context, results)` | After the stream's block workers finish | A pure reduction of owned block results |
| `write(output_folder, result)` | After extension reductions succeed | Files listed in `output_files` |

`MetricDefinition.name` must match the instance name. Its `create` factory
constructs fresh accumulator state for that block; `functools.partial` can
bind resolved sizes and parameters. `window_indices=None` selects every
scheduled window. A snippet metric passes its plan's selected indices.

The accumulator implements two methods:

- `update(window_index, inputs)` consumes the requested products in their
  declared tuple order. Calls are sequential within a block but can arrive
  out of time order. Index time-series outputs explicitly and filter each
  independent window with fresh temporal state.
- `finish()` returns owned block results. Retain the statistics required by
  the estimator: sums and support for additive metrics, or small per-window
  summaries for robust reductions.

Keep configuration free of mutable accumulation state because multiple block
workers share the same metric instance. Accumulators receive read-only borrowed
views in native recording channel order. They must release those inputs after
`update()` and copy any data they need to modify. The pipeline owns trace I/O,
prefetch, and concurrency; extensions process the supplied arrays.

Two complete implementations provide starting points:

- [`lfp_band_power`](../src/aind_ephys_ibl_gui_conversion/metric_extensions.py)
  requests an existing FFT, accumulates power and exposure, and composes the
  common channel writer.
## Functions compose preparation, accumulation, reduction, and saving

The public authoring helpers in `metric_extensions` accept ordinary functions:

| Entry point | Author supplies | Library supplies |
| --- | --- | --- |
| `channel_mean(name=..., source=..., compute=..., units=...)` | `compute(traces)` returns one scalar per native channel | Full raw read requests, accumulation and contact reduction, provenance, standard files |
| `channel_mean(name=..., prepare=..., units=...)` | `prepare(context)` returns a `ChannelComputation` | Geometry, duration, accumulator factory, reduction, provenance, and saving |
| `ChannelComputation(input, compute, window_indices=None, metadata={})` | One raw/FFT request or `SnippetPlan`, its bound function, and method parameters | A declaration resolved once per block; no mutable accumulation state |
| `channel_metric(name=..., define=..., reduce=..., attribute=None)` | Per-block preparation and a pure `ChannelOutput` reduction | Standard filenames and writer; `attribute` defaults to the instance name |
| `Metric(name, define, reduce, write, output_files)` | All callables and output filenames | Immutable configuration satisfying `WindowMetric` |
| `ChannelMeanAccumulator(n_channels, seconds, compute)` | `compute(*inputs)` returns one scalar per native channel | Duration-weighted sums with `update()` and `finish()` |
| `reduce_channel_mean(context, results, units=..., source=None)` | Native totals, with a source on each result or an explicit fallback | Pure contact mapping and sampled-time-weighted pooling |

Choose `source` and `compute` for a function operating directly on raw traces.
Choose `prepare` when setup needs block metadata, an FFT request, or a snippet
plan. Preparation runs once per block before reads; computation runs once
per selected window.

The raw form of `channel_mean()` uses each full existing read without extra
reference or filtering. It optionally accepts `window_indices`. The prepared
form takes its selection from `ChannelComputation` or its snippet plan;
combining `prepare` with `source`, `compute`, or `window_indices` is an error.
Both forms accept JSON `metadata`, `output_scale=1.0`, and an output `attribute`
that defaults to the metric name. Names and files are checked before reads.

A prepared computation's request supplies source geometry, native channel
count, sample count, and offset. The helper derives duration and records
selected sample intervals, analyzed intervals, and the request's full
preprocessing specification. Method metadata is added per block. Derived
sampling facts take precedence over conflicting method metadata.

A `SnippetPlan` supplies filter edges and its own window selection. The
helper validates its starts and sampling rate against the current source,
and derives exposure from the retained core. The function must still perform
filtering and trim the declared edges before calculating values. A plan cannot
also specify separate window indices. Units, normalization, rejection rules,
and the calculation itself remain the author's scientific choices.

`ChannelMeanAccumulator` calls `compute(*inputs)` once per selected window.
The input function is bound once during preparation. It must be stateless and
safe to call concurrently across blocks. Inputs remain borrowed and read-only;
there is no signature inspection, argument injection, or recording handle.
The output has shape `(n_channels,)`. A nonfinite value excludes that channel's
whole window from sum and coverage.

`finish()` returns `ChannelMeanResult(weighted_sum, sampled_seconds,
input_source=None, provenance=None)` in native channel order. The prepared
path fills the input source and exact provenance automatically. A manually
constructed accumulator accepts these as optional keyword arguments; otherwise
its reducer needs an explicit source. A fixed duration applies to every update
within a block. The helper retains O(channels) numerical state plus small input
metadata and performs no recording reads or transforms itself.

`reduce_channel_mean()` maps each result using its recorded source. An explicit
source cannot contradict it. The reducer checks shapes, finite sums,
nonnegative finite coverage, and zero sums for unmeasured channels. Scaling
applies after pooling. Empty results produce NaN values and zero coverage.

For multiple inputs, specialized accumulation, or another pooling rule, use
`channel_metric()` and return a `MetricDefinition` with a fresh accumulator
factory from your preparation function. Its name must match the configured
metric. Bind shared immutable parameters once per block, and keep mutable
state in the accumulator. The general accumulator protocol requires only
`update()` and `finish()`. A custom result can retain robust summaries and
use a separate reduction function while keeping the standard writer.

## Block contexts supply the resolved data contract

Use `BlockContext` to construct requests and accumulator state:

| Context member | Meaning |
| --- | --- |
| `ap`, `lfp` | Per-source sampling rate, read length, integer read starts, native channel IDs, locations, contacts, and shanks |
| `n_windows`, `timestamps` | Schedule length and shared window centers |
| `ap_fft`, `lfp_fft` | Exact existing transform requests, including preprocessing |
| `frequencies(request)` | Frequency coordinates at that request's source rate |
| `lfp_window_power` | Hann energy used by the existing LF transform |
| `block_id`, `block_source` | Unique extraction identity and resolved `main`/`surface` provenance |

Raw arrays contain microvolts; locations contain probe-local micrometres.
Metadata arrays are read-only snapshots. Use per-source integer starts for
sampling provenance, including clamped LFP reads. `block_index` retains the
acquisition-local number and can repeat across main and survey assets.

NP1 sources have independent sampling rates and channel orders. NP2 AP and
LFP contexts refer to the same wideband source. Their transform requests still
define the signal: NP2 AP RMS uses the referenced, demeaned, Hann-windowed
spectrum shared with LF metrics. Inspect the request rather than assuming an
AP label means unreferenced data.

A spectral consumer can request that existing product directly:

```python
request = context.lfp_fft
frequencies = context.frequencies(request)
mask = (frequencies >= low_hz) & (frequencies < high_hz)
```

The corresponding accumulator receives a `Spectrum` and reads its
`coefficients`. Equal requests share one product per window. Changing source,
sample interval, dtype, reference, demeaning, taper, or normalization can
require another FFT. Each metric remains responsible for its normalization
and units. Time-domain event detection uses raw samples because spectral power does
not retain event timing.

## Stream reductions preserve contact identity and estimator semantics

`reduce()` receives `MetricBlockResult` objects in the stream's original
block order. Each pairs a typed value with its `BlockContext`. The
`StreamContext` supplies canonical geometry without an output destination.

Map native columns with `stream_context.channel_rows(block_context.ap)` or
the corresponding LFP source. The mapping follows the existing
[channel-table contract](shank_channel_metadata_spec.md), handles reordered
columns, and rejects contacts absent from the table. Preserve these rows
across overlap, gaps, and changes in recorded banks.

Choose pooling to match the estimator. Band power combines summed power and
sampled time. A robust estimator can instead retain per-window summaries
and combine block estimates with a median. Choose pooling to preserve the
definition of the estimator.

For scalar channel results, compose `channel_metric()` with your preparation
and reduction functions. Return a `ChannelOutput` containing
values, sampled seconds, units, and method metadata. Call
`output.validate(n_rows)` before returning. The composed writer declares
and writes the three files shown above. Custom writers declare simple
filenames relative to the stream folder and implement `write()` themselves.

## Tests cover calculation, mapping, and incremental work

Test a new metric's numerical result against an independent calculation or
known signal. Test missing coverage, rejected channels, out-of-order updates,
and reduction across overlapping or reordered contacts. Check that invalid
configuration fails before reads and that enabling the metric preserves
existing outputs.

The current tests provide examples:

```sh
uv run pytest -q tests/test_metric_api.py tests/test_sample_budget.py
uv run pytest -q tests/test_metric_extensions.py tests/test_functional_metrics.py
```

Measure added reads, chunk accesses, transforms, elapsed time, and process
memory on compressed recordings with concurrent blocks. Extra reductions
and private filters still consume CPU when they reuse I/O. The
[experiment notes](../METRIC_PIPELINE_EXPERIMENT.md) contain reproducible
benchmarks and the current measurements.
