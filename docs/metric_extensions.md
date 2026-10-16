# Writing a continuous metric

Say you want to calculate the mean absolute voltage of each channel in an AP
recording. Write a function that calculates on traces, then pass it to
`channel_mean()`. The library supplies window accumulation, aggregation by
physical contact, and saving. No subclasses are needed.

This tutorial starts with that complete example, switches it to LFP traces,
then works through the library's own FFT-based band-power calculation.

## The building blocks

A **block** is one retained recording segment from an Open Ephys experiment
within a probe stream. Main and survey recordings can supply several blocks
with different contact banks. A **window** is a short interval sampled within
one block. The pipeline schedules windows across the recording and reads them
for all enabled metrics.

The computation and its accumulation are separate:

| Building block | Responsibility |
| --- | --- |
| Computation function | Calculate values from the supplied traces or spectrum |
| Accumulator | Call the computation for each window, retain statistics, and return a block result |
| Preparation function | Choose inputs and bind computation parameters for one block |
| Reduction function | Combine block results into output contact rows |
| Writer | Save the reduced result and metadata |

`channel_mean()` assembles these pieces for a time-weighted mean of raw-trace
or prepared measurements. `channel_metric()` lets you supply preparation and
reduction functions while keeping the standard contact writer. The accumulator
protocol remains available for calculations that need specialized state.

### Choose `compute` or `prepare`

Pass `source` and `compute` when your function works directly on supplied raw
traces with fixed parameters. Use `prepare` when setting up the calculation
requires block metadata or more specific inputs: an FFT frequency mask,
filter coefficients derived from the sampling rate, a snippet budget, or
shank geometry.

`prepare(block)` runs once per block, before trace reads, and returns a
`ChannelComputation` with the input and bound computation function.
`compute(data)` runs once per selected window. The library creates the
accumulator and handles aggregation and saving in either case. The first
example uses `compute` directly; the FFT example introduces `prepare`.

## Example 1: mean absolute AP voltage

Put this complete example in `my_metrics.py` beside your conversion script,
or in your own package. A contribution to this library would live under
`src/aind_ephys_ibl_gui_conversion/`.

```python
import numpy as np

from aind_ephys_ibl_gui_conversion.metric_extensions import channel_mean


def mean_absolute(traces):
    """Return mean absolute voltage for each native channel."""
    return np.mean(np.abs(traces), axis=0, dtype=np.float64)


ap_mean_absolute = channel_mean(
    name="apMeanAbsolute",
    source="ap",
    compute=mean_absolute,
    units="uV",
)
```

`traces` has shape `(samples, native_channels)` and contains microvolts.
Your function returns one value per native channel. The AP source is the
recording's AP stream; the request applies no additional reference or filter.
On NP2, it is wideband. Add filtering in the computation if your measurement
requires a particular passband.

`channel_mean()` requests each existing raw read and constructs a fresh
`ChannelMeanAccumulator` for every block. The accumulator calls your function
once per window and retains two channel-sized arrays: the duration-weighted
sum and valid sampled seconds. A nonfinite value excludes that channel's
entire window from both totals.

Your function should be stateless and safe to call concurrently across blocks.
Inputs are borrowed and read-only. Copy anything you need to modify, and
release inputs after the calculation. The library owns recording I/O,
prefetching, and concurrency.

### Run it and inspect the output

Import the configured metric in the conversion script:

```python
from pathlib import Path

from aind_ephys_ibl_gui_conversion import extract_continuous
from my_metrics import ap_mean_absolute

results = Path("/results")
results.mkdir(parents=True, exist_ok=True)

extract_continuous(
    sorting_folder=Path("/data/session_sorted"),
    session_folder=Path("/data/session"),
    results_folder=results,
    extra_metrics=(ap_mean_absolute,),
)
```

Replace the paths with your data; see the [input layout](../README.md#input-data-expectations).
The usual metrics continue to run. Your metric adds three files in each
probe's output folder:

```text
_iblqc_channels.apMeanAbsolute.npy
_iblqc_channels.apMeanAbsoluteSampledDuration.npy
_iblqc_apMeanAbsolute.metadata.json
```

The arrays have one entry per row of `channels.localCoordinates.npy`, which
identifies a physical contact using its probe-local coordinates. Reduction
maps native trace columns onto these rows, sums the weighted values and valid
durations, then divides. Overlapping banks contribute to the same physical
contact even when it occupies different trace columns. Unmeasured contacts
retain NaN values and zero duration.

The JSON records units, source intervals, window selection, and the pooling
policy. Supply `metadata={...}` to `channel_mean()` to record additional
method parameters. Distinct configurations need distinct names.

## Example 2: use the same function on LFP traces

Append another configured metric to `my_metrics.py`:

```python
lfp_mean_absolute = channel_mean(
    name="lfpMeanAbsolute",
    source="lfp",
    compute=mean_absolute,
    units="uV",
)
```

Pass both instances in `extra_metrics`. They share the stateless function and
have independent accumulators. The library uses the chosen source's sampling
rate and channel geometry. NP1 AP and LFP recordings can have different rates
and channel orders; NP2 supplies the same wideband recording for both, so
these two unfiltered measurements agree there. For an LF frequency band on
NP2, filter the traces or select FFT bins.

## Work with the accumulator directly

You can test the computation and accumulation without a recording:

```python
from aind_ephys_ibl_gui_conversion.metric_extensions import (
    ChannelMeanAccumulator,
)

accumulator = ChannelMeanAccumulator(
    compute=mean_absolute, n_channels=2, seconds=1.0
)
accumulator.update(1, (np.array([[-2.0, 4.0], [2.0, -4.0]]),))
accumulator.update(0, (np.array([[-4.0, 8.0], [4.0, -8.0]]),))
value = accumulator.finish()
np.testing.assert_allclose(value.weighted_sum / value.sampled_seconds, [3, 6])
```

`update(window_index, inputs)` expands the input tuple into positional function
arguments: `compute(*inputs)`. `finish()` returns a `ChannelMeanResult` with
`weighted_sum` and `sampled_seconds` in native channel order. Out-of-order
windows leave this sum unchanged. A time-series accumulator would instead
store results by `window_index`.

## Example 3: prepare an FFT-based computation once per block

Band power needs a frequency mask and normalization derived from the block's
sampling rate and FFT length. Resolve those scientific settings in a
preparation function. Return a `ChannelComputation` containing the input and
bound computation. The library derives the rest of the setup.

The production [`lfp_band_power()`](../src/aind_ephys_ibl_gui_conversion/metric_extensions.py)
uses this calculation:

```python
from functools import partial

from aind_ephys_ibl_gui_conversion.metric_context import BlockContext
from aind_ephys_ibl_gui_conversion.metric_extensions import ChannelComputation
from aind_ephys_ibl_gui_conversion.window_pipeline import Spectrum


def band_power(spectrum: Spectrum, *, mask, scale):
    """Return mean-square power in interior positive-frequency bins."""
    power = np.sum(np.abs(spectrum.coefficients[mask]) ** 2, axis=0)
    return np.asarray(power * scale)
```

`Spectrum.coefficients` has shape `(frequency_bins, native_channels)`.
`BlockContext` supplies source metadata, frequency coordinates, and existing
FFT requests. Preparation binds the band mask and Hann-window normalization:

```python
def prepare_power(context: BlockContext, *, low_hz, high_hz):
    """Resolve the band calculation against one block's existing LF FFT."""
    request = context.lfp_fft
    fs = context.lfp.sampling_frequency
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError("Band must be strictly between DC and Nyquist")
    frequencies = context.frequencies(request)
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if not mask.any():
        raise ValueError("Band contains no FFT bins")
    return ChannelComputation(
        input=request,
        compute=partial(
            band_power,
            mask=mask,
            scale=2.0 / (request.n_samples * context.lfp_window_power),
        ),
        metadata={
            "band_hz": [low_hz, high_hz],
            "high_edge_inclusive": False,
        },
    )
```

Configure it with the same mean constructor as the raw example:

```python
spectral_metric = channel_mean(
    name="power30to100",
    prepare=partial(prepare_power, low_hz=30, high_hz=100),
    units="V^2",
    output_scale=1e-12,
)
```

The library selects channel geometry and sampling rate from the request's
source, then calculates duration as sample count divided by sampling rate.
It creates fresh accumulation state and records exact selected sample
intervals, transform settings, and your method metadata for each block.
`output_scale` converts the pooled µV² result to V². Units and normalization
remain explicit scientific choices.

Requesting `context.lfp_fft` reuses the transform used by default LF metrics,
including its reference, demeaning, and taper. Matching requests share a
transform. Changing a transform setting can require another FFT. The mask
and normalization above are prepared once per block, rather than per window.

The same helper also accepts a `RawRequest` with an offset or a subset of
samples. It derives duration and provenance from that exact interval.
Set `window_indices=(...)` on `ChannelComputation` to select scheduled windows.

The library's band-power factory uses this same preparation path. To run it
in two bands:

```python
from aind_ephys_ibl_gui_conversion.metric_extensions import lfp_band_power

metrics = (
    lfp_band_power(30, 100, name="power30to100"),
    lfp_band_power(100, 200, name="power100to200"),
)
```

Both instances share the FFT and save separate contact arrays. A different
spectral mean can reuse the preparation pattern with another computation and
appropriate normalization.

## A snippet plan carries sampling and analyzed duration

For a bounded raw computation, pass a resolved `SampleBudget` directly as the
computation's input:

```python
from aind_ephys_ibl_gui_conversion.metric_context import SampleBudget

budget = SampleBudget(max_windows=3, seconds=1.0, edge_seconds=0)


def prepare_short_mean(context):
    """Use up to three distributed snippets from the existing AP reads."""
    return ChannelComputation(budget.resolve(context.ap), mean_absolute)


short_mean = channel_mean(
    name="sampledMeanAbsolute", prepare=prepare_short_mean, units="uV"
)
```

The helper obtains selected windows, source, offsets, and analyzed duration
from the plan. It checks the plan against the current block's integer read
starts and sampling rate. You cannot accidentally attach an LF plan to an
incompatible AP source or reuse stale intervals from another block.

This example has no filter edges. For a padded plan, the duration and metadata
exclude both edges automatically, but your computation still performs the
filtering and trims those samples before calculating its result. Bind the
plan's `edge_samples` in your computation function. A plan's window selection
is authoritative, so it cannot be combined with separate `window_indices`.

## Specialized estimators can supply their own accumulation and reduction

The general accumulator interface has two methods:

- `update(window_index, inputs)` consumes one window and retains owned state.
- `finish()` returns one block's result.

For custom accumulation, `channel_metric()` takes a preparation function that
returns a `MetricDefinition` with a factory for an accumulator implementing
these methods. A plain class can hold its state without subclassing a library
class.
`channel_metric()` accepts a custom pure reduction function with signature
`reduce(stream_context, block_results) -> ChannelOutput`. Each block result
pairs the accumulator's value with its original `BlockContext`.

A robust estimator can retain small per-window summaries, compute a block
estimate in `finish()`, and combine block estimates with a median. Such an
estimator needs its own reduction because a time-weighted mean would change
its definition. The [function-composition tests](../tests/test_functional_metrics.py)
include a complete peak accumulator, reduction, and writer without library
base classes.

For other output shapes, construct a `Metric` with your own `define`, `reduce`,
`write`, and declared `output_files`. These callables satisfy the existing
`WindowMetric` protocol. No subclass or registry is required.

## Check correctness and added work

The [tutorial tests](../tests/test_metric_tutorial.py) execute these definitions
through the real worker and writer, checking AP/LFP rates, reordered and
overlapping contacts, missing coverage, and unchanged existing outputs.
The [band-power tests](../tests/test_metric_extensions.py) check a known
sinusoid, agreement with the existing PSD, and unchanged FFT counts.

Default RMS, PSD, and coherence use the same `MetricDefinition` and accumulator
protocol, while retaining their legacy stream assembly and saving. Band power
uses the complete function-composition path described here.

Measure added reads, transforms, elapsed time, and memory on compressed data
with concurrent blocks. Shared I/O still leaves the computation's CPU and
temporary-array costs. The [experiment notes](../METRIC_PIPELINE_EXPERIMENT.md)
include benchmark commands; the [API reference](metric_extension_reference.md)
lists detailed contracts and the [channel contract](shank_channel_metadata_spec.md)
defines output contact rows.
