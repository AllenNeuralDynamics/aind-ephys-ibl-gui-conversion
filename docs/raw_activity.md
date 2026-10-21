# Raw AP activity per channel

Raw activity measures supported negative-event rates in AP traces. Several
contacts can detect the same event, so these rates do not count distinct
neurons. Enable the metric alongside the default continuous metrics:

```python
from aind_ephys_ibl_gui_conversion import extract_continuous
from aind_ephys_ibl_gui_conversion.activity import raw_activity

extract_continuous(
    sorting_folder=sorting_folder,
    results_folder=results_folder,
    extra_metrics=(raw_activity(),),
)
```

The metric uses the sampled extension API with an independent accumulator per
experiment block. It adds three files in each probe folder:

- `_iblqc_channels.rawActivity.npy`: event rates in Hz.
- `_iblqc_channels.rawActivitySampledDuration.npy`: analyzed seconds per
  contact, after filter-edge trimming and channel rejection.
- `_iblqc_rawActivity.metadata.json`: parameters, exact sampled intervals,
  block identities, units, and pooling policy.

Entry `i` refers to row `i` of `channels.localCoordinates.npy`, with shank
identity in `channels.shankInd.npy`. Mixed pitches, depth offsets, split-bank
columns, and overlapping survey coverage retain those coordinates. Missing or
rejected measurements remain `NaN` with zero coverage. Conversion exports
per-contact rates without depth binning, interpolation, bank stacking,
smoothing, or profile normalization.

## Sampling and aggregation

The default selects one one-second snippet per block, centered within the
middle scheduled read. It keeps the original estimator's sampling budget;
the location can differ from its original midpoint chunk. A 25 ms trim at
each edge leaves 0.95 seconds of analyzed data at 30 kHz. The metric adds no
recording reads or FFTs, although filtering and detection still add computation
and temporary arrays.

To select up to three distributed snippets instead:

```python
from aind_ephys_ibl_gui_conversion.metric_context import SampleBudget

activity = raw_activity(sampling=SampleBudget(max_windows=3))
```

Snippets must fit inside existing reads and leave a positive core after
trimming. Configuration is validated before recording I/O. See the
[execution overview](continuous_metric_pipeline.md) and
[authoring tutorial](metric_extensions.md) for the shared machinery.

The accumulator preserves the original detector's per-shank reference,
bandpass, MAD thresholds, artifact vetoes, trimmed mean of snippet rates,
and channel-health gates. With fewer than ten snippets the 10% trimmed mean
removes no tails. With one snippet the temporal noise-stability gate has no
effect. Artifact vetoes suppress events without removing time from the rate
denominator.

Repeated observations of the same physical contact combine as the median of
healthy block rates. Coverage sums analyzed seconds from those blocks. It
describes support for the estimate and is not a pooled-count denominator.
This reduction combines repeated measurements of one contact; contacts at
the same depth remain separate when their lateral coordinates differ.

## Detector limitations remain separate from pipeline integration

This change preserves the detector from PR #46 rather than introducing new
physical-radius parameters. Its spatial support and local-noise gates still
use neighborhoods in depth-sorted channel order. Those neighborhoods can span
physical bank gaps. The original peak selector can also count a flat peak
multiple times and accept peaks closer than the nominal refractory interval.
These are known detector limitations for separate correction and review.

The frozen synthetic test checks original-detector parity on identical
snippets. Integration tests cover unchanged reads and FFT counts, out-of-order
updates, overlap, mixed pitch, split-bank columns, depth offsets, rejected
contacts, and the saved channel-table order. The metric-specific metadata
records the legacy spatial-support rule and the chosen contact pooling.

`scripts/benchmark_activity.py` compares activity disabled, the original
estimator, and the integrated metric with one or three snippets in fresh
processes. It checks complete existing metric arrays and exact read, FFT, and
compressed-chunk counts. For example:

```sh
uv run python scripts/benchmark_activity.py --recording /path/to/recording.zarr --blocks 4 --repeats 3 --output /tmp/activity-comparison.json
```

The original PR commit must be available locally for that comparison. Previous
timing trials ran while the machine was heavily loaded and included proposed
detector corrections; they do not establish this integration's performance.
Repeat timing comparisons on a quiet machine. Inputs remain read-only.
