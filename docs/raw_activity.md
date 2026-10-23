# Raw AP activity per contact

Raw activity measures supported negative-event rates in AP traces. It is a
spatial activity proxy: several contacts can detect the same event, and rates
do not count distinct neurons. Enable it alongside the default metrics:

```python
from aind_ephys_ibl_gui_conversion import extract_continuous
from aind_ephys_ibl_gui_conversion.activity import raw_activity

extract_continuous(
    sorting_folder=sorting_folder,
    results_folder=results_folder,
    extra_metrics=(raw_activity(),),
)
```

Each probe folder gains three files:

- `_iblqc_channels.rawActivity.npy`: event rates in Hz.
- `_iblqc_channels.rawActivitySampledDuration.npy`: analyzed seconds for each
  contact, after filter-edge trimming and channel rejection.
- `_iblqc_rawActivity.metadata.json`: parameters, exact sampled intervals,
  block identities, units, and pooling policy.

Array entry `i` refers to row `i` of `channels.localCoordinates.npy`, with
shank identity in `channels.shankInd.npy`. Mixed pitches, depth offsets,
split-bank columns, and overlapping survey coverage retain those coordinates.
Missing or rejected measurements remain `NaN` with zero coverage. Conversion
exports these per-contact measurements without depth binning, interpolation,
bank stacking, smoothing, or profile normalization.

## Sampling shares the existing recording reads

The default budget selects one one-second snippet per experiment block,
centered within the middle scheduled read. Each snippet
discards 25 ms at either edge after filtering, leaving 0.95 seconds of analyzed
data at 30 kHz. Filtering and event detection still add CPU work and temporary
arrays. A three-snippet budget analyzes 2.85 seconds distributed across the
block without adding recording reads or FFTs.

To change the budget or detector parameters:

```python
from aind_ephys_ibl_gui_conversion.activity import ActivityParams
from aind_ephys_ibl_gui_conversion.metric_context import SampleBudget

activity = raw_activity(
    sampling=SampleBudget(max_windows=3, seconds=1.0, edge_seconds=0.025),
    params=ActivityParams(support_radius_um=40, noise_radius_um=100),
)
```

One snippet selects the middle scheduled window; it need not coincide with
the recording's exact midpoint. Snippets must fit inside existing reads and
leave a positive core after trimming. The metric validates these constraints
before recording I/O. See the [execution overview](continuous_metric_pipeline.md)
and [metric tutorial](metric_extensions.md) for the shared machinery.

## Detection and aggregation

Each snippet is processed independently. The detector subtracts a per-shank
common median, applies a third-order 300–6,000 Hz Butterworth bandpass with
forward/backward filtering, and trims the filter edges. The upper cutoff is
limited to 95% of Nyquist. Per-channel noise is median absolute filtered voltage
divided by 0.67448975, with a floor based on the shank's positive noise values.

A negative peak must exceed 4.5 noise standard deviations. It needs threshold
support above 3 standard deviations from two contacts, including itself, within
40 µm Euclidean distance on its shank. Support uses a 0.25 ms moving window.
Samples where more than 8% of shank contacts exceed 8 noise standard deviations
in either direction trigger an artifact veto, extended by 0.5 ms on both sides.

Flat peaks count once. Eligible peaks on one channel must be at least 0.5 ms
apart; stronger peaks take precedence, with equal-height ties resolved by
SciPy's peak-selection routine. Temporal lengths round to native samples,
with a minimum of three samples for peak spacing and the support window.

The accumulator retains only per-window rates and noise summaries. Block
rates use a 10% trimmed mean; with fewer than ten snippets no tails are removed.
A contact is rejected when its median noise is outside 0.25–4 times the median
noise of contacts within 100 µm on the same shank, or its 90th/10th percentile
noise ratio exceeds five. With one snippet the temporal ratio is one, so that
gate has no effect. Neighborhoods include the contact itself and are prepared
once per block from physical coordinates.

Repeated measurements of one physical contact combine as the median of
healthy block rates. Coverage sums analyzed seconds from those blocks; it is
descriptive support, not the denominator of a pooled event-count rate.
Artifact vetoes suppress events without subtracting time from the denominator.
These thresholds and pooling choices are heuristics that need validation for
the intended recording conditions.

The [validation notes](raw_activity_validation.md) compare the original PR
detector with this implementation and describe the performance measurements.
