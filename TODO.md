# TODO

## Pairwise Metric Support For Correlation And Coherency

Per-channel support is now explicit: RMS arrays are dense in channel-table row
order with `NaN` for rows a block did not record, so consumers join by row
position and cannot mistake "not recorded" for a measured zero. See
`docs/shank_channel_metadata_spec.md` §5.2.1.

Channel *pairs* still encode missing support as `0.0`. A pair is only measurable
inside a block containing both members, so in multi-block streams the
cross-block entries of `band_corr/{band}_mean_corr.npy` and
`{band}_coherency.npy` are zeros that were never estimated — and `0.0` is a
valid measured correlation.

**Deliberately deferred.** No consumer reads those entries: the GUI renders one
image per recording block and its colour scaling samples only within-block
sub-matrices, which always have full support. Changing the fill would alter
published artifacts for no observable benefit. Revisit when something actually
reads the full matrix across blocks.

If it is picked up:

- Fill cross-block pairs with `NaN` in `_assemble_blockwise_coherence`; the
  complex matrix needs `complex(np.nan, np.nan)`, since assigning a bare
  `np.nan` leaves the imaginary part a real zero.
- Do not add a per-pair weight sidecar. Support is derivable — `support(i, j) >
  0` iff some `blocks[]` entry's `rows` contains both `i` and `j` — and dense
  pairwise weights would be ~590k entries per probe.
- Audit consumers for full-matrix reductions first: `np.quantile` and friends
  are NaN-poisoning.

Acceptance criteria:

- Consumers can render unmeasured pairs differently from valid zero or
  low-valued correlations without inferring support from geometry.
- A multi-block stream with disjoint channel maps does not encode unmeasured
  cross-block pairs as zero.
- `channels.localCoordinates.npy`, `channels.contactId.npy`,
  `channels.shankInd.npy`, and channel-table row position remain the join keys
  for every dense metric artifact.
