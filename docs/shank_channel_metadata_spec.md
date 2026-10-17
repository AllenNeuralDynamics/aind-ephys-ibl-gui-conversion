# Ephys Collection And Channel Geometry Contract

Status: current implementation. Conversion produces these artifacts;
preprocessing describes the collection in `datapackage.json`; the alignment
GUI consumes both explicit references and the collection layout.

## Collection Boundary

Each `(recording_id, ephys_collection)` has one ephys directory. Both spike
extraction and continuous-metric extraction write into it. Spikes, clusters,
channel metadata, and ALF QC objects are direct children; correlation and
coherency products live in its `band_corr/` subdirectory.

```text
<recording_id>/<ephys_collection>/
  spikes.*.npy
  clusters.*.npy
  clusters.metrics.csv
  spike_shank_indices.npy
  unit_shank_indices.npy
  channels.{localCoordinates,rawInd,contactId,shankInd}.npy
  _iblqc_ephysTimeRmsAP*.{rms,timestamps}.npy
  _iblqc_ephysTimeRmsLF*.{rms,timestamps}.npy
  _iblqc_ephysSpectralDensityLF*.{power,freqs}.npy
  _iblqc_metrics.method.json
  band_corr/
    <band>_mean_corr.npy
    <band>_coherency.npy
    <band>_shank<k>_mean_corr.npy
    <band>_shank<k>_coherency.npy
    row_channels.json
    channel_blocks.json
```

This illustrates naming conventions, not files guaranteed in every run.
Products depend on recording configuration and available measurements.
`channel_blocks.json` is a legacy multi-block artifact. `Main` variants occur
in multi-block output. See [README.md](../README.md) for metric semantics.

The datapackage explicitly references the collection directory and channel
files. It does **not** enumerate each spike, QC, or matrix file. Their names,
shapes, and placement are a separate producer–consumer contract. Moving QC
requires a corresponding reader change; editing the datapackage alone cannot
redirect individual QC objects. Schema validation does not establish that all
plots have usable inputs. Preserve the complete collection when copying it.

## Channel Table And Join Key

`channel_metadata.build_channel_table()` constructs one table across blocks.
It deduplicates channels by `(x, y)` rounded to the nearest micrometre. Contact
and shank IDs annotate the table; they are not deduplication keys. The first
encountered channel supplies a new row's metadata, so overlapping blocks need
consistent geometry and shank attribution.

| File | Shape | Meaning |
|---|---|---|
| `channels.localCoordinates.npy` | `(N, 2)` | Lateral position and axial depth, in micrometres |
| `channels.rawInd.npy` | `(N,)` | Legacy positional `0..N-1` placeholder |
| `channels.contactId.npy` | `(N,)` | Physical contact IDs; empty strings when unavailable |
| `channels.shankInd.npy` | `(N,)` | Zero-based producer shank ID per row, potentially sparse |

The join key is **table row position**, not `rawInd`, a device channel number,
or contact ID. Order is first occurrence across blocks, not depth order.
Block-local maps translate recording columns into table rows. Changing block
order can change numbering; regenerate table, arrays, and row maps together.

RMS/PSD channel axes use table order. Per-block RMS retains all N columns and
uses `NaN` for contacts not recorded in that block. Missing support must not
become zero, which is a valid measurement. Plot depth comes from
`localCoordinates[rows, 1]`, never row number or assumed uniform spacing.

## Recording Geometry And Surface-Finding Overlap

Acquisitions can use different contact pitches, partial banks, gaps, and
split-bank selections with two columns whose depth rows do not overlap.
The Open Ephys GUI also permits overlapping recordings in surface-finding
mode, and this capability is in use. A surface recording can therefore share
contacts with the main recording; its label does not imply an adjacent bank.

Use each block's physical `(x, y)` coordinates and block-to-table row map to
place measurements. In the channel arrays, entry `i` describes the same
physical contact across coordinates, shank assignments, and channel-aligned
metrics. The alignment GUI uses these coordinates and assignments to select
contacts and place values at physical depths. Array adjacency alone does not
establish a physical neighborhood, especially across gaps or split banks.

New per-contact metrics should follow this existing ordering and retain
missing or rejected measurements as `NaN`. When blocks measure the same
contact, define how repeated measurements are combined or retain separate
block values. Deduplicating geometry does not define a rate-aggregation rule.

A separately resampled depth profile needs explicit physical coordinates and
coverage. Document how it pools columns, reconciles different pitches, and
handles gaps, edge extrapolation, and smoothing. The smallest observed depth
spacing need not divide every other spacing: contacts on 15 µm and 20 µm
grids do not all lie on one 15 µm grid. Keep measured values and coverage
available so display interpolation cannot conceal unsupported regions.

## Shank Identity

`normalized_channel_groups()` uses this precedence:

1. If every channel has a valid prefixed contact ID such as `s2e17`, use its
   physical shank prefix (`2`), preserving potentially non-contiguous IDs.
2. Otherwise normalize SpikeInterface `group` values to contiguous zero-based
   indices in sorted unique-value order.
3. If groups are absent, use one shank (`0`).

Contact IDs come from the recording property or attached probe, projected into
recording order through device channel indices when possible. Incomplete or
malformed contact metadata is not used for physical identity. This preference
matters for surface-finding blocks whose `group` can be flat despite spanning
multiple shanks.

The GUI normalizes producer IDs into local contiguous indices. For
`[2, 2, 5, 5]`, local indices are `[0, 0, 1, 1]`. Physical IDs, local GUI
indices, and one-based display labels are distinct.

## Matrices And Row Maps

Full correlation and complex coherency matrices have shape `(N, N)` in table
order. `band_corr/row_channels.json` version 3 describes them:

```json
{
  "version": 3,
  "matrix_rows": [0, 1, 2, 3],
  "matrix_row_order": "channel_table",
  "shanks": {
    "0": {"rows": [1, 0], "legacy_file_index": 1, "blocks": []},
    "1": {"rows": [3, 2], "legacy_file_index": 2, "blocks": []}
  }
}
```

`shanks.<id>.rows` selects table positions sorted by axial depth, then lateral
position. Select **both** matrix axes with these rows, and use the same rows to
look up physical coordinates. Block entries record `block_index`, `label`,
`rows`, `n_channels`, `duration_s`, and `n_windows`. Window counts supply
aggregation weights; different blocks may cover different rows.

Per-channel `NaN` support does not extend to every pairwise entry: currently
unmeasured channel pairs can be stored as zero correlation/coherency. Zero is
also a valid measured result. Do not infer pairwise support from matrix values
alone; block row membership matters. See the open support work in
[TODO.md](../TODO.md#pairwise-metric-support-for-correlation-and-coherency).

JSON keys are producer shank IDs. Compatibility files use
`legacy_file_index = producer_shank_id + 1` and are sliced from full matrices.
The current GUI matrix reader tries the local-index key, then its one-based
alternative, then filename fallbacks. It does not universally remap sparse
producer IDs: successful channel selection does not guarantee every matrix
plot handles arbitrary sparse IDs. Check both paths when changing numbering.

## Histology Joins And Worked Examples

Preprocessing groups manifest rows by `(recording_id, ephys_collection)` and
records per-track `histology_shank` and `ephys_shank` in `xyz_picks`.

| Acquisition | Manifest mapping | GUI units |
|---|---|---|
| Two shanks in one stream | Same collection; ephys shanks 0 and 1 | Two local shanks within one collection |
| Physical shank 3 in a split stream | `histology_shank=3`, `ephys_shank=0` | One local shank within its collection |

The split stream may retain contact IDs such as `s3e17`; its one-shank table
normalizes to local shank 0. Histology filenames can have one-based physical
shank suffixes; never infer ephys membership from them. Spike/unit shank
vectors come separately from sorting-analyzer enumeration and do not replace
the per-channel shank table.

## Compatibility And Review

Older folders may lack contact IDs, shank metadata, or row maps. The GUI keeps
coordinate and filename fallbacks with narrower geometry assumptions. These
fallbacks do not define new producer output. Merging separate acquisition
streams into probe-wide matrices and coupled multi-shank alignment are outside
this collection-level contract.

Review `channel_metadata.py`, `metrics.py`, and `io.py`; preprocessing's
`datapackage.py`; and GUI `services/ephys_data.py`,
`geometry/channel_geometry.py`, and `plotting/builders/lfp_correlation.py`.
Existing coverage includes `tests/test_fft_metrics.py` and GUI ephys-data,
geometry, and LFP-correlation tests. Check surface/main overlap, non-depth-sorted
channels, and split-stream identity when changing this contract.
