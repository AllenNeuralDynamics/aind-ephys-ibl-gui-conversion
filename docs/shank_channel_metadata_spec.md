# Spec: Ephys Geometry Contract for IBL GUI Consumers

Status: **implementation v1**
Owners: producing side = `aind-ephys-ibl-gui-conversion`; orchestration/datapackage side = `aind-ibl-ephys-alignment-preprocessing`; consuming side = `ibl-ephys-alignment-gui`.

## 1. Problem

Channel → shank → depth geometry is currently **re-derived independently** in ≥2 places:

- Producing side derives shanks from SI `get_property("group")` and matches
  channels across blocks by float location (`|Δloc| < 1e-6`).
- The alignment GUI re-derives shanks from x-coordinate gaps
  (`chn_x[i*2]`, i.e. "exactly 2 columns per shank"), and never uses the
  channel metadata the producing side already writes (`channel_blocks.json`
  is loaded and ignored).

These independent derivations disagree on any geometry that violates the
unstated assumptions (notably non-"quadbase" multi-shank layouts), producing
wrong shank membership, mis-indexed correlation/coherency matrices, and
silent full-probe fallbacks. Every multi-shank display bug we've hit is an
instance of this pattern.

`channels.rawInd.npy` is currently a legacy positional placeholder. This
contract does **not** redefine it; channel-table row position is the join key.
`channels.contactId.npy` carries the physical probe contact identity when
probe metadata provides one. Consumers should tolerate its absence when loading
older result folders; absence means contact-ID stitching is unavailable for
that output.

## 2. Principle

Derive the channel geometry **once**, on the producing side, from the
authoritative probe grouping; **persist it** as explicit per-channel
metadata; and have every consumer **reference it by a stable key** rather
than re-deriving from coordinates. Make the correlation/coherency matrices
**self-describing** (each row maps to a known channel).

## 3. Scope

In scope:
- A per-ephys-collection channel table (source of truth) with explicit shank.
- Self-describing full correlation/coherency matrices (row → channel map).
- GUI consumes the table + row maps; drops the x-gap and size-equality
  heuristics.
- Datapackage uses explicit `logical_probe`, `ephys_collection`,
  `histology_shank`, and `ephys_shank` fields so split-stream quadbase and
  single-stream multi-shank probes are distinguishable.

Out of scope (explicitly deferred; the model must *accommodate* but not
*require* these):
- Probe-level compute domain / merging a quadbase's separate streams into one
  probe-level coherency matrix.
- Joint (coupled) multi-shank alignment.

Quadbase note: with streams left separate, each quadbase shank is its own
probe output folder containing a trivial 1-shank channel table — the contract
still holds; a future probe-merge produces a multi-shank table without
changing the contract.

## 4. The join key

**The join key across all artifacts is the channel-table row position
`0..N-1`.**

- Per-channel arrays (rms, psd) remain in channel-table order; consumers
  index them **positionally**.
- `contactId`, `shankInd`, `rawInd`, and matrix row-maps are descriptive
  fields keyed by that position. `rawInd` remains the legacy positional
  placeholder for now; consumers must not depend on it as the row selector.
- This is deliberately back-compatible: existing per-channel array ordering
  keeps working; we only add metadata and stop deriving shank membership from
  x-coordinate gaps.

### 4.1 Ordering & spatial mapping (important)

- The channel table is in **acquisition order** (first-occurrence of the
  recording's channels after dedup). It is **NOT depth-sorted**, and for
  Neuropixels position is not monotonic in depth.
- Therefore **table position is an identity/join key only — never a spatial
  coordinate.** Every physiology plot's depth/spatial axis MUST be computed from
  `localCoordinates`, not from a row/column index.
- Correlation matrices, coherency matrices, RMS plots, PSD plots, and any
  future physiology plots may use channel-table rows to select data, but
  must use `localCoordinates[rows]` to draw depth/lateral geometry.
- Concretely: a matrix's depth axis is derived from
  `localCoordinates[rows, 1]` (the depths of its rows), not from a linear
  `depth_range / n_rows` scale. Do not assume rows are depth-sorted or
  uniformly spaced. (The producer *constructs* `rows` depth-sorted by
  convention, but consumers must still map via `localCoordinates` so a
  non-uniform or non-monotonic layout renders correctly.)
- Uniform-spacing caveat: a pyqtgraph ImageItem uses a single affine
  scale+offset, which is exact only if the mapped depths are uniformly
  spaced. For a single Neuropixels shank that holds (channels tile depth at
  fixed pitch, typically 2 per depth); derive the affine from the actual
  `localCoordinates[rows]` min/max and count. For genuinely non-uniform
  layouts, resample the matrix onto a uniform depth grid before display.

## 5. Data contract (artifacts)

### 5.1 Channel table (per probe output folder)

| file | dtype/shape | meaning |
|---|---|---|
| `channels.localCoordinates.npy` | float (N, 2) | (x, depth), unchanged |
| `channels.rawInd.npy` | int (N,) | Legacy positional placeholder, unchanged |
| `channels.contactId.npy` | str (N,) | Physical probe contact IDs in channel-table row order; empty string when unavailable |
| `channels.shankInd.npy` | int (N,) | 0-based shank index per channel |

- Canonical order: rows `0..N-1`. All per-channel arrays (rms, psd) MUST be
  in this same column order.

### 5.2 Correlation / coherency matrices

- `band_corr/{band}_mean_corr.npy` — (N, N) float — primary full collection
  matrix in channel-table row order.
- `band_corr/{band}_coherency.npy` — (N, N) complex — primary full collection
  matrix in channel-table row order.
- `band_corr/{band}_shank{k}_mean_corr.npy` — (n_k, n_k) float —
  compatibility view sliced from the full matrix.
- `band_corr/{band}_shank{k}_coherency.npy` — (n_k, n_k) complex —
  compatibility view sliced from the full matrix.
- `band_corr/row_channels.json` — always written (single- and multi-block):

```json
{
  "version": 2,
  "matrix_rows": [0, 1, 2],
  "matrix_row_order": "channel_table",
  "shanks": {
    "0": {
      "rows": [<channel-table positions in display order>],
      "legacy_file_index": 1,
      "blocks": [
        {"label": "main", "rows": [...]},
        {"label": "surface", "rows": [...]}
      ]
    }
  }
}
```

- `matrix_rows` lists the primary full-matrix row order. It is the channel
  table order.
- `shanks.<k>.rows` lists, for each 0-based shank, the channel-table positions
  of that shank in display order.
- `blocks` carries main/surface provenance. `channel_blocks.json` may remain
  as a temporary compatibility alias for multi-block data.
- Producer construction convention: `rows` for shank k = channel-table
  positions with `shankInd == k`, sorted by depth ascending. Persist
  explicitly regardless of the convention.

### 5.3 Versioning

- `row_channels.json.version`. Consumers check for presence + version;
  absence ⇒ legacy fallback (§8).

## 6. Producing side changes (`aind-ephys-ibl-gui-conversion`)

1. **New `build_channel_table(recordings) -> ChannelTable`** — the single
   derivation point. Sources: `get_channel_locations()`,
   `get_property("group")` for `shankInd`, and `get_property("contact_ids")`
   or `get_probe().contact_ids` for `contactId`; `rawInd` remains positional.
2. **`io.py` write paths** (`_assemble_and_save_stream` ~:355 and the
   single-block path ~:484): write `rawInd`, `contactId`, and `shankInd` via the builder
   in **both** paths. Ensure per-channel arrays share the table order.
3. **Metrics**: compute pairwise correlation/coherency over the full ephys
   collection. Do not compute primary pairwise artifacts per group/shank.
4. **`_save_spectral_outputs`**: always emit `row_channels.json`. Save full
   matrices first and derive shank-suffixed compatibility files from them.
5. **Stitching** (`_assemble_blockwise_coherence` / `_build_channel_maps`):
   match channels by normalized shank plus contact ID when available, falling
   back to normalized shank plus rounded local coordinate when contact IDs are
   missing or invalid. Preserve channel-table row order.
6. **Tests** (`tests/test_fft_metrics.py`): assert `channels.shankInd.npy`
   present and consistent with groups; single-block run also writes
   `row_channels.json`; full matrix dims are N x N; shank compatibility matrix
   dims match `row_channels.shanks[*].rows`.

## 7. Consuming side changes (`ibl-ephys-alignment-gui`)

1. **`PlotData.__init__` / `load_data_local`**: shank membership + `n_shanks`
   from `channels.shankInd` (positions where `shankInd == k`). Remove the
   `chn_x[i*2]` 2-column derivation (keep as fallback only, §8). `chn_coords`
   / rms-psd indexing = those positions.
2. **`get_lfp_correlation_data_img` + coherency**: load `row_channels.json`;
   for the active shank take `rows`; select/order matrix rows by `rows`; map
   depth from `localCoordinates[rows, 1]`. **Delete** the
   `n_matrix == len(chn_coords)` heuristic and the full-probe linear fallback.
   In particular, **derive the image's y-scale/offset from
   `localCoordinates[rows]`, not from `depth_range / n_matrix`** — the current
   linear index→depth mapping is a latent bug (it assumes rows are
   depth-sorted and uniformly spaced; see §4.1). rms/psd already do this right
   (depth grid via `chn_full`/`idx_full`); bring the correlation/coherency
   images to the same standard.
3. **Spike correlation** (`get_spike_correlation_data_img`): unchanged
   (computed from filtered spikes, already self-consistent) — but its shank
   membership now also derives from `shankInd`.
4. **Stop loading-and-ignoring** `channel_blocks.json`; use `row_channels.json`.

## 8. Backward compatibility / migration

- Legacy datasets (no `shankInd` / no `row_channels.json`): GUI logs a warning
  and falls back to current heuristics (x-gap shank split; size-equality depth
  mapping). No hard break.
- Existing shank-suffixed filenames remain as compatibility views. New full
  matrix filenames are added. `rawInd` remains unchanged.
- Optionally re-emit metadata for already-converted sessions via a small
  backfill that runs `build_channel_table` against the saved recording.

## 9. Open questions / verify before implementing

- Confirm a real non-quadbase 4-shank `channels.localCoordinates`
  layout (columns per shank) to be sure `get_property("group")` is populated
  correctly upstream and the x-gap fallback is only ever a fallback.
- Verify real AIND SpikeInterface/probeinterface recordings expose contact IDs
  in channel order either as `get_property("contact_ids")` or via
  `get_probe().contact_ids` / `device_channel_indices`.

## 10. Bug classes eliminated by construction

1. GUI mis-slicing on non-quadbase geometry (x-gap heuristic).
2. Producing-vs-GUI shank disagreement (one persisted field).
3. Correlation/coherency row mis-mapping (rows self-describe).
4. Index-as-depth-proxy in physiology rendering (depth always via
   `localCoordinates`; see §4.1).
5. Datapackage ambiguity between physical/logical probe and ephys collection.

## 11. Phasing

- **Phase 1 (this spec):** channel table + self-describing matrices +
  GUI consumes; single-base multi-shank correct; heuristics gone.
- **Phase 2 (deferred):** probe-level compute domain (merge quadbase streams),
  whole-probe coherency, shank as a pure coordinate.
- **Phase 3 (deferred, part research):** coupled multi-shank alignment
  (shared probe pose + regularized per-shank residual warps).
