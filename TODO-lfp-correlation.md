# LFP correlation investigation — findings and open work

Context: the scrambled/checkerboard LFP correlation matrix for probe **46100** in
`SmartSPIM_754372_2025-01-31_11-27-18_preprocessed_2026-08-13_04-31-17`
(session `ecephys_754372_2025-01-14_17-32-49`).

Scratch scripts: `/tmp/claude-.../scratchpad/` (`plot_46100.py`, `multi_probe.py`,
`look_at_traces.py`, `mean_vs_median.py`, `decompose.py`, `phase_shift_test.py`,
`settings_compare.py`, `audit_asset.py`) plus the diagnostic figure
`probe_46100_diagnostic.png`. Copy anything worth keeping somewhere durable.

---

## Root cause: probe 46100's tip reference is compromised. Not our code.

Measured on the **raw zarr with standalone FFT code**, never through
`extract_continuous`, so neither the producer nor the display is in the path.

**The tell is in the depth profile.** On 46100 the theta field is ~280 µV at
*every* depth from 500–2865 µm — completely flat, no laminar structure, the
heatmap is uniform vertical stripes down the whole shank. Healthy 46116 in the
same session goes 4 µV at the tip to ~150 µV at 2400 µm with real structure.
A signal identical in amplitude and phase across 2.4 mm is not from tissue.

With `referenceChannel = Tip`, a compromised tip reference injects one large
common signal onto every channel. Consistent with this, 46100's channels at
0–450 µm (nearest the tip) are erratic, collapsing to 25–50 µV.

Chain of consequences:
1. Bad tip reference → ~280 µV uniform common mode on all channels.
2. Genuine local LFP buried — CMR residual only 13–17 µV.
3. The small per-column offset that **every** probe has (~26 µV here, 5–15 µV on
   healthy 46116) is now the largest thing left in the residual.
4. Depth-sorting interleaves the two columns row by row → checkerboard.

Note 46116 has a *larger relative* column offset (22.7% of signal vs 9.4%) and
shows no checkerboard, because it has real local structure to swamp it. The
column offset is normal; the flat field is not.

Ratio = (same-column 15 µm) / (same-depth 32 µm) correlation, theta, must be ≈1:

| probe | CMR ratio | CAR ratio |
|---|---|---|
| 45883 | 0.98 | 1.02 |
| **46100** | **20.09** | 1.22 |
| 46116 | 1.01 | 1.00 |

`settings.xml` is **identical** across probes (`preset=Bank A`, `ref=Tip`,
`refIdx=1`, same XPOS/YPOS). 46100 (NP2000, port 3, dock 1) matches healthy 46116
(NP2000, port 4, dock 1). 45883 is the hardware odd-one-out (NP2004, NPM_HS_31,
flex 0.1, dock 2) and is fine. No config or metadata difference explains it.

### Ruled out (each tested directly, all negative)
- **CMR / median reference** — works correctly on 2 of 3 probes in the identical
  config. **Do not change `np.median` → `np.mean` in `metrics.py:316-320`.**
  An earlier recommendation to do so was wrong.
- **Column-differential voltage as a primary cause** — 0.041% of power,
  band-independent; it only matters because the real signal is absent.
- **Per-column gain** — `gain_to_uV` identical (0.585) on every channel;
  correcting per-channel β made the ratio *worse* (9.7 → 15.1).
- **PC removal** (k=1,2,3) — leaves a 1.9–2.4× column bias.
- **ADC grouping** — distance-controlled, same-ADC and different-ADC pairs are
  indistinguishable (0.8688 vs 0.8786 at 15 µm).
- **Phase shift / sample skew** — see item 4 below; provably irrelevant.
- **Broken site switching** — no systematic offset; best-match offsets are flat
  and centred near 0, and each column's own sub-matrix is smooth and brain-like.
- **Median nonlinearity in the abstract** — synthetic data with two contacts per
  depth identical up to noise gives ratio 0.99 for median *and* mean, in every
  regime. The median cannot manufacture a column split.

---

## Open work

### 1. Decide what to do with probe 46100  — BLOCKED ON DECISION
- [ ] Check whether 46100 is broken in its **other sessions**
      (`ecephys_754372_2025-01-15_16-40-18`, and both surface-finding recordings)
      — persistent probe fault vs one-session fault.
- [ ] Decide: exclude from alignment, or flag and let the user judge.

### 2. Automated QC flag  — NOT STARTED
Two candidate detectors, both cheap from data the producer already computes:
- **Flat depth profile** (preferred — detects the cause): fraction of LFP
  variance in the common mode, or the dynamic range of per-channel LFP RMS
  across depth. 46100 is flat at ~280 µV; 46116 spans 4→150 µV.
- **Column ratio** (detects the symptom): same-column vs same-depth correlation.
  Separates 46100 (20.1) from neighbours (0.98, 1.01) by >an order of magnitude.
  Only meaningful where ≥2 contacts share a depth.
- [ ] Emit per probe/shank, surface in the datapackage.

### 3. Display depth scale  — DONE (uncommitted, untested in the live GUI)
Real and independent of the 46100 fault. Verified by replicating
`_matrix_depth_geometry` on the asset's real `row_channels.json` +
`channels.localCoordinates.npy`: the 768-row matrix got one uniform
`scale = 11.25 µm/row`, drawing the main block **0→4320 µm** when it is truly
**0→2865**, and surface **4320→8640** when truly **2880→8625** — up to **1.4 mm**
of depth error.

Fix: one image per recording block, each with its own affine.

- [x] `_unique_block_row_groups()` — per-block rows from `row_channels.json`,
      deduplicated by row-set identity; falls back to the single-image path when
      only one unique depth range exists.
- [x] `_load_correlation_files` multi-image path + shared colour scale across
      blocks (`_block_max_corr`) so colours stay comparable.
- [x] `_load_coherency_files` multi-image path (`_coherency_block_payload`,
      `_coherency_phase_rgba`).
- [x] `render_image` accepts a list of images or a single one (back-compatible).
- [x] Tests: multi-block affine + single-image fallback. 619 GUI tests pass.
- [ ] Run in the live GUI on real data — never exercised outside tests.

**Two contacts per depth: resolved, no extra work.** Straddling is what the
per-block affine already produces. Bank A: 384 rows over 192 depths →
`scale = (2865−0+15)/384 = 7.5`, so rows 2k and 2k+1 cover `[15k, 15k+7.5]` and
`[15k+7.5, 15k+15]` — together exactly the 15 µm slab for that depth. Max
positional error ±3.75 µm inside a 15 µm pitch. The lexsort (item 5) makes row 2k
always col0 and 2k+1 always col32, so it is stable and interpretable.

### 4. Phase-shift correction  — DROPPED, not needed
`extract_continuous` does no phase correction (`inter_sample_shift` is present
but unused). It affects **none** of the metrics computed here:
- `rms_ap`, `rms_lfp`, `psd` are per-channel, computed from `|X|²`, which is
  invariant to a time shift.
- Cross-channel metrics top out at 80 Hz (gamma); worst-case skew is 0.94 samples
  = 31 µs = **0.016 rad** there, 1.2e-3 rad at theta.
- Empirically `phase_shift + CMR` = 9.65 vs `raw + CMR` = 9.73.

Revisit only if this code ever computes a metric above ~1 kHz or does
spike-adjacent work.

### 5. Depth sort secondary key  — DONE (uncommitted)
- [x] `metrics.py` `_compute_all_metrics`: `np.argsort(locs[:, 1])` →
      `np.lexsort((locs[:, 0], locs[:, 1]))` (depth, then lateral).
- [x] `channel_metadata.depth_sorted_shank_rows`: same, and the docstring notes
      the two must stay in step.
- [x] 51 conversion tests pass; ruff/interrogate/codespell clean.

### 6. Rewrite commit 6615a9f's rationale  — NOT STARTED
The change (integer-µm geometry as dedup key) is fine and worth keeping — geometry
is the better contract. **Its stated justification is false** and will mislead:
- Claimed "silently doubling the channel table (768 rows for a 384-channel
  probe)". Verified on the asset: 768 rows, **768 distinct (x,y) pairs, zero
  duplicates**. 768 is correct — bank A's 384 plus banks B/C's 384.
- Claimed the zero off-diagonal blocks were evidence of the bug. They are
  structurally correct: main and surface were separate recordings, so those
  correlations were never measured.
- The asset predates the commit, so it was built with the old contact-id-primary
  key and still shows no doubling.
- [ ] Amend the docstring in `channel_metadata.build_channel_table`.

### 7. Reprocess  — BLOCKED on 1
- [ ] Version bumps for `aind-ephys-ibl-gui-conversion` and
      `ibl-ephys-alignment-gui`.
- [ ] Reprocess 754372 once the 46100 decision lands.

---

## Housekeeping
- `s3fs` had been added to **runtime** dependencies for S3 diagnostics; moved to
  the `dev` group so it does not ship in the wheel.
- `scripts/check_contact_ids.py` (untracked scratch from the earlier contact-id
  work) was the only source of the repo's 5 ruff errors and has been deleted;
  `ruff check` is now clean.

## Confirmed correct — no action
- `row_channels.json` per-block `rows` are channel-table row indices, depth-sorted
  within each block.
- `_assemble_blockwise_coherence` n_windows-weighted averaging of overlapping
  contacts across Open Ephys experiments.
- Bank B and bank C in the surface recording **were** co-recorded (B×C quadrant
  100% non-zero, max |r| 0.73) → one contiguous 384×384 block spanning
  2880–8625 µm. Two diagonal blocks is right, not three.
