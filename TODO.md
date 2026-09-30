# TODO

## Spike-Sorting Channel Support Sidecar

The alignment GUI cannot reliably distinguish "spike-sorted channel with no
accepted units" from "channel/chunk was never spike sorted" using the current
ALF spike outputs. Unit depths only describe units that survived sorting/QC, and
the continuous channel table describes recorded contacts rather than the sorter
input support.

Emit explicit spike-sorting support metadata from `extract_spikes` when the
SortingAnalyzer exposes enough recording/channel information. The sidecar should
be written per stream/probe and keyed by recording chunk/experiment where that
identity is available. Each supported row should include enough identity for GUI
joins and future validation:

- canonical channel-table row when it can be resolved
- `raw_ind`
- `contact_id`
- `shank_ind` when available
- local coordinate/depth used to resolve the row
- source analyzer/shank/chunk identifier

Open questions:

- Confirm whether current postprocessed SortingAnalyzer folders retain the
  per-chunk recording identity after sorting. If they only expose a merged
  recording, emit stream-level support and mark chunk-level support as unknown.
- Decide whether missing support metadata should be a warning or an explicit
  empty/unknown sidecar. Prefer explicit unknowns so consumers avoid treating
  recorded geometry as sorter support.
- Keep the GUI fallback conservative until this sidecar is present: unsupported
  should mean "not proven sorted", not "no units found".

Acceptance criteria:

- A stream with sorting for only some chunks can tell the GUI which
  chunk/channel rows were actually sorter inputs.
- Channels included in sorting but containing no accepted units are marked
  supported.
- Existing `spikes.*.npy` and `clusters.*.npy` consumers remain compatible.

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

## Brain-Surface Depth Metric From Interference Lines

The GUI has no ephys-derived estimate of where the probe enters the brain.
`EphysPlotQueries.active_in_brain_depths_for_alignment` derives "in brain" from
the histology alignment — the channels whose aligned CCF annotation is not root
— so it is a consequence of the alignment the user is trying to produce, not
evidence for it. Narrowband interference picked up by the probe supplies that
evidence independently: a contact above the surface sits in gel or air with a
high source impedance and no tissue to shunt it, so it couples to rig
interference far more effectively than one in brain, and the transition across
the boundary is sharp.

Measured on inserted probes, the line profile steps 12.7 dB across the surface
in one 90 um move, against 8.8 dB for the LF/HF amplitude ratio, which is graded
over hundreds of microns by volume conduction and puts its change point about
300 um too shallow. With a tone driven into the animal the boundary localizes to
a single contact row: a 1 kHz carrier at 191 uV rms over 52 s placed the surface
at 2745 um with 0.065 dB of scatter across 348 in-brain contacts. Undriven, the
best single line on that stream (3870 Hz) reached 0.070 dB scatter at a
contrast-to-noise ratio of 110.

It does not always work, and the failure is rig-dependent rather than random.
Across 19 surveyed streams the best line's CNR ran 83–133 on the VR rig, 0–2 on
the dynamic-foraging rig, and 0–4 on Rig 5; 7 of the 19 have no usable line at
all. Roughly half of probes therefore yield no undriven surface, so the metric
must be optional on the producing side and its absence must be ordinary on the
consuming side.

Reference implementation: the sibling checkout `aind-ephys-noise-survey`,
ported with attribution rather than taken as a dependency. The port departs from
it in both halves. Detection is by spectral excess, not a harmonic F-test: the
discriminating lines are narrowband noise whose frequency wanders by hertz, so
their F-statistic is near zero at any segment length. The surface fit replaces
its rank-1 SVD and Otsu split, which assume every line shares one depth-profile
shape; harmonics of one comb step by different amounts, with opposite signs,
and at different depths.

### Why the existing PSD artifact cannot carry this

`_iblqc_ephysSpectralDensityLF.power.npy` is unusable for line work, for two
independent reasons, both in `metrics.py:_compute_all_metrics`:

- **The band is wrong.** `psd_accum` is accumulated over `lfp_mask`, that is
  `lfp_freq_range = (1.0, 300.0)` Hz. Every line that discriminates the surface
  sits above it: the mains comb reaches well into the kHz, the strongest
  discriminating line measured was 3870 Hz, and a 10.000 kHz line seen on the VR
  rig is a 100 kHz stimulus carrier aliasing.
- **CMR has already removed the signal.** `np.subtract(lfp_raw,
  np.median(lfp_raw, axis=1, keepdims=True), out=lfp_raw)` runs before the FFT.
  A rig line is common-mode almost by definition, so the common-median reference
  is exactly what cancels it. `_iblqc_metrics.method.json` records this as
  `"cmr": true`.

A new accumulator and a new artifact are therefore required. Neither the
existing PSD nor its consumers change.

### Conversion side

Compute inside the existing window loop in `_compute_all_metrics`. The data is
already in memory, so this costs no extra I/O:

- `_read_window` already reads the **full-rate** window,
  `recording.get_traces(start_frame=s, end_frame=s + window_samples)`, where
  `window_samples = window_duration * fs` — 4 s at 30 kHz by default. AP RMS
  only transforms the first `ap_samples = ap_rms_duration * fs` (1 s, so 1 Hz
  bins). Transforming the whole 4 s gives 0.25 Hz bins from the same read, which
  is what resolves a line from its background.
- **NP1.0** (`lfp_recording is not None`): `raw` is the AP recording and is
  never common-median referenced. Accumulate from it directly.
- **NP2.0 wideband** (`lfp_recording is None`): `lfp_raw = raw` *aliases the
  same array*, and the CMR subtract, the demean and the Hann multiply all run
  in place on it (`lfp_seg = lfp_raw[: hann.shape[0]]` is a view). Any
  un-referenced spectrum must be taken **before** that subtract, or on a copy.
  Getting this wrong is silent: the metric still computes, and still produces a
  profile, but the common-mode line it depends on has been subtracted out and
  the surface call degrades to noise.

Accumulate unreferenced Hann power per channel over 300 Hz–15 kHz, then, after
the loop (`lines.py`):

- **Detect** lines across all blocks of the stream: per block, the
  across-channel median of each channel's power over a ring of background
  around it (inner 3 Hz, outer 25 Hz half-width); pooled by the largest over
  blocks, thinned to the inner half-width and kept above 3 dB excess, strongest
  `n_lines` first. Every line is measured in every block, so all blocks report
  the same lines whether or not each sees them. Each line's amplitude
  integrates the inner band, background included, in V rms. This keeps each
  block's accumulated power until the stream is saved, about 90 MB per block.
- **Fit transitions** once per shank over every block that recorded it. Each
  line's dB profile along depth is `a_b + c*y + sum_t b_t * logistic((y - y_t) /
  w_t)`: the locations `y_t` are shared by every line, while the linear trend,
  step and the two widths either side of `y_t` are each line's own, and the
  baseline `a_b` is its own per block. A line's level drifts between recordings
  by several dB — 1–5 dB, up to 9, across the bank boundaries of blocks tiling
  a shank, against under 0.5 dB over the same distance within a block — so a
  shared baseline would put a transition at every seam. The cost is that a
  transition only counts where one block has `min_side` channels on each side
  of it; a boundary between tiling blocks is never a candidate. Outlier
  replacement, channel-to-channel noise and neighbour correlation are all taken
  within blocks. A line may
  step by any amount, either sign, or not at all; the trend absorbs smooth depth
  gradients so they are not read as a transition. Widths run from a hard step
  to 120 um. Dead and noisy channels (AP RMS below 0.15× or above 6× the shank
  median) are excluded, and each line's outlying channels are replaced by its
  running median before fitting.
- **Up to two transitions.** A contact can be in tissue, in the bath or above
  it, and line families that couple differently step at different depths, so a
  second boundary is common and can be as strong as the surface. The largest
  number of transitions that all pass is kept.
- **Gate** each transition on per-line support, the weighted fit gain from
  adding it, weighting each line in each block by that block's residual
  variance deflated for its own neighbour correlation, so a noisy block adds
  little evidence and does not discount a clean one. A line agrees when its support beats a BIC
  penalty for its step and two grid-searched widths; the transition passes when
  at least two lines agree and their support beyond the penalty, summed over
  the agreeing lines only, clears `min_delta_bic` (500, provisional). Lines
  that do not step cost nothing, so the gate does not tighten with
  `n_lines`. The BIC penalty alone admits in-tissue
  features, so this gate needs calibrating against insertion depths.
- A **declared** driven tone (`extract_continuous(driven_tone_hz=)`) is also
  fitted alone, one transition and one agreeing line, and kept as a `driven`
  candidate.
- Order channels shallow-first with `np.lexsort((x, -y))` so contacts sharing a
  depth do not flip column between depths.

A driven tone is the strong case: the carrier profile is flat in tissue and
falls above it. Quiet rigs are exactly the ones that need one, since they offer
no endogenous line — a few tens of microvolts rms clear of the mains comb is
ample; the 191 uV case above stood 49 dB over its surroundings.

### Artifacts

Follow `docs/shank_channel_metadata_spec.md` §5.2.1: every per-channel array is
dense in channel-table row order, `NaN` for rows a block did not record, and
row position is the only join key. Write from `io.py` beside
`_save_spectral_outputs`, reusing `_on_channel_table_rows` for the dense fill
and `_build_row_channels_metadata` for provenance.

- `_iblqc_ephysLineAmplitude.amps.npy` — `(n_blocks * n_lines, n_rows)`
  float32, V rms per channel, one row per line per block, `NaN` off that
  block's rows; matching the RMS artifacts' units rather than the PSD's. Rows
  are not merged across blocks, since repeat recordings share channel rows and
  their levels differ.
- `_iblqc_ephysLineAmplitude.freqs.npy` — aligned float64, Hz.
- `brain_surface.json` — per shank, `candidates`: every passing transition,
  strongest `delta_bic` first, each with `method` (`lines` / `driven`),
  `block_indices` (the blocks fitted), `midpoint_um` (probe-local y),
  `width_um` (median 10–90 % extent over agreeing lines), `interval_um`,
  `delta_bic`, `n_agreeing_lines`, `n_lines`, and per line its `line_index`,
  `freq_hz`, `amplitude_rows` (block index to amplitude row), `step_db`, the
  two logistic widths, `support`, `agrees`, and the joint fit's `baseline_db`
  per block at the tip and `trend_db_per_mm`, from which the GUI draws the
  fitted curve. A top-level `lines` table is aligned with the amplitude rows:
  one row per line per block, with `line_index`, `block_index`, `label` and
  that block's `excess_db`. None is chosen as
  the surface. Read it directly, the way `session_notes.txt` and
  `spike_shank_indices.npy` are read in `EphysDataService._load_session_notes`
  and `_attach_optional_shank_arrays`; `alfio.load_object` will not load it.

Both are optional. A stream with no usable line writes the amplitude arrays it
did measure and empty `candidates` — explicit unknown, not a missing file,
matching the preference stated in the spike-sorting sidecar section above.

`BlockMetrics.line_power` carries each block's accumulator to
`_save_line_outputs`, which pools and then frees them; the `lines` table keeps
the per-block provenance that `block_source()` distinguishes. Add the parameters to
`_save_method_metadata` and bump its `"version"` past `"1.1"`.

### GUI side

The sibling checkout is `ibl-ephys-alignment-gui`; all paths below are under
`src/ephys_alignment_gui/`. The existing LFP spectrum plot is the working
model for every step — copy its shape rather than inventing one.

1. **Load.** Add `("line_amp", "ephysLineAmplitude")` to
   `EphysDataService._ALF_OBJECTS` in `services/ephys_data.py`. A missing object
   already becomes `{"exists": False}` there and logs, which is the correct
   behavior for the half of probes that have no line. Load `brain_surface.json`
   in `_attach_optional_shank_arrays` and put the parsed dict on the data under
   its own key.
2. **Build payloads.** Add `plotting/builders/line_amplitude.py` modelled on
   `LfpSpectrumPlotDataBuilder`, with a `line_keys()` that reports the lines
   cheaply without building anything. Two payloads, both in dB relative to each
   line's median within each block, so block offsets do not show as seams: an
   image of every line (one column per line by frequency, split by shank
   column) and one line's depth profile with its fitted curve. Use `array_utils.safe_take` to go from channel-table
   rows to the selected shank, `probe_bank_layout.arrange_channels_to_banks` for
   the image, and `level_policy.probe_colour_levels(..., channel_depths_um=,
   in_brain_depths_um=)` for scaling — the payload keys are `img`, `scale`,
   `offset`, `levels`, `cmap`, `xaxis`, `xrange`, `title`.
3. **Memoize.** Construct the builder in `PlotDataPayloadCache.__init__`
   (`plotting/payload_cache.py`) alongside `self.lfp_spectrum_builder`, and
   expose `get_line_amplitude_data` and `get_line_amplitude_keys`.
4. **Register.** In `plotting/registry.py`, add a `PlotSpec(key=
   "image.line_amplitude")` for the all-lines image and a `DynamicPlotSpec(key=
   "line.line_amplitude", menu="line", children=...)` with one child per line,
   built by `_mapping_child_specs_from_keys` as `_probe_lfp_children` builds
   `probe.lfp_spectrum`. No new plot area: both use the existing image and line
   slots. Clicking a column of the image selects that line's child. Wrap the source in `_memoized_payload`. Availability is
   `_data_exists("line_amp")`; the dynamic parent contributes no menu entries
   when `line_keys()` is empty, so a line-free probe simply shows nothing.
   `plotting/menu_state.py` and `desktop/shell/menu_setup.py` then need no
   per-metric change.
5. **Candidate markers**, only on the two line-amplitude plots, never on
   standard plots such as RMS AP or on histology. They must not read as
   reference lines, which are full-width, draggable, 2 px lines in a random
   palette colour and dash: a candidate is a shape, never a line across the
   plot — a wedge at the plot's right edge at `midpoint_um` with a bar over
   `width_um`, and on the line plot a hollow diamond on the fitted curve with
   the transition's extent of the curve highlighted — in one fixed colour
   outside the reference-line palette, with no hover or drag. Nothing is drawn
   when `candidates` is empty.
6. **Surface-candidates dock.** A `QDockWidget` listing the active shank's
   candidates (midpoint, method, agreeing lines, `delta_bic`, width) and the
   selected one's per-line steps, widths and agreement; selecting a line shows
   it in the line plot. Opened from View. This is the GUI's first dock; build
   the dock mechanism under `desktop/docks/` as a general feature rather than
   for this one use.

The alignment payoff is the candidates, not the probe plots. The GUI's present
notion of "in brain" comes from the alignment; these come from the recording,
so the two can be compared. They are display only, never applied: which
candidate is the surface, where a bath boundary sits above it, is the user's
call, and a driven-tone candidate is a far stronger prior than a `lines` one
near its gate.

Decided: the band is 300 Hz–15 kHz and only the accepted lines are stored; a
driven tone is declared, not inferred from the spectrum; lines are detected and
transitions fitted across blocks, with per-block baselines; no LF/HF fallback
is emitted, since it lands about 300 um too shallow.

Open questions:

- The `min_delta_bic` gate, and whether the upper of two candidates is
  typically the bath surface, need insertion depths to settle.
- `n_lines` (8). On 16 surveyed shanks, pooling reproduces every
  lines-based reference call to within 30 um at 8 or 32 lines, in 0.2–8 s per
  shank at 8. At 32, streams with many lines hit the cap, ΔBIC grows with the
  agreeing lines (841303 probe 46121: 3,269 → 11,622), and a few near-gate
  extras appear (860903 probe 50213 at 352 um, ΔBIC 565), so the choice is
  tied to the gate.

Acceptance criteria:

- A stream with strong interference lines reports surface candidates with
  midpoint, width and per-line support, and the GUI shows the per-line depth
  profiles.
- A stream with no usable line writes empty `candidates` and every GUI plot and
  menu behaves as it does today, with no error and no empty panel.
- A declared driven tone reports a `driven` candidate.
- NP1.0 and NP2.0 streams both produce line amplitudes that are *not* common-
  median referenced; a regression test asserts the NP2.0 path reads the raw
  window before the in-place CMR subtract.
- `_iblqc_ephysSpectralDensityLF.*`, `band_corr/*`, and the RMS artifacts are
  byte-identical to what the current code writes.
