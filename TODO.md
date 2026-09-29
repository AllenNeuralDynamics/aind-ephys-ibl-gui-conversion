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

Reference implementation: the sibling checkout `aind-ephys-noise-survey`. Port
the algorithm with attribution; do not take the dependency. The pieces that
matter are `multitaper.make_taper_bank` / `eigencoefficients` /
`harmonic_f_test` / `line_threshold`, `detect.detect_lines` and
`detect.benjamini_hochberg`, `lines.extract_lines`, and
`channels.classify_channels`, whose `_shared_line_factor` (rank-1 SVD across
lines) and `_surface_index` (Otsu between-class variance) do the surface call.

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

Accumulate `|X|**2` per channel over the AP band the way `psd_accum` does for
the LFP band, then, after the loop:

- Detect candidate lines on the across-channel median spectrum — harmonic
  F-test against the local background, candidates thinned by minimum
  separation, accepted under Benjamini-Hochberg at `detect_q`. Group harmonics
  into families so one mains comb is one candidate, not sixty.
- Build the per-channel amplitude profile for each accepted line.
- Select the surface. Test **each line separately and in both polarities** and
  take the sharpest step; lines disagree about which side of the boundary they
  are stronger on. Keep the rank-1 shared fit as a secondary candidate, gated on
  explained variance, but do not prefer it: on the stream cited above the best
  single line scored a CNR of 110 against 3.9 for the rank-1 fit over 8 lines.
- Gate the result. Below the CNR floor, report no surface rather than a guess.
  `classify_channels` defaults — `min_line_cnr=3.0`, `min_line_rank1=0.5`,
  `min_step_db=4.0`, `max_out_fraction=0.6`, `smooth=7` — are the calibrated
  starting point; `min_step_db` is deliberately low because a **tip** reference
  sits in tissue and mutes the step to about 5 dB.
- Order channels by depth with the existing convention,
  `np.lexsort((locs[:, 0], locs[:, 1]))` in `shank_info`, so probes with several
  contacts at one depth do not flip column between depths.

A driven block, if one exists, supersedes all of this: the carrier profile is
flat in tissue and falls above it, so the surface comes out to one contact row.
Detect it as an unusually loud isolated line and record it as
`surface_method = "driven"`. `driven.py` carries the offsets that separate
carrier from shoulder from background (`CARRIER_HZ, SHOULDER_HZ, FAR_HZ = 8.0,
(8.0, 60.0), (100.0, 140.0)`). Quiet rigs are exactly the ones that need a
driven tone, since they offer no endogenous line to discriminate with — a few
tens of microvolts rms at a frequency clear of the mains comb and its harmonics
is ample, the 191 uV case above stood 49 dB over its surroundings.

### Artifacts

Follow `docs/shank_channel_metadata_spec.md` §5.2.1: every per-channel array is
dense in channel-table row order, `NaN` for rows a block did not record, and
row position is the only join key. Write from `io.py` beside
`_save_spectral_outputs`, reusing `_on_channel_table_rows` for the dense fill
and `_build_row_channels_metadata` for provenance.

- `_iblqc_ephysLineAmplitude.amps.npy` — `(n_lines, n_rows)` float32, V rms per
  channel per line, matching the RMS artifacts' units rather than the PSD's.
- `_iblqc_ephysLineAmplitude.freqs.npy` — `(n_lines,)` float64, Hz.
- `brain_surface.json` — the derived call: `surface_um`, `surface_method`
  (`driven` / `lines` / `lf_hf` / `none`), `cnr`, `step_db`, `n_out_of_brain`,
  the selected line's frequency, and the per-line table behind the choice. Read
  it directly, the way `session_notes.txt` and `spike_shank_indices.npy` are
  read in `EphysDataService._load_session_notes` and
  `_attach_optional_shank_arrays`; `alfio.load_object` will not load it.

Both are optional. A stream with no usable line writes the amplitude arrays it
did measure and a `brain_surface.json` whose `surface_method` is `"none"` —
explicit unknown, not a missing file, matching the preference stated in the
spike-sorting sidecar section above.

Extend `BlockMetrics` in `types.py` with the new arrays; keep the per-block
provenance that `block_source()` already distinguishes, since a short surface
block and a long main block see different interference. Add the parameters to
`_save_method_metadata` (`line_detect_q`, the AP band edges, the window
duration actually used, the CNR gates) and bump its `"version"` past `"1.1"`.

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
   `LfpSpectrumPlotDataBuilder`: a `build(in_brain_depths_um)` returning the
   probe payload mapping, and a `probe_keys()` that reports the available line
   keys cheaply without building anything. Emit dB, one probe strip per line,
   labelled by frequency. Use `array_utils.safe_take` to go from channel-table
   rows to the selected shank, `probe_bank_layout.arrange_channels_to_banks` for
   the image, and `level_policy.probe_colour_levels(..., channel_depths_um=,
   in_brain_depths_um=)` for scaling — the payload keys are `img`, `scale`,
   `offset`, `levels`, `cmap`, `xaxis`, `xrange`, `title`.
3. **Memoize.** Construct the builder in `PlotDataPayloadCache.__init__`
   (`plotting/payload_cache.py`) alongside `self.lfp_spectrum_builder`, and
   expose `get_line_amplitude_data` and `get_line_amplitude_probe_keys`.
4. **Register.** In `plotting/registry.py`, add a `DynamicPlotSpec(key=
   "probe.line_amplitude", menu="probe", children=...)` whose children come from
   `_mapping_child_specs_from_keys`, exactly as `_probe_lfp_children` builds
   `probe.lfp_spectrum`. Wrap the source in `_memoized_payload`. Availability is
   `_data_exists("line_amp")`; the dynamic parent contributes no menu entries
   when `probe_keys()` is empty, so a line-free probe simply shows nothing.
   `plotting/menu_state.py` and `desktop/shell/menu_setup.py` then need no
   per-metric change.
5. **Surface line.** Draw the estimate as a horizontal marker on the depth plots
   with its method and CNR in the label, so a weak `lines` call is visibly
   weaker than a `driven` one. Never draw it when `surface_method` is `"none"`.

The alignment payoff is the surface estimate, not the probe plots. The GUI's
present notion of "in brain" comes from the alignment; this one comes from the
recording, so the two can be compared. Offer it as a starting reference point
for the top boundary rather than applying it — the user's judgment stays
authoritative, and a driven-tone surface good to one contact row is a far
stronger prior than a `lines` call scraping past its CNR gate.

Open questions:

- Which AP band to accumulate. The full Nyquist range is the most informative
  and the most expensive; 300 Hz–15 kHz covers every line observed. Decide
  whether to store the full line spectrum or only the accepted lines. Storing
  only the accepted lines is ~10 rows per probe instead of tens of thousands,
  but makes the detection un-revisable without a rerun.
- Whether a driven block should be recognized automatically from its spectrum or
  declared. Automatic detection risks promoting a loud rig line to "driven" and
  reporting a false confidence; a declaration needs a producer-side flag that
  does not exist yet.
- Whether per-block or per-stream. A stream's surface should not move between
  blocks, but a short surface block and a long main block will disagree on
  amplitude. Emitting per-block and letting the consumer choose is the
  conservative option and matches `row_channels.json` provenance.
- Whether the LF/HF ratio fallback is worth emitting at all, given that it lands
  about 300 um too shallow. It may be better to report no surface than one known
  to be biased.

Acceptance criteria:

- A stream with a strong interference line reports a surface depth, the line it
  came from, and a CNR, and the GUI shows the per-line depth profiles.
- A stream with no usable line writes `surface_method = "none"` and every GUI
  plot and menu behaves as it does today, with no error and no empty panel.
- A driven-tone block reports `surface_method = "driven"` and a surface whose
  scatter is well under a decibel.
- NP1.0 and NP2.0 streams both produce line amplitudes that are *not* common-
  median referenced; a regression test asserts the NP2.0 path reads the raw
  window before the in-place CMR subtract.
- `_iblqc_ephysSpectralDensityLF.*`, `band_corr/*`, and the RMS artifacts are
  byte-identical to what the current code writes.
