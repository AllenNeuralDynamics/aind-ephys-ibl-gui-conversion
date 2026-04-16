# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run all checks (formatting, linting, docstring coverage, tests)
./scripts/run_linters_and_checks.sh -c

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_ephys.py

# Run a single test by name
uv run pytest -k "test_name"

# Formatting
uv run ruff format

# Linting
uv run ruff check
uv run ruff check --fix

# Spell checking
uv run codespell --check-filenames

# Docstring coverage (100% required)
uv run interrogate -v src
```

Always use `uv run` to execute commands, `uv add` to add dependencies, and `uv sync` to set up the environment. Never use bare `pip` or `python`.

## Architecture

This is a Python package using a `src/` layout. Source lives in
`src/aind_ephys_ibl_gui_conversion/`, tests in `tests/`. No CLI entry points --
the package is a library consumed by other AIND pipelines (notably
`aind-ibl-ephys-alignment-preprocessing`).

- Build system: hatchling
- Linting/formatting: ruff (line length 79)
- Testing: pytest with coverage reporting
- Type checking: mypy (strict mode); the code is partially type-hinted and
  does not currently pass strict checking, so `run-mypy: false` in CI.
  Local `uv run mypy` still works for incremental cleanup.
- Docstring coverage: interrogate, **fail-under 100%**
- Versioning: commitizen (semantic versioning via conventional commits)

### Module layout

```
src/aind_ephys_ibl_gui_conversion/
  __init__.py           # Public API re-exports (extract_spikes, extract_continuous, + internal helpers)
  types.py              # Dataclasses: ExperimentBlock, ProbeStream, BlockMetrics, ShankChannels
  ephys.py              # extract_continuous(): FFT-based RMS/PSD/coherence computation
  spikes.py             # extract_spikes(): SpikeInterface SortingAnalyzer -> IBL ALF
  io.py                 # Streaming recording loaders, ALF writers
  metrics.py            # COHERENCE_BANDS, Parseval RMS, coherence assembly
  recording_utils.py    # Stream name parsing, segment selection, asset merging
  utils.py              # Frequency-domain helpers (copied from ibl-neuropixel, Feb 2024)
  histology.py          # Slicer fCSV generation, probe trajectory -> fCSV (supplementary)
```

### Key entry points

```python
from aind_ephys_ibl_gui_conversion import extract_spikes, extract_continuous
```

- **`extract_spikes(sorting_folder, results_folder, stream_to_use=None, min_duration_secs=300)`**
  -- reads `<sorting_folder>/postprocessed/*.zarr` SortingAnalyzer output and
  writes `spikes.*.npy`, `clusters.*.npy`, etc. per probe.
- **`extract_continuous(sorting_folder, results_folder, ...)`** -- reads
  `ecephys_compressed/*.zarr` raw recordings and writes RMS time series, PSD,
  per-band coherence, and channel metadata.

### Data layout assumed

```
<session>/
  ecephys_clipped/                # Open Ephys binary
  ecephys_compressed/             # Zarr + WavPack
    experiment<N>_<stream>.zarr
  <session>_sorted/               # sorting_folder argument
    postprocessed/
      experiment<N>_<stream>_recording<M>.zarr              # Single-shank
      experiment<N>_<stream>_recording<M>_group<K>.zarr     # Multi-shank
```

### Stream name parsing

Open Ephys stream IDs follow
`Record Node <id>#Neuropix-PXI-<model>.<probe_name>[-AP|-LFP]`. Parsed in
`recording_utils._stream_to_probe_name()` via regex.

### Key patterns

- **Neuropixels 1.0 vs 2.0**: 1.0 probes have separate AP and LFP streams;
  2.0 probes use combined wideband. `ExperimentBlock.is_1_0` distinguishes
  them. Metrics handle both.
- **Multi-block recordings**: When `probe_surface_finding` is provided,
  recordings include surface-finding blocks in addition to the main block.
  `ProbeStream.has_surface` flags this; output filenames gain a `Main` infix
  and `band_corr/channel_blocks.json` is written.
- **FFT-based metrics**: RMS is computed via Parseval's theorem
  (band-limited integration of the PSD) rather than time-domain filtering.
  See `metrics._parseval_rms()` and `_compute_all_metrics()`.
- **Coherence bands**: `delta, theta, alpha, beta, gamma` defined in
  `metrics.COHERENCE_BANDS`.
- **SI 0.104 compatibility**: SpikeInterface renamed metrics
  (`peak_to_valley` -> `peak_to_trough_duration`). There is a monkey-patch
  (`_patch_si_deprecated_metric_validation()`) for backward compatibility
  with legacy waveform extractors.
- **Auto-adjusted window interval**: `extract_continuous` shortens
  `rms_window_interval` for short recordings to guarantee at least 20
  windows (needed for coherence statistics).

## Code style

- **Line length**: 79
- **Target**: Python 3.10+
- **Type hints**: Partially present; not all modules pass strict mypy yet
  (~80 errors). Add hints when editing; don't remove existing ones.
- **Ruff rules**: `Q, RUF100, C90, I, F, E, W, UP, PYI` (no `ANN` or `D`)
- **McCabe complexity**: max 14
- **Docstrings**: Required on all public functions/classes (interrogate
  fail-under 100%). Module docstrings are exempt.

## Testing

- `tests/test_ephys.py` -- unit tests for stream parsing, segment selection,
  asset merging
- `tests/test_fft_metrics.py` -- integration tests using synthetic
  `si.NumpyRecording` objects (multi-frequency sinusoids + noise) to validate
  FFT-based RMS against time-domain RMS, and output shape/timestamp
  correctness
- No pytest fixtures; uses local `_make_synthetic_recording()` helpers
- Coverage target: not enforced (`cov-fail-under=0` in pyproject)

## Dependencies

- **spikeinterface[full]** >= 0.103.0 -- Recording/Sorting abstractions,
  SortingAnalyzer, quality metrics, Phy export
- **iblatlas** -- Atlas transforms
- **one-api** >= 2.6.0 -- IBL data API
- **antspyx** >= 0.4.2 -- Used by `histology.py` only
- **aind-mri-utils** -- Probe trajectory fitting in `histology.py`
- **wavpack-numcodecs** -- Decompresses WavPack-compressed zarr recordings
- **tqdm, numpy** -- Standard utilities

## Related packages

- Consumed by `aind-ibl-ephys-alignment-preprocessing`, which imports
  `extract_spikes` and `extract_continuous` to produce ALF output for the
  IBL alignment GUI.
- The `utils.py` module is vendored from IBL's `ibl-neuropixel` (Feb 2024);
  if upgrading, check upstream for bug fixes in `WindowGenerator`, `bp`,
  `lp`, `hp`, `fscale`, `fexpand`, `rms`, and `fcn_cosine`.
