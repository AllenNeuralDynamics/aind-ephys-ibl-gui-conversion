# Contributing

This is the main development guide for human contributors and coding agents.
Agent-specific instructions are in [AGENTS.md](AGENTS.md).

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

Before changing channel metadata, shank attribution, array ordering, metric
filenames, or output directories, read
[the ephys collection and geometry contract](docs/shank_channel_metadata_spec.md).
The GUI depends on collection layout as well as explicit datapackage paths.

For continuous metrics, start with the
[authoring tutorial](docs/metric_extensions.md). It builds an AP-trace metric
from a function, adapts it to LFP, and demonstrates FFT preparation with derived
sampling and provenance. The linked API reference covers custom accumulators,
pure reductions, and saving.

The [sampling and execution overview](docs/continuous_metric_pipeline.md)
explains experiment blocks, time windows, shared reads and FFTs, worker limits,
and stream assembly.

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
  channel_metadata.py   # Canonical channel table, block row maps, physical shank identity
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
  and `band_corr/row_channels.json` maps full matrices to channel-table rows.
  `band_corr/channel_blocks.json` is retained as a legacy multi-block alias.
- **FFT-based metrics**: RMS is computed via Parseval's theorem
  (band-limited integration of the PSD) rather than time-domain filtering.
  See `metrics._parseval_rms()` and `_compute_all_metrics()`.
- **Coherence bands**: `delta, theta, alpha, beta, gamma` defined in
  `metrics.COHERENCE_BANDS`.
- **Legacy waveform compatibility**: `_load_analyzer()` delegates legacy
  folders to SpikeInterface's waveform loader. The locked SpikeInterface
  0.105.3 includes the upstream load-order fix (SpikeInterface PR #4509).
  Converter PR #34's deprecated metric-name workaround was closed as
  unnecessary; it should not be restored. The declared minimum of 0.104.0
  still permits older installations and is separate from the lockfile.
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

The package supports Python 3.10 and newer. CI tests Python 3.10 and 3.13;
keep the oldest tested version aligned with `requires-python`.

- **spikeinterface[metrics]** >= 0.104.0 -- Recording/Sorting abstractions,
  SortingAnalyzer, and quality metrics. The converter uses the `metrics` extra;
  the `full` extra also installs unused CUDA, sorting, and visualization
  dependencies.
- **pandas, scipy** -- Direct dependencies for spike metadata tables and
  continuous-signal calculations
- **iblatlas** -- Atlas transforms
- **one-api** >= 2.6.0 -- IBL data API
- **antspyx** >= 0.4.2 -- Used by `histology.py` only
- **aind-mri-utils** -- Probe trajectory fitting in `histology.py`
- **wavpack-numcodecs** -- Decompresses WavPack-compressed zarr recordings
- **tqdm, numpy** -- Standard utilities

## Development branch map

Audited and cleaned against main at `09153d2` on 2026-10-21. An absent merge
commit does not establish that work is missing: several changes were squashed
or cherry-picked under new hashes. Remote branches were not deleted.

### Current feature work

| Branch | Meaning and next step |
| --- | --- |
| `main` | Contains the sampled metric API (`685f3cd`), dependency reduction (`b954978`), and revised execution guide (`74179f7`). Start new work here. |
| `feat/brain-surface-lines-main` | Port onto main's sampled API, including contact mapping and regression tests. Ready for real-data testing and integration; see [usage](docs/brain_surface_lines.md). Candidates still need calibration and downstream GUI integration. |
| `feat/brain-surface-lines` | Original detector at `09976dd`, retained as the reference for testing the port. |
| `feat/raw-activity-api` | Raw activity adapted to the sampled API, at `91e44a0`; checked out in the sibling `raw-activity-worktree`. This contains work beyond main. |
| `experiment/raw-activity-detector-corrections` | Follow-up detector corrections at `8713500`, branching from the activity implementation at `ea025eb`: physical neighborhoods prevent distant banks supporting each other, flat peaks count once, and eligible events obey the full refractory interval. This is a separate development line from the later activity documentation commit. |
| `origin/legacy-data-compat` / `origin/fix/legacy-data-compat` | Ben Hardcastle's open [PR #45](https://github.com/AllenNeuralDynamics/aind-ephys-ibl-gui-conversion/pull/45) targets the older `legacy-data-compat` branch (`3492a0b`). The newer `fix/legacy-data-compat` commit (`1707c35`) uses current loader helpers and explicit property precedence. Both recover QC and decoder labels from curated sorting objects when absent on the analyzer. They still require the quality-metrics extension. Port with coverage for absent curated folders and mismatched units; the older PR also uses Boolean `or` on NumPy arrays, which can fail. |

Arjun Sridhar opened [PR #34](https://github.com/AllenNeuralDynamics/aind-ephys-ibl-gui-conversion/pull/34)
for deprecated metric names. It was closed on 2026-04-15 because
[SpikeInterface PR #4509](https://github.com/SpikeInterface/spikeinterface/pull/4509)
fixed the underlying load order. The locked installation runs compatibility
migration before metric validation. The old monkey-patch is obsolete for that
installation; the remote branch remains historical.

The `yb-flexible-recording-number` proposal was explicitly discarded on
October 21. Main continues to discover analyzer names automatically and
retain the longest recording segment.

### Archives

| Branch | What it preserves | Assessment |
| --- | --- | --- |
| `archive/metric-prototype-with-detectors` (`f94446f`) | One preservation commit on top of `09976dd`: combined metric API, activity detector, surface detector, design review, numerical fixtures, and six benchmark JSON reports. | Contains unique experimental evidence and older detector implementations. The API was subsequently separated and landed on main; activity and surface features now have dedicated branches. Retain until those ports and benchmark records are accounted for. |
| `archive/pre-api-main-drafts` (`ff5f8e9`) | Child of `f9e08e4`: earlier execution/dependency prose plus a saved Zarr SortingAnalyzer extension reload test in `tests/test_spikes.py`. | The reload test remains unique. The Python 3.11/3.13 CI proposal was explicitly discarded on October 21: retain Python 3.10 in CI while the package supports it. Its execution guide predates the API and incorrectly describes the current pipeline as lacking configurable metrics. Do not merge the whole archive. |

The October 16 archives were explicitly created to preserve experiments and
uncommitted drafts while separating the sampled API from detector work. Their
large diffs against the merge base include earlier versions of work now on
main, so those diff sizes overstate the amount of missing implementation.

### Historical backup

`backup/feat-simplify-data-access-pre-squash` (`820509d`) retains detailed
development history and `REFACTOR.md`. Its runtime tree matches the unified
FFT squash (`ba9a482`) already on main. There is no active feature to port;
the branch preserves history that would otherwise lose its named reference.

### Removed local branches

| Branch | Former tip | Reason |
| --- | --- | --- |
| `archive/agent-refactor` | `68b38fa` | Superseded module split and FFT architecture. |
| `archive/pending-python-ci` | `f9e08e4` | Commit remains reachable through `archive/pre-api-main-drafts`. |
| `backup/fix-si-deprecated-template-metrics` | `cfd1582` | Ancestor of the retained compatibility fix branch. |
| `feat/sampled-metric-api` | `264d006` | Both staging commits are patch-equivalent to commits on main. |
| `experiment/metric-pipeline` | `09976dd` | Exact duplicate of `feat/brain-surface-lines`. |
| `feat/ephys-geometry-contract` | `720c18c` | Merged into main. |
| `fix/deprecated-waveforms-loader` | `ba9a482` | Merged into main. |
| `docs/readme-claude-and-copier-update` | `42c5889` | Both commits are patch-equivalent to commits on main. |
| `fix/constrain-parallelism` | `2e128d1` | Adjusted the removed filter pipeline; the FFT pipeline budgets workers directly. |
| `fix/spike-interface-104` | `5aa164a` | Adjusted removed filter code/tests; main already requires SpikeInterface >=0.104. |
| `feat/simplify-data-access` | `8d68596` | Exact tree match to `ba9a482`, already on main. |
| `fix/si-deprecated-template-metrics` | `2f27e26` | PR #34 was closed after the upstream fix; locked SpikeInterface 0.105.3 contains that fix. Full branch history remains on the remote. |
| `update-packaging` | `f85af1c` | Exact tree match to the squash merge `9af1263` from PR #28, merged February 25 Pacific time. No remaining changes to port. |
| `yb-flexible-recording-number` | `57b8da4` | Explicitly discarded; keep main's automatic analyzer discovery and longest-segment policy. |

Use `git show <commit>` for a preservation commit and `git diff main <branch>
-- <path>` to inspect the current content difference. Use `git cherry` to
detect equivalent patches; ahead/behind counts alone cannot identify work
that landed with different commit hashes. The table above records the local
branch cleanup; retained archives still contain the identified unique work.

## Related packages

- Consumed by `aind-ibl-ephys-alignment-preprocessing`, which imports
  `extract_spikes` and `extract_continuous` to produce ALF output for the
  IBL alignment GUI.
- The `utils.py` module is vendored from IBL's `ibl-neuropixel` (Feb 2024);
  if upgrading, check upstream for bug fixes in `WindowGenerator`, `bp`,
  `lp`, `hp`, `fscale`, `fexpand`, `rms`, and `fcn_cosine`.
