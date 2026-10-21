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
  folders to SpikeInterface's waveform loader. The additional deprecated
  metric-name patch exists on `fix/si-deprecated-template-metrics` and its
  backup, and is absent from main; see the branch map below.
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

Audited against main at `09153d2` on 2026-10-21. Branch names below describe
local refs; an absent merge commit does not establish that their work is
missing. Several changes were squashed or cherry-picked under new hashes.

### Current feature work

| Branch | Meaning and next step |
| --- | --- |
| `main` | Contains the sampled metric API (`685f3cd`), dependency reduction (`b954978`), and revised execution guide (`74179f7`). Start new work here. |
| `feat/sampled-metric-api` | Old API staging branch. Its two commits (`6926e6a`, `264d006`) are patch-equivalent to commits on main (`git cherry main feat/sampled-metric-api` marks both `-`). No feature remains to port. |
| `feat/brain-surface-lines` | Original detector at `09976dd`, based before the API. `experiment/metric-pipeline` points to the same commit. Use the port on `feat/brain-surface-lines-main` for the current API; see [usage](docs/brain_surface_lines.md). |
| `feat/raw-activity-api` | Raw activity adapted to the sampled API, at `91e44a0`; checked out in the sibling `raw-activity-worktree`. This contains work beyond main. |
| `experiment/raw-activity-detector-corrections` | Follow-up detector corrections at `8713500`, branching from the activity implementation at `ea025eb`. This is a separate development line from the later activity documentation commit. |

### Archives

| Branch | What it preserves | Assessment |
| --- | --- | --- |
| `archive/agent-refactor` (`68b38fa`) | Early split of `ephys.py` into types, I/O, metrics, spikes, and recording utilities. | Superseded by the later module split (`541fa42`) and unified FFT rewrite. Main has that architecture. Keep as historical reference; merging this old snapshot would not help. |
| `archive/metric-prototype-with-detectors` (`f94446f`) | One preservation commit on top of `09976dd`: combined metric API, activity detector, surface detector, design review, numerical fixtures, and six benchmark JSON reports. | Contains unique experimental evidence and older detector implementations. The API was subsequently separated and landed on main; activity and surface features now have dedicated branches. Retain until those ports and benchmark records are accounted for. |
| `archive/pending-python-ci` (`f9e08e4`) | A single commit on top of the old API staging tip `264d006`, proposing Python 3.11/3.13 in place of 3.10/3.13. | The CI change is still absent from audited main. Review that single commit independently of its already-landed API ancestors. This commit is also reachable through `archive/pre-api-main-drafts`. |
| `archive/pre-api-main-drafts` (`ff5f8e9`) | Child of `f9e08e4`: earlier execution/dependency prose plus a saved Zarr SortingAnalyzer extension reload test in `tests/test_spikes.py`. | Main contains revised documentation, but the reload test and CI proposal remain unique. Its execution guide predates the API and incorrectly describes the current pipeline as lacking configurable metrics. Do not merge the whole archive. |

The October 16 archives were explicitly created to preserve experiments and
uncommitted drafts while separating the sampled API from detector work. Their
large diffs against the merge base include earlier versions of work now on
main, so those diff sizes overstate the amount of missing implementation.

### Backups

| Branch | Assessment |
| --- | --- |
| `backup/feat-simplify-data-access-pre-squash` (`820509d`) | Detailed development history before the unified FFT squash. Its final tree differs from `feat/simplify-data-access` (`8d68596`) only by `REFACTOR.md`; it preserves history and a plan, rather than a separate runtime feature. |
| `backup/fix-si-deprecated-template-metrics` (`cfd1582`) | Earlier version of the legacy waveform metric-name compatibility fix. The later remote branch (`origin/fix/si-deprecated-template-metrics`, `404748d`) differs only in `spikes.py` formatting, a docstring, and log expression construction. Main's `_load_analyzer()` delegates to SpikeInterface and does not contain this patch, despite the older development guide claiming it did. The backup is redundant with the later fix branch for runtime behavior, but that fix itself remains separate work to evaluate. |

Use `git show <commit>` for a preservation commit and `git diff main <branch>
-- <path>` to inspect the current content difference. Use `git cherry` to
detect equivalent patches; ahead/behind counts alone cannot identify work
that landed with different commit hashes. No archive or backup refs were
deleted during this audit.

## Related packages

- Consumed by `aind-ibl-ephys-alignment-preprocessing`, which imports
  `extract_spikes` and `extract_continuous` to produce ALF output for the
  IBL alignment GUI.
- The `utils.py` module is vendored from IBL's `ibl-neuropixel` (Feb 2024);
  if upgrading, check upstream for bug fixes in `WindowGenerator`, `bp`,
  `lp`, `hp`, `fscale`, `fexpand`, `rms`, and `fcn_cosine`.
