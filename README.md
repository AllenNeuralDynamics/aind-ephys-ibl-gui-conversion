# AIND Ephys IBL GUI Conversion

![CI](https://github.com/AllenNeuralDynamics/aind-ephys-ibl-gui-conversion/actions/workflows/ci-call.yml/badge.svg)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border.json)](https://github.com/copier-org/copier)

Convert AIND Neuropixels electrophysiology data into
[IBL ALF format](https://int-brain-lab.github.io/ONE/alf_intro.html) for
consumption by the
[IBL ephys alignment GUI](https://github.com/AllenNeuralDynamics/ibl-ephys-alignment-gui).
It extracts spike data from SpikeInterface sorting output and computes
continuous QC metrics (RMS, spectral density, per-band coherence) from raw
Neuropixels recordings.

## Installation

From source (no PyPI release):

```bash
git clone https://github.com/AllenNeuralDynamics/aind-ephys-ibl-gui-conversion.git
cd aind-ephys-ibl-gui-conversion
pip install .
```

Or add as a dependency:

```bash
uv add "aind-ephys-ibl-gui-conversion @ git+https://github.com/AllenNeuralDynamics/aind-ephys-ibl-gui-conversion.git"
```

## Usage

This package is a library -- there is no CLI. The two top-level entry points
are `extract_spikes()` and `extract_continuous()`, both imported from the
package root:

```python
from pathlib import Path
from aind_ephys_ibl_gui_conversion import extract_spikes, extract_continuous

sorting_folder = Path("/path/to/<session>_sorted")
results_folder = Path("/path/to/output")

# Convert sorted spikes into IBL ALF format
extract_spikes(sorting_folder, results_folder)

# Compute continuous QC metrics (RMS, PSD, coherence)
extract_continuous(sorting_folder, results_folder)
```

Both functions write one subdirectory per probe under `results_folder`.

### `extract_spikes(sorting_folder, results_folder, ...)`

Reads the `postprocessed/` directory inside `sorting_folder` and writes
IBL ALF spike files per probe.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sorting_folder` | -- | Path to the sorted session folder (name typically ends with `_sorted`) |
| `results_folder` | -- | Output directory (one subfolder per probe will be created) |
| `stream_to_use` | `None` | Filter to a single Open Ephys stream name (e.g. `"Record Node 104#Neuropix-PXI-100.ProbeA-AP"`). If `None`, all streams are processed. |
| `min_duration_secs` | `300` | Skip recordings shorter than this many seconds |

### `extract_continuous(sorting_folder, results_folder, ...)`

Reads compressed Neuropixels recordings from `ecephys_compressed/` and writes
per-band RMS, PSD, and coherence files in IBL ALF format.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sorting_folder` | -- | Path to the sorted session folder |
| `results_folder` | -- | Output directory |
| `stream_to_use` | `None` | Filter to a single Open Ephys stream |
| `main_recording_min_secs` | `600` | Minimum duration (s) for the main recording block |
| `probe_surface_finding` | `None` | Optional path to a separate surface-finding asset |
| `rms_window_interval` | `30.0` | Gap between analysis windows (s); auto-shortened on short recordings to guarantee >= 20 windows |
| `rms_window_duration` | `4.0` | Length of each analysis window (s) |
| `num_parallel_jobs` | `4` | Concurrent block processing |

Coherence bands (from `metrics.COHERENCE_BANDS`):

| Band | Range (Hz) |
|------|------------|
| delta | 0.5 -- 4 |
| theta | 4 -- 12 |
| alpha | 12 -- 30 |
| beta | 30 -- 100 |
| gamma | 100 -- 300 |

## Input data expectations

Both entry points expect the AIND session layout:

```
<session>/
|-- ecephys_clipped/                    # Open Ephys binary (read by SpikeInterface)
|-- ecephys_compressed/                 # Zarr + WavPack compressed recordings
|   |-- experiment1_<stream>.zarr
|   +-- experiment2_<stream>.zarr
|
+-- <session>_sorted/                   # SORTING_FOLDER argument points here
    +-- postprocessed/
        |-- experiment1_<stream>_recording1.zarr        # SortingAnalyzer
        +-- experiment1_<stream>_recording1_group<N>.zarr  # Multi-shank (one per shank)
```

The package handles both Neuropixels 1.0 (separate AP/LFP streams) and 2.0
(combined wideband) probes, as well as multi-shank probes.

### Stream naming

Open Ephys stream names follow the pattern:

```
Record Node <id>#Neuropix-PXI-<model>.<probe_name>[-AP|-LFP]
```

For example, `Record Node 104#Neuropix-PXI-100.ProbeA-AP` is parsed into
`probe_name = "ProbeA"`.

## Output format

All outputs are written under `results_folder/<probe_name>/` in IBL ALF
format.

### From `extract_spikes`

| File | Shape / dtype | Description |
|------|---------------|-------------|
| `spikes.times.npy` | (n_spikes,) float64 | Spike times in seconds |
| `spikes.clusters.npy` | (n_spikes,) | Cluster ID per spike |
| `spikes.amps.npy` | (n_spikes,) float64 | Spike amplitudes (negated) |
| `spikes.depths.npy` | (n_spikes,) | Spike depths along the probe (um) |
| `spike_shank_indices.npy` | (n_spikes,) | 0-based shank index per spike |
| `clusters.channels.npy` | (n_clusters,) | Peak channel per cluster |
| `clusters.peakToTrough.npy` | (n_clusters,) | Peak-to-trough duration (samples) |
| `clusters.waveforms.npy` | (n_clusters, n_samples, n_channels) | Mean waveforms |
| `clusters.metrics.csv` | -- | SpikeInterface quality metrics |
| `unit_shank_indices.npy` | (n_clusters,) | 0-based shank index per cluster |

### From `extract_continuous`

| File | Description |
|------|-------------|
| `_iblqc_ephysTimeRmsAP.rms.npy` (`*Main*` if multi-block) | AP-band RMS time series, shape (n_windows, n_channels) |
| `_iblqc_ephysTimeRmsAP.timestamps.npy` | Window centers (s) |
| `_iblqc_ephysTimeRmsLF*.rms.npy` | LFP-band RMS time series |
| `_iblqc_ephysTimeRmsLF*.timestamps.npy` | Window centers (s) |
| `_iblqc_ephysSpectralDensityLF.power.npy` | PSD (V^2/Hz), shape (n_freqs, n_channels) |
| `_iblqc_ephysSpectralDensityLF.freqs.npy` | Frequencies (Hz) |
| `band_corr/<band>_shank<i>_mean_corr.npy` | Per-band, per-shank correlation matrices (real, power-normalized) |
| `band_corr/<band>_shank<i>_coherency.npy` | Per-band, per-shank complex coherency |
| `band_corr/channel_blocks.json` | Multi-block channel map (only if surface-finding blocks are present) |
| `channels.localCoordinates.npy` | Unique channel positions (x, y) |
| `channels.rawInd.npy` | Channel indices |
| `_iblqc_metrics.method.json` | Method metadata (FFT params, CMR flag) |

## Metrics method

RMS and PSD are computed via FFT, not time-domain filtering:

- Hann-windowed segments, Welch-style PSD
- Band-limited RMS via Parseval's theorem (integrate PSD over each band)
- AP band: > 300 Hz (neural noise); LFP band: 1 -- 300 Hz
- Common-mode rejection (CMR) is applied

## Adapting to your own data

The package is tightly coupled to the AIND session layout and Open Ephys
stream naming. To adapt it to a different setup, the main integration points
are:

1. **Recording discovery**: `recording_utils.get_ecephys_stream_names()` and
   `io.load_probe_streams()` enumerate compressed `.zarr` recordings under
   `ecephys_compressed/`. Replace these if your recordings live elsewhere or
   use a different container.
2. **Stream name parsing**: `recording_utils._stream_to_probe_name()` extracts
   the probe name via a regex over the Open Ephys stream ID. Replace this if
   your stream names follow a different convention.
3. **Sorting output**: `extract_spikes()` assumes SpikeInterface
   `SortingAnalyzer` output in `<sorting_folder>/postprocessed/`. If you have
   a different SortingAnalyzer location or a different post-processed format,
   point at it directly.
4. **Surface-finding recordings**: If you use separate "surface finding"
   recordings (short probe-movement blocks used to map the cortical surface),
   pass the asset path via `probe_surface_finding`. Otherwise leave it
   unset.

The output format is IBL ALF and is decoupled from input assumptions -- it
will work with any IBL GUI consumer.

## Development

Set up the environment:

```bash
uv sync
```

Run the full check suite:

```bash
./scripts/run_linters_and_checks.sh -c
```

Or run individual checks:

```bash
uv run --frozen ruff format                          # Code formatting
uv run --frozen ruff check                           # Linting
uv run --frozen interrogate -v src                   # Docstring coverage (100% required)
uv run --frozen codespell --check-filenames          # Spell checking
uv run --frozen pytest --cov aind_ephys_ibl_gui_conversion  # Tests with coverage
```

## License

This project is licensed under the MIT License -- see the [LICENSE](LICENSE)
file for details.
