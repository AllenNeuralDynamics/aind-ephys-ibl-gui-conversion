# Brain surface candidates from interference lines

Enable the line detector through the sampled metric API:

```python
from aind_ephys_ibl_gui_conversion import extract_continuous
from aind_ephys_ibl_gui_conversion.surface_lines import brain_surface_lines

extract_continuous(
    sorting_folder,
    results_folder,
    extra_metrics=(brain_surface_lines(),),
)
```

For a known tone driven into the animal, configure
`brain_surface_lines(driven_tone_hz=1000.0)`. Detection and fit settings are
available as `brain_surface_lines(params=LineParams(...))`, importing
`LineParams` from `aind_ephys_ibl_gui_conversion.lines`.

## Processing

The metric detects peaks of spectral excess over neighboring frequency bins
in unreferenced, demeaned, Hann-windowed AP data. This is a spectral excess
test; wandering narrowband interference is not assumed to be a stable
sinusoid. NP2 shares the existing wideband FFT and restores its median
reference in owned band buffers. NP1 requests a full AP-window Hann FFT,
independent of its separate LF recording. No extra recording reads are needed.

Blocks with the same frequency grid pool detected frequencies and measure
them in every compatible block. Fits use each contact's physical depth,
with separate block baselines to allow line amplitudes to drift between
recordings. AP RMS above 300 Hz rejects dead or noisy contacts; contacts
without the lines and transitions pinned to shank ends are also excluded.
This rejection band is fixed independently of the extraction's AP RMS setting.

The fit retains passing shared line transitions. A shallow collapse in line
amplitudes can be labeled `air`; below it, departure from the bath plateau
can supply a `plateau` candidate. A declared driven tone is additionally fit
alone as `driven`. Candidates are ordered by support, with air last.

## Output contract

Four files are written in each ephys collection:

| File | Contents |
| --- | --- |
| `_iblqc_ephysLineAmplitude.amps.npy` | `(line/block rows, channel-table rows)`, RMS volts |
| `_iblqc_ephysLineAmplitude.freqs.npy` | Frequency in Hz of each amplitude row |
| `brain_surface.json` | Line row identities and candidates by producer shank ID |
| `_iblqc_ephysLineAmplitude.metadata.json` | Detector settings, units, block identity, and sampled intervals |

Amplitude columns join to `channels.localCoordinates.npy`. Each detected line
has a separate row for each block; unrecorded contacts are NaN, including
when blocks overlap or have different native channel orders. Amplitudes are
not averaged across blocks. JSON block indices address stream result order;
the metadata records original experiment indices and block IDs.

Candidate depths are probe-local micrometres along `localCoordinates[:, 1]`,
increasing from the tip. Per-line fit records refer to amplitude rows and
include block baselines, steps, and widths. A stream without usable lines
writes an empty `(0, N)` amplitude array and empty candidate lists.

These are experimental candidates for inspection. No candidate is selected
automatically as the brain surface, and the support threshold still needs
calibration against insertion depths. Some rigs supply no useful lines;
empty results are expected. GUI display and downstream selection are
separate integration work.

## Branch provenance

This ports `feat/brain-surface-lines` at `09976dd` onto main's sampled metric
API (`685f3cd`). It preserves the detector and output shapes while replacing
the old direct hooks into `metrics.py` and `io.py` with a `Metric` configuration.
Enable it with `extra_metrics`; the earlier branch ran it by default and took
`driven_tone_hz` directly on `extract_continuous`. Method details now live in
the detector's own metadata file.
