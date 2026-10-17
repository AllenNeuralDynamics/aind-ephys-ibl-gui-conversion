# Raw activity validation

The rewrite integrates the estimator from conversion PR #46 with the sampled
metric API and fixes three reproducible event-detection errors. It saves event
rates on physical contacts, leaving display-profile construction to the GUI.

## Comparison with the original detector

The reference is PR head `7731283a10ed8b9fd7aa1a0ef9a49b8a289682fa`.
Commit `ea025eb` first ports its detector unchanged into the extension API:
seven tests passed, including frozen per-channel rates on twelve identical
snippets with out-of-order updates, noise rejection, and trimmed reduction.
Subsequent detector corrections deliberately change some rates.

Direct tests of the original `_supported_candidates()` and the rewrite give:

| Input at 30 kHz | Original events | Rewritten events |
| --- | ---: | ---: |
| One four-sample flat negative peak with neighboring support | 4 | 1 |
| Supported peaks on one channel, 12 samples (0.4 ms) apart | 2 | 1 |
| A peak supported only by an array neighbor 700 µm away | 1 | 0 |

The original minimum filter covers approximately half the nominal refractory
interval on either side and retains equal minima. The rewrite selects one
peak per plateau and enforces the full minimum spacing among eligible peaks.
Support and local-noise neighborhoods use physical distances within each
shank, replacing depth-sorted index windows.

The geometry tests cover channel permutations, separate shanks, lateral
separation, bank gaps, mixed pitches, depth offsets, overlapping main/survey
contacts, and rejected measurements. The saved arrays retain one value per
canonical contact without interpolation or a synthetic bank span. A separate
test checks that tail trimming leaves analyzed duration unchanged.

These tests establish event-count and output-contract behavior. Real recording
data supplies no independent ground truth for neuron counts or detector
sensitivity. The 40 µm support radius, 100 µm noise radius, thresholds, and
median pooling across blocks remain explicit choices for scientific review.

On three identical one-second snippets from the real recording, the
integration-only accumulator matched the original rates exactly on all 384
contacts. The corrected detector changed 48 rates, with a maximum absolute
change of 1.40 Hz and no change in channel validity. To reproduce both this
comparison and the three synthetic cases above:

```sh
uv run python scripts/check_activity_reference.py --recording /path/to/recording.zarr
```

## Performance methodology

`scripts/benchmark_activity.py` compares four cases against the same current
pipeline: activity disabled, the original one-snippet estimator plus profile
composition, and the rewrite with one or three snippets. Each trial runs in a
fresh process. Case order is shuffled deterministically for each repetition.
The rewrite includes contact reduction; file writing is outside the timed
region for all cases. FFT and BLAS use one thread; decoder defaults are retained.

The input is a lossless Zarr/WavPack copy of a 600-second, 384-channel, 30 kHz
AP excerpt from recording `786867_50213-1`, available locally under `/mnt/Data/ephys`.
The concurrent case divides it into four contiguous 150-second blocks and runs
four block workers. These artificial boundaries exercise parallel processing;
they are not independent experiments. Page-cache state is uncontrolled, so
timings describe local warm-cache execution rather than remote-storage speed.
The machine was under substantial unrelated load during these runs. The
timings below are diagnostic observations; comparative overhead requires an
idle-machine rerun.

Each trial compares complete existing RMS, PSD, correlation, and complex
coherency arrays with activity disabled. FFT counts must match in every case.
For the rewrite, exact recording-read intervals and compressed trace-chunk
fetch counts must also match. The original estimator retains its separate
recording reads. Peak RSS measures the entire worker process, including
prefetched traces, decompression, and existing spectral accumulators.

To reproduce with this recording or another compatible compressed recording:

```sh
uv run python scripts/benchmark_activity.py --recording /path/to/recording.zarr --blocks 1 --repeats 3 --output /tmp/activity-serial.json
uv run python scripts/benchmark_activity.py --recording /path/to/recording.zarr --blocks 4 --repeats 3 --output /tmp/activity-concurrent.json
```

The JSON preserves individual measurements, event-rate summaries, and work
counters. The benchmark loads the original estimator directly from its Git
commit, so that commit must be present in the checkout.

## Serial measurements

Three shuffled trials per case gave these medians:

| Case | Median seconds | Peak RSS range, MiB |
| --- | ---: | ---: |
| Activity disabled | 14.76 | 2,993–3,247 |
| Original, one snippet | 17.24 | 3,149–3,252 |
| Rewrite, one snippet | 17.06 | 3,103–3,548 |
| Rewrite, three snippets | 18.33 | 3,332–3,462 |

Each baseline/rewrite run made 20 recording reads and 480 compressed chunk
fetches. The original added one recording read and six chunk fetches. All
existing numerical arrays and FFT counts matched the baseline.

Three snippets increase temporal coverage and computation; the factory keeps
the original one-snippet budget as its default. Peak RSS includes the existing
read prefetch and decoder allocations and varies between fresh processes.

## Four-worker measurements

Three trials per case gave these ranges under the same machine load:

| Case | Seconds, min–max | Peak RSS range, MiB |
| --- | ---: | ---: |
| Activity disabled | 28.23–37.74 | 9,697–11,955 |
| Original, one snippet per block | 30.65–37.17 | 9,975–10,772 |
| Rewrite, one snippet per block | 27.20–46.25 | 10,102–12,887 |
| Rewrite, three snippets per block | 30.47–46.10 | 10,025–13,266 |

The overlapping ranges and reversals between trials preclude a performance
ranking. Each baseline/rewrite run made 80 recording reads and 2,160 compressed
chunk fetches; the original made 84 reads and 2,184 fetches. All existing
numerical arrays and FFT counts matched the baseline. Short-block scheduling
selects twenty windows per block, so this case processes four times as many
windows as the serial case. It is not a serial-versus-parallel speed comparison.

All 154 tests pass on Python 3.13.16. Lint, formatting, spelling, docstring
coverage, and strict typing of the new activity module also pass.
