# Frozen metric baseline

`metric_pipeline_baseline.npz` contains 30 arrays captured by running
`metrics.py` from main commit `202625c204d1cd260f52f319e8eaba49f6c28e1f` on
`make_block("np1")` and `make_block("np2")` from `test_window_pipeline.py`, with
`window_duration=2`. It includes AP/LFP RMS, timestamps, PSD power/frequencies,
and all five bands of real/complex coherency.

The capture used Python 3.13.16, NumPy 2.3.5, SciPy 1.15.3, and
SpikeInterface 0.104.1. The helper fixes the random seeds and creates two
physical shanks. Tests load these numerical results rather than a second copy
of the implementation. Changes to this fixture require an explicit scientific
or output-contract change, not automatic regeneration by the tests.
