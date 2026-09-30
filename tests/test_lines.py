"""Tests for interference-line detection and brain-surface transitions."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import spikeinterface as si

from aind_ephys_ibl_gui_conversion.io import _assemble_and_save_stream
from aind_ephys_ibl_gui_conversion.lines import (
    LineAccumulator,
    LineParams,
    fit_transitions,
    pool_lines,
)
from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from aind_ephys_ibl_gui_conversion.types import ExperimentBlock, ProbeStream

FS = 30000.0
LINE_HZ = 3870.3
N_CHANNELS = 64
PITCH_UM = 20.0
SECOND_LINE_HZ = 7503.1
# Channels above this y sit out of the brain.
SURFACE_UM = 1000.0
# Transition midpoints sit halfway between rows.
SURFACE_MID_UM = SURFACE_UM + PITCH_UM / 2


def _depths() -> np.ndarray:
    return np.arange(N_CHANNELS) * PITCH_UM


def _locations() -> np.ndarray:
    return np.c_[np.zeros(N_CHANNELS), _depths()]


def _traces(
    duration_s: float,
    line_rms_uv: np.ndarray,
    seed: int = 0,
    second_rms_uv: np.ndarray | None = None,
) -> np.ndarray:
    """White noise plus common-phase lines of per-channel rms."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * FS)
    t = np.arange(n) / FS
    x = rng.normal(0, 10, (n, N_CHANNELS)).astype(np.float32)
    for hz, rms in ((LINE_HZ, line_rms_uv), (SECOND_LINE_HZ, second_rms_uv)):
        if rms is None:
            continue
        tone = np.sin(2 * np.pi * hz * t).astype(np.float32)
        x += (np.sqrt(2) * rms).astype(np.float32)[None, :] * tone[:, None]
    return x


def _stepped_rms() -> np.ndarray:
    return np.where(_depths() > SURFACE_UM, 20.0, 2.0)


def _jittered(rms: np.ndarray, sd_db: float = 0.3, seed: int = 0):
    """Line rms in volts with independent per-channel dB noise."""
    rng = np.random.default_rng(seed)
    return rms * 10 ** (rng.normal(0, sd_db, rms.shape) / 20) * 1e-6


def _recording(traces: np.ndarray, fs: float = FS) -> si.NumpyRecording:
    rec = si.NumpyRecording([traces], fs)
    rec.set_property("location", _locations())
    rec.set_property("group", np.zeros(N_CHANNELS, dtype=int))
    return rec


def _save(
    block: ExperimentBlock | list[ExperimentBlock], out: Path, **kwargs
) -> dict:
    blocks = block if isinstance(block, list) else [block]
    stream = ProbeStream("s", "ProbeA", blocks, out)
    _assemble_and_save_stream(
        stream, [_compute_all_metrics(b) for b in blocks], 30.0, 4.0, **kwargs
    )
    return json.loads((out / "brain_surface.json").read_text())


class TestLineAccumulator:
    """Detection and per-channel measurement."""

    def _accumulate(self, rms_uv: np.ndarray, n_windows: int = 4):
        n = int(4.0 * FS)
        acc = LineAccumulator(n, FS, N_CHANNELS, LineParams())
        for w in range(n_windows):
            window = _traces(4.0, rms_uv, seed=w)
            acc.add_spectrum(window)
        return acc.finalize()

    def test_line_amplitude_recovered_per_channel(self):
        """A stepped line is found and measured in V rms on each channel."""
        found = self._accumulate(_stepped_rms())
        assert found.freqs_hz.size >= 1
        assert abs(found.freqs_hz[0] - LINE_HZ) < 0.25
        out = _depths() > SURFACE_UM
        amps_uv = found.amps_v[0] * 1e6
        assert np.median(amps_uv[out]) == pytest.approx(20.0, rel=0.05)
        assert np.median(amps_uv[~out]) == pytest.approx(2.0, rel=0.1)

    def test_pooled_line_measured_in_every_block(self):
        """A line seen in one block only is reported by both, at background
        level where it is absent."""
        n = int(4.0 * FS)
        accs = [LineAccumulator(n, FS, N_CHANNELS, LineParams()) for _ in "ab"]
        for w in range(4):
            accs[0].add_spectrum(_traces(4.0, np.zeros(N_CHANNELS), seed=w))
            accs[1].add_spectrum(
                _traces(4.0, np.full(N_CHANNELS, 5.0), seed=10 + w)
            )
        quiet, loud = pool_lines(accs, LineParams())
        assert quiet.freqs_hz == pytest.approx(loud.freqs_hz)
        assert abs(loud.freqs_hz[0] - LINE_HZ) < 0.25
        assert np.median(loud.amps_v[0]) * 1e6 == pytest.approx(5.0, rel=0.05)
        assert np.median(quiet.amps_v[0]) * 1e6 < 1.0

    def test_noise_detects_no_line(self):
        """White noise yields no line under the FDR gate."""
        found = self._accumulate(np.zeros(N_CHANNELS))
        assert found.freqs_hz.size == 0
        assert found.amps_v.shape == (0, N_CHANNELS)


class TestFitTransitions:
    """Shared-location transition fit and its gates."""

    def test_step_located_with_per_line_sign(self):
        """Lines stepping either way by different amounts share one
        midpoint, and each keeps its own step."""
        up = _stepped_rms()
        down = np.where(_depths() > SURFACE_UM, 2.0, 12.0)
        found = fit_transitions(
            _jittered(np.vstack([up, down, up])), _locations(), LineParams()
        )
        assert len(found) == 1
        assert found[0]["midpoint_um"] == SURFACE_MID_UM
        steps = [line["step_db"] for line in found[0]["lines"]]
        assert steps == pytest.approx([20.0, -15.6, 20.0], abs=1.0)

    def test_noise_and_depth_trends_give_no_candidate(self):
        """Noisy lines with linear depth trends of any size are absorbed by
        each line's trend; a pooled contrast would pass them at sqrt(12)."""
        rng = np.random.default_rng(1)
        n = 96
        locations = np.c_[np.zeros(n), np.arange(n) * 15.0]
        for _ in range(100):
            ramp = rng.normal(0, 3, (8, 1)) * np.linspace(-1, 1, n)
            db = rng.normal(0, 1, (8, n)) * rng.uniform(0.2, 3, (8, 1))
            amps = 10 ** ((db + ramp) / 20) * 1e-6
            assert fit_transitions(amps, locations, LineParams()) == []

    def test_lines_without_the_step_do_not_move_it(self):
        """Lines graded smoothly along depth neither shift the midpoint nor
        count as agreeing."""
        graded = np.tile(10 ** (np.linspace(0, 6, N_CHANNELS) / 20), (5, 1))
        rms = np.vstack([graded, _stepped_rms(), _stepped_rms()])
        found = fit_transitions(_jittered(rms), _locations(), LineParams())
        assert [t["midpoint_um"] for t in found] == [SURFACE_MID_UM]
        agrees = [line["agrees"] for line in found[0]["lines"]]
        assert agrees == [False] * 5 + [True, True]

    def test_two_boundaries(self):
        """Line groups stepping at different depths give two candidates."""
        deep_um = 600.0
        deep = np.where(_depths() > deep_um, 2.0, 12.0)
        rms = np.vstack([deep, deep, _stepped_rms(), _stepped_rms()])
        found = fit_transitions(_jittered(rms), _locations(), LineParams())
        assert [t["midpoint_um"] for t in found] == [
            deep_um + PITCH_UM / 2,
            SURFACE_MID_UM,
        ]

    def test_asymmetric_transition_is_one_candidate(self):
        """A transition sharper below than above is one candidate, not a
        step plus a second boundary."""
        d = _depths() - SURFACE_MID_UM
        rise = 1 / (1 + np.exp(-d / np.where(d < 0, 10.0, 80.0)))
        rms = np.tile(2.0 * 10 ** (rise * 20 / 20), (3, 1))
        found = fit_transitions(_jittered(rms), _locations(), LineParams())
        assert [t["midpoint_um"] for t in found] == [SURFACE_MID_UM]
        assert (
            found[0]["lines"][0]["width_above_um"]
            > found[0]["lines"][0]["width_below_um"]
        )

    def test_outlier_channels_do_not_move_it(self):
        """A few dead-looking channels on every line leave the midpoint."""
        rms = np.tile(_stepped_rms(), (3, 1))
        rms[:, [8, 30, 31, 44]] = 1e-3
        found = fit_transitions(_jittered(rms), _locations(), LineParams())
        assert [t["midpoint_um"] for t in found] == [SURFACE_MID_UM]

    def test_one_line_needs_a_declared_tone(self):
        """A single stepped line cannot agree with itself unless declared."""
        amps = _jittered(_stepped_rms()[None, :])
        assert fit_transitions(amps, _locations(), LineParams()) == []
        found = fit_transitions(
            amps, _locations(), LineParams(), min_agreeing_lines=1
        )
        assert found[0]["midpoint_um"] == SURFACE_MID_UM

    def test_too_few_channels(self):
        """Fewer channels than two sides' minimum give nothing."""
        found = fit_transitions(
            np.ones((2, 10)), _locations()[:10], LineParams()
        )
        assert found == []


class TestPooledBlocks:
    """One fit over every block that recorded a shank."""

    @staticmethod
    def _tiled(rms: np.ndarray, offsets_db: list[float]):
        """Lower and upper halves of the shank as two blocks, each block's
        lines shifted by its offset."""
        blocks = (np.arange(N_CHANNELS) >= N_CHANNELS // 2).astype(int)
        gain = 10 ** (np.asarray(offsets_db)[blocks] / 20)
        return _jittered(rms) * gain, blocks

    def test_offset_between_blocks_is_not_a_transition(self):
        """Lines whose level differs between two blocks tiling the shank
        give no candidate, where one block would place one at the seam."""
        amps, blocks = self._tiled(np.full((3, N_CHANNELS), 5.0), [0.0, 8.0])
        params = LineParams()
        assert fit_transitions(amps, _locations(), params, blocks=blocks) == []
        seam = _depths()[N_CHANNELS // 2] - PITCH_UM / 2
        found = fit_transitions(amps, _locations(), params)
        assert [t["midpoint_um"] for t in found] == [seam]

    def test_surface_found_across_offset_blocks(self):
        """A step inside one block is located with each block's own
        baseline."""
        rms = np.tile(_stepped_rms(), (3, 1))
        amps, blocks = self._tiled(rms, [0.0, -6.0])
        found = fit_transitions(
            amps, _locations(), LineParams(), blocks=blocks
        )
        assert [t["midpoint_um"] for t in found] == [SURFACE_MID_UM]
        baseline = found[0]["lines"][0]["baseline_db"]
        assert baseline[1] - baseline[0] == pytest.approx(-6.0, abs=1.0)
        assert found[0]["lines"][0]["step_db"] == pytest.approx(20.0, abs=1.0)

    def test_repeat_blocks_are_replicates(self):
        """Two recordings of the same channels at different line levels
        agree on one transition."""
        rms = np.tile(_stepped_rms(), (2, 1))
        amps = np.hstack(
            [_jittered(rms, seed=1), _jittered(rms, seed=2) * 10 ** (4 / 20)]
        )
        found = fit_transitions(
            amps,
            np.vstack([_locations()] * 2),
            LineParams(),
            blocks=np.repeat([0, 1], N_CHANNELS),
        )
        assert [t["midpoint_um"] for t in found] == [SURFACE_MID_UM]

    def test_noisy_block_adds_little_evidence(self):
        """A far noisier replicate block neither discounts nor inflates the
        clean block's evidence."""
        rms = np.tile(_stepped_rms(), (3, 1))
        clean = _jittered(rms, seed=1)
        alone = fit_transitions(clean, _locations(), LineParams())
        pooled = fit_transitions(
            np.hstack([clean, _jittered(rms, sd_db=3.0, seed=2)]),
            np.vstack([_locations()] * 2),
            LineParams(),
            blocks=np.repeat([0, 1], N_CHANNELS),
        )
        assert [t["midpoint_um"] for t in pooled] == [SURFACE_MID_UM]
        assert pooled[0]["delta_bic"] == pytest.approx(
            alone[0]["delta_bic"], rel=0.05
        )


class TestUnreferencedLines:
    """Lines must be measured before the common-median reference."""

    def test_np2_lines_read_before_cmr(self):
        """A uniform common-mode line, which CMR cancels, is measured at
        full amplitude on a wideband 2.0 recording."""
        uniform = np.full(N_CHANNELS, 5.0)
        block = ExperimentBlock(_recording(_traces(12.0, uniform)), None, 0)
        found = _compute_all_metrics(block).line_power.finalize()
        assert abs(found.freqs_hz[0] - LINE_HZ) < 0.25
        assert np.median(found.amps_v[0]) * 1e6 == pytest.approx(5.0, rel=0.05)

    def test_np1_lines_from_ap_stream(self):
        """A 1.0 recording measures lines on its AP stream."""
        uniform = np.full(N_CHANNELS, 5.0)
        lfp = _recording(
            np.random.default_rng(9)
            .normal(0, 10, (int(12.0 * 2500), N_CHANNELS))
            .astype(np.float32),
            fs=2500.0,
        )
        block = ExperimentBlock(_recording(_traces(12.0, uniform)), lfp, 0)
        found = _compute_all_metrics(block).line_power.finalize()
        assert np.median(found.amps_v[0]) * 1e6 == pytest.approx(5.0, rel=0.05)


class TestLineOutputs:
    """Saved artifacts and brain_surface.json."""

    @staticmethod
    def _stepped_block(
        gain: float = 1.0, seed: int = 0, block_index: int = 0
    ) -> ExperimentBlock:
        down = np.where(_depths() > SURFACE_UM, 2.0, 12.0)
        traces = _traces(
            12.0, gain * _stepped_rms(), seed=seed, second_rms_uv=gain * down
        )
        return ExperimentBlock(_recording(traces), None, block_index)

    def test_candidates_saved(self):
        """Two lines stepping at the surface give one lines candidate whose
        rows index the saved amplitudes."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            surface = _save(self._stepped_block(), out)
            amps = np.load(out / "_iblqc_ephysLineAmplitude.amps.npy")
            freqs = np.load(out / "_iblqc_ephysLineAmplitude.freqs.npy")
        (candidate,) = surface["shanks"]["0"]["candidates"]
        assert candidate["method"] == "lines"
        assert candidate["midpoint_um"] == SURFACE_MID_UM
        assert amps.shape == (freqs.size, N_CHANNELS)
        assert len(surface["lines"]) == freqs.size
        stepped = [
            line["amplitude_rows"]["0"]
            for line in candidate["lines"]
            if line["agrees"]
        ]
        assert freqs[stepped] == pytest.approx(
            [LINE_HZ, SECOND_LINE_HZ], abs=0.25
        )

    def test_blocks_pooled_into_one_candidate(self):
        """Two recordings of the shank at different line levels share their
        lines and give one candidate drawing on both."""
        blocks = [
            self._stepped_block(),
            self._stepped_block(gain=2.0, seed=5, block_index=1),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            surface = _save(blocks, out)
            freqs = np.load(out / "_iblqc_ephysLineAmplitude.freqs.npy")
        (candidate,) = surface["shanks"]["0"]["candidates"]
        assert candidate["midpoint_um"] == SURFACE_MID_UM
        assert candidate["block_indices"] == [0, 1]
        n_lines = candidate["n_lines"]
        assert freqs.size == 2 * n_lines
        for line in candidate["lines"]:
            rows = line["amplitude_rows"]
            assert freqs[[rows["0"], rows["1"]]] == pytest.approx(
                [line["freq_hz"]] * 2
            )
        offsets = [
            line["baseline_db"]["1"] - line["baseline_db"]["0"]
            for line in candidate["lines"]
            if line["agrees"]
        ]
        assert offsets == pytest.approx([6.0] * len(offsets), abs=1.0)

    def test_declared_driven_tone(self):
        """A declared tone is also fitted alone as a driven candidate."""
        with tempfile.TemporaryDirectory() as tmp:
            surface = _save(
                self._stepped_block(), Path(tmp), driven_tone_hz=LINE_HZ
            )
        candidates = surface["shanks"]["0"]["candidates"]
        driven = [c for c in candidates if c["method"] == "driven"]
        assert len(driven) == 1
        assert driven[0]["midpoint_um"] == SURFACE_MID_UM
        assert len(driven[0]["lines"]) == 1

    def test_no_line_gives_no_candidate(self):
        """A line-free stream writes empty amplitudes and no candidates."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            block = ExperimentBlock(
                _recording(_traces(12.0, np.zeros(N_CHANNELS))), None, 0
            )
            surface = _save(block, out)
            amps = np.load(out / "_iblqc_ephysLineAmplitude.amps.npy")
        assert amps.shape == (0, N_CHANNELS)
        assert surface["shanks"]["0"]["candidates"] == []
