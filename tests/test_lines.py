"""Tests for interference-line detection and brain-surface transitions."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import spikeinterface as si
from probeinterface import Probe

from aind_ephys_ibl_gui_conversion.io import _assemble_and_save_stream
from aind_ephys_ibl_gui_conversion.lines import (
    LineAccumulator,
    LineParams,
    fit_transitions,
    plateau_onset,
    pool_lines,
)
from aind_ephys_ibl_gui_conversion.metrics import _compute_all_metrics
from aind_ephys_ibl_gui_conversion.surface_lines import brain_surface_lines
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
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(
        _locations(), shapes="circle", shape_params={"radius": 5}
    )
    probe.set_device_channel_indices(np.arange(N_CHANNELS))
    rec.set_probe(probe, in_place=True)
    rec.set_property("group", np.zeros(N_CHANNELS, dtype=int))
    return rec


def _save(
    block: ExperimentBlock | list[ExperimentBlock], out: Path, **kwargs
) -> dict:
    blocks = block if isinstance(block, list) else [block]
    stream = ProbeStream("s", "ProbeA", blocks, out)
    metric = brain_surface_lines(**kwargs)
    metrics = (metric,)
    _assemble_and_save_stream(
        stream,
        [_compute_all_metrics(b, extra_metrics=metrics) for b in blocks],
        30.0,
        4.0,
        extra_metrics=metrics,
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
        found = (
            _compute_all_metrics(block, extra_metrics=(brain_surface_lines(),))
            .metric_results["brainSurfaceLines"]
            .power.finalize()
        )
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
        found = (
            _compute_all_metrics(block, extra_metrics=(brain_surface_lines(),))
            .metric_results["brainSurfaceLines"]
            .power.finalize()
        )
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


def _two_lines(rms: np.ndarray) -> np.ndarray:
    return np.vstack([_jittered(rms, seed=0), _jittered(rms * 0.5, seed=1)])


def test_step_at_the_end_of_the_shank_is_not_a_candidate():
    rms = np.full(N_CHANNELS, 10.0)
    rms[-3:] = 3.0
    assert fit_transitions(_two_lines(rms), _locations(), LineParams()) == []


def test_step_at_the_end_does_not_displace_the_surface():
    rms = _stepped_rms()
    rms[-3:] /= 4.0
    found = fit_transitions(_two_lines(rms), _locations(), LineParams())
    assert [t["midpoint_um"] for t in found] == [SURFACE_MID_UM]


def test_channels_without_the_lines_are_dropped():
    rms = _stepped_rms()
    rms[20:22] = 1e-9  # adjacent, so outlier replacement keeps them
    found = fit_transitions(_two_lines(rms), _locations(), LineParams())
    assert [t["midpoint_um"] for t in found] == [SURFACE_MID_UM]


# Bath between these depths, air above it, tissue grading the lines away
# from their bath level below it.
BATH_BOTTOM_UM = 600.0
AIR_UM = 1000.0


def _bath_profile(sign: float) -> np.ndarray:
    y = _depths()
    db = sign * 0.01 * np.clip(BATH_BOTTOM_UM - y, 0, None)
    db = np.where(y > AIR_UM, db - 15.0, db)
    return 10.0 * 10 ** (db / 20)


def _bath_lines(seed: int = 0) -> np.ndarray:
    return np.vstack(
        [
            _jittered(_bath_profile(-1.0), seed=2 * seed),
            _jittered(_bath_profile(1.0), seed=2 * seed + 1),
        ]
    )


def test_line_collapse_at_the_top_is_the_air_boundary():
    found = fit_transitions(_bath_lines(), _locations(), LineParams())
    assert found[-1]["midpoint_um"] == AIR_UM + PITCH_UM / 2
    assert found[-1]["boundary"] == "air"
    assert all(t["boundary"] is None for t in found[:-1])


def test_lines_rising_at_the_surface_are_not_air():
    found = fit_transitions(
        _two_lines(_stepped_rms()), _locations(), LineParams()
    )
    assert [t["boundary"] for t in found] == [None]


def test_plateau_onset_is_where_the_lines_leave_the_bath():
    params = LineParams()
    errors = []
    for seed in range(10):
        amps = _bath_lines(seed)
        air = fit_transitions(amps, _locations(), params)[-1]
        assert air["boundary"] == "air"
        onset = plateau_onset(amps, _locations(), air, params)
        assert onset is not None
        assert onset["n_agreeing_lines"] == 2
        # Opposite grading, so opposite steps.
        steps = [line["step_db"] for line in onset["lines"]]
        assert steps[0] > 0 > steps[1]
        errors.append(onset["midpoint_um"] - BATH_BOTTOM_UM)
    # The trace back up rides chance runs of the noise for a window or so.
    assert np.median(np.abs(errors)) <= 2 * PITCH_UM


def test_no_plateau_onset_when_the_lines_stay_at_the_bath_level():
    params = LineParams()
    flat = np.where(_depths() > AIR_UM, 10.0 * 10 ** (-15 / 20), 10.0)
    amps = _two_lines(flat)
    air = fit_transitions(amps, _locations(), params)[-1]
    assert air["boundary"] == "air"
    assert plateau_onset(amps, _locations(), air, params) is None


def test_plateau_candidate_saved_and_air_listed_last():
    traces = _traces(
        12.0,
        _bath_profile(-1.0),
        second_rms_uv=_bath_profile(1.0),
    )
    with tempfile.TemporaryDirectory() as tmp:
        surface = _save(
            ExperimentBlock(_recording(traces), None, 0), Path(tmp)
        )
    candidates = surface["shanks"]["0"]["candidates"]
    assert candidates[-1]["boundary"] == "air"
    (plateau,) = [c for c in candidates if c["method"] == "plateau"]
    assert abs(plateau["midpoint_um"] - BATH_BOTTOM_UM) <= 3 * PITCH_UM
    assert plateau["block_indices"] == [0]


@pytest.mark.parametrize("separate_lfp", [False, True])
def test_surface_extension_preserves_existing_metrics(separate_lfp):
    """Borrowed spectra stay unchanged and NP2 reuses existing FFTs."""
    rec = _recording(_traces(8.0, _stepped_rms()))
    lfp = None
    if separate_lfp:
        traces = np.random.default_rng(2).normal(0, 10, (20000, N_CHANNELS))
        lfp = _recording(traces.astype(np.float32), fs=2500.0)
    block = ExperimentBlock(rec, lfp, 0)
    original = _compute_all_metrics(block)
    extended = _compute_all_metrics(
        block, extra_metrics=(brain_surface_lines(),)
    )
    for attribute in ("rms_ap", "rms_lfp", "psd_power", "psd_freqs"):
        np.testing.assert_array_equal(
            getattr(original, attribute), getattr(extended, attribute)
        )
    for band in original.correlation:
        np.testing.assert_array_equal(
            original.correlation[band], extended.correlation[band]
        )
        np.testing.assert_array_equal(
            original.coherency[band], extended.coherency[band]
        )
    if not separate_lfp:
        assert (
            original.metric_stats.fft_calls == extended.metric_stats.fft_calls
        )


def test_reordered_overlapping_banks_keep_native_amplitude_rows(tmp_path):
    """Partial banks join by contact coordinates and retain missing NaNs."""
    rec = _recording(_traces(8.0, _stepped_rms()))
    first = rec.select_channels(np.arange(40)[::-1])
    second = rec.select_channels(np.arange(24, N_CHANNELS))
    surface = _save(
        [ExperimentBlock(first, None, 0), ExperimentBlock(second, None, 1)],
        tmp_path,
    )
    coordinates = np.load(tmp_path / "channels.localCoordinates.npy")
    amplitudes = np.load(tmp_path / "_iblqc_ephysLineAmplitude.amps.npy")
    for row, line in enumerate(surface["lines"]):
        present = (
            coordinates[:, 1] <= _depths()[39]
            if line["block_index"] == 0
            else coordinates[:, 1] >= _depths()[24]
        )
        assert np.isfinite(amplitudes[row, present]).all()
        assert np.isnan(amplitudes[row, ~present]).all()
        if abs(line["freq_hz"] - LINE_HZ) < 0.25:
            expected = np.where(coordinates[present, 1] > SURFACE_UM, 20, 2)
            np.testing.assert_allclose(
                amplitudes[row, present] * 1e6, expected, rtol=0.12
            )
    metadata = json.loads(
        (tmp_path / "_iblqc_ephysLineAmplitude.metadata.json").read_text()
    )
    assert metadata["method"] == "unreferenced_spectral_excess"
    assert metadata["cmr"] is False
    assert len(metadata["blocks"]) == 2


def test_unavailable_line_band_fails_before_reads():
    """A search above Nyquist has an explicit configuration error."""
    with pytest.raises(ValueError, match="no AP FFT bins"):
        LineAccumulator(100, 500, N_CHANNELS, LineParams())
