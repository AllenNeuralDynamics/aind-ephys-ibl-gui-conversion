"""
Tests for FFT-based ephys metric computation.

Unit tests use synthetic data. Integration test uses a real zarr from S3.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from aind_ephys_ibl_gui_conversion.channel_metadata import (
    build_channel_table,
    normalized_channel_groups,
)
from aind_ephys_ibl_gui_conversion.io import (
    process_stream_fft,
)
from aind_ephys_ibl_gui_conversion.metrics import (
    COHERENCE_BANDS,
    _compute_all_metrics,
    _parseval_rms,
)
from aind_ephys_ibl_gui_conversion.types import ExperimentBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_recording(
    duration_sec: float = 10.0,
    fs: float = 30000.0,
    n_channels: int = 32,
    freqs_hz: list[float] | None = None,
    groups: np.ndarray | None = None,
    locations: np.ndarray | None = None,
):
    """Create a mock SI-like recording backed by a numpy array."""
    import spikeinterface as si

    if freqs_hz is None:
        freqs_hz = [10.0, 100.0, 500.0]

    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs

    # Generate signal: sum of sinusoids with per-channel amplitude variation
    rng = np.random.default_rng(42)
    traces = np.zeros((n_samples, n_channels), dtype=np.float32)
    for f in freqs_hz:
        amps = rng.uniform(0.5, 2.0, size=n_channels).astype(np.float32)
        traces += amps[None, :] * np.sin(2 * np.pi * f * t)[:, None]

    # Add some noise
    traces += rng.normal(0, 0.1, size=traces.shape).astype(np.float32)

    recording = si.NumpyRecording(
        traces_list=[traces],
        sampling_frequency=fs,
    )

    # Set channel locations (depth along y-axis)
    if locations is None:
        locations = np.zeros((n_channels, 2))
        locations[:, 1] = np.arange(n_channels) * 20.0  # 20 um spacing
    recording.set_property("location", locations)

    # Set channel group / shank metadata.
    if groups is None:
        groups = np.zeros(n_channels, dtype=int)
    recording.set_property("group", np.asarray(groups, dtype=int))

    return recording, traces


class _FakeRecording:
    """Minimal recording exposing geometry/contact_ids for metadata tests."""

    def __init__(
        self,
        *,
        locations: np.ndarray,
        groups: np.ndarray,
        contact_ids: np.ndarray | None = None,
        channel_ids: np.ndarray | None = None,
        probe=None,
    ) -> None:
        self._locations = locations
        self._groups = groups
        self._contact_ids = contact_ids
        self._channel_ids = channel_ids
        self._probe = probe

    def get_num_channels(self):
        return self._locations.shape[0]

    def get_channel_locations(self):
        return self._locations

    def get_property(self, name):
        if name == "group":
            return self._groups
        if name == "contact_ids" and self._contact_ids is not None:
            return self._contact_ids
        raise KeyError(name)

    def get_channel_ids(self):
        if self._channel_ids is None:
            return np.arange(self.get_num_channels())
        return self._channel_ids

    def get_probe(self):
        if self._probe is None:
            raise ValueError("No probe attached.")
        return self._probe


class _FakeProbe:
    """Minimal probe exposing contact IDs and device channel indices."""

    def __init__(
        self, *, contact_ids: np.ndarray, device_channel_indices: np.ndarray
    ):
        self.contact_ids = contact_ids
        self.device_channel_indices = device_channel_indices


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestParsevalRms:
    """Tests for _parseval_rms function."""

    def test_matches_time_domain_rms(self):
        """FFT-based RMS should match time-domain RMS for a pure tone."""
        fs = 1000.0
        duration = 2.0
        n = int(fs * duration)
        t = np.arange(n) / fs

        # Pure 50 Hz sine wave, amplitude 1
        signal = np.sin(2 * np.pi * 50 * t).reshape(-1, 1)
        expected_rms = np.sqrt(np.mean(signal**2, axis=0))

        import scipy.fft

        X = scipy.fft.rfft(signal, axis=0)
        freqs = scipy.fft.rfftfreq(n, d=1.0 / fs)

        # Full-band RMS
        all_mask = freqs > 0  # exclude DC
        fft_rms = _parseval_rms(X, all_mask, n)

        np.testing.assert_allclose(fft_rms, expected_rms, rtol=0.02)

    def test_band_limited_rms(self):
        """RMS in a specific band should only capture that band's energy."""
        fs = 1000.0
        n = 2000
        t = np.arange(n) / fs

        # 10 Hz (amplitude 3) + 200 Hz (amplitude 4)
        signal = (
            3.0 * np.sin(2 * np.pi * 10 * t)
            + 4.0 * np.sin(2 * np.pi * 200 * t)
        ).reshape(-1, 1)

        import scipy.fft

        X = scipy.fft.rfft(signal, axis=0)
        freqs = scipy.fft.rfftfreq(n, d=1.0 / fs)

        # RMS of just the 10 Hz component
        low_mask = (freqs >= 5) & (freqs <= 15)
        rms_low = _parseval_rms(X, low_mask, n)
        expected_low = 3.0 / np.sqrt(2)  # RMS of sine = A/sqrt(2)

        np.testing.assert_allclose(rms_low, expected_low, rtol=0.05)


class TestSparseRms:
    """Tests for sparse RMS computation."""

    def test_output_shapes(self):
        """Test that output arrays have expected shapes."""
        recording, _ = _make_synthetic_recording(duration_sec=10.0)
        block = ExperimentBlock(
            recording=recording, lfp_recording=None, block_index=0
        )
        result = _compute_all_metrics(
            block,
            window_interval=2.0,
            window_duration=2.0,
        )

        # 10s duration, 2s interval, 2s window
        n_expected = len(np.arange(0, 10.0 - 2.0, 2.0))
        assert result.rms_ap.shape == (n_expected, 32)
        assert result.rms_lfp.shape == (n_expected, 32)
        assert result.timestamps.shape == (n_expected,)

    def test_timestamps_span_recording(self):
        """Test that timestamps span recording duration."""
        recording, _ = _make_synthetic_recording(duration_sec=10.0)
        block = ExperimentBlock(
            recording=recording, lfp_recording=None, block_index=0
        )
        result = _compute_all_metrics(
            block,
            window_interval=2.0,
            window_duration=2.0,
        )

        assert result.timestamps[0] >= 0
        assert result.timestamps[-1] <= 10.0

    def test_lfp_rms_captures_low_freq(self):
        """Low-freq content should give LFP RMS > AP RMS."""
        recording, _ = _make_synthetic_recording(
            duration_sec=10.0, freqs_hz=[10.0, 50.0]
        )
        block = ExperimentBlock(
            recording=recording, lfp_recording=None, block_index=0
        )
        result = _compute_all_metrics(
            block,
            window_interval=2.0,
            window_duration=2.0,
        )

        mean_lfp = np.mean(result.rms_lfp)
        mean_ap = np.mean(result.rms_ap)
        assert mean_lfp > mean_ap

    def test_separate_lfp_recording_1_0(self):
        """For 1.0 probes, LFP RMS should come from the LFP recording."""
        ap_rec, _ = _make_synthetic_recording(
            duration_sec=10.0,
            fs=30000.0,
            n_channels=32,
            freqs_hz=[500.0, 1000.0],
        )
        lfp_rec, _ = _make_synthetic_recording(
            duration_sec=10.0,
            fs=2500.0,
            n_channels=32,
            freqs_hz=[5.0, 50.0],
        )

        block = ExperimentBlock(
            recording=ap_rec, lfp_recording=lfp_rec, block_index=0
        )
        result = _compute_all_metrics(
            block,
            window_interval=2.0,
            window_duration=2.0,
        )

        assert np.mean(result.rms_ap) > 0  # values in volts after µV→V
        assert np.mean(result.rms_lfp) > 0

        block_no_lfp = ExperimentBlock(
            recording=ap_rec, lfp_recording=None, block_index=0
        )
        result_no_lfp = _compute_all_metrics(
            block_no_lfp,
            window_interval=2.0,
            window_duration=2.0,
        )
        assert np.mean(result.rms_lfp) > np.mean(result_no_lfp.rms_lfp) * 10


class TestCoherenceAndPsd:
    """Tests for coherence and PSD computation."""

    def test_output_structure(self):
        """Test that result contains expected attributes."""
        recording, _ = _make_synthetic_recording(duration_sec=10.0)
        block = ExperimentBlock(
            recording=recording, lfp_recording=None, block_index=0
        )
        result = _compute_all_metrics(
            block,
            window_interval=2.0,
            window_duration=2.0,
        )

        assert result.correlation is not None
        assert result.coherency is not None
        assert result.psd_power is not None
        assert result.psd_freqs is not None

        assert result.psd_power.ndim == 2
        assert result.psd_power.shape[1] == 32
        assert result.psd_freqs.ndim == 1

    def test_correlation_shape_and_range(self):
        """Re(coherency) should be in [-1, 1] with diagonal ~1."""
        recording, _ = _make_synthetic_recording(duration_sec=10.0)
        block = ExperimentBlock(
            recording=recording, lfp_recording=None, block_index=0
        )
        result = _compute_all_metrics(
            block,
            window_interval=2.0,
            window_duration=2.0,
        )

        for band_name in COHERENCE_BANDS:
            assert band_name in result.correlation
            corr = result.correlation[band_name]
            assert corr.shape == (32, 32)
            np.testing.assert_allclose(np.diag(corr), 1.0, atol=0.05)
            assert np.all(corr >= -1.0 - 1e-6)
            assert np.all(corr <= 1.0 + 1e-6)

    def test_multi_shank_correlation_is_full_matrix(self):
        """Pairwise metrics are full collection matrices, not per-shank."""
        groups = np.repeat([0, 1], 4)
        locations = np.zeros((8, 2))
        locations[:, 0] = groups * 250.0
        locations[:, 1] = np.tile(np.arange(4) * 20.0, 2)
        recording, _ = _make_synthetic_recording(
            duration_sec=6.0,
            fs=2500.0,
            n_channels=8,
            groups=groups,
            locations=locations,
        )
        block = ExperimentBlock(
            recording=recording, lfp_recording=None, block_index=0
        )

        result = _compute_all_metrics(
            block,
            window_interval=2.0,
            window_duration=2.0,
        )

        for band_name in COHERENCE_BANDS:
            assert result.correlation[band_name].shape == (8, 8)
            assert result.coherency[band_name].shape == (8, 8)
        assert [s.shank_index for s in result.shank_channels] == [1, 2]

    def test_coherency_is_complex(self):
        """Complex coherency should have magnitude <= 1."""
        recording, _ = _make_synthetic_recording(duration_sec=10.0)
        block = ExperimentBlock(
            recording=recording, lfp_recording=None, block_index=0
        )
        result = _compute_all_metrics(
            block,
            window_interval=2.0,
            window_duration=2.0,
        )

        for coh in result.coherency.values():
            assert np.iscomplexobj(coh)
            assert np.all(np.abs(coh) <= 1.0 + 1e-6)


class TestProcessStreamFft:
    """Tests for process_stream_fft convenience wrapper."""

    def test_writes_expected_files(self):
        """Test that all expected output files are created."""
        recording, _ = _make_synthetic_recording(duration_sec=10.0)
        block = ExperimentBlock(
            recording=recording, lfp_recording=None, block_index=0
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            process_stream_fft(
                block,
                output,
                compute_coherence=True,
                rms_window_interval=2.0,
                rms_window_duration=2.0,
                tag="",
            )

            # Check RMS files
            assert (output / "_iblqc_ephysTimeRmsAP.rms.npy").exists()
            assert (output / "_iblqc_ephysTimeRmsAP.timestamps.npy").exists()
            assert (output / "_iblqc_ephysTimeRmsLF.rms.npy").exists()
            assert (output / "_iblqc_ephysTimeRmsLF.timestamps.npy").exists()

            # Check PSD files
            assert (
                output / "_iblqc_ephysSpectralDensityLF.power.npy"
            ).exists()
            assert (
                output / "_iblqc_ephysSpectralDensityLF.freqs.npy"
            ).exists()

            # Check coherence files
            band_corr = output / "band_corr"
            assert band_corr.is_dir()
            for band in COHERENCE_BANDS:
                assert (band_corr / f"{band}_mean_corr.npy").exists()
                assert (band_corr / f"{band}_coherency.npy").exists()
                assert (band_corr / f"{band}_shank1_mean_corr.npy").exists()
                assert (band_corr / f"{band}_shank1_coherency.npy").exists()

            row_channels_path = band_corr / "row_channels.json"
            assert row_channels_path.exists()
            with open(row_channels_path) as f:
                row_channels = json.load(f)
            assert row_channels["version"] == 2
            assert row_channels["matrix_row_order"] == "channel_table"
            assert row_channels["matrix_rows"] == list(range(32))
            assert row_channels["shanks"]["0"]["rows"] == list(range(32))

            # Check channel metadata
            assert (output / "channels.localCoordinates.npy").exists()
            assert (output / "channels.rawInd.npy").exists()
            assert (output / "channels.contactId.npy").exists()
            assert (output / "channels.shankInd.npy").exists()

    def test_writes_multi_shank_geometry_contract(self):
        """Saved outputs include full matrices and explicit shank row maps."""
        groups = np.repeat([0, 1], 4)
        locations = np.zeros((8, 2))
        locations[:, 0] = groups * 250.0
        locations[:, 1] = np.tile(np.arange(4) * 20.0, 2)
        recording, _ = _make_synthetic_recording(
            duration_sec=6.0,
            fs=2500.0,
            n_channels=8,
            groups=groups,
            locations=locations,
        )
        block = ExperimentBlock(
            recording=recording, lfp_recording=None, block_index=0
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            process_stream_fft(
                block,
                output,
                compute_coherence=True,
                rms_window_interval=2.0,
                rms_window_duration=2.0,
            )

            band_corr = output / "band_corr"
            full_corr = np.load(band_corr / "theta_mean_corr.npy")
            shank1_corr = np.load(band_corr / "theta_shank1_mean_corr.npy")
            shank2_corr = np.load(band_corr / "theta_shank2_mean_corr.npy")
            assert full_corr.shape == (8, 8)
            assert shank1_corr.shape == (4, 4)
            assert shank2_corr.shape == (4, 4)

            np.testing.assert_array_equal(
                np.load(output / "channels.shankInd.npy"), groups
            )
            assert np.load(output / "channels.contactId.npy").shape == (8,)
            with open(band_corr / "row_channels.json") as f:
                row_channels = json.load(f)
            assert row_channels["shanks"]["0"]["rows"] == [0, 1, 2, 3]
            assert row_channels["shanks"]["1"]["rows"] == [4, 5, 6, 7]

    def test_channel_table_uses_contact_ids_for_cross_block_stitching(self):
        """Stable contact IDs stitch blocks before rounded coordinates."""
        groups = np.array([0, 0])
        rec1 = _FakeRecording(
            locations=np.array([[0.0, 0.0], [0.0, 20.0]]),
            groups=groups,
            contact_ids=np.array(["s0e0", "s0e1"]),
        )
        rec2 = _FakeRecording(
            locations=np.array([[0.0, 0.1], [0.0, 20.1]]),
            groups=groups,
            contact_ids=np.array(["s0e0", "s0e1"]),
        )

        table, block_maps = build_channel_table([rec1, rec2])

        assert table.local_coordinates.shape[0] == 2
        np.testing.assert_array_equal(
            table.contact_id, np.array(["s0e0", "s0e1"])
        )
        np.testing.assert_array_equal(block_maps[0], np.array([0, 1]))
        np.testing.assert_array_equal(block_maps[1], np.array([0, 1]))

    def test_channel_table_projects_probe_contact_ids_by_device_channel(self):
        """Probe contact vectors are projected through device channel order."""
        probe = _FakeProbe(
            contact_ids=np.array(["e0", "e1"]),
            device_channel_indices=np.array([0, 1]),
        )
        rec = _FakeRecording(
            locations=np.array([[0.0, 20.0], [0.0, 0.0]]),
            groups=np.array([0, 0]),
            channel_ids=np.array(["1", "0"]),
            probe=probe,
        )

        table, block_maps = build_channel_table([rec])

        np.testing.assert_array_equal(table.contact_id, np.array(["e1", "e0"]))
        np.testing.assert_array_equal(block_maps[0], np.array([0, 1]))

    def test_channel_table_falls_back_to_geometry_for_invalid_contact_ids(
        self,
    ):
        """Partial contact ID vectors are not used as stable stitch keys."""
        groups = np.array([0, 0])
        probe = _FakeProbe(
            contact_ids=np.array(["e0", None], dtype=object),
            device_channel_indices=np.array([0, 1]),
        )
        rec1 = _FakeRecording(
            locations=np.array([[0.0, 0.0], [0.0, 20.0]]),
            groups=groups,
            channel_ids=np.array(["0", "1"]),
            probe=probe,
        )
        rec2 = _FakeRecording(
            locations=np.array([[0.0, 0.1], [0.0, 20.1]]),
            groups=groups,
            channel_ids=np.array(["0", "1"]),
            probe=probe,
        )

        table, block_maps = build_channel_table([rec1, rec2])

        assert table.local_coordinates.shape[0] == 4
        np.testing.assert_array_equal(
            table.contact_id, np.array(["", "", "", ""])
        )
        np.testing.assert_array_equal(block_maps[0], np.array([0, 1]))
        np.testing.assert_array_equal(block_maps[1], np.array([2, 3]))

    def test_normalized_channel_groups_prefers_contact_id_shank(self):
        """Flat group + shank-prefixed contact ids -> physical shank wins.

        Surface-finding blocks often report a flat ``group`` (all 0) even
        though they span every shank; the physical shank must come from the
        contact id prefix.
        """
        rec = _FakeRecording(
            locations=np.array(
                [[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]]
            ),
            groups=np.array([0, 0, 0, 0]),
            contact_ids=np.array(["s0e0", "s0e1", "s1e0", "s1e1"]),
        )
        np.testing.assert_array_equal(
            normalized_channel_groups(rec), np.array([0, 0, 1, 1])
        )

    def test_channel_table_surface_finding_flat_group_uses_contact_shank(self):
        """Surface block with flat group must not collapse shanks.

        Reproduces the 4-shank surface-finding defect where the surface block's
        flat ``group`` (a) labeled every shank's channels as shank 0 and (b)
        failed to dedup the same physical contact against the main block,
        leaving shanks 2-4 with only a redundant bottom-bank slice.
        """
        contact_ids = np.array(["s0e0", "s0e1", "s1e0", "s1e1"])
        locations = np.array(
            [[0.0, 0.0], [0.0, 20.0], [250.0, 0.0], [250.0, 20.0]]
        )
        surface = _FakeRecording(  # flat group across all shanks
            locations=locations,
            groups=np.array([0, 0, 0, 0]),
            contact_ids=contact_ids,
        )
        main = _FakeRecording(  # correct per-shank groups
            locations=locations,
            groups=np.array([0, 0, 1, 1]),
            contact_ids=contact_ids,
        )

        table, block_maps = build_channel_table([surface, main])

        # Same physical contacts across blocks dedup to 4 rows (not 6).
        assert table.local_coordinates.shape[0] == 4
        np.testing.assert_array_equal(table.contact_id, contact_ids)
        # shank_ind follows the contact-id physical shank, not the flat group.
        np.testing.assert_array_equal(table.shank_ind, np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(block_maps[0], np.array([0, 1, 2, 3]))
        np.testing.assert_array_equal(block_maps[1], np.array([0, 1, 2, 3]))

    def test_main_tag_no_coherence(self):
        """Test tagged output without coherence."""
        recording, _ = _make_synthetic_recording(duration_sec=10.0)
        block = ExperimentBlock(
            recording=recording, lfp_recording=None, block_index=0
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir)
            process_stream_fft(
                block,
                output,
                compute_coherence=False,
                rms_window_interval=2.0,
                rms_window_duration=2.0,
                tag="Main",
            )

            assert (output / "_iblqc_ephysTimeRmsAPMain.rms.npy").exists()
            assert (output / "_iblqc_ephysTimeRmsLFMain.rms.npy").exists()

            # No PSD or coherence
            assert not (
                output / "_iblqc_ephysSpectralDensityLFMain.power.npy"
            ).exists()
            assert not (output / "band_corr").exists()

    def test_all_metrics_unified(self):
        """_compute_all_metrics returns BlockMetrics with all fields."""
        recording, _ = _make_synthetic_recording(duration_sec=10.0)
        block = ExperimentBlock(
            recording=recording, lfp_recording=None, block_index=0
        )
        result = _compute_all_metrics(
            block,
            window_interval=2.0,
            window_duration=2.0,
        )

        # All attributes present and non-None
        assert result.rms_ap is not None
        assert result.rms_lfp is not None
        assert result.timestamps is not None
        assert result.correlation is not None
        assert result.coherency is not None
        assert result.psd_power is not None
        assert result.psd_freqs is not None

        # Shapes consistent
        n_win = len(result.timestamps)
        assert result.rms_ap.shape[0] == n_win
        assert result.rms_lfp.shape[0] == n_win

        # Correlation in [-1, 1]
        for corr in result.correlation.values():
            assert np.all(corr >= -1.0 - 1e-6)
            assert np.all(corr <= 1.0 + 1e-6)

        # Coherency is complex
        for coh in result.coherency.values():
            assert np.iscomplexobj(coh)
