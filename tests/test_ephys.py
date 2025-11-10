"""Tests for ephys module."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import spikeinterface as si
import spikeinterface.preprocessing as spre
from spikeinterface.extractors import toy_example

from aind_ephys_ibl_gui_conversion.ephys import (
    _stream_to_probe_name,
    get_concatenated_recordings,
    get_largest_segment_recordings,
    get_main_recording_from_list,
    get_neuropixel_lfp_stream,
    process_lfp_stream,
    process_raw_data,
)


class TestStreamToProbeNameFunction(unittest.TestCase):
    """Test cases for _stream_to_probe_name function."""

    def test_probe_name_with_ap_suffix(self):
        """Test extraction of probe name from stream with -AP suffix."""
        stream_name = "Record Node 104#Neuropix-PXI-100.ProbeA-AP"
        expected = "ProbeA"
        result = _stream_to_probe_name(stream_name)
        self.assertEqual(result, expected)

    def test_probe_name_without_suffix(self):
        """Test extraction of probe name from stream without -AP/-LFP
        suffix."""
        stream_name = "Record Node 109#Neuropix-PXI-100.45883-1"
        expected = "45883-1"
        result = _stream_to_probe_name(stream_name)
        self.assertEqual(result, expected)

    def test_probe_name_with_lfp_suffix(self):
        """Test extraction of probe name from stream with -LFP suffix."""
        stream_name = "Record Node 104#Neuropix-PXI-100.ProbeA-LFP"
        expected = "ProbeA"
        result = _stream_to_probe_name(stream_name)
        self.assertEqual(result, expected)

    def test_numeric_probe_name_with_ap_suffix(self):
        """Test extraction of numeric probe name with -AP suffix."""
        stream_name = "Record Node 109#Neuropix-PXI-100.12345-AP"
        expected = "12345"
        result = _stream_to_probe_name(stream_name)
        self.assertEqual(result, expected)

    def test_invalid_format_returns_none(self):
        """Test that invalid stream format returns None."""
        stream_name = "InvalidStreamFormat"
        result = _stream_to_probe_name(stream_name)
        self.assertIsNone(result)

    def test_alphanumeric_probe_name(self):
        """Test extraction of alphanumeric probe name."""
        stream_name = "Record Node 104#Neuropix-PXI-100.Probe1A-AP"
        expected = "Probe1A"
        result = _stream_to_probe_name(stream_name)
        self.assertEqual(result, expected)


class TestExtractContinuous(unittest.TestCase):
    """Tests extract_continuous orchestrator"""

    @classmethod
    def setUpClass(cls):
        """Set up small synthetic recordings for reuse."""
        rec, _ = toy_example(
            duration=[5.0, 10.0], num_channels=4, seed=0
        )
        rec_lfp = spre.decimate(
            spre.bandpass_filter(rec, 0.1, 300), decimation_factor=10
        )
        cls.rec_ap = rec
        cls.rec_lfp = rec_lfp
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="test_extract_continuous_"))

    @classmethod
    def tearDownClass(cls):
        """TearDown"""
        shutil.rmtree(cls.tmpdir)

    # --------------------------
    # get_neuropixel_lfp_stream
    # --------------------------

    @patch("aind_ephys_ibl_gui_conversion.ephys.si.read_zarr")
    def test_get_neuropixel_lfp_stream_1_0(self, mock_read_zarr):
        """Tests getting neuropixel LFP stream for 1.0"""
        mock_read_zarr.return_value = self.rec_lfp

        stream_name = "probeA.AP"
        lfp_stream, is_1_0 = get_neuropixel_lfp_stream(
            self.rec_ap, stream_name, self.tmpdir, block_index=0
        )

        self.assertTrue(is_1_0)
        self.assertIsInstance(lfp_stream, si.BaseRecording)
        mock_read_zarr.assert_called_once()

    def test_get_neuropixel_lfp_stream_2_0(self):
        """Tests getting neuropixel LFP stream for 2.0"""
        stream_name = "probeB"
        lfp_stream, is_1_0 = get_neuropixel_lfp_stream(
            self.rec_ap, stream_name, self.tmpdir, block_index=0
        )
        self.assertFalse(is_1_0)
        self.assertIs(lfp_stream, self.rec_ap)

    # --------------------------
    # process_lfp_stream
    # --------------------------

    def test_process_lfp_stream_bandpass_only(self):
        """Tests bandpass filter to LFP"""
        processed = process_lfp_stream(
            self.rec_ap,
            is_1_0_probe=True,
            freq_min=0.1,
            freq_max=300,
            decimation_factor=10,
        )
        self.assertIsInstance(processed, si.BaseRecording)
        self.assertAlmostEqual(
            processed.sampling_frequency, self.rec_ap.sampling_frequency
        )

    def test_process_lfp_stream_bandpass_and_decimate(self):
        """Tests bandpass and decimate"""
        processed = process_lfp_stream(
            self.rec_ap,
            is_1_0_probe=False,
            freq_min=0.1,
            freq_max=300,
            decimation_factor=4,
        )
        self.assertEqual(
            processed.sampling_frequency, self.rec_ap.sampling_frequency / 4
        )

    # --------------------------
    # get_concatenated_recordings
    # --------------------------

    @patch(
        "aind_ephys_ibl_gui_conversion.ephys.remove_overlapping_channels",
        side_effect=lambda x: x,
    )
    def test_get_concatenated_recordings(self, mock_remove):
        """Tests getting concatenated recordings"""
        combined = get_concatenated_recordings([self.rec_ap], [self.rec_lfp])
        self.assertIsInstance(combined, si.BaseRecording)
        self.assertEqual(
            combined.get_num_channels(),
            self.rec_ap.get_num_channels() + self.rec_lfp.get_num_channels(),
        )
        mock_remove.assert_called_once()

    # --------------------------
    # get_main_recording_from_list
    # --------------------------

    def test_get_main_recording_from_list(self):
        """Tests getting main recording"""
        rec_short = self.rec_ap.frame_slice(
            0, self.rec_ap.get_num_samples() // 2
        )
        main = get_main_recording_from_list([rec_short, self.rec_ap])
        self.assertIs(main, self.rec_ap)

    # --------------------------
    # process_raw_data
    # --------------------------

    @patch("aind_ephys_ibl_gui_conversion.ephys.save_rms_and_lfp_spectrum")
    def test_process_raw_data(self, mock_save):
        """Tests process raw data saves"""
        process_raw_data(
            main_recording=self.rec_ap,
            recording_combined=None,
            stream_name="probeA.ap",
            results_folder=self.tmpdir,
            is_lfp=False,
        )

        mock_save.assert_called()
        files = list(self.tmpdir.glob("**/*"))
        self.assertTrue(any(f.is_dir() for f in files))

    # --------------------------
    # get_largest_segment_recordings
    # --------------------------

    def test_get_largest_segment_recordings(self):
        """Tests extracting only the largest segment from each recording."""
        # Create recordings with multiple segments by slicing
        largest_segments = get_largest_segment_recordings(
            [self.rec_ap, self.rec_lfp]
        )

        # Check that the result is a list of recordings
        self.assertIsInstance(largest_segments, list)
        self.assertTrue(
            all(isinstance(r, si.BaseRecording) for r in largest_segments)
        )

        # Check that each recording now has only 1 segment
        self.assertEqual(largest_segments[0].get_num_segments(), 1)
        self.assertEqual(largest_segments[1].get_num_segments(), 1)

        # Check that the largest segment corresponds to the longer one
        self.assertEqual(
            largest_segments[0].get_num_samples(),
            max(
                self.rec_ap.get_num_samples(0), self.rec_ap.get_num_samples(1)
            ),
        )
        self.assertEqual(
            largest_segments[1].get_num_samples(),
            max(
                self.rec_lfp.get_num_samples(0),
                self.rec_lfp.get_num_samples(1),
            ),
        )


if __name__ == "__main__":
    unittest.main()
