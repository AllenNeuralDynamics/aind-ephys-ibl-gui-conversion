"""Tests for ephys module."""

import unittest
from collections import defaultdict

import spikeinterface as si
from spikeinterface.extractors import toy_example

from aind_ephys_ibl_gui_conversion.recording_utils import (
    _merge_separate_asset_recording_dicts,
    _stream_to_probe_name,
    get_largest_segment_recordings,
    get_main_recording_from_list,
)


class TestMergeMainAndSurfaceRecordingDicts(unittest.TestCase):
    """Tests for _merge_main_and_surface_recording_dicts."""

    def test_overlapping_keys_are_concatenated(self):
        """
        Overlapping keys must have their list values concatenated.

        This is the core behavior required when surface ephys recordings
        are stored as a separate data asset.
        """
        d1 = defaultdict(list, {"probeA": [1, 2]})
        d2 = defaultdict(list, {"probeA": [3]})

        merged = _merge_separate_asset_recording_dicts(d1, d2)

        self.assertEqual(merged["probeA"], [1, 2, 3])

    def test_non_overlapping_keys_are_preserved(self):
        """
        Non-overlapping keys should be carried through unchanged.

        Keys that only appear in one input dict must still be present
        in the merged result.
        """
        d1 = defaultdict(list, {"probeA": [1]})
        d2 = defaultdict(list, {"probeB": [2]})

        merged = _merge_separate_asset_recording_dicts(d1, d2)

        self.assertEqual(merged["probeA"], [1])
        self.assertEqual(merged["probeB"], [2])

    def test_differs_from_dict_union_behavior(self):
        """
        The union operator overwrites values for overlapping keys,
        whereas this function must concatenate them.
        """
        d1 = defaultdict(list, {"probeA": [1, 2]})
        d2 = defaultdict(list, {"probeA": [3]})

        merged = _merge_separate_asset_recording_dicts(d1, d2)
        union = d1 | d2

        self.assertEqual(merged["probeA"], [1, 2, 3])
        self.assertEqual(union["probeA"], [3])

    def test_default_factory_is_preserved(self):
        """
        The merged defaultdict should preserve the default_factory.

        This ensures missing keys still produce empty lists and that
        defaultdict semantics are not lost during the merge.
        """
        d1 = defaultdict(list)
        d2 = defaultdict(list)

        merged = _merge_separate_asset_recording_dicts(d1, d2)

        self.assertIs(merged.default_factory, list)


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
        suffix.
        """
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


class TestRecordingUtils(unittest.TestCase):
    """Tests for recording utility functions."""

    @classmethod
    def setUpClass(cls):
        """Set up small synthetic recordings for reuse."""
        rec, _ = toy_example(num_segments=1, num_channels=4, seed=0)
        cls.rec_ap = rec

    def test_get_largest_segment_recordings(self):
        """Tests extracting only the largest segment from each recording."""
        multi_seg, _ = toy_example(
            num_segments=2,
            duration=[5.0, 10.0],
            num_channels=4,
            seed=0,
        )
        largest_segments = get_largest_segment_recordings([multi_seg])

        self.assertIsInstance(largest_segments, list)
        self.assertTrue(
            all(isinstance(r, si.BaseRecording) for r in largest_segments)
        )

        self.assertEqual(largest_segments[0].get_num_segments(), 1)

        self.assertEqual(
            largest_segments[0].get_num_samples(),
            max(
                multi_seg.get_num_samples(0),
                multi_seg.get_num_samples(1),
            ),
        )

    def test_get_main_recording_from_list(self):
        """Tests getting main recording."""
        rec_short = self.rec_ap.frame_slice(
            0, self.rec_ap.get_num_samples() // 2
        )
        main = get_main_recording_from_list([rec_short, self.rec_ap])
        self.assertIs(main, self.rec_ap)


if __name__ == "__main__":
    unittest.main()
