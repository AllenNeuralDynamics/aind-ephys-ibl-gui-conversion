"""Tests for ephys module."""

import unittest

from aind_ephys_ibl_gui_conversion.ephys import _stream_to_probe_name


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


if __name__ == "__main__":
    unittest.main()
