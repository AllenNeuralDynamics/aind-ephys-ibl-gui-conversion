"""Tests for spikes module."""

import tempfile
import unittest
from pathlib import Path

from aind_ephys_ibl_gui_conversion.spikes import _find_analyzer_folder


class TestFindAnalyzerFolder(unittest.TestCase):
    """Tests for _find_analyzer_folder folder discovery."""

    def setUp(self):
        """Create a temporary postprocessed directory."""
        self._tmp = tempfile.TemporaryDirectory()
        self.postprocessed = Path(self._tmp.name)
        self.stream = "Record Node 104#Neuropix-PXI-100.ProbeA-AP"

    def tearDown(self):
        """Remove the temporary directory."""
        self._tmp.cleanup()

    def _make(self, name):
        """Create an empty folder in the postprocessed directory."""
        folder = self.postprocessed / name
        folder.mkdir()
        return folder

    def test_finds_recording_numbered_zarr(self):
        """A recording-numbered .zarr folder is found (legacy layout)."""
        expected = self._make(f"experiment1_{self.stream}_recording1.zarr")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertEqual(result, expected)

    def test_finds_folder_without_recording_number(self):
        """A folder omitting the recording number is found."""
        expected = self._make(f"experiment1_{self.stream}.zarr")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertEqual(result, expected)

    def test_finds_non_zarr_recording_folder(self):
        """A non-.zarr (waveforms) folder is found when no .zarr exists."""
        expected = self._make(f"experiment1_{self.stream}_recording1")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertEqual(result, expected)

    def test_arbitrary_experiment_and_recording_index(self):
        """Experiment and recording indices other than 1 are matched."""
        expected = self._make(f"experiment3_{self.stream}_recording7.zarr")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertEqual(result, expected)

    def test_zarr_preferred_over_non_zarr(self):
        """When both exist, the .zarr folder is returned first."""
        self._make(f"experiment1_{self.stream}_recording1")
        expected = self._make(f"experiment1_{self.stream}_recording1.zarr")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertEqual(result, expected)

    def test_returns_none_when_missing(self):
        """None is returned when no matching folder exists."""
        self._make(f"experiment1_{self.stream}_recording1_group0.zarr")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertIsNone(result)

    def test_single_shank_ignores_group_folders(self):
        """A shankless lookup must not match a multi-shank group folder."""
        self._make(f"experiment1_{self.stream}_recording1_group0.zarr")
        expected = self._make(f"experiment1_{self.stream}_recording1.zarr")

        result = _find_analyzer_folder(self.postprocessed, self.stream)

        self.assertEqual(result, expected)

    def test_finds_shank_group_folder(self):
        """A group folder is found when shank_index is given."""
        expected = self._make(f"experiment1_{self.stream}_recording1_group1")

        result = _find_analyzer_folder(
            self.postprocessed, self.stream, shank_index=1
        )

        self.assertEqual(result, expected)

    def test_finds_shank_group_without_recording_number(self):
        """A group folder without a recording number is found."""
        expected = self._make(f"experiment1_{self.stream}_group2.zarr")

        result = _find_analyzer_folder(
            self.postprocessed, self.stream, shank_index=2
        )

        self.assertEqual(result, expected)

    def test_shank_lookup_matches_only_requested_group(self):
        """The requested shank index must be selected, not another."""
        self._make(f"experiment1_{self.stream}_recording1_group0.zarr")
        expected = self._make(
            f"experiment1_{self.stream}_recording1_group1.zarr"
        )

        result = _find_analyzer_folder(
            self.postprocessed, self.stream, shank_index=1
        )

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
