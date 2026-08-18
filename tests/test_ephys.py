"""Tests for ephys module."""

import unittest
from collections import defaultdict
from pathlib import Path

import spikeinterface as si
from spikeinterface.extractors import toy_example

from aind_ephys_ibl_gui_conversion.ephys import _results_in_block_order
from aind_ephys_ibl_gui_conversion.recording_utils import (
    _merge_separate_asset_recording_dicts,
    _stream_matches,
    _stream_to_probe_name,
    get_largest_segment_recordings,
    get_main_recording_from_list,
    merge_probe_streams,
)
from aind_ephys_ibl_gui_conversion.types import (
    BlockMetrics,
    ExperimentBlock,
    ProbeStream,
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


class TestStreamMatchesFunction(unittest.TestCase):
    """Test cases for the _stream_matches filter predicate."""

    def test_none_filter_selects_all(self):
        """A None filter selects every stream."""
        stream_name = "Record Node 104#Neuropix-PXI-100.ProbeA-AP"
        self.assertTrue(_stream_matches(stream_name, None))

    def test_exact_stream_name_matches(self):
        """The full Open Ephys stream name selects itself."""
        stream_name = "Record Node 104#Neuropix-PXI-100.ProbeA-AP"
        self.assertTrue(_stream_matches(stream_name, stream_name))

    def test_probe_token_matches_ap_stream(self):
        """The probe/collection token selects the AP stream."""
        stream_name = "Record Node 104#Neuropix-PXI-100.ProbeA-AP"
        self.assertTrue(_stream_matches(stream_name, "ProbeA"))

    def test_probe_token_matches_paired_lfp_stream(self):
        """The token drops the suffix, so it also selects the LFP stream."""
        stream_name = "Record Node 104#Neuropix-PXI-100.ProbeA-LFP"
        self.assertTrue(_stream_matches(stream_name, "ProbeA"))

    def test_wrong_token_does_not_match(self):
        """A token for a different probe is not selected."""
        stream_name = "Record Node 104#Neuropix-PXI-100.ProbeA-AP"
        self.assertFalse(_stream_matches(stream_name, "ProbeB"))

    def test_numeric_probe_token_matches(self):
        """A numeric probe/collection token selects its stream."""
        stream_name = "Record Node 109#Neuropix-PXI-100.45883-1"
        self.assertTrue(_stream_matches(stream_name, "45883-1"))


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


def _block_metrics(block):
    """Build a BlockMetrics carrying only the block identity."""
    return BlockMetrics(
        block=block,
        rms_ap=None,
        rms_lfp=None,
        timestamps=None,
        correlation=None,
        coherency=None,
        psd_power=None,
        psd_freqs=None,
        shank_channels=None,
    )


class TestSurfaceFindingBlockCounts(unittest.TestCase):
    """A surface recording has its own experiment count (TODO 15).

    754372's 2025-01-15 session records main as experiment1+experiment2 and
    surface as experiment1 only. The loader used to be handed the *main*
    count for both, so it asked the surface asset for an experiment2 zarr
    that does not exist and 5 of 12 units failed. A surface recording that
    happens to match the main count passed silently, so this was latent
    across the cohort rather than specific to one mouse.
    """

    @staticmethod
    def _stream(name, block_indices):
        """Build a ProbeStream with placeholder blocks at given indices."""
        return ProbeStream(
            stream_name=name,
            probe_name=name,
            blocks=[
                ExperimentBlock(
                    recording=None, lfp_recording=None, block_index=i
                )
                for i in block_indices
            ],
            output_folder=Path("/tmp/unused"),
        )

    def test_merge_accepts_unequal_block_counts(self):
        """Main with 2 experiments merges with surface having only 1."""
        main = self._stream("probeA", [0, 1])
        surface = self._stream("probeA", [0])

        merged = merge_probe_streams([main], [surface])

        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].blocks), 3)
        # the surface block's index collides with the first main block's
        self.assertEqual([b.block_index for b in merged[0].blocks], [0, 1, 0])

    def test_results_follow_block_order_not_block_index(self):
        """Colliding indices must not reorder results vs stream.blocks.

        The canonical channel table is built by first-seen order from both
        stream.blocks and these results; if the two disagree the saved
        channel rows stop matching the coherence matrix rows silently.
        """
        stream = self._stream("probeA", [0, 1, 0])
        main0, main1, surface0 = stream.blocks

        def _result(block):
            return _block_metrics(block)

        # as_completed returns in arbitrary order; sorting by block_index
        # would give [main0, surface0, main1].
        scrambled = [
            (stream, _result(surface0)),
            (stream, _result(main1)),
            (stream, _result(main0)),
        ]

        ordered = _results_in_block_order(stream, scrambled)

        self.assertEqual([r.block for r in ordered], [main0, main1, surface0])

    def test_results_ignore_other_streams(self):
        """Only the requested stream's results are collected."""
        a = self._stream("probeA", [0])
        b = self._stream("probeB", [0])

        def _result(block):
            return _block_metrics(block)

        pairs = [(b, _result(b.blocks[0])), (a, _result(a.blocks[0]))]

        ordered = _results_in_block_order(a, pairs)

        self.assertEqual([r.block for r in ordered], [a.blocks[0]])
