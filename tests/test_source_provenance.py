"""Provenance is inspected as JSON; serialized objects are never loaded."""

import pytest

from aind_ephys_ibl_gui_conversion.source_provenance import (
    parse_preprocessing_source,
    source_matches,
)


def _source():
    return {
        "class": "spikeinterface.core.zarrextractors.ZarrRecordingExtractor",
        "kwargs": {
            "folder_path": "../data/raw/ecephys/experiment1_ProbeA.zarr"
        },
    }


def _select(parent, indices):
    return {
        "class": "spikeinterface.core.segmentutils.SelectSegmentRecording",
        "kwargs": {"recording": parent, "segment_indices": indices},
    }


def test_explicit_segment_indices_compose_without_loading_recordings():
    result = parse_preprocessing_source(
        _select(_select(_source(), [2, 4]), [1])
    )
    assert result["segment_indices"] == [4]
    assert result["status"] == "observed"
    assert source_matches(
        result, {"path": "ecephys/experiment1_ProbeA.zarr"}, {"key": "raw"}
    )
    assert not source_matches(
        result,
        {"path": "ecephys/experiment1_ProbeA.zarr"},
        {"key": "other-raw"},
    )


def test_retained_channel_slice_parent_form_is_supported():
    source = _select(_source(), [0])
    for wrapper, key in [
        ("PhaseShiftRecording", "recording"),
        ("HighpassFilterRecording", "recording"),
        ("ChannelSliceRecording", "parent_recording"),
        ("CommonReferenceRecording", "recording"),
        ("ChannelSliceRecording", "parent_recording"),
    ]:
        source = {
            "class": "spikeinterface." + wrapper,
            "kwargs": {key: source},
        }
    result = parse_preprocessing_source(source)
    assert result["status"] == "observed"
    assert result["segment_indices"] == [0]


@pytest.mark.parametrize("indices", [[True], [-1], ["0"], [], None])
def test_invalid_segment_selection_cannot_fall_back(indices):
    result = parse_preprocessing_source(_select(_source(), indices))
    assert result["status"] == "invalid"


def test_cropping_and_unknown_wrappers_are_not_assumed_to_preserve_time():
    result = parse_preprocessing_source(
        {
            "class": "spikeinterface.core.FrameSliceRecording",
            "kwargs": {"recording": _source(), "start_frame": 500},
        }
    )
    assert result["status"] == "unsupported"


def test_nested_segment_index_must_exist():
    result = parse_preprocessing_source(_select(_select(_source(), [2]), [1]))
    assert result["status"] == "invalid"


def test_all_segments_are_explicit_when_no_segment_wrapper_exists():
    result = parse_preprocessing_source(_source())
    assert result["status"] == "observed"
    assert result["segment_indices"] is None
