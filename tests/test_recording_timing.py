"""Validate NP1 edge offsets using timestamp evidence."""

import struct

import pytest

from aind_ephys_ibl_gui_conversion.recording_timing import (
    observe_timing,
    validate_pair,
)


def _segment(count, rate, start=20.0, scale=1.0):
    indices = sorted(
        {
            0,
            (count - 1) // 4,
            (count - 1) // 2,
            3 * (count - 1) // 4,
            count - 1,
        }
    )
    return {
        "sample_count": count,
        "timing": {
            "status": "observed",
            "samples": [[i, start + i / rate * scale] for i in indices],
        },
    }


@pytest.mark.parametrize(
    "offset,lf_count",
    [(0.0096, 24976), (0.0096, 25000), (0, 25024), (-0.0096, 25000)],
)
def test_small_validated_edge_difference_is_accepted(offset, lf_count):
    timing = validate_pair(
        _segment(300000, 30000),
        _segment(lf_count, 2500, 20 + offset),
        30000,
        2500,
    )
    assert timing["lfp_offset_s"] == pytest.approx(offset)
    assert timing["overlap_start_s"] == pytest.approx(max(0, offset))
    assert timing["overlap_end_s"] <= 10.0


@pytest.mark.parametrize("offset", [0.051, -0.051, 5.0])
def test_larger_boundary_shift_is_rejected(offset):
    with pytest.raises(ValueError, match="boundary skew"):
        validate_pair(
            _segment(300000, 30000),
            _segment(25000, 2500, 20 + offset),
            30000,
            2500,
        )


def test_matching_durations_do_not_waive_clock_disagreement():
    with pytest.raises(ValueError, match="clocks disagree"):
        validate_pair(
            _segment(300000, 30000),
            _segment(25000, 2500, scale=1.0001),
            30000,
            2500,
        )


def test_interior_timestamp_jump_is_not_hidden_by_matching_endpoints():
    ap = _segment(300000, 30000)
    ap["timing"]["samples"][2][1] += 0.005
    with pytest.raises(ValueError, match="not affine"):
        validate_pair(ap, _segment(25000, 2500), 30000, 2500)


def test_missing_timing_keeps_the_strict_legacy_duration_rule():
    timing = validate_pair(
        {"sample_count": 300000}, {"sample_count": 25000}, 30000, 2500
    )
    assert timing["evidence"] == "legacy_duration_agreement"
    with pytest.raises(ValueError, match="without timestamp evidence"):
        validate_pair(
            {"sample_count": 300000}, {"sample_count": 25024}, 30000, 2500
        )


def test_one_timestamped_source_cannot_be_paired_with_an_unknown_clock():
    with pytest.raises(ValueError, match="incomplete or invalid"):
        validate_pair(
            _segment(300000, 30000), {"sample_count": 25000}, 30000, 2500
        )


def test_unsupported_encoding_cannot_use_legacy_duration_fallback():
    ap, lf = {"sample_count": 300000}, {"sample_count": 25000}
    for segment in (ap, lf):
        segment["timing"] = {
            "status": "unavailable",
            "reason": "unsupported_timestamp_encoding",
        }
    with pytest.raises(ValueError, match="encoding cannot be validated"):
        validate_pair(ap, lf, 30000, 2500)


def test_timestamp_observation_reads_only_five_eight_byte_ranges():
    class Store:
        def __init__(self):
            self.reads = []

        def read_json(self, path):
            assert path == "recording.zarr/times_seg0/.zarray"
            return {
                "zarr_format": 2,
                "compressor": None,
                "filters": None,
                "dtype": "<f8",
                "shape": [1000000000],
                "chunks": [1000],
            }

        def read_bytes(self, path, *, offset, size):
            assert path.startswith("recording.zarr/times_seg0/")
            assert size == 8
            self.reads.append((path, offset, size))
            index = int(path.rsplit("/", 1)[1]) * 1000 + offset // 8
            return struct.pack("<d", 20 + index / 30000)

    store = Store()
    observed = observe_timing(store, "recording.zarr", 0, 1000000000, 30000)
    assert observed["status"] == "observed"
    assert len(store.reads) == 5
    assert sum(read[2] for read in store.reads) == 40
    assert observed["samples"][-1][0] == 999999999


def test_absent_timestamp_arrays_are_transport_independent():
    class LegacyStore:
        def read_json(self, path):
            raise FileNotFoundError(path)

    assert observe_timing(LegacyStore(), "recording.zarr", 0, 100, 30000) == {
        "status": "absent"
    }
