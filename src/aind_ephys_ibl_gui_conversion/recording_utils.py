"""Utility functions for recording management."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar

import spikeinterface as si
import spikeinterface.extractors as se

if TYPE_CHECKING:
    from aind_ephys_ibl_gui_conversion.types import ProbeStream

STREAM_PROBE_REGEX = re.compile(r"^Record Node \d+#[^.]+\.(.+?)(-AP|-LFP)?$")
T = TypeVar("T")


def _merge_separate_asset_recording_dicts(
    d1: defaultdict[str, list[T]],
    d2: defaultdict[str, list[T]],
) -> defaultdict[str, list[T]]:
    """Merge recordings from separate data assets.

    Keys are expected to overlap; values are concatenated.

    .. deprecated::
        Use :func:`merge_probe_streams` instead.  Retained only for
        backward compatibility; not used in production code paths.
    """
    merged = defaultdict(d1.default_factory)
    for k, v in d1.items():
        merged[k].extend(v)
    for k, v in d2.items():
        merged[k].extend(v)
    return merged


def _stream_to_probe_name(stream_name: str) -> str | None:
    """Extract probe name from Open Ephys stream name.

    Examples
    --------
    >>> _stream_to_probe_name(
    ...     "Record Node 104#Neuropix-PXI-100.ProbeA-AP"
    ... )
    'ProbeA'
    >>> _stream_to_probe_name(
    ...     "Record Node 109#Neuropix-PXI-100.45883-1"
    ... )
    '45883-1'
    """
    m = STREAM_PROBE_REGEX.match(stream_name)
    if m is not None:
        return m.group(1)
    return None


def get_ecephys_stream_names(
    base_folder,
) -> tuple[list[str], object, int]:
    """Discover Neuropixels stream names in an ecephys folder.

    Returns
    -------
    tuple[list[str], Path, int]
        Stream names, compressed folder path, number of blocks.
    """
    from pathlib import Path

    base_folder = Path(base_folder)
    ecephys_folder = base_folder / "ecephys_clipped"
    if ecephys_folder.is_dir():
        ecephys_compressed_folder = base_folder / "ecephys_compressed"
    else:
        ecephys_folder = base_folder / "ecephys" / "ecephys_clipped"
        ecephys_compressed_folder = (
            base_folder / "ecephys" / "ecephys_compressed"
        )
    print(f"ecephys folder: {ecephys_folder}")
    print(f"ecephys compressed folder: {ecephys_compressed_folder}")

    stream_names, stream_ids = se.get_neo_streams(
        "openephysbinary", ecephys_folder
    )
    num_blocks = se.get_neo_num_blocks("openephysbinary", ecephys_folder)

    neuropix_streams = [s for s in stream_names if "Neuropix" in s]

    return neuropix_streams, ecephys_compressed_folder, num_blocks


def get_largest_segment_recordings(
    recordings: list[si.BaseRecording],
) -> list[si.BaseRecording]:
    """Return recordings reduced to their largest segment only."""
    recordings_largest_segment = []

    for rec in recordings:
        segment_lengths = [
            rec.get_num_samples(seg) for seg in range(rec.get_num_segments())
        ]
        max_index = segment_lengths.index(max(segment_lengths))
        largest_seg_rec = rec.select_segments(max_index)
        recordings_largest_segment.append(largest_seg_rec)

    return recordings_largest_segment


def get_main_recording_from_list(
    recordings: list[si.BaseRecording],
) -> si.BaseRecording:
    """Return the recording with the largest number of samples."""
    if len(recordings) > 1:
        logging.warning(
            "Multiple main recordings of "
            f"length {len(recordings)} found. "
            "Defaulting to selecting recording with "
            "largest number of samples"
        )
    return max(recordings, key=lambda r: r.get_num_samples())


def merge_probe_streams(
    a: list[ProbeStream],
    b: list[ProbeStream],
) -> list[ProbeStream]:
    """Merge two lists of ProbeStreams by stream name.

    Streams with matching ``stream_name`` have their blocks
    combined. Streams that appear only in one list are kept
    unchanged.

    Parameters
    ----------
    a : list[ProbeStream]
        First set of probe streams.
    b : list[ProbeStream]
        Second set of probe streams (e.g. from surface-finding asset).

    Returns
    -------
    list[ProbeStream]
        Merged probe streams.
    """
    by_name: dict[str, ProbeStream] = {}
    for stream in a:
        by_name[stream.stream_name] = stream
    for stream in b:
        if stream.stream_name in by_name:
            by_name[stream.stream_name].blocks.extend(stream.blocks)
        else:
            by_name[stream.stream_name] = stream
    return list(by_name.values())
