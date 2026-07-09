"""Helpers for producer-owned ephys channel geometry metadata."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import spikeinterface as si

from aind_ephys_ibl_gui_conversion.types import ChannelTable


def normalized_channel_groups(recording: si.BaseRecording) -> np.ndarray:
    """Return 0-based contiguous shank groups for a recording's channels.

    SpikeInterface propagates probe shanks through the ``group`` property when
    probeinterface metadata is available. Missing groups are treated as a
    single-shank recording.
    """
    try:
        groups = recording.get_property("group")
    except Exception:
        groups = None

    if groups is None:
        return np.zeros(recording.get_num_channels(), dtype=np.int64)

    group_array = np.asarray(groups)
    if group_array.shape[0] != recording.get_num_channels():
        raise ValueError(
            "Channel group metadata length "
            f"({group_array.shape[0]}) does not match channel count "
            f"({recording.get_num_channels()})."
        )

    _, normalized = np.unique(group_array, return_inverse=True)
    return normalized.astype(np.int64, copy=False)


def build_channel_table(
    recordings: Sequence[si.BaseRecording],
) -> tuple[ChannelTable, list[np.ndarray]]:
    """Build a canonical channel table and per-block row maps.

    Channels are deduplicated by normalized shank and local coordinate,
    preserving first occurrence across recordings. ``raw_ind`` intentionally
    remains the legacy positional ``0..N-1`` placeholder in this phase; the
    table row position is the join key.
    """
    if not recordings:
        raise ValueError("At least one recording is required.")

    row_by_key: dict[tuple[object, ...], int] = {}
    local_coordinates: list[np.ndarray] = []
    shank_ind: list[int] = []
    block_maps: list[np.ndarray] = []

    for recording in recordings:
        locations = np.asarray(recording.get_channel_locations(), dtype=float)
        if (
            locations.ndim != 2
            or locations.shape[0] != recording.get_num_channels()
        ):
            raise ValueError(
                "Channel locations must have shape "
                f"({recording.get_num_channels()}, n_dims); got "
                f"{locations.shape}."
            )
        groups = normalized_channel_groups(recording)
        block_map = np.empty(recording.get_num_channels(), dtype=np.int64)

        for channel_index, (location, group) in enumerate(
            zip(locations[:, :2], groups, strict=True)
        ):
            key = _channel_key(location, int(group))
            row = row_by_key.get(key)
            if row is None:
                row = len(local_coordinates)
                row_by_key[key] = row
                local_coordinates.append(location.copy())
                shank_ind.append(int(group))
            block_map[channel_index] = row

        block_maps.append(block_map)

    n_channels = len(local_coordinates)
    table = ChannelTable(
        raw_ind=np.arange(n_channels, dtype=np.int64),
        local_coordinates=np.asarray(local_coordinates, dtype=np.float64),
        shank_ind=np.asarray(shank_ind, dtype=np.int64),
    )
    return table, block_maps


def depth_sorted_shank_rows(
    table: ChannelTable,
    shank_index: int,
) -> np.ndarray:
    """Return channel-table rows for a shank in matrix display order."""
    rows = np.where(table.shank_ind == shank_index)[0].astype(np.int64)
    depths = table.local_coordinates[rows, 1]
    order = np.argsort(depths, kind="stable")
    return rows[order]


def _channel_key(location: np.ndarray, shank_index: int) -> tuple[object, ...]:
    """Stable key for matching the same channel across blocks."""
    rounded = np.round(np.asarray(location, dtype=float), decimals=6)
    return (shank_index, *rounded.tolist())
