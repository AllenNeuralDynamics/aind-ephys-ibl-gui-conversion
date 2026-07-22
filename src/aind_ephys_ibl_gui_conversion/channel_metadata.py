"""Helpers for producer-owned ephys channel geometry metadata."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import spikeinterface as si

from aind_ephys_ibl_gui_conversion.types import ChannelTable

# Multi-shank probeinterface contact ids are ``s{shank}e{elec}`` (e.g.
# ``s2e17``). The shank prefix is the physical shank and is stable across
# recording blocks, unlike the recording ``group`` property.
_SHANK_PREFIX_RE = re.compile(r"^s(\d+)e", re.IGNORECASE)


def _shank_from_contact_id(contact_id: object) -> int | None:
    """Physical 0-based shank encoded in a probeinterface contact id.

    Returns the shank integer parsed from an ``s{shank}e{elec}`` id, or
    ``None`` when the id is missing or carries no shank prefix.
    """
    if contact_id is None:
        return None
    match = _SHANK_PREFIX_RE.match(str(contact_id))
    return int(match.group(1)) if match else None


def _shanks_from_contact_ids(contact_ids: np.ndarray) -> np.ndarray | None:
    """Per-channel physical shank from contact ids, or ``None`` if unusable.

    Requires *every* channel to carry a shank-prefixed contact id; a single
    missing/unprefixed id means contact ids can't be trusted for shank
    identity and the caller falls back to the recording ``group``.
    """
    shanks = [_shank_from_contact_id(cid) for cid in contact_ids]
    if any(shank is None for shank in shanks):
        return None
    return np.asarray(shanks, dtype=np.int64)


def normalized_channel_groups(recording: si.BaseRecording) -> np.ndarray:
    """Return the 0-based physical shank index for a recording's channels.

    Prefers the shank encoded in probeinterface contact ids
    (``s{shank}e{elec}``): it is the true physical shank and is consistent
    across recording blocks. In particular, surface-finding blocks frequently
    report a *flat* ``group`` property (all channels group 0) even though they
    span every shank; deriving the shank from ``group`` there collapses all
    shanks to 0 and, on merge, both mislabels shanks and fails to deduplicate
    the same physical contact recorded in multiple blocks.

    Falls back to the SpikeInterface ``group`` property (normalized to 0-based
    contiguous) when contact ids are unavailable/unprefixed, and finally to a
    single shank.
    """
    contact_shanks = _shanks_from_contact_ids(
        normalized_contact_ids(recording)
    )
    if contact_shanks is not None:
        return contact_shanks

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

    Channels are deduplicated by normalized shank plus contact id when
    available, falling back to local coordinate. ``raw_ind`` intentionally
    remains the legacy positional ``0..N-1`` placeholder in this phase; the
    table row position is the join key.
    """
    if not recordings:
        raise ValueError("At least one recording is required.")

    row_by_key: dict[tuple[object, ...], int] = {}
    contact_ids: list[str] = []
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
        ids = normalized_contact_ids(recording)
        block_map = np.empty(recording.get_num_channels(), dtype=np.int64)

        for channel_index, (location, group, contact_id) in enumerate(
            zip(locations[:, :2], groups, ids, strict=True)
        ):
            key = _channel_key(location, int(group), contact_id)
            row = row_by_key.get(key)
            if row is None:
                row = len(local_coordinates)
                row_by_key[key] = row
                contact_ids.append(contact_id or "")
                local_coordinates.append(location.copy())
                shank_ind.append(int(group))
            block_map[channel_index] = row

        block_maps.append(block_map)

    n_channels = len(local_coordinates)
    table = ChannelTable(
        raw_ind=np.arange(n_channels, dtype=np.int64),
        contact_id=np.asarray(contact_ids, dtype=str),
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


def normalized_contact_ids(recording: si.BaseRecording) -> np.ndarray:
    """Return probe contact ids in recording channel order.

    Contact ids identify physical electrode sites. They may be exposed as a
    per-channel property, or only on the attached probe. If they are
    unavailable, malformed, or cannot be confidently projected into recording
    channel order, return ``None`` sentinels so callers fall back to geometry.
    """
    prop_ids = _contact_ids_property(recording)
    if prop_ids is not None:
        return prop_ids

    try:
        probe = recording.get_probe()
    except Exception:
        return np.full(recording.get_num_channels(), None, dtype=object)

    contact_ids = np.asarray(getattr(probe, "contact_ids", None))
    device_indices = getattr(probe, "device_channel_indices", None)
    if device_indices is not None:
        projected = _project_contact_ids_by_device_channel(
            contact_ids,
            np.asarray(device_indices),
            recording,
        )
        if projected is not None:
            return projected

    direct = _valid_contact_ids(contact_ids, recording.get_num_channels())
    if direct is not None:
        return direct
    return np.full(recording.get_num_channels(), None, dtype=object)


def _contact_ids_property(recording: si.BaseRecording) -> np.ndarray | None:
    """Return per-channel contact ids from the recording, else None."""
    try:
        ids = np.asarray(recording.get_property("contact_ids"))
    except Exception:
        return None
    return _valid_contact_ids(ids, recording.get_num_channels())


def _valid_contact_ids(ids: np.ndarray, n_channels: int) -> np.ndarray | None:
    """Validate a 1-D contact-id vector; return string ids or ``None``.

    Rejects wrong-length, missing, NaN/empty, or non-unique ids so callers
    only stitch blocks on trustworthy per-channel identifiers.
    """
    if ids is None or ids.ndim != 1 or ids.shape[0] != n_channels:
        return None
    values: list[str] = []
    for contact_id in ids:
        if contact_id is None:
            return None
        try:
            if bool(np.isnan(contact_id)):
                return None
        except TypeError:
            pass
        value = str(contact_id)
        normalized = value.strip().lower()
        if normalized in {"", "nan", "none"}:
            return None
        values.append(value)
    if len(set(values)) != n_channels:
        return None
    return np.asarray(values, dtype=object)


def _project_contact_ids_by_device_channel(
    contact_ids: np.ndarray,
    device_channel_indices: np.ndarray,
    recording: si.BaseRecording,
) -> np.ndarray | None:
    """Reorder probe contact ids into recording channel order, or ``None``.

    Maps each probe contact to its recording row via the device channel index;
    returns ``None`` if any recording row is left unresolved.
    """
    n_channels = recording.get_num_channels()
    if contact_ids.ndim != 1 or device_channel_indices.ndim != 1:
        return None
    if contact_ids.shape[0] != device_channel_indices.shape[0]:
        return None
    try:
        channel_ids = [
            str(channel_id) for channel_id in recording.get_channel_ids()
        ]
    except Exception:
        return None
    row_by_device = {
        channel_id: row for row, channel_id in enumerate(channel_ids)
    }
    projected: list[object | None] = [None] * n_channels
    for contact_id, device_index in zip(
        contact_ids, device_channel_indices, strict=True
    ):
        try:
            device_key = str(int(device_index))
        except Exception:
            continue
        if int(device_index) < 0:
            continue
        row = row_by_device.get(device_key)
        if row is not None:
            projected[row] = contact_id
    if any(contact_id is None for contact_id in projected):
        return None
    return _valid_contact_ids(np.asarray(projected, dtype=object), n_channels)


def _channel_key(
    location: np.ndarray,
    shank_index: int,
    contact_id: str | None,
) -> tuple[object, ...]:
    """Stable key for matching the same channel across blocks."""
    if contact_id:
        return ("contact", shank_index, contact_id)
    rounded = np.round(np.asarray(location, dtype=float), decimals=6)
    return ("loc", shank_index, *rounded.tolist())
