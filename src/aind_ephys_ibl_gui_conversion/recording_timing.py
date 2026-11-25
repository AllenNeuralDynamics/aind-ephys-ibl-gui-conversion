"""Observe bounded timestamp samples and validate paired NP1 clocks."""

from __future__ import annotations

import math
import struct
from typing import Any

from numcodecs import get_codec

MAX_BOUNDARY_SKEW_S = 0.05
MAX_TIMESTAMP_CHUNK_BYTES = 8 * 1024 * 1024


class _MissingBoundedReader(NotImplementedError):
    """The legacy metadata transport cannot read timestamp scalars."""


def _value(
    store: Any, path: str, header: dict, index: int, *, allow_nan: bool = False
) -> float:
    """Read one v2 timestamp scalar, bounding Blosc chunk decoding."""
    if not callable(getattr(store, "read_bytes", None)):
        raise _MissingBoundedReader
    if (
        header.get("zarr_format") != 2
        or header.get("filters")
        or header.get("dtype") not in {"<f8", ">f8"}
        or len(header.get("shape", [])) != 1
        or len(header.get("chunks", [])) != 1
    ):
        raise NotImplementedError("Unsupported timestamp encoding")
    chunk = header["chunks"][0]
    if (
        type(chunk) is not int
        or chunk <= 0
        or not 0 <= index < header["shape"][0]
    ):
        raise ValueError("Invalid timestamp array bounds")
    compressor = header.get("compressor")
    if compressor is not None:
        if compressor.get("id") != "blosc" or not callable(
            getattr(store, "read_chunk", None)
        ):
            raise NotImplementedError("Unsupported timestamp compressor")
        size = 8 * chunk
        if size > MAX_TIMESTAMP_CHUNK_BYTES:
            raise NotImplementedError("Timestamp chunk exceeds decode limit")
        encoded = store.read_chunk(f"{path}/{index // chunk}")
        if len(encoded) > MAX_TIMESTAMP_CHUNK_BYTES:
            raise ValueError("Timestamp chunk exceeds read limit")
        decoded = bytearray(size)
        get_codec(compressor).decode(encoded, out=decoded)
        content = decoded[8 * (index % chunk) : 8 * (index % chunk) + 8]
    else:
        content = store.read_bytes(
            f"{path}/{index // chunk}", offset=8 * (index % chunk), size=8
        )
    if len(content) != 8:
        raise ValueError("Incomplete timestamp scalar")
    value = struct.unpack(header["dtype"][0] + "d", content)[0]
    if not math.isfinite(value) and not (allow_nan and math.isnan(value)):
        raise ValueError("Nonfinite timestamp")
    return value


def observe_timing(
    store: Any, path: str, segment: int, count: int, rate: float
) -> dict:
    """Sample five timestamp positions, or an explicit segment start time."""
    try:
        array = f"{path}/times_seg{segment}"
        try:
            header = store.read_json(array + "/.zarray")
        except FileNotFoundError:
            array = f"{path}/t_starts"
            try:
                header = store.read_json(array + "/.zarray")
            except FileNotFoundError:
                return {"status": "absent"}
            start = _value(store, array, header, segment, allow_nan=True)
            if math.isnan(start):
                return {"status": "absent"}
            return {
                "status": "observed",
                "representation": "t_start",
                "samples": [
                    [0, start],
                    [count - 1, start + (count - 1) / rate],
                ],
            }
        if header.get("shape") != [count] or count < 2:
            raise ValueError("Timestamp and trace sample counts disagree")
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
            "status": "observed",
            "representation": "explicit_vector",
            "samples": [[i, _value(store, array, header, i)] for i in indices],
        }
    except _MissingBoundedReader:
        return {
            "status": "unavailable",
            "reason": "bounded_reader_unavailable",
        }
    except NotImplementedError:
        return {
            "status": "unavailable",
            "reason": "unsupported_timestamp_encoding",
        }
    except Exception as exc:
        # Storage errors can contain signed URLs; keep the safe exception type.
        return {"status": "invalid", "reason": type(exc).__name__}


def _clock(segment: dict, rate: float) -> tuple[float, float]:
    """Check sampled timestamps against their own affine sample clock."""
    points = segment["timing"]["samples"]
    count = segment["sample_count"]
    first, last = points[0], points[-1]
    if first[0] != 0 or last[0] != count - 1 or count < 2:
        raise ValueError("Timestamp endpoints do not cover the segment")
    step = (last[1] - first[1]) / (count - 1)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Invalid timestamp clock")
    if any(
        not math.isfinite(t) or abs(t - (first[1] + i * step)) > 1 / rate
        for i, t in points
    ):
        raise ValueError("Sampled timestamps are not affine")
    return first[1], step * rate


def validate_pair(ap: dict, lf: dict, ap_rate: float, lf_rate: float) -> dict:
    """Bound edge loss and clock disagreement in the AP sample-zero clock."""
    statuses = [s.get("timing", {}).get("status", "absent") for s in (ap, lf)]
    if statuses != ["observed", "observed"]:
        if "invalid" in statuses or "observed" in statuses:
            raise ValueError(
                "AP/LFP timestamp evidence is incomplete or invalid"
            )
        if any(
            s.get("timing", {}).get("status") == "unavailable"
            and s["timing"].get("reason") != "bounded_reader_unavailable"
            for s in (ap, lf)
        ):
            raise ValueError("AP/LFP timestamp encoding cannot be validated")
        if not math.isclose(
            ap["sample_count"] / ap_rate,
            lf["sample_count"] / lf_rate,
            rel_tol=0,
            abs_tol=1 / min(ap_rate, lf_rate),
        ):
            raise ValueError(
                "AP/LFP duration mismatch without timestamp evidence"
            )
        return {"evidence": "legacy_duration_agreement", "lfp_offset_s": 0.0}
    ap_start, ap_scale = _clock(ap, ap_rate)
    lf_start, lf_scale = _clock(lf, lf_rate)
    ap_duration = ap["sample_count"] / ap_rate
    lf_duration = lf["sample_count"] / lf_rate
    offset = (lf_start - ap_start) / ap_scale
    relative_scale = lf_scale / ap_scale
    drift = abs(relative_scale - 1) * min(ap_duration, lf_duration)
    if drift > 1 / lf_rate + 1e-9:
        raise ValueError("AP/LFP clocks disagree by more than one LFP sample")
    end_delta = offset + lf_duration * relative_scale - ap_duration
    if max(abs(offset), abs(end_delta)) > MAX_BOUNDARY_SKEW_S:
        raise ValueError("AP/LFP boundary skew exceeds 50 ms")
    overlap_start = max(0.0, offset)
    # Scheduling uses nominal source rates; conservatively stay inside both
    # the timestamp-derived and nominal-rate LFP endpoints.
    overlap_end = min(
        ap_duration,
        offset + lf_duration,
        offset + lf_duration * relative_scale,
    )
    if overlap_end <= overlap_start:
        raise ValueError("AP/LFP have no common interval")
    return {
        "evidence": "sampled_timestamp_agreement",
        "lfp_offset_s": offset,
        "lfp_end_delta_s": end_delta,
        "relative_clock_drift_s": drift,
        "overlap_start_s": overlap_start,
        "overlap_end_s": overlap_end,
        "max_boundary_skew_s": MAX_BOUNDARY_SKEW_S,
        "clock_tolerance_s": 1 / lf_rate,
    }
