"""Inspect retained SpikeInterface source descriptions as plain JSON."""

from __future__ import annotations

from typing import Any

# These wrappers preserve segment membership and sample counts. Unknown or
# time-changing operations must not silently acquire that interpretation.
_PRESERVE_SEGMENTS = {
    "ChannelSliceRecording",
    "CommonReferenceRecording",
    "HighpassFilterRecording",
    "BandpassFilterRecording",
    "PhaseShiftRecording",
    "ScaleRecording",
    "WhitenRecording",
    "CenterRecording",
    "UnsignedToSignedRecording",
    "AstypeRecording",
    "InterpolateMotionRecording",
    "RemoveArtifactsRecording",
}


def parse_preprocessing_source(document: dict[str, Any]) -> dict[str, Any]:
    """Resolve one Zarr source and compose explicit segment selections."""
    node = document
    selection: list[int] | None = None
    for _ in range(100):
        if not isinstance(node, dict):
            return {"status": "invalid", "reason": "recording_not_object"}
        name = str(node.get("class", "")).rsplit(".", 1)[-1]
        kwargs = node.get("kwargs")
        if not isinstance(kwargs, dict):
            return {"status": "invalid", "reason": "missing_recording_kwargs"}
        if name == "ZarrRecordingExtractor":
            folder = kwargs.get("folder_path") or kwargs.get("zarr_path")
            if not isinstance(folder, str) or not folder:
                return {"status": "invalid", "reason": "missing_zarr_path"}
            return {
                "status": "observed",
                "folder_path": folder,
                "segment_indices": selection,
            }
        if name == "SelectSegmentRecording":
            indices = kwargs.get("segment_indices")
            if (
                not isinstance(indices, list)
                or not indices
                or any(type(i) is not int or i < 0 for i in indices)
            ):
                return {
                    "status": "invalid",
                    "reason": "invalid_segment_indices",
                }
            if selection is not None:
                if any(i >= len(indices) for i in selection):
                    return {
                        "status": "invalid",
                        "reason": "segment_out_of_range",
                    }
                selection = [indices[i] for i in selection]
            else:
                selection = indices
        elif name not in _PRESERVE_SEGMENTS:
            return {
                "status": "unsupported",
                "reason": "unsupported_wrapper",
                "class": name,
            }
        parents = [
            kwargs[key]
            for key in ("recording", "parent_recording")
            if key in kwargs
        ]
        if len(parents) != 1:
            return {
                "status": "invalid",
                "reason": "recording_parent_unresolved",
            }
        node = parents[0]
    return {"status": "invalid", "reason": "recording_chain_too_deep"}


def source_matches(
    source: dict[str, Any], recording: dict[str, Any], asset: dict[str, Any]
) -> bool | None:
    """Compare a producer path to an inventoried source and asset boundary."""
    if source.get("status") != "observed":
        return False if source.get("status") == "invalid" else None
    folder = source["folder_path"].replace("\\", "/").rstrip("/")
    names = {asset.get("name"), asset["key"], *(asset.get("aliases") or [])}
    # Require the asset boundary as well as the complete Zarr path. A matching
    # basename in a different acquisition is not source evidence.
    return any(
        folder.endswith("/" + name + "/" + recording["path"])
        or folder == name + "/" + recording["path"]
        for name in names
        if isinstance(name, str) and name
    )
