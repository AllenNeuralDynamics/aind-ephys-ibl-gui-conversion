"""Observe recording membership without choosing conversion inputs.

Only JSON metadata is read. Sorting analyzers are never instantiated and no
signal, spike, extension, or pickled payload is decoded. Storage names supply
candidate relationships, not authoritative acquisition identities.
"""

from __future__ import annotations

import hashlib
import json
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from aind_ephys_ibl_gui_conversion.probe_metadata import probe_metadata

INVENTORY_VERSION = "1.0.0"
INVENTORY_FILENAME = "ephys_inventory.json"
MAX_METADATA_BYTES = 8 * 1024 * 1024


class MetadataStore(Protocol):
    """Bounded metadata access relative to one asset root."""

    def directories(self, path: str) -> list[str]:
        """Return immediate child directory names in deterministic order."""
        ...

    def read_json(self, path: str) -> Any:
        """Read bounded JSON, raising FileNotFoundError for an absent key."""
        ...


def relative_path(path: str) -> str:
    """Reject references escaping an asset, including absolute paths."""
    value = PurePosixPath(path)
    if value.is_absolute() or ".." in value.parts or "\\" in path:
        raise ValueError("Metadata paths must stay within their asset")
    return value.as_posix()


class LocalMetadataStore:
    """Read metadata from an explicitly supplied local asset root."""

    def __init__(self, root: Path):
        """Bind the store to a deployment-specific root."""
        self.root = root.resolve()

    def _path(self, path: str) -> Path:
        """Keep references inside the root after resolving symlinks."""
        candidate = (self.root / relative_path(path)).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError("Metadata symlink escapes asset")
        return candidate

    def directories(self, path: str) -> list[str]:
        """List one level without walking signal chunks."""
        directory = self._path(path)
        if not directory.exists():
            return []
        return sorted(p.name for p in directory.iterdir() if p.is_dir())

    def read_json(self, path: str) -> Any:
        """Read at most the metadata size limit."""
        with self._path(path).open("rb") as handle:
            content = handle.read(MAX_METADATA_BYTES + 1)
        if len(content) > MAX_METADATA_BYTES:
            raise ValueError("Metadata exceeds size limit")
        return json.loads(content)


def _id(asset: str, path: str, kind: str) -> str:
    """Keep identity stable when assets move between deployment roots."""
    return hashlib.sha256(
        json.dumps([asset, path, kind], separators=(",", ":")).encode()
    ).hexdigest()[:24]


def _finding(
    inventory: dict[str, Any],
    code: str,
    asset: str,
    path: str,
    *,
    record: str | None = None,
    detail: str | None = None,
) -> None:
    """Retain diagnostic evidence without dropping its source record."""
    inventory["findings"].append(
        dict(code=code, asset=asset, path=path, record=record, detail=detail)
    )


def _read(
    store: MetadataStore,
    path: str,
    inventory: dict[str, Any],
    asset: str,
    *,
    optional: bool = False,
) -> Any:
    """Distinguish absent metadata from unreadable or unsupported metadata."""
    try:
        value = store.read_json(path)
        if not isinstance(value, dict):
            raise ValueError("Metadata must be a JSON object")
        return value
    except FileNotFoundError:
        if not optional:
            _finding(inventory, "missing_metadata", asset, path)
    except Exception as exc:
        # Remote errors may expose credentials. Persist only their type.
        _finding(
            inventory,
            "unreadable_metadata",
            asset,
            path,
            detail=type(exc).__name__,
        )
    return None


def _geometry(
    inventory: dict[str, Any],
    asset: str,
    path: str,
    probe_group: Any,
    annotations: Any = None,
) -> str:
    """Preserve producer channel ordering and physical identity verbatim."""
    config_id = _id(asset, path, "channels")
    probes = probe_metadata(probe_group, annotations)
    inventory["channel_configurations"].append(
        dict(id=config_id, asset=asset, metadata_path=path, probes=probes)
    )
    return config_id


def _headers(
    store: MetadataStore, base: str, inventory: dict[str, Any], asset: str
) -> None:
    """Observe every declared stream in each recording header."""
    for node in store.directories(base):
        for experiment in store.directories(f"{base}/{node}"):
            for recording in store.directories(f"{base}/{node}/{experiment}"):
                path = (
                    f"{base}/{node}/{experiment}/{recording}/structure.oebin"
                )
                header = _read(store, path, inventory, asset)
                if header is None:
                    continue
                seen: set[str] = set()
                for index, stream in enumerate(header.get("continuous", [])):
                    name = stream.get("folder_name", "").rstrip("/")
                    if name in seen:
                        _finding(
                            inventory,
                            "duplicate_stream_declaration",
                            asset,
                            path,
                        )
                    seen.add(name)
                    inventory["recorded_streams"].append(
                        dict(
                            id=_id(asset, path, f"{name}:{index}"),
                            asset=asset,
                            metadata_path=path,
                            record_node=node,
                            experiment=experiment,
                            recording=recording,
                            stream=f"{node}#{name}",
                            channel_count=stream.get("num_channels"),
                            sampling_frequency=stream.get("sample_rate"),
                        )
                    )


def _compressed(
    store: MetadataStore, base: str, inventory: dict[str, Any], asset: str
) -> None:
    """Describe stored segments without assigning acquisition roles."""
    for name in store.directories(base):
        try:
            _compressed_record(store, base, inventory, asset, name)
        except Exception as exc:
            _finding(
                inventory,
                "record_scan_incomplete",
                asset,
                f"{base}/{name}",
                detail=type(exc).__name__,
            )


def _compressed_record(
    store: MetadataStore,
    base: str,
    inventory: dict[str, Any],
    asset: str,
    name: str,
) -> None:
    """Observe one stored recording without losing later malformed siblings."""
    if not name.endswith(".zarr"):
        return
    path = f"{base}/{name}"
    attrs = _read(store, f"{path}/.zattrs", inventory, asset)
    if attrs is None:
        return
    match = re.fullmatch(r"(experiment[^_]+)_(.+)\.zarr", name)
    experiment, stream = match.groups() if match else (None, None)
    segment_count = attrs.get("num_segments")
    if not isinstance(segment_count, int) or not 1 <= segment_count <= 10000:
        _finding(inventory, "unknown_segment_structure", asset, path)
        segment_count = 0
    segments = []
    for index in range(segment_count):
        array_path = f"{path}/traces_seg{index}/.zarray"
        array = _read(store, array_path, inventory, asset)
        shape = array.get("shape", []) if array else []
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(not isinstance(n, int) or n < 0 for n in shape)
        ):
            _finding(inventory, "invalid_array_shape", asset, array_path)
            shape = []
        segments.append(
            dict(
                index=index,
                metadata_path=array_path,
                sample_count=shape[0] if shape else None,
                channel_count=shape[1] if len(shape) == 2 else None,
            )
        )
    record_id = _id(asset, path, "compressed")
    inventory["recordings"].append(
        dict(
            id=record_id,
            asset=asset,
            path=path,
            experiment=experiment,
            stream=stream,
            identity_evidence="storage_name" if match else "unknown",
            annotated_experiment=(attrs.get("annotations") or {}).get(
                "experiment_name"
            ),
            sampling_frequency=attrs.get("sampling_frequency"),
            segments=segments,
            declared_segment_count=attrs.get("num_segments"),
            channel_configuration=_geometry(
                inventory,
                asset,
                f"{path}/.zattrs",
                attrs.get("probe"),
                attrs.get("annotations"),
            ),
            acquisition_role=None,
        )
    )


def _raw(store: MetadataStore, inventory: dict[str, Any], asset: str) -> None:
    """Scan format containers without recursively walking trace chunks."""
    # Enumerate known format containers; both layouts may coexist. Do not pick
    # the first or walk recursively into signal storage.
    found = False
    for prefix in ("", "ecephys/"):
        children = store.directories(prefix or ".")
        if "ecephys_clipped" in children:
            found = True
            _headers(store, prefix + "ecephys_clipped", inventory, asset)
        if "ecephys_compressed" in children:
            found = True
            _compressed(store, prefix + "ecephys_compressed", inventory, asset)
    if not found:
        _finding(inventory, "no_recording_layout", asset, ".")


def _analyzer_metadata(
    store: MetadataStore, path: str, inventory: dict[str, Any], asset: str
) -> Any:
    """Use consolidated metadata or bounded v2 attribute fallbacks."""
    consolidated = _read(
        store, f"{path}/.zmetadata", inventory, asset, optional=True
    )
    if consolidated is not None:
        return consolidated["metadata"]
    # Older unconsolidated v2 analyzers: read just the required attributes and
    # array headers, never extension payloads or sorting_provenance/0.
    keys = (
        ".zattrs",
        "recording_info/.zattrs",
        "sorting/.zattrs",
        "sorting/unit_ids/.zarray",
    )
    return {
        key: _read(store, f"{path}/{key}", inventory, asset) for key in keys
    }


def _sortings(
    store: MetadataStore, inventory: dict[str, Any], asset: str
) -> None:
    """Describe every analyzer independently of other groups or experiments."""
    names = store.directories("postprocessed")
    if not names:
        _finding(inventory, "no_analyzers", asset, "postprocessed")
    for name in names:
        try:
            _sorting_record(store, inventory, asset, name)
        except Exception as exc:
            _finding(
                inventory,
                "record_scan_incomplete",
                asset,
                f"postprocessed/{name}",
                detail=type(exc).__name__,
            )


def _sorting_record(
    store: MetadataStore, inventory: dict[str, Any], asset: str, name: str
) -> None:
    """Observe one analyzer without instantiating its stored objects."""
    path = f"postprocessed/{name}"
    meta = _analyzer_metadata(store, path, inventory, asset)
    for required in (
        "recording_info/.zattrs",
        "sorting/.zattrs",
        "sorting/unit_ids/.zarray",
    ):
        if not isinstance(meta.get(required), dict):
            _finding(
                inventory,
                "missing_analyzer_metadata",
                asset,
                f"{path}/{required}",
            )
    info = meta.get("recording_info/.zattrs") or {}
    recording = info.get("recording_attributes") or {}
    sorting = meta.get("sorting/.zattrs") or {}
    match = re.fullmatch(
        r"(experiment[^_]+)_(.+)_recording([^_\.]*)(?:_group([^\.]+))?(?:\.zarr)?",
        name,
    )
    experiment, stream, recording_name, group = (
        match.groups() if match else (None,) * 4
    )
    source = (
        (sorting.get("annotations") or {}).get("__sorting_info__") or {}
    ).get("recording") or {}
    units = (meta.get("sorting/unit_ids/.zarray") or {}).get("shape")
    inventory["sortings"].append(
        dict(
            id=_id(asset, path, "sorting"),
            asset=asset,
            path=path,
            experiment=experiment,
            stream=stream,
            recording=(
                "recording" + recording_name
                if recording_name is not None
                else None
            ),
            group=group,
            identity_evidence="storage_name" if match else "unknown",
            annotated_experiment=(source.get("annotations") or {}).get(
                "experiment_name"
            ),
            source_reference=dict(
                class_name=source.get("class"),
                folder_path=(source.get("kwargs") or {}).get("folder_path"),
            ),
            sampling_frequency=recording.get(
                "sampling_frequency", sorting.get("sampling_frequency")
            ),
            sample_counts=recording.get("num_samples"),
            segment_count=sorting.get("num_segments"),
            channel_count=recording.get("num_channels"),
            channel_ids=recording.get("channel_ids"),
            unit_count=units[0] if units else None,
            channel_configuration=_geometry(
                inventory,
                asset,
                f"{path}/recording_info/.zattrs",
                info.get("probegroup"),
                source.get("annotations"),
            ),
        )
    )


def _contacts(
    config: dict[str, Any],
) -> dict[tuple[str, str, str], Any] | None:
    """Require sufficient physical identity before comparing contacts."""
    contacts = {}
    for probe in config["probes"]:
        serial = probe["serial"]
        if serial is None or not isinstance(probe["units"], str):
            return None
        ids, positions = probe["contact_ids"], probe["contact_positions"]
        if (
            not isinstance(ids, list)
            or not isinstance(positions, list)
            or len(ids) != len(positions)
        ):
            return None
        for contact, position in zip(ids, positions):
            key = (str(serial), str(contact), probe["units"])
            if key in contacts or not contact:
                return None
            contacts[key] = position
    return contacts or None


def _compare(
    inventory: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    configs: dict[str, Any],
) -> dict[str, Any]:
    """Report agreement, disagreement, or unknown for candidate sources."""
    checks = {}
    for field in ("sampling_frequency",):
        a, b = left.get(field), right.get(field)
        checks[field] = a == b if a is not None and b is not None else None
    samples = [s["sample_count"] for s in right["segments"]]
    checks["sample_counts"] = (
        left["sample_counts"] == samples
        if left["sample_counts"] is not None and None not in samples
        else None
    )
    a = _contacts(configs[left["channel_configuration"]])
    b = _contacts(configs[right["channel_configuration"]])
    checks["contact_subset"] = (
        a.keys() <= b.keys() if a is not None and b is not None else None
    )
    checks["contact_positions"] = (
        all(b.get(k) == v for k, v in a.items())
        if a is not None and b is not None
        else None
    )
    return checks


def _relationships(inventory: dict[str, Any]) -> None:
    """Retain all candidate joins, including conflicts and ambiguities."""
    configs = {c["id"]: c for c in inventory["channel_configurations"]}
    assets = {a["key"]: a for a in inventory["assets"]}
    for recording in inventory["recordings"]:
        candidates = [
            h
            for h in inventory["recorded_streams"]
            if h["asset"] == recording["asset"]
            and h["experiment"] == recording["experiment"]
            and h["stream"] == recording["stream"]
        ]
        inventory["relationships"].append(
            dict(
                kind="header_membership",
                source=recording["id"],
                candidates=[h["id"] for h in candidates],
                evidence="storage_name",
                status="candidate" if candidates else "unresolved",
            )
        )
        if not candidates:
            _finding(
                inventory,
                "compressed_without_header",
                recording["asset"],
                recording["path"],
                record=recording["id"],
            )
        for header in candidates:
            counts = {s["channel_count"] for s in recording["segments"]}
            if (counts and counts != {header["channel_count"]}) or header[
                "sampling_frequency"
            ] != recording["sampling_frequency"]:
                _finding(
                    inventory,
                    "header_metadata_conflict",
                    recording["asset"],
                    recording["path"],
                    record=recording["id"],
                )
    referenced_headers = {
        c for link in inventory["relationships"] for c in link["candidates"]
    }
    for header in inventory["recorded_streams"]:
        if header["id"] not in referenced_headers:
            _finding(
                inventory,
                "header_without_compressed_recording",
                header["asset"],
                header["metadata_path"],
                record=header["id"],
            )
    for sorting in inventory["sortings"]:
        source_keys = assets[sorting["asset"]].get("source_keys", [])
        candidates = [
            r
            for r in inventory["recordings"]
            if (not source_keys or r["asset"] in source_keys)
            and r["experiment"] is not None
            and r["experiment"] == sorting["experiment"]
            and r["stream"] == sorting["stream"]
        ]
        comparisons = {
            r["id"]: _compare(inventory, sorting, r, configs)
            for r in candidates
        }
        status = (
            "candidate"
            if len(candidates) == 1
            else "ambiguous"
            if candidates
            else "unresolved"
        )
        if any(False in checks.values() for checks in comparisons.values()):
            status = "conflict"
        annotated = sorting["annotated_experiment"]
        if annotated is not None and annotated != sorting["experiment"]:
            status = "conflict"
        inventory["relationships"].append(
            dict(
                kind="sorting_source",
                source=sorting["id"],
                candidates=[r["id"] for r in candidates],
                status=status,
                evidence=["storage_name", "source_experiment_annotation"]
                if annotated
                else ["storage_name"],
                checks=comparisons,
            )
        )
        if status != "candidate":
            _finding(
                inventory,
                f"sorting_source_{status}",
                sorting["asset"],
                sorting["path"],
                record=sorting["id"],
            )
    linked = {
        c
        for link in inventory["relationships"]
        if link["kind"] == "sorting_source"
        for c in link["candidates"]
    }
    for recording in inventory["recordings"]:
        if recording["id"] not in linked:
            _finding(
                inventory,
                "no_sorting_observed",
                recording["asset"],
                recording["path"],
                record=recording["id"],
            )


def build_inventory(
    sources: list[tuple[dict[str, Any], MetadataStore | None]],
) -> dict[str, Any]:
    """Inventory supplied assets, preserving failures as observations.

    Asset descriptors require ``key`` and ``role`` (``raw`` or ``sorted``).
    Optional ``id``, ``name`` and ``source_keys`` qualify identity and links.
    Stores resolve locations separately; local mounts and credentials are not
    persisted. A None store records an inaccessible source, not an empty one.
    """
    try:
        reader_version = version("aind-ephys-ibl-gui-conversion")
    except PackageNotFoundError:
        reader_version = "uninstalled"
    inventory: dict[str, Any] = dict(
        inventory_version=INVENTORY_VERSION,
        reader_version=reader_version,
        assets=[],
        recorded_streams=[],
        recordings=[],
        channel_configurations=[],
        sortings=[],
        relationships=[],
        findings=[],
    )
    keys = [asset["key"] for asset, _ in sources]
    if len(set(keys)) != len(keys):
        raise ValueError("Inventory source keys must be unique")
    for descriptor, store in sorted(sources, key=lambda item: item[0]["key"]):
        asset: dict[str, Any] = {
            k: descriptor[k]
            for k in ("key", "role", "id", "name", "aliases", "source_keys")
            if k in descriptor
        }
        asset["metadata_status"] = "complete"
        inventory["assets"].append(asset)
        key = asset["key"]
        before = len(inventory["findings"])
        if store is None:
            _finding(inventory, "source_unavailable", key, ".")
        else:
            try:
                if asset["role"] == "sorted":
                    _sortings(store, inventory, key)
                elif asset["role"] == "raw":
                    _raw(store, inventory, key)
                else:
                    _finding(inventory, "unsupported_source_role", key, ".")
            except Exception as exc:
                _finding(
                    inventory,
                    "source_scan_incomplete",
                    key,
                    ".",
                    detail=type(exc).__name__,
                )
        if len(inventory["findings"]) != before:
            asset["metadata_status"] = "incomplete"
    _relationships(inventory)
    inventory["metadata_complete"] = bool(inventory["assets"]) and all(
        a["metadata_status"] == "complete" for a in inventory["assets"]
    )
    return inventory


def inventory_slice(
    inventory: dict[str, Any],
    collection: str | None,
    sorted_recording: str | None = None,
    surface_finding: str | None = None,
) -> dict[str, Any]:
    """Keep a collection's observations with self-contained references."""
    assets = inventory["assets"]
    if sorted_recording is not None:
        selected = [
            a
            for a in assets
            if a["role"] == "sorted"
            and sorted_recording
            in [a["key"], a.get("name"), *a.get("aliases", [])]
        ]
        allowed = {a["key"] for a in selected}
        allowed.update(k for a in selected for k in a.get("source_keys", []))
        allowed.update(
            a["key"]
            for a in assets
            if surface_finding is not None
            and surface_finding
            in [a["key"], a.get("name"), *a.get("aliases", [])]
        )
        assets = [a for a in assets if a["key"] in allowed]
    else:
        allowed = {a["key"] for a in assets}

    def matches(record: dict[str, Any]) -> bool:
        """Retain unparsed identities as evidence within the source scope."""
        if record["asset"] not in allowed:
            return False
        if collection is None:
            return True
        stream = record.get("stream") or ""
        if not stream:
            return True
        token = stream.split("#")[-1].split(".", 1)[-1]
        return re.sub(r"-(AP|LFP)$", "", token) == collection

    result = dict(inventory)
    result["assets"] = assets
    for key in ("recordings", "recorded_streams", "sortings"):
        result[key] = [r for r in inventory[key] if matches(r)]
    # Preserve candidate references even when source-asset identity is unknown.
    for kind, target in (
        ("sortings", "recordings"),
        ("recordings", "recorded_streams"),
    ):
        source_ids = {r["id"] for r in result[kind]}
        needed = {
            c
            for link in inventory["relationships"]
            if link["source"] in source_ids
            for c in link["candidates"]
        }
        present = {r["id"] for r in result[target]}
        result[target].extend(
            r
            for r in inventory[target]
            if r["id"] in needed and r["id"] not in present
        )
    allowed.update(
        r["asset"]
        for key in ("recordings", "recorded_streams", "sortings")
        for r in result[key]
    )
    assets = [a for a in inventory["assets"] if a["key"] in allowed]
    result["assets"] = assets
    ids = {
        r["id"]
        for key in ("recordings", "recorded_streams", "sortings")
        for r in result[key]
    }
    channels = {
        r["channel_configuration"]
        for key in ("recordings", "sortings")
        for r in result[key]
    }
    result["channel_configurations"] = [
        c for c in inventory["channel_configurations"] if c["id"] in channels
    ]
    result["relationships"] = [
        r for r in inventory["relationships"] if r["source"] in ids
    ]
    result["findings"] = [
        f
        for f in inventory["findings"]
        if f["asset"] in allowed
        and (f["record"] is None or f["record"] in ids)
    ]
    result["scope"] = dict(
        collection=collection, sorted_recording=sorted_recording
    )
    result["metadata_complete"] = bool(assets) and all(
        a["metadata_status"] == "complete" for a in assets
    )
    return result


def write_inventory(inventory: dict[str, Any], path: Path) -> None:
    """Write portable, deterministic JSON beside stage outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
