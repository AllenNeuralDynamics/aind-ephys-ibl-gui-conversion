"""Bind a saved decision to mounted inputs before producing ALF arrays."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import spikeinterface as si

from aind_ephys_ibl_gui_conversion.channel_metadata import build_channel_table
from aind_ephys_ibl_gui_conversion.conversion_plan import (
    PLAN_VERSION,
    _contacts,
    inventory_digest,
    selected_segment,
)
from aind_ephys_ibl_gui_conversion.inventory import write_inventory
from aind_ephys_ibl_gui_conversion.probe_metadata import probe_metadata
from aind_ephys_ibl_gui_conversion.recording_utils import _stream_matches
from aind_ephys_ibl_gui_conversion.spikes import _load_analyzer
from aind_ephys_ibl_gui_conversion.types import (
    ChannelTable,
    ExperimentBlock,
    ProbeStream,
)

Row = dict[str, Any]
EXECUTION_FILENAME = "ephys_extraction.json"


def resolve_reference(reference: Row, roots: dict[str, Path]) -> Path:
    """Resolve an exact asset-relative path without escaping its mount."""
    relative = Path(reference["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid inventory path: {relative}")
    root = Path(roots[reference["asset"]]).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Inventory reference escapes its asset: {relative}")
    return path


def _live_geometry(obj: Any) -> Row:
    """Use the same portable probe fields as discovery, including serials."""
    group = obj.get_probegroup().to_dict(array_as_list=True)
    if hasattr(obj, "sorting"):
        source = (obj.sorting.get_annotation("__sorting_info__") or {}).get(
            "recording"
        ) or {}
        annotations = source.get("annotations") or {}
    else:
        annotations = dict(probes_info=obj.get_annotation("probes_info"))
    return dict(probes=probe_metadata(group, annotations))


def _geometry_in_channel_order(
    config: Row, count: int
) -> list[tuple[Any, Any]]:
    """Project physical identities onto recording columns, retaining order."""
    contacts = _contacts(config)
    if contacts is None or len(contacts) != count:
        raise ValueError("Incomplete live channel geometry")
    ordered: dict[int, tuple[Any, Any]] = {}
    for probe in config["probes"]:
        for contact, device in zip(
            probe["contact_ids"], probe["device_channel_indices"], strict=True
        ):
            if device < 0:
                continue
            key = (str(probe["serial"]), str(contact))
            if device in ordered:
                raise ValueError("Duplicate device channel in geometry")
            ordered[device] = (key, contacts[key])
    if set(ordered) != set(range(count)):
        raise ValueError("Unverified probe-to-recording channel mapping")
    return [ordered[i] for i in range(count)]


def _validate_live(obj: Any, observed: Row, configs: dict[str, Row]) -> None:
    """Reject changed samples, rate, channels, or geometry before array IO."""
    analyzer = "sample_counts" in observed
    counts = (
        observed["sample_counts"]
        if analyzer
        else [s["sample_count"] for s in observed["segments"]]
    )
    if not analyzer and [s["index"] for s in observed["segments"]] != list(
        range(len(counts))
    ):
        raise ValueError("Unverified segment index mapping")
    rate = obj.sampling_frequency if analyzer else obj.get_sampling_frequency()
    if (
        rate != observed["sampling_frequency"]
        or obj.get_num_segments() != len(counts)
        or [obj.get_num_samples(i) for i in range(len(counts))] != counts
    ):
        raise ValueError(f"Live sampling metadata changed: {observed['id']}")
    count = obj.get_num_channels()
    expected_count = (
        observed["channel_count"]
        if analyzer
        else observed["segments"][0]["channel_count"]
    )
    expected = configs[observed["channel_configuration"]]
    if count != expected_count or _geometry_in_channel_order(
        _live_geometry(obj), count
    ) != _geometry_in_channel_order(expected, count):
        raise ValueError(f"Live channel geometry changed: {observed['id']}")
    if analyzer and (
        len(obj.unit_ids) != observed["unit_count"]
        or list(obj.channel_ids) != observed["channel_ids"]
    ):
        raise ValueError(f"Live analyzer identity changed: {observed['id']}")


def _observed(reference: Row, rows: dict[str, Row]) -> Row:
    """Require selected IDs, source assets and paths to match inventory."""
    row = rows[reference["id"]]
    if any(row[key] != reference[key] for key in ("asset", "path")):
        raise ValueError("Plan reference disagrees with inventory")
    return row


def validate_execution_plan(plan: Row, inventory: Row) -> None:
    """Allow known absent sorting shanks, rejecting unresolved decisions."""
    if plan.get("plan_version") != PLAN_VERSION:
        raise ValueError("Unsupported extraction plan version")
    if plan.get("inventory_digest") != inventory_digest(inventory):
        raise ValueError("Extraction plan does not match inventory")
    unresolved = [
        f["code"]
        for f in plan["findings"]
        if f["code"] != "main_shanks_without_sorting"
    ]
    if (
        plan["status"] not in {"ready", "incomplete"}
        or unresolved
        or not plan["main_recording"]
        or not plan["analyzers"]
    ):
        raise ValueError(f"Conversion plan cannot execute: {unresolved}")
    families = {f["id"]: f for f in plan["sorting_families"]}
    family = families[plan["sorting_family"]]
    if set(family["analyzer_ids"]) != {a["id"] for a in plan["analyzers"]}:
        raise ValueError("Plan analyzer family membership changed")
    main = plan["main_recording"]["id"]
    selected_main = [
        r
        for r in plan["continuous_inputs"]
        if r["selected"] and r["id"] == main and r["role"] == "main"
    ]
    if len(selected_main) != 1:
        raise ValueError("Selected main recording is missing from QC inputs")


@dataclass(frozen=True)
class PlannedExtraction:
    """One preflight result shared by spike and continuous extraction."""

    plan: Row
    stream: ProbeStream
    analyzers: tuple[tuple[Row, Any, Path], ...]
    channel_table: ChannelTable
    window_duration: float
    skipped_blocks: tuple[Row, ...]

    def check_request(
        self, stream: str | None, window: float | None = None
    ) -> None:
        """Reject changes to stream selection or channel-table membership."""
        if not _stream_matches(self.stream.stream_name, stream):
            raise ValueError("Extraction request does not match plan stream")
        if window is not None and window != self.window_duration:
            raise ValueError(
                "Analysis window differs from extraction preflight"
            )

    def write_receipt(
        self, destination: Path, status: str, *, error: str | None = None
    ) -> None:
        """Record activation, original decisions, and runtime QC exclusions."""
        write_inventory(
            dict(
                plan_version=PLAN_VERSION,
                execution_enabled=True,
                status=status,
                error=error,
                plan=self.plan,
                blocks=[
                    dict(
                        recording_id=b.recording_id,
                        segment_index=b.segment_index,
                        role=b.source,
                        selected_main=b.selected_main,
                    )
                    for b in self.stream.blocks
                ],
                skipped_blocks=list(self.skipped_blocks),
            ),
            destination / EXECUTION_FILENAME,
        )


def _selected_main_index(
    plan: Row, main: Row, sortings: dict[str, Row]
) -> int:
    """Validate the saved segment decision without substituting another one."""
    segment = plan.get("main_segment")
    if segment is not None:
        expected = selected_segment(
            main, [_observed(a, sortings) for a in plan["analyzers"]]
        )
        if segment != expected:
            raise ValueError("Saved main segment does not match inventory")
        return int(segment["raw_segment_index"])
    if len(main["segments"]) != 1:
        raise ValueError("Main multisegment time mapping is unverified")
    return 0


def _qc_segments(
    reference: Row,
    observed: Row,
    main: Row,
    main_index: int,
    skipped: list[Row],
) -> list[Row]:
    """Keep the selected main clock and retain independent secondary clocks."""
    if reference["id"] != main["id"]:
        return list(observed["segments"])
    skipped.extend(
        dict(
            recording_id=reference["id"],
            segment_index=s["index"],
            reason="not_selected_main_segment",
        )
        for s in observed["segments"]
        if s["index"] != main_index
    )
    return [s for s in observed["segments"] if s["index"] == main_index]


def prepare_extraction(
    plan: Row,
    inventory: Row,
    roots: dict[str, Path],
    results_folder: Path,
    *,
    window_duration: float = 4.0,
) -> PlannedExtraction:
    """Load planned paths and validate inputs before output starts."""
    validate_execution_plan(plan, inventory)
    for finding in plan["findings"]:
        logging.warning(
            "Planned extraction retains missing sorting coverage: %s", finding
        )
    if not np.isfinite(window_duration) or window_duration <= 0:
        raise ValueError("QC window must be finite and positive")
    plan = copy.deepcopy(plan)
    records = {r["id"]: r for r in inventory["recordings"]}
    sortings = {r["id"]: r for r in inventory["sortings"]}
    configs = {r["id"]: r for r in inventory["channel_configurations"]}
    main = _observed(plan["main_recording"], records)
    main_index = _selected_main_index(plan, main, sortings)
    selected = sorted(
        (r for r in plan["continuous_inputs"] if r["selected"]),
        key=lambda r: (r["id"] != main["id"], r["id"]),
    )
    blocks: list[ExperimentBlock] = []
    skipped: list[Row] = []
    for reference in selected:
        observed = _observed(reference, records)
        recording = si.read_zarr(resolve_reference(reference, roots))
        _validate_live(recording, observed, configs)
        lfp = None
        if reference["paired_lfp"] is not None:
            lfp_ref = reference["paired_lfp"]
            lfp = si.read_zarr(resolve_reference(lfp_ref, roots))
            _validate_live(lfp, _observed(lfp_ref, records), configs)
            if _geometry_in_channel_order(
                _live_geometry(recording), recording.get_num_channels()
            ) != _geometry_in_channel_order(
                _live_geometry(lfp), lfp.get_num_channels()
            ):
                raise ValueError("AP and LFP channel order differs")
        segments = _qc_segments(reference, observed, main, main_index, skipped)
        for source_segment in segments:
            index = source_segment["index"]
            ap_segment = recording.select_segments([index])
            lf_segment = lfp.select_segments([index]) if lfp else None
            is_main = reference["id"] == main["id"] and index == main_index
            if any(
                r.get_num_samples()
                < int(window_duration * r.get_sampling_frequency())
                for r in (ap_segment, lf_segment)
                if r is not None
            ):
                if is_main:
                    raise ValueError("Selected main is shorter than QC window")
                skipped.append(
                    dict(
                        recording_id=reference["id"],
                        segment_index=index,
                        reason="short_window",
                    )
                )
                continue
            ap_segment.reset_times()
            if lf_segment is not None:
                lf_segment.reset_times()
            blocks.append(
                ExperimentBlock(
                    ap_segment,
                    lf_segment,
                    len(blocks),
                    source="unknown"
                    if reference["role"] == "main" and not is_main
                    else reference["role"],
                    recording_id=reference["id"],
                    segment_index=index,
                    selected_main=is_main,
                )
            )
    analyzers = []
    for reference in plan["analyzers"]:
        observed = _observed(reference, sortings)
        path = resolve_reference(reference, roots)
        analyzer = _load_analyzer(path)
        _validate_live(analyzer, observed, configs)
        if analyzer.get_num_segments() != 1:
            raise ValueError(
                "Analyzer multisegment time mapping is unverified"
            )
        if (
            analyzer.get_num_samples(0)
            != main["segments"][main_index]["sample_count"]
        ):
            raise ValueError("Analyzer does not match selected main segment")
        analyzers.append((reference, analyzer, path))
    table, _ = build_channel_table([b.recording for b in blocks])
    # Validate analyzer channel joins before either consumer starts writing.
    for _, analyzer, _ in analyzers:
        analyzer_rows(analyzer, table)
    collection = plan["scope"]["collection"]
    stream = ProbeStream(
        main["stream"], collection, blocks, Path(results_folder) / collection
    )
    return PlannedExtraction(
        plan, stream, tuple(analyzers), table, window_duration, tuple(skipped)
    )


def analyzer_rows(analyzer: Any, table: ChannelTable) -> np.ndarray:
    """Translate analyzer-local columns into canonical output channel rows."""
    locations = np.asarray(analyzer.get_channel_locations())[:, :2]
    keys = {
        tuple(np.round(p).astype(int)): i
        for i, p in enumerate(table.local_coordinates)
    }
    try:
        return np.asarray(
            [keys[tuple(np.round(p).astype(int))] for p in locations],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError("Analyzer channel absent from QC table") from exc
