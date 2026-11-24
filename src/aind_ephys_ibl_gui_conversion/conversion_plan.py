"""Resolve inventory observations into recorded, reviewable policy decisions.

This layer performs no data IO or extraction. A plan identifies one main
recording and its analyzer family; it never substitutes another experiment's
sorting for a missing shank. Extraction explicitly adopts these decisions and
records activation separately from the immutable discovery plan.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any

from aind_ephys_ibl_gui_conversion.inventory import inventory_slice

PLAN_VERSION = "1.0.0"
PLAN_FILENAME = "ephys_conversion_plan.json"
Row = dict[str, Any]
ContactKey = tuple[str, str]


def inventory_digest(inventory: Row) -> str:
    """Bind decisions to the precise portable inventory used to make them."""
    content = json.dumps(inventory, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def _issue(plan: Row, code: str, severity: str, **details: Any) -> None:
    """Keep uncertainty separate from a decision's recorded rationale."""
    plan["findings"].append(dict(code=code, severity=severity, **details))


def _token(stream: str | None) -> str | None:
    """Compare collection tokens while preserving full stream identity."""
    if stream is None:
        return None
    token = stream.split("#")[-1].split(".", 1)[-1]
    return re.sub(r"-(AP|LFP)$", "", token)


def _duration(recording: Row) -> float | None:
    """Measure a stored block by its total recorded samples, excluding gaps."""
    rate = recording.get("sampling_frequency")
    segments = recording.get("segments", [])
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (float, int))
        or not math.isfinite(rate)
        or rate <= 0
        or not segments
        or recording.get("declared_segment_count") != len(segments)
    ):
        return None
    samples = [s.get("sample_count") for s in segments]
    if any(type(n) is not int or n < 0 for n in samples) or sum(samples) <= 0:
        return None
    if len({s["index"] for s in segments}) != len(segments):
        return None
    return float(sum(samples) / rate)


def _reference(record: Row) -> Row:
    """Use asset-relative references, never producer or runtime mount paths."""
    return {key: record[key] for key in ("id", "asset", "path")}


def _main_recording(
    plan: Row, candidates: list[Row], override: str | None
) -> Row | None:
    """Choose the longest block before inspecting analyzer availability."""
    ranked = [
        dict(**_reference(r), duration_s=_duration(r)) for r in candidates
    ]
    ranked.sort(key=lambda r: r["id"])
    plan["main_candidates"] = ranked
    if override is not None:
        selected = [r for r in candidates if r["id"] == override]
        if not selected or _duration(selected[0]) is None:
            _issue(
                plan, "invalid_main_override", "error", recording_id=override
            )
            return None
        return selected[0]
    if not candidates:
        _issue(plan, "no_main_recording", "error")
        return None
    if len({r.get("stream") for r in candidates}) != 1:
        _issue(plan, "ambiguous_primary_stream", "error")
        return None
    if any(r["duration_s"] is None for r in ranked):
        _issue(plan, "main_duration_unknown", "error")
        return None
    longest = max(r["duration_s"] for r in ranked)
    winners = [
        r
        for r in ranked
        if math.isclose(r["duration_s"], longest, rel_tol=0, abs_tol=1e-9)
    ]
    if len(winners) != 1:
        _issue(
            plan,
            "main_duration_tie",
            "error",
            candidates=[r["id"] for r in winners],
        )
        return None
    return next(r for r in candidates if r["id"] == winners[0]["id"])


def _family_id(analyzer: Row) -> str:
    """Keep experiments, recording identifiers, and source assets distinct."""
    identity = [
        analyzer.get(k) for k in ("asset", "experiment", "stream", "recording")
    ]
    if None in identity:
        identity.append(analyzer["id"])
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]


def _family_candidates(
    plan: Row, inventory: Row, main: Row, sorted_key: str
) -> list[Row]:
    """Group analyzers only within one experiment/stream/recording identity."""
    links = {
        r["source"]: r
        for r in inventory["relationships"]
        if r["kind"] == "sorting_source"
    }
    grouped: dict[str, list[Row]] = defaultdict(list)
    for analyzer in inventory["sortings"]:
        link = links.get(analyzer["id"], {})
        if analyzer["asset"] == sorted_key and main["id"] in link.get(
            "candidates", []
        ):
            grouped[_family_id(analyzer)].append(analyzer)
    result = []
    for family_id, analyzers in sorted(grouped.items()):
        analyzers = sorted(analyzers, key=lambda a: a["id"])
        first = analyzers[0]
        result.append(
            dict(
                id=family_id,
                asset=sorted_key,
                experiment=first.get("experiment"),
                stream=first.get("stream"),
                recording=first.get("recording"),
                analyzer_ids=[a["id"] for a in analyzers],
            )
        )
    plan["sorting_families"] = result
    return result


def _contacts(config: Row) -> dict[ContactKey, Row] | None:
    """Read active physical contacts without interpreting analyzer groups."""
    result = {}
    for probe in config.get("probes", []):
        ids, positions = (
            probe.get("contact_ids"),
            probe.get("contact_positions"),
        )
        serial, units = probe.get("serial"), probe.get("units")
        if serial is None or not isinstance(units, str):
            return None
        if (
            not isinstance(ids, list)
            or not isinstance(positions, list)
            or len(ids) != len(positions)
        ):
            return None
        channels = probe.get("device_channel_indices")
        shanks = probe.get("shank_ids")
        if channels is not None and (
            not isinstance(channels, list)
            or len(channels) != len(ids)
            or any(type(n) is not int for n in channels)
        ):
            return None
        if shanks is not None and (
            not isinstance(shanks, list) or len(shanks) != len(ids)
        ):
            return None
        single_shank_ids = all(re.fullmatch(r"e\d+", str(c)) for c in ids)
        for index, (contact, position) in enumerate(zip(ids, positions)):
            if channels is not None and channels[index] < 0:
                continue
            key = (str(serial), str(contact))
            if not contact or key in result or not isinstance(position, list):
                return None
            if len(position) < 2 or any(
                type(n) not in (int, float) or not math.isfinite(n)
                for n in position
            ):
                return None
            match = re.match(r"^s(\d+)e", str(contact))
            physical = int(match[1]) if match else None
            recorded = str(shanks[index]) if shanks is not None else ""
            if recorded.isdigit():
                if physical is not None and physical != int(recorded):
                    return None
                physical = int(recorded)
            if physical is None and not recorded and single_shank_ids:
                physical = 0
            result[key] = dict(position=position, units=units, shank=physical)
    return result or None


def _contact_agrees(raw: Row | None, other: Row) -> bool:
    """Distinguish a conflicting shank assignment from a missing assignment."""
    return (
        raw is not None
        and raw["position"] == other["position"]
        and raw["units"] == other["units"]
        and (
            raw["shank"] is None
            or other["shank"] is None
            or raw["shank"] == other["shank"]
        )
    )


def selected_segment(main: Row, analyzers: list[Row]) -> Row | None:
    """Resolve one local sample clock within an already identified source.

    Lengths resolve segments only after asset/experiment/stream and physical
    source checks. Every analyzer group must identify the same unique segment.
    """
    segments = main["segments"]
    if (
        _duration(main) is None
        or [s["index"] for s in segments] != list(range(len(segments)))
        or not analyzers
        or any(a.get("segment_count") != 1 for a in analyzers)
    ):
        return None
    matches = [
        s
        for s in segments
        if s["sample_count"] > 0
        and all(
            a.get("sample_counts") == [s["sample_count"]] for a in analyzers
        )
    ]
    if len(matches) != 1:
        return None
    segment = matches[0]
    return dict(
        raw_segment_index=segment["index"],
        analyzer_segment_index=0,
        sample_count=segment["sample_count"],
        duration_s=segment["sample_count"] / main["sampling_frequency"],
        evidence="single_segment"
        if len(segments) == 1
        else "unique_sample_count_with_source_checks",
    )


def _validate_family(
    plan: Row, inventory: Row, main: Row, family: Row, explicit_main: bool
) -> list[Row]:
    """Check groups together and preserve uncertainty in their source maps."""
    by_id = {a["id"]: a for a in inventory["sortings"]}
    links = {
        r["source"]: r
        for r in inventory["relationships"]
        if r["kind"] == "sorting_source"
    }
    selected = [by_id[key] for key in family["analyzer_ids"]]
    plan["main_segment"] = selected_segment(main, selected)
    groups = [a.get("group") for a in selected]
    if len(set(groups)) != len(groups) or (None in groups and len(groups) > 1):
        _issue(
            plan,
            "overlapping_sorting_representations",
            "error",
            family_id=family["id"],
        )
    for analyzer in selected:
        link = links[analyzer["id"]]
        checks = dict(link.get("checks", {}).get(main["id"], {}))
        # Discovery compares whole segment vectors. A single-segment analyzer
        # can instead match one unique segment of this identified raw source.
        if (
            len(main["segments"]) > 1
            and plan["main_segment"] is not None
            and "sample_counts" in checks
        ):
            checks["sample_counts"] = True
        annotated = analyzer.get("annotated_experiment")
        if False in checks.values() or (
            annotated is not None and annotated != main.get("experiment")
        ):
            _issue(
                plan,
                "sorting_source_conflict",
                "error",
                analyzer_id=analyzer["id"],
                checks=checks,
            )
        elif not {
            "sampling_frequency",
            "sample_counts",
            "contact_subset",
            "contact_positions",
        }.issubset(checks) or any(v is None for v in checks.values()):
            _issue(
                plan,
                "sorting_source_unverified",
                "incomplete",
                analyzer_id=analyzer["id"],
            )
        if len(link["candidates"]) > 1 and not explicit_main:
            _issue(
                plan,
                "ambiguous_sorting_source",
                "error",
                analyzer_id=analyzer["id"],
            )
        if (
            analyzer.get("experiment") is None
            or analyzer.get("recording") is None
        ):
            _issue(
                plan,
                "sorting_identity_unknown",
                "error",
                analyzer_id=analyzer["id"],
            )
    if plan["main_segment"] is None:
        _issue(plan, "segment_mapping_unverified", "incomplete")
    _validate_acquisition_mapping(plan, inventory, main, family)
    return selected


def _validate_acquisition_mapping(
    plan: Row, inventory: Row, main: Row, family: Row
) -> None:
    """Keep header membership separate from segment correspondence."""
    header_ids = {
        candidate
        for link in inventory["relationships"]
        if link["kind"] == "header_membership" and link["source"] == main["id"]
        for candidate in link["candidates"]
    }
    headers = [
        h for h in inventory["recorded_streams"] if h["id"] in header_ids
    ]
    # The dispatcher labels split Zarr segments recording1, recording2, ...
    # independently of the original Open Ephys directory names. One acquired
    # recording and one raw segment have an unambiguous correspondence even
    # when, for example, acquisition recording2 became sorting recording1.
    single = (
        len(headers) == 1
        and len(main["segments"]) == 1
        and plan.get("main_segment") is not None
    )
    matching = (
        headers
        if single
        else [h for h in headers if h["recording"] == family.get("recording")]
    )
    if (
        family.get("recording") in (None, "recording")
        or plan.get("main_segment") is None
        or not headers
        or len(matching) > 1
        or (
            len(headers) > 1
            and (
                plan.get("main_segment") is None
                or len(headers) != len(main["segments"])
            )
        )
    ):
        _issue(plan, "acquisition_recording_mapping_unverified", "incomplete")
    elif not matching:
        _issue(plan, "acquisition_recording_mapping_unverified", "incomplete")
    else:
        header = matching[0]
        if any(
            header.get(k) is None
            for k in ("sampling_frequency", "channel_count")
        ):
            _issue(
                plan, "acquisition_header_metadata_unverified", "incomplete"
            )
        elif header["sampling_frequency"] != main["sampling_frequency"] or {
            s["channel_count"] for s in main["segments"]
        } != {header["channel_count"]}:
            _issue(plan, "acquisition_header_metadata_conflict", "error")
        else:
            plan["acquisition_header_id"] = header["id"]
            plan["acquisition_recording_mapping"] = dict(
                header_id=header["id"],
                acquisition_recording=header["recording"],
                sorting_recording=family.get("recording"),
                raw_segment_index=plan["main_segment"]["raw_segment_index"],
                analyzer_segment_index=plan["main_segment"][
                    "analyzer_segment_index"
                ],
                evidence="unique_header_for_single_raw_segment"
                if single
                else "matching_recording_name",
            )


def _shank_coverage(
    plan: Row, main: Row, analyzers: list[Row], configs: dict[str, Row]
) -> None:
    """Report missing main shanks without borrowing another sorting."""
    raw = _contacts(configs.get(main["channel_configuration"], {}))
    sorted_contacts: dict[ContactKey, Row] = {}
    unknown = raw is None
    for analyzer in analyzers:
        contacts = _contacts(
            configs.get(analyzer["channel_configuration"], {})
        )
        if contacts is None:
            unknown = True
            continue
        if analyzer.get("channel_count") != len(contacts):
            _issue(
                plan,
                "analyzer_channel_geometry_mismatch",
                "error",
                analyzer_id=analyzer["id"],
            )
        if raw is not None and any(
            not _contact_agrees(raw.get(k), v) for k, v in contacts.items()
        ):
            _issue(
                plan,
                "analyzer_contact_geometry_conflict",
                "error",
                analyzer_id=analyzer["id"],
            )
        if sorted_contacts.keys() & contacts.keys():
            _issue(
                plan,
                "overlapping_analyzer_contacts",
                "error",
                analyzer_id=analyzer["id"],
            )
        sorted_contacts.update(contacts)
    if unknown or any(
        c["shank"] is None
        for c in [*(raw or {}).values(), *sorted_contacts.values()]
    ):
        plan["shank_coverage"] = None
        _issue(plan, "shank_coverage_unknown", "incomplete")
        return
    assert raw is not None
    raw_shanks = {c["shank"] for c in raw.values()}
    observed = {c["shank"] for c in sorted_contacts.values()}
    missing = sorted(raw_shanks - observed)
    plan["shank_coverage"] = dict(
        recorded=sorted(raw_shanks), sorted=sorted(observed), missing=missing
    )
    if missing:
        _issue(
            plan,
            "main_shanks_without_sorting",
            "incomplete",
            physical_shanks=missing,
        )


def _paired_lfp(
    inventory: Row, recording: Row, indices: list[int] | None = None
) -> tuple[Row | None, str | None]:
    """Keep NP1 AP/LFP pairing within the same asset and experiment."""
    stream = recording.get("stream") or ""
    if not stream.endswith("-AP"):
        return None, None
    pairs = [
        r
        for r in inventory["recordings"]
        if r["asset"] == recording["asset"]
        and r.get("experiment") == recording.get("experiment")
        and r.get("stream") == stream[:-3] + "-LFP"
    ]
    if len(pairs) != 1 or _duration(pairs[0]) is None:
        return None, "missing_or_ambiguous_lfp_pair"
    paired = pairs[0]
    if len(paired["segments"]) != len(recording["segments"]):
        return paired, "lfp_segment_mismatch"
    ap_rate, lfp_rate = (
        recording["sampling_frequency"],
        paired["sampling_frequency"],
    )
    if any(
        not math.isclose(
            a["sample_count"] / ap_rate,
            b["sample_count"] / lfp_rate,
            rel_tol=0,
            abs_tol=1 / min(ap_rate, lfp_rate),
        )
        for index, (a, b) in enumerate(
            zip(recording["segments"], paired["segments"])
        )
        if indices is None or index in indices
    ):
        return paired, "lfp_duration_mismatch"
    return paired, None


def _continuous_inputs(
    plan: Row,
    inventory: Row,
    main: Row,
    configs: dict[str, Row],
    surface_keys: set[str],
) -> None:
    """List compatible QC blocks and preserve unknown acquisition roles."""
    main_contacts = _contacts(configs.get(main["channel_configuration"], {}))
    main_serials = {key[0] for key in main_contacts} if main_contacts else None
    inputs = []
    for recording in sorted(inventory["recordings"], key=lambda r: r["id"]):
        stream = recording.get("stream") or ""
        if _token(stream) != plan["scope"]["collection"] or stream.endswith(
            "-LFP"
        ):
            continue
        contacts = _contacts(
            configs.get(recording["channel_configuration"], {})
        )
        reason = None
        if _duration(recording) is None:
            reason = "duration_unknown"
        elif contacts is None or main_contacts is None:
            reason = "geometry_unknown"
        elif {key[0] for key in contacts} != main_serials:
            reason = "different_probe_identity"
        elif recording.get("sampling_frequency") != main.get(
            "sampling_frequency"
        ):
            reason = "different_sampling_frequency"
        elif any(
            not _contact_agrees(main_contacts[key], contacts[key])
            for key in main_contacts.keys() & contacts.keys()
        ):
            reason = "contact_geometry_conflict"
        indices = (
            [plan["main_segment"]["raw_segment_index"]]
            if recording["id"] == main["id"]
            and plan.get("main_segment") is not None
            else [s["index"] for s in recording["segments"]]
        )
        paired, pair_problem = (
            _paired_lfp(inventory, recording, indices)
            if reason is None
            else (None, None)
        )
        reason = reason or pair_problem
        role = (
            "main"
            if recording["id"] == main["id"]
            else "surface"
            if recording["asset"] in surface_keys
            else "unknown"
        )
        inputs.append(
            dict(
                **_reference(recording),
                stream=recording.get("stream"),
                segments=recording["segments"],
                segment_indices=indices,
                paired_lfp=_reference(paired) if paired is not None else None,
                channel_configuration=recording["channel_configuration"],
                selected=reason is None,
                exclusion_reason=reason,
                role=role,
                role_evidence=plan["main_recording"]["reason"]
                if role == "main"
                else "explicit_surface_asset"
                if role == "surface"
                else None,
            )
        )
        if reason is not None:
            _issue(
                plan,
                "continuous_input_unresolved",
                "incomplete",
                recording_id=recording["id"],
                reason=reason,
            )
    plan["continuous_inputs"] = inputs


def _sorted_asset(plan: Row, inventory: Row, name: str | None) -> Row | None:
    """Require one sorting asset rather than mixing captures or sessions."""
    candidates: list[Row] = [
        a
        for a in inventory["assets"]
        if a["role"] == "sorted"
        and (
            name is None
            or name in [a["key"], a.get("name"), *a.get("aliases", [])]
        )
    ]
    if len(candidates) != 1:
        _issue(
            plan,
            "sorting_asset_ambiguous",
            "error",
            candidates=[a["key"] for a in candidates],
        )
        return None
    return candidates[0]


def _status(plan: Row) -> Row:
    """Keep a partial decision inspectable even when it cannot be executed."""
    levels = {f["severity"] for f in plan["findings"]}
    plan["status"] = (
        "blocked"
        if "error" in levels
        else "incomplete"
        if "incomplete" in levels
        else "ready"
    )
    return copy.deepcopy(plan)


def _choose_families(
    plan: Row,
    inventory: Row,
    main: Row,
    families: list[Row],
    override: str | None,
) -> list[Row]:
    """Choose the longest raw segment before inspecting its sortings."""
    if override is not None:
        return [f for f in families if f["id"] == override]
    if len(main["segments"]) == 1:
        return families
    maximum = max(s["sample_count"] for s in main["segments"])
    if sum(s["sample_count"] == maximum for s in main["segments"]) != 1:
        _issue(plan, "main_segment_duration_tie", "error")
        _issue(plan, "segment_mapping_unverified", "incomplete")
        return []
    analyzers = {a["id"]: a for a in inventory["sortings"]}
    chosen = [
        f
        for f in families
        if all(
            analyzers[key].get("segment_count") == 1
            and analyzers[key].get("sample_counts") == [maximum]
            for key in f["analyzer_ids"]
        )
    ]
    if not chosen:
        _issue(plan, "segment_mapping_unverified", "incomplete")
    return chosen


def build_conversion_plan(
    inventory: Row,
    collection: str,
    sorted_recording: str | None = None,
    *,
    main_recording_id: str | None = None,
    sorting_family_id: str | None = None,
    surface_finding: str | None = None,
) -> Row:
    """Select the longest raw block and one compatible analyzer family.

    Duration is the sum of stored segment samples divided by sample rate.
    Missing durations and exact ties require an override. The main block is
    chosen before considering available sortings; no shorter block fills a
    missing main shank. A single sorted segment can map to one uniquely
    identified raw segment; all segment clocks remain local.
    """
    if inventory.get("inventory_version", "").split(".")[0] != "1":
        raise ValueError("Unsupported inventory version")
    scoped = inventory_slice(
        inventory, collection, sorted_recording, surface_finding
    )
    plan: Row = dict(
        plan_version=PLAN_VERSION,
        inventory_digest=inventory_digest(scoped),
        execution_enabled=False,
        scope=dict(collection=collection, sorted_recording=sorted_recording),
        policy=dict(
            main="longest_recorded_duration",
            duration="sum_segment_samples_over_rate",
            time_basis="segment_local",
            main_segment="longest_recorded_segment",
            non_main_role="unknown",
            sorting="one_family_from_selected_block",
            qc="compatible_primary_blocks",
        ),
        overrides=dict(
            main_recording_id=main_recording_id,
            sorting_family_id=sorting_family_id,
        ),
        findings=[],
        main_candidates=[],
        main_recording=None,
        main_segment=None,
        acquisition_header_id=None,
        acquisition_recording_mapping=None,
        sorting_families=[],
        sorting_family=None,
        analyzers=[],
        continuous_inputs=[],
        shank_coverage=None,
    )
    asset = _sorted_asset(plan, scoped, sorted_recording)
    if asset is None:
        return _status(plan)
    source_keys = set(asset.get("source_keys", []))
    if not source_keys and main_recording_id is None:
        _issue(plan, "main_source_asset_unknown", "error")
        return _status(plan)
    candidates = [
        r
        for r in scoped["recordings"]
        if _token(r.get("stream")) == collection
        and not (r.get("stream") or "").endswith("-LFP")
        and (not source_keys or r["asset"] in source_keys)
    ]
    main = _main_recording(plan, candidates, main_recording_id)
    if main is None:
        return _status(plan)
    plan["main_recording"] = dict(
        **_reference(main),
        experiment=main.get("experiment"),
        stream=main.get("stream"),
        segments=main["segments"],
        duration_s=_duration(main),
        reason="explicit_override"
        if main_recording_id
        else "longest_recorded_duration",
    )
    if not scoped["metadata_complete"]:
        _issue(plan, "inventory_metadata_incomplete", "incomplete")
    families = _family_candidates(plan, scoped, main, asset["key"])
    chosen = _choose_families(plan, scoped, main, families, sorting_family_id)
    if len(chosen) != 1:
        _issue(
            plan,
            "sorting_family_unresolved",
            "error",
            candidates=[f["id"] for f in families],
        )
    else:
        family = chosen[0]
        plan["sorting_family"] = family["id"]
        analyzers = _validate_family(
            plan, scoped, main, family, main_recording_id is not None
        )
        plan["analyzers"] = [
            dict(
                **_reference(a),
                group=a.get("group"),
                unit_count=a.get("unit_count"),
                channel_configuration=a["channel_configuration"],
                sample_counts=a.get("sample_counts"),
            )
            for a in analyzers
        ]
        _shank_coverage(
            plan,
            main,
            analyzers,
            {c["id"]: c for c in scoped["channel_configurations"]},
        )
    surface_keys = {
        a["key"]
        for a in scoped["assets"]
        if surface_finding is not None
        and surface_finding in [a["key"], a.get("name"), *a.get("aliases", [])]
    }
    _continuous_inputs(
        plan,
        scoped,
        main,
        {c["id"]: c for c in scoped["channel_configurations"]},
        surface_keys,
    )
    return _status(plan)
