"""Duration selects a recording; analyzer availability never changes it."""

import copy
import gzip
import json
from pathlib import Path

import pytest

from aind_ephys_ibl_gui_conversion.conversion_plan import build_conversion_plan
from aind_ephys_ibl_gui_conversion.inventory import build_inventory

from .test_inventory import sources

STREAM = "Record Node 103#Neuropix-PXI-111.ProbeA"


@pytest.fixture
def inventory():
    path = Path(__file__).parent / "fixtures/inventory_858398.json.gz"
    with gzip.open(path, "rt") as handle:
        return build_inventory(sources(json.load(handle)))


def _recording(inventory, experiment):
    return next(
        r
        for r in inventory["recordings"]
        if r["stream"] == STREAM
        and r["experiment"] == f"experiment{experiment}"
    )


def _codes(plan):
    return {f["code"] for f in plan["findings"]}


def _plan(inventory, **kwargs):
    return build_conversion_plan(inventory, "ProbeA", "sort", **kwargs)


def test_retained_main_is_chosen_by_duration_and_missing_shanks_remain_missing(
    inventory,
):
    before = copy.deepcopy(inventory)
    plan = _plan(inventory)
    assert inventory == before
    assert plan["execution_enabled"] is False
    assert plan["main_recording"]["experiment"] == "experiment1"
    assert plan["main_recording"]["duration_s"] == pytest.approx(5634.26)
    assert plan["main_recording"]["reason"] == "longest_recorded_duration"
    assert len(plan["analyzers"]) == 2
    assert {a["group"] for a in plan["analyzers"]} == {"0", "1"}
    assert all("experiment1_" in a["path"] for a in plan["analyzers"])
    assert plan["shank_coverage"] == {
        "recorded": [0, 1, 2, 3],
        "sorted": [0, 1],
        "missing": [2, 3],
    }
    assert plan["status"] == "incomplete"
    assert _codes(plan) == {"main_shanks_without_sorting"}
    assert len(plan["continuous_inputs"]) == 11
    assert all(r["selected"] for r in plan["continuous_inputs"])
    assert sum(r["role"] == "main" for r in plan["continuous_inputs"]) == 1
    assert sum(r["role"] == "unknown" for r in plan["continuous_inputs"]) == 10


def test_split_stream_keeps_physical_shank_identity(inventory):
    plan = build_conversion_plan(inventory, "ProbeB-2", "sort")
    assert plan["status"] == "ready"
    assert plan["main_recording"]["experiment"] == "experiment1"
    assert plan["shank_coverage"] == {
        "recorded": [1],
        "sorted": [1],
        "missing": [],
    }
    assert len(plan["analyzers"]) == 1
    assert len(plan["continuous_inputs"]) == 4


def test_explicit_short_recording_override_selects_its_own_family(inventory):
    plan = _plan(inventory, main_recording_id=_recording(inventory, 5)["id"])
    assert plan["status"] == "ready"
    assert plan["main_recording"]["experiment"] == "experiment5"
    assert plan["main_recording"]["reason"] == "explicit_override"
    assert len(plan["analyzers"]) == 1
    assert "experiment5_" in plan["analyzers"][0]["path"]
    assert plan["shank_coverage"]["recorded"] == [1]
    assert plan["shank_coverage"]["sorted"] == [1]


def test_longest_recording_is_not_replaced_when_it_has_no_sorting(inventory):
    inventory["sortings"] = [
        s
        for s in inventory["sortings"]
        if not (s["experiment"] == "experiment1" and s["stream"] == STREAM)
    ]
    plan = _plan(inventory)
    assert plan["main_recording"]["experiment"] == "experiment1"
    assert plan["status"] == "blocked"
    assert plan["analyzers"] == []
    assert "sorting_family_unresolved" in _codes(plan)


def test_exact_duration_ties_require_an_override(inventory):
    main = _recording(inventory, 1)
    _recording(inventory, 2)["segments"] = copy.deepcopy(main["segments"])
    assert _plan(inventory)["status"] == "blocked"
    assert "main_duration_tie" in _codes(_plan(inventory))
    chosen = _plan(inventory, main_recording_id=main["id"])
    assert chosen["main_recording"]["id"] == main["id"]


def test_unknown_duration_does_not_silently_drop_a_main_candidate(inventory):
    _recording(inventory, 2)["segments"][0]["sample_count"] = None
    plan = _plan(inventory)
    assert plan["main_recording"] is None
    assert "main_duration_unknown" in _codes(plan)
    overridden = _plan(
        inventory, main_recording_id=_recording(inventory, 1)["id"]
    )
    assert overridden["main_recording"] is not None
    assert any(
        r["exclusion_reason"] == "duration_unknown"
        for r in overridden["continuous_inputs"]
    )


@pytest.mark.parametrize(
    "override",
    [{"main_recording_id": "wrong"}, {"sorting_family_id": "wrong"}],
)
def test_invalid_override_is_reported_without_falling_back(
    inventory, override
):
    assert _plan(inventory, **override)["status"] == "blocked"


def _another_family(inventory, recording_name):
    original = next(
        s
        for s in inventory["sortings"]
        if s["stream"] == STREAM and s["experiment"] == "experiment1"
    )
    other = copy.deepcopy(original)
    other.update(
        id="another-analyzer",
        recording=recording_name,
        path="postprocessed/another.zarr",
    )
    inventory["sortings"].append(other)
    link = next(
        r for r in inventory["relationships"] if r["source"] == original["id"]
    )
    inventory["relationships"].append(
        dict(copy.deepcopy(link), source=other["id"])
    )
    return original


def test_distinct_recording_families_require_selection(inventory):
    _another_family(inventory, "recording2")
    plan = _plan(inventory)
    assert plan["status"] == "blocked"
    assert len(plan["sorting_families"]) == 2
    selected = next(
        f for f in plan["sorting_families"] if f["recording"] == "recording1"
    )
    chosen = _plan(inventory, sorting_family_id=selected["id"])
    assert chosen["status"] == "incomplete"  # shanks 2/3 still absent
    assert len(chosen["analyzers"]) == 2


def test_duplicate_group_or_overlapping_contacts_block_a_family(inventory):
    _another_family(inventory, "recording1")
    plan = _plan(inventory)
    assert plan["status"] == "blocked"
    assert "overlapping_sorting_representations" in _codes(plan)
    assert "overlapping_analyzer_contacts" in _codes(plan)


def test_conflicting_source_metadata_is_not_overridden_by_duration(inventory):
    link = next(
        r
        for r in inventory["relationships"]
        if r["kind"] == "sorting_source"
        and _recording(inventory, 1)["id"] in r["candidates"]
    )
    link["checks"][_recording(inventory, 1)["id"]]["sample_counts"] = False
    plan = _plan(inventory)
    assert plan["main_recording"]["experiment"] == "experiment1"
    assert plan["status"] == "blocked"
    assert "sorting_source_conflict" in _codes(plan)


def test_unknown_geometry_never_becomes_group_based_shank_identity(inventory):
    target = next(s for s in inventory["sortings"] if s["stream"] == STREAM)
    config = next(
        c
        for c in inventory["channel_configurations"]
        if c["id"] == target["channel_configuration"]
    )
    config["probes"] = []
    plan = _plan(inventory)
    assert plan["shank_coverage"] is None
    assert "shank_coverage_unknown" in _codes(plan)


def test_segments_keep_local_time_and_mapping_uncertainty(inventory):
    main = _recording(inventory, 1)
    segment = dict(main["segments"][0], index=1, sample_count=30000)
    main["segments"].append(segment)
    main["declared_segment_count"] = 2
    for a in inventory["sortings"]:
        if a["stream"] == STREAM and a["experiment"] == "experiment1":
            a["sample_counts"].append(30000)
            a["segment_count"] = 2
    plan = _plan(inventory)
    assert plan["main_recording"]["duration_s"] == pytest.approx(5635.26)
    assert plan["policy"]["time_basis"] == "segment_local"
    assert "segment_mapping_unverified" in _codes(plan)
    assert all("offset" not in s for s in plan["main_recording"]["segments"])


def _make_multisegment_main(inventory, counts):
    main = _recording(inventory, 1)
    original = main["segments"][0]
    main["segments"] = [
        dict(original, index=i, sample_count=n) for i, n in enumerate(counts)
    ]
    main["declared_segment_count"] = len(counts)
    for link in inventory["relationships"]:
        if link["kind"] == "sorting_source" and main["id"] in link.get(
            "checks", {}
        ):
            link["checks"][main["id"]]["sample_counts"] = False
    return main


@pytest.mark.parametrize("other_factor", [0, 0.5, 2])
def test_unique_sorted_segment_preserves_source_and_local_clock(
    inventory, other_factor
):
    count = _recording(inventory, 1)["segments"][0]["sample_count"]
    _make_multisegment_main(inventory, [int(count * other_factor), count])
    plan = _plan(inventory)
    assert _codes(plan) == {"main_shanks_without_sorting"}
    assert plan["main_segment"] == dict(
        raw_segment_index=1,
        analyzer_segment_index=0,
        sample_count=count,
        duration_s=count / 30000,
        evidence="unique_sample_count_with_source_checks",
    )
    assert plan["acquisition_header_id"] is not None


def test_repeated_lengths_do_not_select_a_segment_by_position(inventory):
    count = _recording(inventory, 1)["segments"][0]["sample_count"]
    _make_multisegment_main(inventory, [count, count])
    plan = _plan(inventory)
    assert plan["main_segment"] is None
    assert "segment_mapping_unverified" in _codes(plan)
    assert plan["status"] == "blocked"


def test_analyzer_groups_must_identify_the_same_segment(inventory):
    count = _recording(inventory, 1)["segments"][0]["sample_count"]
    _make_multisegment_main(inventory, [30000, count])
    analyzer = next(
        a
        for a in inventory["sortings"]
        if a["stream"] == STREAM and a["experiment"] == "experiment1"
    )
    analyzer["sample_counts"] = [30000]
    plan = _plan(inventory)
    assert plan["main_segment"] is None
    assert "segment_mapping_unverified" in _codes(plan)


@pytest.mark.parametrize(
    "check,value", [("sampling_frequency", False), ("contact_positions", None)]
)
def test_segment_length_never_waives_other_source_checks(
    inventory, check, value
):
    count = _recording(inventory, 1)["segments"][0]["sample_count"]
    main = _make_multisegment_main(inventory, [30000, count])
    link = next(
        r
        for r in inventory["relationships"]
        if r["kind"] == "sorting_source" and main["id"] in r.get("checks", {})
    )
    link["checks"][main["id"]][check] = value
    plan = _plan(inventory)
    assert (
        "sorting_source_conflict"
        if value is False
        else "sorting_source_unverified"
    ) in _codes(plan)


def test_acquisition_header_is_matched_by_identity_for_multiple_segments(
    inventory,
):
    count = _recording(inventory, 1)["segments"][0]["sample_count"]
    main = _make_multisegment_main(inventory, [30000, count])
    link = next(
        r
        for r in inventory["relationships"]
        if r["kind"] == "header_membership" and r["source"] == main["id"]
    )
    header = next(
        h
        for h in inventory["recorded_streams"]
        if h["id"] == link["candidates"][0]
    )
    other = dict(header, id="second-header", recording="recording2")
    inventory["recorded_streams"].append(other)
    link["candidates"].append(other["id"])
    for analyzer in inventory["sortings"]:
        if (
            analyzer["stream"] == STREAM
            and analyzer["experiment"] == "experiment1"
        ):
            analyzer["recording"] = "recording2"
    plan = _plan(inventory)
    assert _codes(plan) == {"main_shanks_without_sorting"}
    assert plan["acquisition_header_id"] == "second-header"
    # Duplicate identity cannot be disambiguated by segment length.
    header["recording"] = "recording2"
    assert "acquisition_recording_mapping_unverified" in _codes(
        _plan(inventory)
    )


@pytest.mark.parametrize(
    "field,value", [("sampling_frequency", 2500), ("channel_count", None)]
)
def test_selected_acquisition_header_metadata_must_agree(
    inventory, field, value
):
    main = _recording(inventory, 1)
    link = next(
        r
        for r in inventory["relationships"]
        if r["kind"] == "header_membership" and r["source"] == main["id"]
    )
    header = next(
        h
        for h in inventory["recorded_streams"]
        if h["id"] == link["candidates"][0]
    )
    header[field] = value
    plan = _plan(inventory)
    code = (
        "acquisition_header_metadata_unverified"
        if value is None
        else "acquisition_header_metadata_conflict"
    )
    assert code in _codes(plan)
    assert plan["acquisition_header_id"] is None


def test_unsupported_inventory_version_is_rejected(inventory):
    inventory["inventory_version"] = "2.0.0"
    with pytest.raises(ValueError, match="version"):
        _plan(inventory)


def test_an_explicit_surface_asset_is_not_a_main_candidate(inventory):
    original = _recording(inventory, 1)
    surface = dict(
        copy.deepcopy(original), asset="surface", id="surface-recording"
    )
    surface["segments"][0]["sample_count"] *= 2
    inventory["recordings"].append(surface)
    inventory["assets"].append(
        dict(
            key="surface",
            role="raw",
            name="surface",
            metadata_status="complete",
        )
    )
    plan = _plan(inventory, surface_finding="surface")
    assert plan["main_recording"]["id"] == original["id"]
    entry = next(
        r for r in plan["continuous_inputs"] if r["id"] == surface["id"]
    )
    assert entry["role"] == "surface"
    assert entry["selected"]


def test_inventory_order_does_not_choose_the_main_or_family(inventory):
    expected = _plan(inventory)
    for field in ("recordings", "sortings", "assets", "relationships"):
        inventory[field].reverse()
    actual = _plan(inventory)
    for field in (
        "main_recording",
        "analyzers",
        "sorting_family",
        "findings",
        "continuous_inputs",
    ):
        assert actual[field] == expected[field]


def test_returned_plan_does_not_alias_inventory_observations(inventory):
    before = copy.deepcopy(inventory)
    plan = _plan(inventory)
    plan["main_recording"]["segments"][0]["sample_count"] = 1
    plan["analyzers"][0]["sample_counts"][0] = 2
    assert inventory == before


def _np1_streams(inventory):
    paired = []
    for recording in inventory["recordings"]:
        if recording["stream"] == STREAM:
            recording["stream"] += "-AP"
            lfp = copy.deepcopy(recording)
            lfp.update(
                id=lfp["id"] + "-lfp",
                stream=STREAM + "-LFP",
                sampling_frequency=2500,
            )
            for segment in lfp["segments"]:
                segment["sample_count"] //= 12
            paired.append(lfp)
    for sorting in inventory["sortings"]:
        if sorting["stream"] == STREAM:
            sorting["stream"] += "-AP"
    inventory["recordings"].extend(paired)
    return paired


def test_np1_pairs_lfp_per_experiment_without_ranking_it_as_main(inventory):
    _np1_streams(inventory)
    plan = _plan(inventory)
    assert len(plan["main_candidates"]) == 11
    assert len(plan["continuous_inputs"]) == 11
    assert all(
        r["paired_lfp"] is not None and r["selected"]
        for r in plan["continuous_inputs"]
    )


def test_np1_missing_pair_is_reported_on_the_affected_block(inventory):
    paired = _np1_streams(inventory)
    main_lfp = next(r for r in paired if r["experiment"] == "experiment1")
    inventory["recordings"].remove(main_lfp)
    plan = _plan(inventory)
    main = next(r for r in plan["continuous_inputs"] if r["role"] == "main")
    assert main["selected"] is False
    assert main["exclusion_reason"] == "missing_or_ambiguous_lfp_pair"
    assert "continuous_input_unresolved" in _codes(plan)


def test_qc_does_not_merge_a_different_physical_probe(inventory):
    other = _recording(inventory, 5)
    config = copy.deepcopy(
        next(
            c
            for c in inventory["channel_configurations"]
            if c["id"] == other["channel_configuration"]
        )
    )
    config["id"] = "other-probe"
    config["probes"][0]["serial"] = "another-serial"
    inventory["channel_configurations"].append(config)
    other["channel_configuration"] = config["id"]
    plan = _plan(inventory)
    entry = next(
        r for r in plan["continuous_inputs"] if r["id"] == other["id"]
    )
    assert entry["selected"] is False
    assert entry["exclusion_reason"] == "different_probe_identity"


def test_unknown_source_assets_are_not_inferred_from_duration(inventory):
    asset = next(a for a in inventory["assets"] if a["role"] == "sorted")
    asset["source_keys"] = []
    plan = _plan(inventory)
    assert plan["status"] == "blocked"
    assert "main_source_asset_unknown" in _codes(plan)


def test_retained_geometry_supports_planned_channel_projection(inventory):
    from aind_ephys_ibl_gui_conversion.inventory import inventory_slice
    from aind_ephys_ibl_gui_conversion.planned_extraction import (
        _geometry_in_channel_order,
        validate_execution_plan,
    )

    for collection in ("ProbeA", "ProbeB-2", "ProbeC-4"):
        plan = build_conversion_plan(inventory, collection, "sort")
        scoped = inventory_slice(inventory, collection, "sort")
        validate_execution_plan(plan, scoped)
        configs = {c["id"]: c for c in scoped["channel_configurations"]}
        for row in scoped["recordings"] + scoped["sortings"]:
            count = (
                row.get("channel_count") or row["segments"][0]["channel_count"]
            )
            projected = _geometry_in_channel_order(
                configs[row["channel_configuration"]], count
            )
            assert len(projected) == count
