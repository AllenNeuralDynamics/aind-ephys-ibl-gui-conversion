"""Production metadata exercises the original multisegment case."""

import gzip
import json
from pathlib import Path

from aind_ephys_ibl_gui_conversion.conversion_plan import build_conversion_plan
from aind_ephys_ibl_gui_conversion.inventory import (
    build_inventory,
    inventory_slice,
)
from aind_ephys_ibl_gui_conversion.planned_extraction import (
    _geometry_in_channel_order,
    validate_execution_plan,
)
from aind_ephys_ibl_gui_conversion.probe_metadata import probe_metadata

from .test_inventory import sources


def test_original_multisegment_regression_selects_longest_recording_family():
    path = Path(__file__).parent / "fixtures/inventory_762286.json.gz"
    with gzip.open(path, "rt") as handle:
        inventory = build_inventory(sources(json.load(handle)))
    plan = build_conversion_plan(inventory, "ProbeA", "sort")
    assert inventory["metadata_complete"]
    assert plan["status"] == "ready"
    assert plan["findings"] == []
    assert len(plan["sorting_families"]) == 2
    assert len(plan["analyzers"]) == 1
    assert "recording1.zarr" in plan["analyzers"][0]["path"]
    assert plan["main_segment"]["raw_segment_index"] == 0
    assert plan["main_segment"]["sample_count"] == 93133656
    assert plan["shank_coverage"] == dict(recorded=[0], sorted=[0], missing=[])
    (primary,) = plan["continuous_inputs"]
    assert primary["segment_indices"] == [0]
    assert primary["paired_lfp"] is not None
    scoped = inventory_slice(inventory, "ProbeA", "sort")
    validate_execution_plan(plan, scoped)
    configs = {c["id"]: c for c in inventory["channel_configurations"]}
    for row in inventory["recordings"] + inventory["sortings"]:
        count = row.get("channel_count") or row["segments"][0]["channel_count"]
        config = configs[row["channel_configuration"]]
        assert config["probes"][0]["serial"] == "22112107221"
        assert len(_geometry_in_channel_order(config, count)) == count


def test_explicit_and_legacy_probe_serial_conflict_stays_unknown():
    group = dict(probes=[dict(annotations=dict(serial_number="modern"))])
    legacy = dict(probes_info=[dict(serial_number="legacy")])
    (probe,) = probe_metadata(group, legacy)
    assert probe["serial"] is None
    assert probe["serial_conflict"] == ["modern", "legacy"]


def test_mismatched_legacy_probe_count_does_not_assign_identity_by_guessing():
    group = dict(probes=[dict(annotations={}), dict(annotations={})])
    probes = probe_metadata(
        group, dict(probes_info=[dict(serial_number="one")])
    )
    assert all(p["serial"] is None for p in probes)
