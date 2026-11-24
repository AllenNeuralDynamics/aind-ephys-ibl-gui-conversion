"""Discovery observations remain independent of conversion selection."""

import copy
import gzip
import json
from pathlib import Path

import pytest

from aind_ephys_ibl_gui_conversion.inventory import (
    LocalMetadataStore,
    build_inventory,
    inventory_slice,
    relative_path,
    write_inventory,
)

STREAM = "Record Node 103#Neuropix-PXI-111.ProbeA"


class MemoryStore:
    """Model a metadata-only remote store and reject unexpected reads."""

    def __init__(self, files):
        self.files = files
        self.reads = []

    def directories(self, path):
        prefix = "" if path == "." else path.rstrip("/") + "/"
        return sorted(
            {
                name[len(prefix) :].split("/")[0]
                for name in self.files
                if name.startswith(prefix) and "/" in name[len(prefix) :]
            }
        )

    def read_json(self, path):
        self.reads.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        assert path.endswith(
            (".zattrs", ".zarray", ".zmetadata", ".oebin", ".json")
        )
        return copy.deepcopy(self.files[path])


@pytest.fixture
def retained():
    path = Path(__file__).parent / "fixtures/inventory_858398.json.gz"
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def sources(files):
    return [
        ({"key": "raw", "role": "raw"}, MemoryStore(files["raw"])),
        (
            {"key": "sort", "role": "sorted", "source_keys": ["raw"]},
            MemoryStore(files["sorted"]),
        ),
    ]


def test_retained_heterogeneous_recording(retained):
    inventory = build_inventory(sources(retained))
    # 43 probe blocks and 11 auxiliary NI-DAQ blocks are observed equally.
    assert len(inventory["recorded_streams"]) == 54
    assert len(inventory["recordings"]) == 54
    assert len(inventory["sortings"]) == 40
    assert inventory["metadata_complete"]
    assert all(r["acquisition_role"] is None for r in inventory["recordings"])
    joins = [
        r for r in inventory["relationships"] if r["kind"] == "sorting_source"
    ]
    assert len(joins) == 40
    assert all(r["status"] == "candidate" for r in joins)
    assert all(
        all(v is True for v in checks.values())
        for r in joins
        for checks in r["checks"].values()
    )
    first = [
        s
        for s in inventory["sortings"]
        if s["stream"] == STREAM and s["experiment"] == "experiment1"
    ]
    assert {s["group"] for s in first} == {"0", "1"}
    assert {s["unit_count"] for s in first} == {111, 152}
    # Missing sorting is observed, without declaring an acquisition purpose.
    missing = {
        r["id"]
        for r in inventory["recordings"]
        if r["stream"] == STREAM
        and r["experiment"]
        in {"experiment2", "experiment3", "experiment10", "experiment11"}
    }
    reported = {
        f["record"]
        for f in inventory["findings"]
        if f["code"] == "no_sorting_observed"
    }
    assert missing <= reported


def test_local_remote_parity_and_relocation(retained, tmp_path):
    expected = build_inventory(sources(retained))
    local = []
    for descriptor, remote in sources(retained):
        root = tmp_path / "relocated" / descriptor["key"]
        for name, content in remote.files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content))
        local.append((descriptor, LocalMetadataStore(root)))
    assert build_inventory(local) == expected
    path = tmp_path / "output/inventory.json"
    write_inventory(expected, path)
    assert json.loads(path.read_text()) == expected
    assert str(tmp_path) not in path.read_text()


def test_collection_slice_is_self_contained(retained):
    inventory = build_inventory(sources(retained))
    sliced = inventory_slice(inventory, "ProbeB-2")
    assert len(sliced["recordings"]) == 4
    assert len(sliced["sortings"]) == 4
    ids = {
        r["id"]
        for key in ("recordings", "recorded_streams", "sortings")
        for r in sliced[key]
    }
    assert all(
        r["source"] in ids and set(r["candidates"]) <= ids
        for r in sliced["relationships"]
    )
    configs = {c["id"] for c in sliced["channel_configurations"]}
    assert all(
        r["channel_configuration"] in configs
        for key in ("recordings", "sortings")
        for r in sliced[key]
    )


def test_missing_compressed_metadata_is_not_valid_absence(retained):
    broken = copy.deepcopy(retained)
    target = next(
        k
        for k in broken["raw"]
        if k.endswith(".zattrs") and "experiment1_" in k
    )
    del broken["raw"][target]
    inventory = build_inventory(sources(broken))
    assert not inventory["metadata_complete"]
    assert any(
        f["code"] == "missing_metadata" and f["path"] == target
        for f in inventory["findings"]
    )
    assert any(
        f["code"] == "header_without_compressed_recording"
        for f in inventory["findings"]
    )


def test_conflicting_geometry_is_reported(retained):
    files = copy.deepcopy(retained)
    first = next(iter(files["sorted"].values()))["metadata"]
    probe = first["recording_info/.zattrs"]["probegroup"]["probes"][0]
    probe["contact_positions"][0][1] += 123
    inventory = build_inventory(sources(files))
    assert any(
        f["code"] == "sorting_source_conflict" for f in inventory["findings"]
    )


def test_multiple_raw_candidates_stay_ambiguous(retained):
    entries = sources(retained)
    entries[1][0]["source_keys"].append("other-raw")
    entries.append(
        ({"key": "other-raw", "role": "raw"}, MemoryStore(retained["raw"]))
    )
    inventory = build_inventory(entries)
    assert all(
        r["status"] == "ambiguous"
        for r in inventory["relationships"]
        if r["kind"] == "sorting_source"
    )


def test_unavailable_source_is_explicit_and_does_not_hide_other_sources(
    retained,
):
    entries = sources(retained)
    entries.append(({"key": "unavailable", "role": "raw"}, None))
    inventory = build_inventory(entries)
    assert len(inventory["sortings"]) == 40
    assert not inventory["metadata_complete"]
    assert any(
        f["code"] == "source_unavailable" for f in inventory["findings"]
    )


def test_no_payload_access_and_deterministic_order(retained):
    entries = sources(retained)
    inventory = build_inventory(entries)
    assert build_inventory(entries[::-1]) == inventory
    assert all(
        not name.endswith("/0") for _, store in entries for name in store.reads
    )


@pytest.mark.parametrize(
    "path", ["../outside", "/absolute", "a/../../b", "a\\b"]
)
def test_reject_paths_outside_asset(path):
    with pytest.raises(ValueError):
        relative_path(path)


def test_symlink_cannot_escape_asset(tmp_path):
    root = tmp_path / "asset"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        LocalMetadataStore(root).read_json("escape/other.json")


def test_multiple_segments_and_recordings_are_preserved(retained):
    files = copy.deepcopy(retained)
    path = f"ecephys/ecephys_compressed/experiment10_{STREAM}.zarr"
    attrs = files["raw"][path + "/.zattrs"]
    attrs["num_segments"] = 2
    files["raw"][path + "/traces_seg1/.zarray"] = {"shape": [300, 384]}
    header = (
        "ecephys/ecephys_clipped/Record Node 103/"
        "experiment10/recording1/structure.oebin"
    )
    files["raw"][header.replace("recording1", "recording7")] = files["raw"][
        header
    ]
    inventory = build_inventory(sources(files))
    record = next(r for r in inventory["recordings"] if r["path"] == path)
    assert [s["index"] for s in record["segments"]] == [0, 1]
    candidates = next(
        r["candidates"]
        for r in inventory["relationships"]
        if r["source"] == record["id"]
    )
    assert len(candidates) == 2
    # No inferred segment-to-recording order or renumbering is introduced.
    assert all("recording" not in s for s in record["segments"])


def test_unconsolidated_analyzer_metadata(retained):
    files = copy.deepcopy(retained)
    key = next(iter(files["sorted"]))
    meta = files["sorted"].pop(key)["metadata"]
    base = key.removesuffix(".zmetadata")
    files["sorted"].update({base + k: v for k, v in meta.items()})
    inventory = build_inventory(sources(files))
    assert len(inventory["sortings"]) == 40
    assert inventory["metadata_complete"]


def test_malformed_analyzer_does_not_hide_later_siblings(retained):
    files = copy.deepcopy(retained)
    key = next(iter(files["sorted"]))
    files["sorted"][key] = {"metadata": []}
    inventory = build_inventory(sources(files))
    assert len(inventory["sortings"]) == 39
    assert not inventory["metadata_complete"]
    assert any(
        f["code"] == "record_scan_incomplete" for f in inventory["findings"]
    )


@pytest.mark.parametrize(
    "asset_name,expected", [("raw", True), ("wrong-asset", False)]
)
def test_preprocessing_source_is_read_and_checked_against_asset(
    retained, asset_name, expected
):
    files = copy.deepcopy(retained)
    initial = build_inventory(sources(files))
    analyzer = initial["sortings"][0]
    raw = next(
        r
        for r in initial["recordings"]
        if r["experiment"] == analyzer["experiment"]
        and r["stream"] == analyzer["stream"]
    )
    basename = analyzer["path"].split("/", 1)[1].removesuffix(".zarr")
    filename = f"preprocessed/{basename}.json"
    files["sorted"][filename] = {
        "class": "spikeinterface.core.segmentutils.SelectSegmentRecording",
        "kwargs": {
            "segment_indices": [0],
            "recording": {
                "class": (
                    "spikeinterface.core.zarrextractors.ZarrRecordingExtractor"
                ),
                "kwargs": {
                    "folder_path": f"../data/{asset_name}/{raw['path']}"
                },
            },
        },
    }
    inventory = build_inventory(sources(files))
    observed = next(
        a for a in inventory["sortings"] if a["id"] == analyzer["id"]
    )
    assert observed["preprocessing_source"]["metadata_path"] == filename
    assert observed["preprocessing_source"]["segment_indices"] == [0]
    link = next(
        link
        for link in inventory["relationships"]
        if link["source"] == analyzer["id"]
    )
    assert link["checks"][raw["id"]]["preprocessing_source_path"] is expected


def test_unknown_shank_metadata_is_not_filled_from_group(retained):
    files = copy.deepcopy(retained)
    meta = next(iter(files["sorted"].values()))["metadata"]
    meta["recording_info/.zattrs"]["probegroup"] = {}
    inventory = build_inventory(sources(files))
    record = next(
        r
        for r in inventory["sortings"]
        if r["channel_configuration"]
        in {
            c["id"]
            for c in inventory["channel_configurations"]
            if not c["probes"] and c["asset"] == "sort"
        }
    )
    link = next(
        r for r in inventory["relationships"] if r["source"] == record["id"]
    )
    assert all(c["contact_subset"] is None for c in link["checks"].values())
