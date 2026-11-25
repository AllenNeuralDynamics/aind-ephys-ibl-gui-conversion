"""Saved selection controls sample, unit and channel space."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import spikeinterface as si
from probeinterface import Probe

from aind_ephys_ibl_gui_conversion import extract_continuous, extract_spikes
from aind_ephys_ibl_gui_conversion.conversion_plan import inventory_digest
from aind_ephys_ibl_gui_conversion.planned_extraction import (
    _live_geometry,
    prepare_extraction,
    resolve_reference,
)


def _recording(seconds, contacts, positions):
    rng = np.random.default_rng(17)
    rec = si.NumpyRecording(
        rng.normal(size=(int(seconds * 30000), len(contacts))).astype(
            "float32"
        ),
        sampling_frequency=30000,
    )
    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(
        positions,
        shapes="circle",
        shape_params={"radius": 5},
        shank_ids=[c[1] for c in contacts],
    )
    probe.set_contact_ids(contacts)
    probe.set_device_channel_indices(np.arange(len(contacts)))
    probe.annotate(serial_number="synthetic")
    rec.set_probe(probe)
    rec.set_channel_gains(1)
    rec.set_channel_offsets(0)
    return rec


@pytest.fixture
def planned(tmp_path, request):
    raw, sorted_root = tmp_path / "raw", tmp_path / "sort"
    raw.mkdir()
    sorted_root.mkdir()
    # Sparse physical shanks by default; split collections have one shank.
    shanks = getattr(request, "param", (2, 0))
    main = _recording(
        5,
        [f"s{s}e{e}" for s in shanks for e in (1, 0)],
        [[s * 250, e * 20] for s in shanks for e in (1, 0)],
    )
    first = shanks[0]
    secondary = _recording(
        6,
        [f"s{first}e2", f"s{first}e1"],
        [[first * 250, 40], [first * 250, 20]],
    )
    short = _recording(1, [f"s{first}e3"], [[first * 250, 60]])
    records, configs, continuous = [], [], []
    for name, rec, role in [
        ("main", main, "main"),
        ("secondary", secondary, "unknown"),
        ("short", short, "unknown"),
    ]:
        configs.append(dict(id=name, **_live_geometry(rec)))
        path = f"ecephys/{name}#probe.zarr"
        (raw / "ecephys").mkdir(exist_ok=True)
        rec.save(format="zarr", folder=raw / path, verbose=False)
        row = dict(
            id=name,
            asset="raw",
            path=path,
            stream="ProbeA",
            experiment=name,
            sampling_frequency=30000,
            channel_configuration=name,
            segments=[
                dict(
                    index=0,
                    sample_count=rec.get_num_samples(),
                    channel_count=rec.get_num_channels(),
                )
            ],
        )
        records.append(row)
        continuous.append(
            dict(**row, selected=True, role=role, paired_lfp=None)
        )
    sortings, analyzer_refs = [], []
    for i in range(len(shanks)):
        channels = [2 * i, 2 * i + 1]
        recording = main.select_channels(channels)
        times = np.array([30000, 90000]) if i == 0 else np.array([15000])
        sorting = si.NumpySorting.from_unit_dict({7: times}, 30000)
        analyzer = si.create_sorting_analyzer(
            sorting,
            recording,
            format="memory",
            sparse=False,
        )
        analyzer.compute("random_spikes", max_spikes_per_unit=20)
        analyzer.compute(
            "templates", ms_before=0.5 if i == 0 else 1.0, ms_after=0.5
        )
        analyzer.compute("spike_amplitudes")
        analyzer.compute("spike_locations", method="center_of_mass")
        analyzer.compute("quality_metrics", metric_names=["num_spikes"])
        path = f"postprocessed/group{i}#probe.zarr"
        (sorted_root / "postprocessed").mkdir(exist_ok=True)
        analyzer.save_as(format="zarr", folder=sorted_root / path)
        config_id = f"analyzer{i}"
        configs.append(dict(id=config_id, **_live_geometry(analyzer)))
        row = dict(
            id=config_id,
            asset="sort",
            path=path,
            group=str(i),
            sampling_frequency=30000,
            sample_counts=[150000],
            channel_count=2,
            channel_ids=analyzer.channel_ids.tolist(),
            unit_count=1,
            channel_configuration=config_id,
        )
        sortings.append(row)
        analyzer_refs.append(row)
    inv = dict(
        recordings=records, sortings=sortings, channel_configurations=configs
    )
    plan = dict(
        plan_version="1.0.0",
        inventory_digest=inventory_digest(inv),
        execution_enabled=False,
        status="ready",
        findings=[],
        scope=dict(collection="ProbeA"),
        main_recording=records[0],
        sorting_family="family",
        sorting_families=[
            dict(id="family", analyzer_ids=[a["id"] for a in sortings])
        ],
        continuous_inputs=continuous,
        analyzers=analyzer_refs,
    )
    roots = dict(raw=raw, sort=sorted_root)
    return plan, inv, roots, tmp_path / "output"


def _prepare(planned):
    return prepare_extraction(*planned)


@pytest.mark.parametrize("planned", [(0,), (1,), (2,), (3,)], indirect=True)
def test_quadbase_spikes_remain_visible_to_local_shank_zero(planned):
    execution = _prepare(planned)
    extract_spikes(Path("unused"), Path("unused"), execution=execution)
    output = execution.stream.output_folder
    spike_shanks = np.load(output / "spike_shank_indices.npy")
    unit_shanks = np.load(output / "unit_shank_indices.npy")
    # The GUI normalizes this single physical shank to local index zero and
    # selects spikes with this predicate. No producer IDs may leak into it.
    physical = execution.channel_table.shank_ind
    assert len(np.unique(physical)) == 1
    assert spike_shanks.size > 0
    assert np.count_nonzero(spike_shanks == 0) == spike_shanks.size
    np.testing.assert_array_equal(unit_shanks, np.zeros(len(unit_shanks)))


def test_planned_spikes_use_global_units_rows_and_local_shanks(planned):
    execution = _prepare(planned)
    extract_spikes(Path("unused"), Path("unused"), execution=execution)
    output = execution.stream.output_folder
    load = lambda name: np.load(output / name)  # noqa: E731
    np.testing.assert_array_equal(load("spikes.times.npy"), [0.5, 1, 3])
    np.testing.assert_array_equal(load("spikes.clusters.npy"), [1, 0, 0])
    np.testing.assert_array_equal(load("spike_shank_indices.npy"), [0, 1, 1])
    np.testing.assert_array_equal(load("unit_shank_indices.npy"), [1, 0])
    assert load("spikes.amps.npy").shape == (3,)
    assert load("clusters.waveforms.npy").shape == (2, 45, 1)
    assert np.isnan(load("clusters.waveforms.npy")[0, :15]).all()
    assert load("clusters.waveformTimes.npy")[30] == 0
    a0, a1 = [a for _, a, _ in execution.analyzers]
    amps0 = a0.get_extension("spike_amplitudes").get_data()
    amps1 = a1.get_extension("spike_amplitudes").get_data()
    np.testing.assert_allclose(load("spikes.amps.npy"), -np.r_[amps1, amps0])
    y0 = a0.get_extension("spike_locations").get_data()["y"]
    y1 = a1.get_extension("spike_locations").get_data()["y"]
    np.testing.assert_allclose(load("spikes.depths.npy"), np.r_[y1, y0])
    peaks = load("clusters.channels.npy")
    assert peaks[0] in [0, 1] and peaks[1] in [2, 3]
    frame = pd.read_csv(output / "clusters.metrics.csv")
    assert frame.cluster_id.tolist() == [0, 1]
    assert frame.source_unit_id.tolist() == [7, 7]
    assert frame.analyzer_id.nunique() == 2


def test_continuous_uses_same_rows_and_preserves_explicit_main(
    planned, monkeypatch
):
    import aind_ephys_ibl_gui_conversion.ephys as module

    execution = _prepare(planned)
    monkeypatch.setattr(
        module,
        "get_ecephys_stream_names",
        lambda *_: pytest.fail("Planned extraction must not enumerate Neo"),
    )
    extract_continuous(
        Path("unused"),
        Path("unused"),
        execution=execution,
        extra_metrics=(),
        num_parallel_jobs=1,
    )
    output = execution.stream.output_folder
    np.testing.assert_array_equal(
        np.load(output / "channels.localCoordinates.npy"),
        execution.channel_table.local_coordinates,
    )
    assert len(execution.stream.blocks) == 2
    assert execution.stream.main_block.recording_id == "main"
    assert execution.stream.blocks[1].source == "unknown"
    assert execution.skipped_blocks[0]["recording_id"] == "short"
    rows = json.loads((output / "band_corr/row_channels.json").read_text())
    assert {
        b["label"] for s in rows["shanks"].values() for b in s["blocks"]
    } == {"main", "unknown"}
    rms = np.load(output / "_iblqc_ephysTimeRmsAPMain.rms.npy")
    assert rms.shape[1] == 5
    assert np.isnan(rms[:, 4]).all()
    execution.write_receipt(output, "complete")
    receipt = json.loads((output / "ephys_extraction.json").read_text())
    assert receipt["execution_enabled"] is True
    assert receipt["plan"]["execution_enabled"] is False


@pytest.mark.parametrize(
    "change", ["hash", "conflict", "geometry", "samples", "path"]
)
def test_invalid_or_changed_inputs_fail_before_output(planned, change):
    plan, inv, roots, output = planned
    if change == "hash":
        plan["inventory_digest"] = "stale"
    elif change == "conflict":
        plan["findings"] = [dict(code="shank_coverage_unknown")]
    elif change == "geometry":
        inv["channel_configurations"][0]["probes"][0]["contact_positions"][0][
            1
        ] += 1
        plan["inventory_digest"] = inventory_digest(inv)
    elif change == "samples":
        inv["recordings"][0]["segments"][0]["sample_count"] += 1
        plan["inventory_digest"] = inventory_digest(inv)
    else:
        plan["analyzers"] = copy.deepcopy(plan["analyzers"])
        plan["analyzers"][0]["path"] = "../elsewhere"
    with pytest.raises(ValueError):
        _prepare(planned)
    assert not output.exists()


def test_reference_rejects_symlink_escape(tmp_path):
    root = tmp_path / "asset"
    root.mkdir()
    (root / "outside").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        resolve_reference(dict(asset="a", path="outside"), dict(a=root))


def test_missing_shank_converts_available_main_sorting_only(planned):
    plan = planned[0]
    plan["analyzers"] = plan["analyzers"][:1]
    plan["sorting_families"][0]["analyzer_ids"] = ["analyzer0"]
    plan["status"] = "incomplete"
    plan["findings"] = [
        dict(code="main_shanks_without_sorting", physical_shanks=[0])
    ]
    execution = _prepare(planned)
    extract_spikes(Path("unused"), Path("unused"), execution=execution)
    output = execution.stream.output_folder
    assert set(np.load(output / "unit_shank_indices.npy")) == {1}
    assert set(execution.channel_table.shank_ind) == {0, 2}


def test_collection_audit_rejects_physical_ids_in_local_spike_contract(
    planned,
):
    from aind_ephys_ibl_gui_conversion.output_validation import (
        audit_collection,
    )

    execution = _prepare(planned)
    extract_continuous(
        Path("unused"),
        Path("unused"),
        execution=execution,
        extra_metrics=(),
        num_parallel_jobs=1,
    )
    extract_spikes(Path("unused"), Path("unused"), execution=execution)
    output = execution.stream.output_folder
    execution.write_receipt(output, "complete")
    plan = {
        **execution.plan,
        "main_segment": {"duration_s": 5.0},
        "shank_coverage": {
            "recorded": [0, 2],
            "sorted": [0, 2],
            "missing": [],
        },
    }
    result = audit_collection(output, plan)
    assert result["unit_shanks"] == [0, 1]
    assert result["physical_unit_shanks"] == [0, 2]
    peaks = np.load(output / "clusters.channels.npy")
    np.save(
        output / "unit_shank_indices.npy",
        execution.channel_table.shank_ind[peaks],
    )
    with pytest.raises(AssertionError):
        audit_collection(output, plan)


def test_unknown_secondary_is_not_an_implicit_surface(planned):
    execution = _prepare(planned)
    with pytest.raises(ValueError, match="no matching blocks"):
        extract_continuous(
            Path("unused"),
            Path("unused"),
            execution=execution,
            coherence_block_source="surface",
        )
    assert not planned[-1].exists()


def test_explicit_probe_shank_ids_survive_without_contact_prefixes():
    from aind_ephys_ibl_gui_conversion.channel_metadata import (
        normalized_channel_groups,
    )

    rec = _recording(1, ["s2e0", "s2e1"], [[500, 0], [500, 20]])
    probe = rec.get_probe()
    probe.set_contact_ids(["e0", "e1"])
    rec.set_probe(probe)
    np.testing.assert_array_equal(normalized_channel_groups(rec), [2, 2])


def test_secondary_segments_stay_separate_with_local_clocks(planned):
    plan, inv, roots, _ = planned
    row = inv["recordings"][1]
    path = roots["raw"] / row["path"]
    original = si.read_zarr(path)
    rec = si.NumpyRecording(
        [
            np.zeros((150000, 2), dtype="float32"),
            np.zeros((180000, 2), dtype="float32"),
        ],
        30000,
    )
    rec.set_probe(original.get_probe())
    shutil.rmtree(path)
    rec.save(format="zarr", folder=path, verbose=False)
    row["segments"][:] = [
        dict(index=i, sample_count=n, channel_count=2)
        for i, n in enumerate((150000, 180000))
    ]
    plan["inventory_digest"] = inventory_digest(inv)
    execution = _prepare(planned)
    blocks = execution.stream.blocks
    assert [(b.recording_id, b.segment_index) for b in blocks] == [
        ("main", 0),
        ("secondary", 0),
        ("secondary", 1),
    ]
    assert [b.duration for b in blocks] == [5, 5, 6]
    assert all(b.recording.get_times()[0] == 0 for b in blocks)


def test_paired_lfp_uses_exact_reference_and_own_rate(planned):
    plan, inv, roots, _ = planned
    main = si.read_zarr(roots["raw"] / inv["recordings"][0]["path"])
    rec = si.NumpyRecording(np.zeros((12500, 4), dtype="float32"), 2500)
    rec.set_probe(main.get_probe())
    row = dict(
        inv["recordings"][0],
        id="lfp",
        path="ecephys/lfp.zarr",
        sampling_frequency=2500,
        segments=[dict(index=0, sample_count=12500, channel_count=4)],
    )
    rec.save(format="zarr", folder=roots["raw"] / row["path"], verbose=False)
    inv["recordings"].append(row)
    plan["continuous_inputs"][0]["paired_lfp"] = row
    plan["inventory_digest"] = inventory_digest(inv)
    execution = _prepare(planned)
    block = execution.stream.main_block
    assert block.is_1_0
    assert block.lfp_recording.get_sampling_frequency() == 2500
    assert block.lfp_recording.get_num_samples() == 12500


def test_timestamp_pair_preserves_ap_origin_and_revalidates_evidence(planned):
    from aind_ephys_ibl_gui_conversion.inventory import LocalMetadataStore
    from aind_ephys_ibl_gui_conversion.recording_timing import (
        observe_timing,
        validate_pair,
    )

    plan, inv, roots, _ = planned
    row = inv["recordings"][0]
    path = roots["raw"] / row["path"]
    original = si.read_zarr(path)
    data, probe = original.get_traces(), original.get_probe()
    ap = si.NumpyRecording(data, 30000)
    ap.set_probe(probe)
    ap.set_channel_gains(1)
    ap.set_channel_offsets(0)
    ap.set_times(20 + np.arange(len(data)) / 30000)
    shutil.rmtree(path)
    ap.save(format="zarr", folder=path, verbose=False)
    lf = si.NumpyRecording(np.zeros((12476, 4), dtype="float32"), 2500)
    lf.set_probe(probe)
    lf.set_channel_gains(1)
    lf.set_channel_offsets(0)
    lf.set_times(20.0096 + np.arange(12476) / 2500)
    lf_row = dict(
        row,
        id="lfp",
        path="ecephys/lfp.zarr",
        sampling_frequency=2500,
        segments=[dict(index=0, sample_count=12476, channel_count=4)],
    )
    lf.save(format="zarr", folder=roots["raw"] / lf_row["path"], verbose=False)
    store = LocalMetadataStore(roots["raw"])
    row["segments"][0]["timing"] = observe_timing(
        store, row["path"], 0, len(data), 30000
    )
    lf_row["segments"][0]["timing"] = observe_timing(
        store, lf_row["path"], 0, 12476, 2500
    )
    inv["recordings"].append(lf_row)
    reference = plan["continuous_inputs"][0]
    reference["paired_lfp"] = lf_row
    reference["lfp_timing"] = {
        "0": validate_pair(
            row["segments"][0], lf_row["segments"][0], 30000, 2500
        )
    }
    plan["inventory_digest"] = inventory_digest(inv)
    execution = _prepare(planned)
    block = execution.stream.main_block
    assert block.recording.get_times()[0] == 0
    assert block.lfp_timing["lfp_offset_s"] == pytest.approx(0.0096)
    reference["lfp_timing"]["0"]["lfp_offset_s"] = 0
    with pytest.raises(ValueError, match="Plan AP/LFP timing"):
        _prepare(planned)


def _multisegment_main(planned):
    plan, inv, roots, _ = planned
    row = inv["recordings"][0]
    path = roots["raw"] / row["path"]
    original = si.read_zarr(path)
    traces = original.get_traces()
    rec = si.NumpyRecording(
        [np.zeros((180000, 4), dtype="float32"), traces], 30000
    )
    rec.set_probe(original.get_probe())
    rec.set_channel_gains(1)
    rec.set_channel_offsets(0)
    shutil.rmtree(path)
    rec.save(format="zarr", folder=path, verbose=False)
    row["segments"][:] = [
        dict(index=i, sample_count=n, channel_count=4)
        for i, n in enumerate((180000, 150000))
    ]
    row["declared_segment_count"] = 2
    for analyzer in inv["sortings"]:
        analyzer["segment_count"] = 1
    plan["main_segment"] = dict(
        raw_segment_index=1,
        analyzer_segment_index=0,
        sample_count=150000,
        duration_s=5.0,
        evidence="unique_sample_count_with_source_checks",
    )
    plan["inventory_digest"] = inventory_digest(inv)


def test_single_sorted_segment_controls_qc_and_spike_clock(planned):
    _multisegment_main(planned)
    execution = _prepare(planned)
    blocks = execution.stream.blocks
    assert blocks[0].selected_main
    assert blocks[0].segment_index == 1
    assert blocks[0].duration == 5
    assert blocks[1].recording_id == "secondary"
    assert any(
        s["recording_id"] == "main"
        and s["segment_index"] == 0
        and s["reason"] == "not_selected_main_segment"
        for s in execution.skipped_blocks
    )
    extract_continuous(
        Path("unused"),
        Path("unused"),
        execution=execution,
        extra_metrics=(),
        num_parallel_jobs=1,
    )
    extract_spikes(Path("unused"), Path("unused"), execution=execution)
    output = execution.stream.output_folder
    np.testing.assert_array_equal(
        np.load(output / "spikes.times.npy"), [0.5, 1, 3]
    )
    rms = np.load(output / "_iblqc_ephysTimeRmsAPMain.rms.npy")
    assert np.all(rms[:, :4] > 0)
    assert execution.stream.main_block is blocks[0]
    execution.write_receipt(output, "complete")
    receipt = json.loads((output / "ephys_extraction.json").read_text())
    assert receipt["plan"]["main_segment"]["raw_segment_index"] == 1


def test_worker_rejects_an_inconsistent_saved_segment_choice(planned):
    _multisegment_main(planned)
    planned[0]["main_segment"]["raw_segment_index"] = 0
    with pytest.raises(ValueError, match="Saved main segment"):
        _prepare(planned)
    assert not planned[-1].exists()
