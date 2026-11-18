"""Validate planned extraction against read-only mounted production assets."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface as si

from aind_ephys_ibl_gui_conversion.conversion_plan import build_conversion_plan
from aind_ephys_ibl_gui_conversion.inventory import (
    LocalMetadataStore,
    build_inventory,
    inventory_slice,
    write_inventory,
)


def audit_collection(folder: Path, plan: dict) -> dict:
    """Check the consumer-facing joins and preserve quantitative evidence."""
    load = lambda name: np.load(folder / name, mmap_mode="r")  # noqa: E731
    times = load("spikes.times.npy")
    clusters = load("spikes.clusters.npy")
    channels = load("clusters.channels.npy")
    coordinates = load("channels.localCoordinates.npy")
    unit_shanks = load("unit_shank_indices.npy")
    channel_shanks = load("channels.shankInd.npy")
    spike_shanks = load("spike_shank_indices.npy")
    metrics = pd.read_csv(folder / "clusters.metrics.csv")
    duration = plan["main_segment"]["duration_s"]
    assert times.ndim == 1 and np.isfinite(times).all()
    assert np.all(np.diff(times) >= 0)
    assert not len(times) or (times[0] >= 0 and times[-1] < duration)
    assert len(times) == len(clusters) == len(spike_shanks)
    for name in ("amps", "depths"):
        assert load(f"spikes.{name}.npy").shape == times.shape
    assert len(channels) == len(unit_shanks) == len(metrics)
    assert metrics.cluster_id.tolist() == list(range(len(channels)))
    assert metrics[["analyzer_id", "source_unit_id"]].duplicated().sum() == 0
    assert not len(clusters) or (
        clusters.min() >= 0 and clusters.max() < len(channels)
    )
    assert np.all((channels >= 0) & (channels < len(coordinates)))
    np.testing.assert_array_equal(unit_shanks, channel_shanks[channels])
    np.testing.assert_array_equal(spike_shanks, unit_shanks[clusters])
    coverage = plan["shank_coverage"]
    assert set(unit_shanks) <= set(coverage["sorted"])
    assert not set(unit_shanks) & set(coverage["missing"])
    expected = sum(a["unit_count"] for a in plan["analyzers"])
    assert len(channels) == expected
    rows = json.loads((folder / "band_corr/row_channels.json").read_text())
    assert rows["matrix_rows"] == list(range(len(coordinates)))
    for shank, entry in rows["shanks"].items():
        assert np.all(channel_shanks[entry["rows"]] == int(shank))
    for path in (folder / "band_corr").glob("*_mean_corr.npy"):
        if "_shank" not in path.name:
            assert (
                np.load(path, mmap_mode="r").shape == (len(coordinates),) * 2
            )
    receipt = json.loads((folder / "ephys_extraction.json").read_text())
    assert receipt["status"] == "complete"
    assert sum(b["selected_main"] for b in receipt["blocks"]) == 1
    return dict(
        status="passed",
        spike_count=len(times),
        unit_count=len(channels),
        channel_rows=len(coordinates),
        unit_shanks=np.unique(unit_shanks).tolist(),
        channel_shanks=np.unique(channel_shanks).tolist(),
        shank_coverage=coverage,
        main_segment=plan["main_segment"],
        first_spike_s=float(times[0]) if len(times) else None,
        last_spike_s=float(times[-1]) if len(times) else None,
        qc_blocks=receipt["blocks"],
        skipped_blocks=receipt["skipped_blocks"],
    )


def main() -> None:
    """Run the worker wrapper and audit each output collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--sorted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collection", action="append", required=True)
    parser.add_argument("--mouse", default="858398")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    from aind_ibl_ephys_alignment_preprocessing.discovery import (
        prepare_result_dirs,
    )
    from aind_ibl_ephys_alignment_preprocessing.ephys import (
        run_ephys_for_stream,
    )

    logging.basicConfig(level=logging.INFO)
    args.output.mkdir(parents=True, exist_ok=True)
    raw_name = json.loads((args.raw / "data_description.json").read_text())[
        "name"
    ]
    sources = [
        (
            dict(key="raw", name=raw_name, role="raw"),
            LocalMetadataStore(args.raw),
        ),
        (
            dict(
                key="sort",
                name=args.sorted.name,
                role="sorted",
                source_keys=["raw"],
            ),
            LocalMetadataStore(args.sorted),
        ),
    ]
    inventory = build_inventory(sources)
    write_inventory(inventory, args.output / "ephys_inventory.json")
    multisegment = [
        dict(id=r["id"], path=r["path"], segments=r["segments"])
        for r in inventory["recordings"]
        if len(r["segments"]) > 1
    ]
    write_inventory(
        dict(recordings=multisegment),
        args.output / "multisegment_candidates.json",
    )
    provenance = dict(
        python=platform.python_version(),
        spikeinterface=si.__version__,
        conversion_commit=os.environ.get("IBL_CONVERSION_COMMIT"),
        preprocessing_commit=os.environ.get("IBL_PREPROCESSING_COMMIT"),
    )
    report = dict(provenance=provenance, collections={})
    outputs = prepare_result_dirs(args.mouse, args.output / "workers")
    failures = []
    for collection in args.collection:
        scoped = inventory_slice(inventory, collection, "sort")
        plan = build_conversion_plan(inventory, collection, "sort")
        diagnostics = args.output / "decisions" / collection
        write_inventory(scoped, diagnostics / "ephys_inventory.json")
        write_inventory(plan, diagnostics / "ephys_conversion_plan.json")
        try:
            run_ephys_for_stream(
                args.sorted.name,
                raw_name,
                collection,
                None,
                outputs,
                args.raw.parent,
                num_parallel_jobs=args.jobs,
                conversion_decisions=(plan, scoped),
            )
            folder = outputs.tracks_root.parent / raw_name / collection
            report["collections"][collection] = audit_collection(folder, plan)
        except Exception:
            failures.append(collection)
            report["collections"][collection] = dict(
                status="failed", traceback=traceback.format_exc()
            )
            logging.exception("Validation failed for %s", collection)
        write_inventory(report, args.output / "validation_report.json")
    if failures:
        raise RuntimeError(f"Validation failed for {failures}")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
