"""Validate planned extraction against read-only mounted production assets."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import traceback
from pathlib import Path

import spikeinterface as si

from aind_ephys_ibl_gui_conversion.conversion_plan import build_conversion_plan
from aind_ephys_ibl_gui_conversion.inventory import (
    LocalMetadataStore,
    build_inventory,
    inventory_slice,
    write_inventory,
)
from aind_ephys_ibl_gui_conversion.output_validation import audit_collection


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
