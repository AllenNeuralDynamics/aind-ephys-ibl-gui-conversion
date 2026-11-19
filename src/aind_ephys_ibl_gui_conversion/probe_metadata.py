"""Normalize modern and legacy JSON probe identity without loading objects."""

from __future__ import annotations

from typing import Any


def probe_metadata(
    group: Any, recording_annotations: Any = None
) -> list[dict[str, Any]]:
    """Restore ``probes_info`` identity, retaining conflicts as unknown."""
    probes = group.get("probes", []) if isinstance(group, dict) else []
    annotations = recording_annotations or {}
    legacy = annotations.get("probes_info", [])
    if not isinstance(legacy, list) or len(legacy) != len(probes):
        legacy = [{} for _ in probes]
    result = []
    for probe, old in zip(probes, legacy, strict=True):
        explicit = probe.get("annotations") or {}
        old = old if isinstance(old, dict) else {}
        identity = {}
        for target, field in (
            ("serial", "serial_number"),
            ("model", "part_number"),
        ):
            a, b = explicit.get(field), old.get(field)
            identity[target] = a if a is not None else b
            if a is not None and b is not None and str(a) != str(b):
                identity[target] = None
                identity[target + "_conflict"] = [a, b]
        result.append(
            dict(
                **identity,
                contact_ids=probe.get("contact_ids"),
                contact_positions=probe.get("contact_positions"),
                device_channel_indices=probe.get("device_channel_indices"),
                shank_ids=probe.get("shank_ids"),
                units=probe.get("si_units"),
            )
        )
    return result
