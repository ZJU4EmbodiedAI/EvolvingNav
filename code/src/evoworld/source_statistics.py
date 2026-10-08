"""Summarize local human-trace assets without copying trajectories.

The benchmark consumes aggregate counts and category histograms only.  This
module deliberately records missing sources rather than silently treating an
empty directory as calibrated evidence.
"""
from __future__ import annotations

import json
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .sources import casas_histogram


_EXPECTED = ("CASAS", "ARAS", "OPPORTUNITY", "HD_EPIC", "ParaHome", "HOMER_PLUS")


def _jsonl_count(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _source_records_root(root: Path) -> Path | None:
    candidates = sorted(
        root.glob("P4D-HSSD/p4d_hssd_*/source_records"),
        key=lambda path: path.parent.name,
    )
    return candidates[-1] if candidates else None


def build_source_statistics(root: str | Path) -> dict[str, Any]:
    """Build a provenance manifest for available local source statistics.

    Only aggregate statistics are returned; ``trajectory_copied`` is always
    false as a guard against accidentally using raw trajectories in episodes.
    """
    root = Path(root)
    sources: dict[str, Any] = {}
    hour_counts = Counter()
    category_hours = defaultdict(Counter)
    category_activities = defaultdict(Counter)
    category_actions = defaultdict(Counter)
    casas = root / "casas" / "extracted" / "data" / "aruba.csv"
    if casas.exists():
        histogram = casas_histogram(casas)
        with casas.open() as handle:
            for row in csv.reader(handle):
                if len(row) >= 4 and row[-1].strip().upper() == 'ON':
                    try:
                        hour = int(row[1].split(':')[0])
                    except (ValueError, IndexError):
                        continue
                    if 0 <= hour < 24:
                        hour_counts[hour] += 1
        sources["CASAS"] = {
            "status": "available",
            "records": int(sum(histogram["room_counts"].values())),
            "room_counts": histogram["room_counts"],
            "adapter": "casas_histogram",
            "trajectory_copied": False,
        }
    records = _source_records_root(root)
    record_names = {"HD_EPIC": "hd_epic.jsonl", "ParaHome": "parahome.jsonl", "HOMER_PLUS": "homer_plus.jsonl"}
    for source, filename in record_names.items():
        path = records / filename if records else None
        if path and path.exists():
            count = 0
            with path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line); count += 1
                    category = record.get('object_category')
                    if not category:
                        continue
                    activity = record.get('canonical_activity') or record.get('raw_activity')
                    if activity:
                        category_activities[category][activity] += 1
                    action = record.get('canonical_action')
                    if action:
                        category_actions[category][action] += 1
                    if record.get('time_minutes') is not None:
                        hour = int(float(record['time_minutes']) // 60) % 24
                        category_hours[category][hour] += 1
            sources[source] = {
                "status": "available",
                "records": count,
                "path": str(path),
                "adapter": "p4d_source_record_count",
                "trajectory_copied": False,
            }
    # Keep explicit source presence information for assets that may exist
    # outside the P4D export, while never claiming calibration from presence.
    direct_presence = {
        "HD_EPIC": (root / "hd_epic_annotations").exists(),
        "ParaHome": (root / "parahome_data").exists(),
        "HOMER_PLUS": (root / "HOMER_PLUS").exists(),
        "ARAS": (root / "aras").exists(),
        "OPPORTUNITY": (root / "opportunity").exists(),
    }
    for source, present in direct_presence.items():
        if source not in sources and present:
            sources[source] = {"status": "present_unparsed", "records": None, "trajectory_copied": False}
    missing = [source for source in _EXPECTED if source not in sources]
    def weights(counts):
        total = sum(counts.values())
        return [counts.get(h, 0)/total if total else 0. for h in range(24)]
    behavior = {
        'hour_weights': weights(hour_counts + sum(category_hours.values(), Counter())),
        'category_hour_weights': {c: weights(h) for c,h in category_hours.items()},
        'category_activity_counts': {c: dict(v) for c,v in category_activities.items()},
        'category_action_counts': {c: dict(v) for c,v in category_actions.items()},
        'parameterization': 'aggregate activity/category and hour marginals; no trajectory replay',
        'limitations': ['HOMER_PLUS is a simulated routine source, not observed real residents',
                       'CASAS sensor activations are occupancy proxies, not object relocation rates',
                       'movement frequency remains a synthetic controlled parameter'],
    }
    return {
        "root": str(root),
        "sources": sources,
        "behavior_model": behavior,
        "missing_sources": missing,
        "calibration_status": "aggregate_only",
        "trajectory_copied": False,
        "notes": [
            "Statistics constrain synthetic generation; raw human trajectories are not replayed.",
            "ARAS and OPPORTUNITY adapters are absent unless supplied under the source root.",
        ],
    }


def write_source_statistics(root: str | Path, output: str | Path) -> dict[str, Any]:
    result = build_source_statistics(root)
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
