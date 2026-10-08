"""Fast independent audit for public/private JSONL streams."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from .io import read_jsonl


_PRIVATE_KEYS = {
    "current_state", "hidden_future_events", "all_events", "world", "hidden_transition",
    "future_activity", "oracle_target_location", "private_answer",
}


def _forbidden(value):
    if isinstance(value, dict):
        if value.get('visibility_kind') == 'isolated_silhouette':
            return True
        for key, child in value.items():
            if key in _PRIVATE_KEYS or (key == "mobility_regime" and child != "hidden"):
                return True
            if _forbidden(child):
                return True
    elif isinstance(value, list):
        return any(_forbidden(child) for child in value)
    return False


def audit_stream(root: str | Path) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    held_out = set(manifest.get("generator_config", {}).get("held_out_scene_ids", []))
    counts = Counter()
    split_counts = Counter()
    ids = set()
    private_ids = set()
    public_forbidden = id_mismatch = time_error = 0
    positive = negative = total = heldout_n5 = nonheldout_n5 = 0
    private_iter = (row for path in sorted((root / "private").glob("*.jsonl")) for row in read_jsonl(path))
    for path in sorted((root / "episodes").glob("*.jsonl")):
        split = path.stem
        for row in read_jsonl(path):
            total += 1
            split_counts[split] += 1
            task = row.get("task_type")
            counts[task] += 1
            ident = row.get("id")
            if ident in ids:
                id_mismatch += 1
            ids.add(ident)
            if _forbidden(row):
                public_forbidden += 1
            timestamps = [h.get("timestamp") for h in row.get("history", [])]
            if not timestamps or any(t is None for t in timestamps) or timestamps != sorted(set(timestamps)):
                time_error += 1
            if timestamps and any(not (0 <= t < row.get("query_time", 0)) for t in timestamps):
                time_error += 1
            visible = [bool(h.get("visible", True)) for h in row.get("history", [])]
            if any(visible):
                positive += 1
            if any(not item for item in visible):
                negative += 1
            if task == "N5":
                if row.get("scene") in held_out:
                    heldout_n5 += 1
                else:
                    nonheldout_n5 += 1
            truth = next(private_iter, None)
            if truth is None or truth.get("id") != ident:
                id_mismatch += 1
            elif truth.get("id") in private_ids:
                id_mismatch += 1
            elif truth.get("id"):
                private_ids.add(truth["id"])
    if next(private_iter, None) is not None:
        id_mismatch += 1
    expected = int(manifest.get("episodes", 0))
    passed = (total == expected and bool(total) and not id_mismatch and not time_error
              and not public_forbidden and nonheldout_n5 == 0)
    return {
        "counts": dict(split_counts), "tasks": dict(counts), "total": total,
        "heldout_n5": heldout_n5, "nonheldout_n5": nonheldout_n5,
        "id_mismatch": id_mismatch, "time_error": time_error,
        "public_forbidden": public_forbidden,
        "patrol": {"positive": positive, "negative": negative},
        "pass": passed, "release_eligible": False,
    }
