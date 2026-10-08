#!/usr/bin/env python3
"""Evaluate predictions and optional agent traces without oracle-path shortcuts.

The default last-seen predictor produces only prediction metrics. Navigation
and dynamic metrics are reported only when an external agent supplies an
action-result JSONL; distances are never fabricated by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evoworld.evaluation import dynamic_metrics, navigation_metrics, prediction_metrics


def _last_seen(row):
    states = [str(c["state_id"]) for c in row.get("candidate_states", [])]
    observed = [h.get("location") for h in row.get("history", []) if h.get("visible") and h.get("location")]
    if observed and observed[-1] in states:
        return {state: 1.0 if state == observed[-1] else 0.0 for state in states}
    probability = 1.0 / len(states) if states else 1.0
    return {state: probability for state in states} or {"unknown": 1.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--actions", type=Path, help="optional evaluator action-result JSONL")
    args = parser.parse_args()
    root = args.dataset
    public = [json.loads(line) for line in (root / "episodes" / f"{args.split}.jsonl").open(encoding="utf-8")]
    private = {}
    with (root / "private" / f"{args.split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            private[record["id"]] = record
    grouped = defaultdict(list)
    for row in public:
        if row.get("task_type") == "EQA":
            continue
        truth = private[row["id"]]
        grouped[row.get("task_type", "unknown")].append({
            "probabilities": _last_seen(row),
            "label": truth["current_state"],
        })
    output = {"split": args.split, "episodes": len(public), "prediction": {
        task: prediction_metrics(rows) for task, rows in sorted(grouped.items())
    }}
    if args.actions:
        action_rows = [json.loads(line) for line in args.actions.open(encoding="utf-8")]
        output["navigation"] = navigation_metrics(action_rows)
        output["dynamic"] = dynamic_metrics(action_rows)
    else:
        output["navigation"] = {"status": "not_evaluated", "reason": "provide --actions evaluator results"}
        output["dynamic"] = {"status": "not_evaluated", "reason": "provide --actions evaluator results"}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
