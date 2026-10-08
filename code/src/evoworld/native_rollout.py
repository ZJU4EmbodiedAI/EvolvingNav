"""Small, reproducible Habitat diagnostic rollout for native episodes.

This runner writes an agent-facing trace and evaluator diagnostics.  It produces an agent-facing
 RGB-D trace plus evaluator-only semantic diagnostics, which are consumed by
a later private evaluator after visibility calibration.
"""
from __future__ import annotations

import json
from pathlib import Path

from .habitat_online import NativeHabitatSession
from .evaluation.oracle import dynamic_oracle_distance


def select_indices(rows, task_types, limit=None):
    """Select protocol rows in source order for deterministic batch runs."""
    selected = [index for index, row in enumerate(rows) if row.get("task_type") in set(task_types)]
    return selected if limit is None else selected[:max(0, int(limit))]


def execution_world(world, task_type, *, query_time):
    """Freeze post-query evolution except in the N4 online protocol."""
    if task_type == "N4":
        return world
    return {**world, "events": [event for event in world.get("events", [])
                                 if event["timestamp"] <= query_time]}


def last_seen_candidate(row):
    candidate_ids = [str(candidate["state_id"]) for candidate in row.get("candidate_states", [])]
    for history in reversed(row.get("history", [])):
        if history.get("visible") and history.get("location") in candidate_ids:
            return history["location"]
    return next((state for state in candidate_ids if state != "unknown"), "unknown")


def run_last_seen(catalog, target, row, truth, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = last_seen_candidate(row)
    start_view = row["start"]["view_id"]
    world = execution_world(truth["world"], row.get("task_type"), query_time=row["query_time"])
    with NativeHabitatSession(
        catalog, target, world, query_time=row["query_time"], start_view=start_view,
        max_steps=row["navigation_budget"]["max_steps"],
        max_distance_m=row["navigation_budget"]["max_distance_m"],
        max_inspections=row["navigation_budget"]["max_inspections"],
        max_seconds=row["navigation_budget"]["max_seconds"],
    ) as session:
        session.move_to(candidate)
        # The target candidate's native viewpoint is the only route endpoint;
        # the private semantic test stays inside the session.
        observation = session.inspect()
        private = session.private_result()
        public_trace = {
            "episode_id": row["id"],
            "visited_candidate": candidate,
            "observation": {
                "timestamp": observation["timestamp"],
                "view_id": observation["view_id"],
                "camera_pose": observation["camera_pose"],
                "rgb": str(output_dir / f"{row['id']}.rgb.npy"),
                "depth": str(output_dir / f"{row['id']}.depth.npy"),
            },
            "actions": session.public_trace(),
        }
        import numpy as np
        np.save(public_trace["observation"]["rgb"], observation["rgb"])
        np.save(public_trace["observation"]["depth"], observation["depth"])
        private_diagnostic = dict(private)
        private_diagnostic["task_type"] = row.get("task_type")
        private_diagnostic["oracle_reference_distance_m"] = dynamic_oracle_distance(
            target, world, query_time=row["query_time"], start_view=start_view,
            speed_mps=1.0,
            max_seconds=row["navigation_budget"]["max_seconds"],
            max_distance_m=row["navigation_budget"]["max_distance_m"],
            max_steps=row["navigation_budget"]["max_steps"],
        )
        private_diagnostic["oracle_kind"] = "time_feasible_catalog_oracle"
        private_diagnostic.update({
            "evaluation_status": "diagnostic_only",
            "ground_truth_state": truth["current_state"],
            "success_rule": "calibrated target visible fraction >= 0.20",
        })
        return {"public_trace": public_trace, "private_diagnostic": private_diagnostic}


def write_rollout(result, path):
    path = Path(path)
    path.write_text(json.dumps(result["public_trace"], indent=2, default=str) + "\n", encoding="utf-8")
    path.with_name(path.stem + ".private.json").write_text(
        json.dumps(result["private_diagnostic"], indent=2, default=str) + "\n", encoding="utf-8")
