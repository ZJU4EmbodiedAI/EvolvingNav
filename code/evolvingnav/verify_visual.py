"""Render one scored success and confirm the benchmark visibility contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evolvingnav.backend import HabitatInspectionBackend
from evolvingnav.habitat_utils import set_agent
from evolvingnav.run import rows


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="a completed N1/N2/N3 run directory")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--navmesh-root", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = arguments()
    passed = next((row for row in rows(args.run / "scores.jsonl") if row["success"]), None)
    if passed is None:
        raise ValueError("run has no successful episode to verify")
    episode_id = passed["base_episode_id"]
    episode = next(row for row in rows(args.tasks / "public/base_episodes.jsonl") if row["base_episode_id"] == episode_id)
    truth = next(row["evaluation_private"] for row in rows(args.tasks / "private/evaluation_gt.jsonl") if row["base_episode_id"] == episode_id)
    target = next(row for row in rows(args.tasks / "catalogs/object_instances.jsonl") if row["instance_uuid"] == episode["target"]["object_id"])
    stop_state = int(passed["inspection_order"][-1])
    candidate_catalog = json.loads(
        (args.tasks / "catalogs/candidate_states_navigation.json").read_text()
    )
    viewpoints = {
        int(row["state_id"]): row["navigation_viewpoint"]
        for row in candidate_catalog["states"] if row.get("navigation_eligible")
    }
    scene_id = episode["scene_id"]
    navmesh = args.navmesh_root / f"{scene_id}.navmesh"
    backend = HabitatInspectionBackend(args.hssd_root, scene_id, navmesh, viewpoints)
    try:
        backend.prepare(truth, target, [stop_state])
        start = episode["agent_start"]
        set_agent(backend.real.get_agent(0), start["position_xyz"], start["rotation_xyzw"])
        start_pixels = int((np.asarray(backend.real.get_sensor_observations()["semantic"]) == int(target["semantic_instance_id"])).sum())
        viewpoint = viewpoints[stop_state]
        observation = backend.inspect(stop_state, viewpoint["position_xyz"])
    finally:
        backend.close()
    result = {
        "base_episode_id": episode_id,
        "stop_state_id": stop_state,
        "start_target_pixels": start_pixels,
        "stop_target_pixels": observation["visible_target_pixels"],
        "projected_target_pixels": observation["projected_target_pixels"],
        "visible_fraction": observation["visible_fraction"],
        "distance_to_valid_goal_m": observation["distance_to_valid_goal_m"],
        "start_hidden": start_pixels == 0,
        "stop_visible": observation["visible_fraction"] >= 0.20,
        "valid_stop_distance": observation["distance_to_valid_goal_m"] <= 1.0,
        "stop_action": passed["actions"][-1] == "STOP",
        "matches_record": observation["visible_fraction"] == passed["inspection_evidence"][-1]["visible_fraction"],
    }
    result["passed"] = all(result[key] for key in (
        "start_hidden", "stop_visible", "valid_stop_distance", "stop_action", "matches_record",
    ))
    (args.run / "visual_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
