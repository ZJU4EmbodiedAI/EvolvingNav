#!/usr/bin/env python3
"""Collect held-out RGB-D detector observations for logistic recall calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evolvingnav.backend import HabitatInspectionBackend
from evolvingnav.coverage import (
    candidate_surface_samples, visible_sample_ids, view_features,
)
from evolvingnav.perception import GroundedSAMInspector
from evolvingnav.run import rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--navmesh-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--perception-config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/perception.yaml")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    catalog = json.loads((args.tasks / "catalogs/candidate_states_navigation.json").read_text())
    viewpoints = {int(row["state_id"]): row["navigation_viewpoint"]
                  for row in catalog["states"] if row.get("navigation_eligible")}
    centers = {int(row["state_id"]): np.asarray(row["state_center"], dtype=float)
               for row in catalog["states"]}
    surface_points = {
        int(row["state_id"]): [slot["point"] for slot in row.get("sampled_place_points", [])]
        for row in rows(args.tasks / "catalogs/receptacles.jsonl")
    }
    objects = {row["instance_uuid"]: row
               for row in rows(args.tasks / "catalogs/object_instances.jsonl")}
    selected = []
    for record in rows(args.dataset / "records/val_queries.jsonl"):
        state = int(record["supervision"]["current_state_id"])
        if state in viewpoints and record["input"]["target"]["instance_uuid"] in objects:
            selected.append(record)
        if len(selected) >= args.limit:
            break
    if len(selected) < args.limit:
        raise ValueError(f"only {len(selected)} validation records have a public viewpoint")
    inspector = GroundedSAMInspector(
        args.perception_config, dino_model=args.dino_model, sam_model=args.sam_model
    )
    backend = HabitatInspectionBackend(
        args.hssd_root, selected[0]["scene_id"],
        args.navmesh_root / f"{selected[0]['scene_id']}.navmesh",
        viewpoints, detector=inspector,
    )
    observations = []
    try:
        for record in selected:
            target = record["input"]["target"]
            state = int(record["supervision"]["current_state_id"])
            center = centers[state]
            nearest = sorted(viewpoints, key=lambda candidate: float(np.linalg.norm(
                np.asarray(viewpoints[candidate]["position_xyz"]) - center
            )))
            selected_views = list(dict.fromkeys(
                [state, *nearest[:3], nearest[len(nearest) // 2], nearest[-1]]
            ))
            truth = {
                "target_position_xyz": record["supervision"]["current_position"],
                "current_state_id": state,
                "valid_goal_viewpoints": [viewpoints[state]],
            }
            backend.prepare(truth, objects[target["instance_uuid"]], selected_views)
            samples = candidate_surface_samples(center, place_points=surface_points.get(state))
            for view_state in selected_views:
                viewpoint = viewpoints[view_state]
                backend.inspect(view_state, viewpoint["position_xyz"])
                depth = backend.last_observation["depth"]
                covered = visible_sample_ids(
                    samples, viewpoint["position_xyz"], viewpoint["rotation_xyzw"],
                    depth, 79.0,
                )
                semantic = backend.last_semantic == int(objects[target["instance_uuid"]]["semantic_instance_id"])
                target_pixels = int(np.count_nonzero(semantic))
                overlaps = [int(np.count_nonzero(detection.mask & semantic))
                            for detection in backend.last_detections]
                features, _ = view_features(samples, covered, viewpoint["position_xyz"],
                    viewpoint["rotation_xyzw"], depth, backend.last_observation["rgb"],
                    category=target["category"])
                observations.append({
                    **features,
                    "detected": bool(target_pixels >= 20 and max(overlaps, default=0) / target_pixels >= 0.1),
                })
            backend.clear()
    finally:
        backend.close()
        inspector.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in observations:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({
        "observations": len(observations),
        "detected": sum(row["detected"] for row in observations),
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
