"""Run P4D-HSSD navigation episodes with belief or closed-loop Agent control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evolvingnav.backend import HabitatInspectionBackend
from evolvingnav.agent import Agent, AgentConfig
from evolvingnav.calibration import DetectionCalibrator
from evolvingnav.controller import LunaToolController
from evolvingnav.evaluate import aggregate_metrics, oracle_distance, score_agent
from evolvingnav.memory import VersionedMemory
from evolvingnav.perception import GroundedSAMInspector
from evolvingnav.policy import load_belief, model_input_batch, pack_public_query, predict_public
from evolvingnav.transition import IdentityTransition
from evolvingnav.transition_model import NeuralTransition, TransitionHead
from evolvingnav.world import HabitatAgentWorld

CODE_ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("n1", "n2", "n3", "n4", "n5"), default="n3")
    parser.add_argument("--protocol", choices=("n1", "n2", "n3", "n4"))
    parser.add_argument("--agent", action="store_true", help="Run the event-driven Agent for N1/N2 as well")
    parser.add_argument("--controller", choices=("utility", "luna"), default="luna")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--world", choices=("routine", "random", "static"), default="routine")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--transition-checkpoint", type=Path)
    parser.add_argument("--inspection", choices=("semantic-oracle", "grounded-sam"), default="grounded-sam")
    parser.add_argument("--hssd-root", type=Path, required=True)
    parser.add_argument("--navmesh-root", type=Path, required=True)
    parser.add_argument("--grounding-dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam2-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--perception-config", type=Path, default=CODE_ROOT / "configs/perception.yaml")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    args.protocol = args.protocol if args.task == "n5" else args.task
    if args.protocol is None:
        parser.error("N5 requires --protocol")
    if args.protocol == "n4" and args.transition_checkpoint is None:
        parser.error("N4 requires --transition-checkpoint")
    return args


def public_features(states: list[dict], schema: dict) -> dict:
    states = sorted(states, key=lambda row: int(row["state_id"]))
    if [int(row["state_id"]) for row in states] != list(range(len(states))):
        raise ValueError("public state IDs must be dense and zero-based")
    return {
        "candidate_region_category_id": np.asarray([
            schema["region_category_to_id"].get(row["region_category"], schema["region_category_to_id"]["unknown"])
            for row in states], dtype=np.int64),
        "candidate_receptacle_category_id": np.asarray([
            schema["receptacle_category_to_id"].get(row["receptacle_category"], schema["receptacle_category_to_id"]["unknown"])
            for row in states], dtype=np.int64),
        "candidate_center_xyz": np.asarray([row.get("state_center") or [0., 0., 0.]
                                             for row in states], dtype=np.float32),
        "candidate_is_unknown": np.asarray([row.get("is_unknown", row.get("region_id") == "unknown")
                                             for row in states]),
    }


def episode_config(episode: dict, protocol: str, unknown_state: int | None) -> AgentConfig:
    budget = episode["episode_budget"]
    return AgentConfig(
        protocol=protocol,
        max_inspections=1 if protocol == "n1" else int(budget["max_candidate_inspections"]),
        max_path_m=float(budget["max_path_length_m"]),
        max_time_s=float(budget.get("max_time_s", 3600.)),
        max_steps=int(budget.get("max_steps", 500)),
        max_explorations=int(budget.get("max_explorations", 10))
            if "EXPLORE" in episode["public_refs"].get("action_space", []) else 0,
        unknown_state=unknown_state,
    )


def validate_episode_contract(episode: dict, protocol: str) -> None:
    spec = episode["success_spec"]
    if (spec.get("min_visible_fraction") != .20
            or spec.get("max_geodesic_distance_m") != 1.
            or not spec.get("require_target_visible") or not spec.get("require_stop_action")):
        raise ValueError("episode success contract does not match the benchmark")
    if protocol == "n4" and "max_time_s" not in episode["episode_budget"]:
        raise ValueError("N4 requires a fixed predeclared evaluation window")


def main() -> int:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.inspection == "grounded-sam" and args.calibration is None:
        raise ValueError("Grounded-SAM execution requires --calibration")
    episode_path = args.tasks / f"public/episodes_{args.task}.jsonl"
    if args.task == "n5" and not episode_path.exists():
        episode_path = args.tasks / f"public/episodes_{args.protocol}.jsonl"
    episodes = [row for row in rows(episode_path) if row["world_variant"] == args.world][:args.limit]
    if len(episodes) != args.limit:
        raise ValueError(f"only found {len(episodes)} matching episodes")
    for episode in episodes:
        validate_episode_contract(episode, args.protocol)
    wanted_queries = {row["query_id"] for row in episodes}
    queries = {row["query_id"]: row for row in rows(args.tasks / "public/query_inputs.jsonl")
               if row["query_id"] in wanted_queries}
    active_checkpoint = args.transition_checkpoint if args.protocol == "n4" else args.checkpoint
    model, schema = load_belief(active_checkpoint, args.dataset)
    transition_head = None
    if args.protocol == "n4":
        import torch
        checkpoint = torch.load(active_checkpoint, map_location="cpu", weights_only=True)
        transition_head = TransitionHead(checkpoint["model_config"]["hidden_dim"])
        transition_head.load_state_dict(checkpoint["transition_head"])
        transition_head.eval()

    catalog = json.loads((args.tasks / "catalogs/candidate_states_navigation.json").read_text())
    features = public_features(catalog["states"], schema)
    scene_schema = {**schema, "state_count_including_unknown": len(catalog["states"])}
    viewpoints = {int(row["state_id"]): row["navigation_viewpoint"]
                  for row in catalog["states"] if row.get("navigation_eligible")}
    goals = {state: row["position_xyz"] for state, row in viewpoints.items()}
    centers = {int(row["state_id"]): row["state_center"] for row in catalog["states"]
               if row.get("state_center") is not None}
    surface_points = {int(row["state_id"]): [slot["point"] for slot in row.get("sampled_place_points", [])]
                      for row in rows(args.tasks / "catalogs/receptacles.jsonl")}
    objects = {row["instance_uuid"]: row for row in rows(args.tasks / "catalogs/object_instances.jsonl")}
    wanted = {row["base_episode_id"] for row in episodes}
    private = {row["base_episode_id"]: row["evaluation_private"]
               for row in rows(args.tasks / "private/evaluation_gt.jsonl")
               if row["base_episode_id"] in wanted}
    unknown_ids = np.flatnonzero(features["candidate_is_unknown"])
    unknown = int(unknown_ids[0]) if len(unknown_ids) else None
    inspector = GroundedSAMInspector(args.perception_config,
        dino_model=args.grounding_dino_model, sam_model=args.sam2_model
    ) if args.inspection == "grounded-sam" else None
    calibrator = DetectionCalibrator.load(args.calibration) if args.calibration else None
    controller = LunaToolController() if args.controller == "luna" else None
    decisions, scores = [], []
    backend = None
    try:
        for episode in episodes:
            if episode["scene_id"] != catalog["scene_id"]:
                raise ValueError("episode and public scene catalog disagree")
            query = queries[episode["query_id"]]
            packed = pack_public_query(query, scene_schema, features)
            batch = model_input_batch(packed, scene_schema)
            probabilities = predict_public(model, scene_schema, packed)
            allowed = {int(state) for state in episode["public_refs"]["candidate_state_ids"]}
            if unknown is not None:
                allowed.add(unknown)
            prior = {state: mass for state, mass in probabilities.items() if state in allowed}
            prior = {state: mass / sum(prior.values()) for state, mass in prior.items()}
            decisions.append({"base_episode_id": episode["base_episode_id"],
                              "query_id": episode["query_id"], "task": args.task, "belief": prior})
            truth = private[episode["base_episode_id"]]
            schedule = truth.get("target_motion_schedule", []) if args.protocol == "n4" else []
            if args.protocol == "n4" and not schedule:
                raise ValueError("N4 requires a predeclared target_motion_schedule")
            required = {"time_s", "target_position_xyz", "current_state_id", "valid_goal_viewpoints"}
            if any(required - set(event) for event in schedule):
                raise ValueError("incomplete target motion event")
            if backend is None:
                backend = HabitatInspectionBackend(args.hssd_root, episode["scene_id"],
                    args.navmesh_root / f"{episode['scene_id']}.navmesh", viewpoints, detector=inspector)
            backend.prepare(truth, objects[episode["target"]["object_id"]],
                            list(set(prior) & set(goals)), dynamic=args.protocol == "n4")
            config = episode_config(episode, args.protocol, unknown)
            phases = [(0., [goal["position_xyz"] for goal in truth["valid_goal_viewpoints"]])]
            phases += [(float(event["time_s"]), [goal["position_xyz"] for goal in event["valid_goal_viewpoints"]])
                       for event in schedule if 0 < event["time_s"] <= config.max_time_s]
            reference = oracle_distance(start=episode["agent_start"]["position_xyz"],
                phases=phases, distance=backend.distance, max_time_s=config.max_time_s,
                max_path_m=config.max_path_m, max_steps=config.max_steps)
            if reference is None:
                raise ValueError("episode has no budget-feasible oracle path")

            target = query["input"]["target"]
            target_id = target["instance_uuid"]
            memory = VersionedMemory()
            for index, event in enumerate(query["input"].get("target_history", [])):
                if event["event_type"] == "positive_observation":
                    state = int(event["observed_state_id"])
                    memory.observe(target_id, state, float(event["timestamp_s"]),
                        float(event["detector_confidence"]) * float(event["instance_match_confidence"]),
                        f"{query['query_id']}:history:{index}",
                        np.asarray(event.get("world_point", centers.get(state, [0., 0., 0.]))),
                        tuple(event.get("visual_feature", ())), category=target["category"],
                        attributes=event.get("attributes"), relations={"at": str(state)})
                elif event["event_type"] == "candidate_inspection":
                    memory.record_negative(f"{query['query_id']}:history:{index}",
                        int(event["candidate_state_id"]), float(event["timestamp_s"]),
                        event.get("pose_xyz", [0., 0., 0.]))
            world = HabitatAgentWorld(backend, viewpoints,
                {state: centers[state] for state in viewpoints},
                episode["agent_start"]["position_xyz"], episode["agent_start"]["rotation_xyzw"],
                calibrator=calibrator, known_states=set(prior),
                motion_schedule=schedule, surface_points=surface_points)
            transition = NeuralTransition(model, transition_head, batch) if args.protocol == "n4" else IdentityTransition()
            result = Agent(prior, goals, world, transition, config,
                sample_count={state: world.sample_count(state) for state in prior if state in viewpoints},
                controller=controller, memory=memory, target_id=target_id,
                time_origin_s=float(query["input"]["query"]["query_time_s"]),
                target={**target, "instruction": episode["target"].get("instruction", "")}).run()
            score = score_agent(result, world.private_inspections, reference_distance=reference,
                initial_state=int(truth["current_state_id"]), schedule=schedule, max_time_s=config.max_time_s)
            scores.append({**score, "base_episode_id": episode["base_episode_id"], "task": args.task,
                "protocol": args.protocol, "actions": result.actions, "path_m": result.path_m,
                "elapsed_s": result.elapsed_s, "steps": result.steps, "inspection_order": result.inspections,
                "covered_inspections": result.covered_inspections, "inspection_evidence": world.private_inspections,
                "posterior": result.posterior, "termination": result.termination,
                "evidence_trace": result.evidence_trace})
            backend.clear()
    finally:
        if backend is not None:
            backend.close()
        if inspector is not None:
            inspector.close()

    args.output.mkdir(parents=True)
    for name, records in (("policy", decisions), ("scores", scores)):
        with (args.output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {**aggregate_metrics(scores), "task": args.task, "protocol": args.protocol,
               "world": args.world, "controller": args.controller,
               "track": f"high_level_event_agent_{args.inspection}", "min_visible_fraction": .20,
               "dataset": str(args.tasks), "checkpoint": str(active_checkpoint)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
