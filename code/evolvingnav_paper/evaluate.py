"""Evaluator-only truth access and high-level N1/N2 scoring."""

from __future__ import annotations

import math
import heapq
from typing import Callable


def oracle_distance(*, start, phases, distance, max_time_s, max_path_m,
                    inspection_s=1., speed_mps=1., max_steps=500):
    """Private time-feasible shortest travel path, including stationary waiting."""
    phases = sorted(phases, key=lambda phase: phase[0])
    nodes = [start, *(goal for _, goals in phases for goal in goals)]
    heap = [(0., 0., 0, 0)]
    seen = {}
    while heap:
        travelled, now, node, steps = heapq.heappop(heap)
        phase = max(i for i, (time, _) in enumerate(phases) if time <= now)
        key = node, phase
        labels = seen.setdefault(key, [])
        if any(d <= travelled and t <= now and s <= steps for d, t, s in labels):
            continue
        labels.append((travelled, now, steps))
        end = phases[phase+1][0] if phase+1 < len(phases) else max_time_s + 1e-9
        if now + inspection_s < end and now + inspection_s <= max_time_s:
            if any(distance(nodes[node], goal) <= 1e-6 for goal in phases[phase][1]):
                return travelled
        if phase+1 < len(phases) and phases[phase+1][0] <= max_time_s:
            heapq.heappush(heap, (travelled, phases[phase+1][0], node, steps))
        for destination, goal in enumerate(nodes):
            if destination == node:
                continue
            leg = float(distance(nodes[node], goal))
            if not math.isfinite(leg):
                continue
            arrival = now + leg/speed_mps
            next_steps = steps + max(1, math.ceil(leg/.25))
            if travelled + leg <= max_path_m and arrival + inspection_s <= max_time_s and next_steps < max_steps:
                heapq.heappush(heap, (travelled+leg, arrival, destination, next_steps))
    return None


def aggregate_metrics(rows: list[dict]) -> dict:
    def rate(values, key):
        return sum(bool(row.get(key)) for row in values)/len(values) if values else 0.
    dynamic = [row for row in rows if row.get("dynamic_eligible")]
    recovery = [row for row in rows if row.get("recovery_eligible")]
    online = [row for row in rows if row.get("online_recovery_eligible")]
    revisits = sum(row.get("revisit_count", 0) for row in dynamic)
    return {
        "episodes": len(rows), "successes": sum(bool(row["success"]) for row in rows),
        "sr": rate(rows, "success"), "spl": sum(row["spl"] for row in rows)/len(rows) if rows else 0.,
        "first_inspection_sr": rate(rows, "first_inspection_success"),
        "recovery_sr": rate(recovery, "recovered"), "dynamic_sr": rate(dynamic, "success"),
        "online_recovery_sr": rate(online, "online_recovered"),
        "revisit_success": sum(row.get("revisit_success_count", 0) for row in dynamic)/revisits if revisits else 0.,
        "excess_distance": sum(row["distance"]-row["reference_distance"] for row in dynamic)/len(dynamic) if dynamic else 0.,
    }


def score_agent(result, frames, *, reference_distance, initial_state,
                schedule=(), max_time_s=3600.):
    final = frames[-1] if frames else {}
    success = bool(result.found and result.actions[-1:] == ["STOP"]
                   and final.get("identified_target", False)
                   and final.get("visible_fraction", 0.) >= .20
                   and final.get("distance_to_valid_goal_m", math.inf) <= 1.)
    if reference_distance is None:
        raise ValueError("episode has no budget-feasible oracle path")
    first = result.covered_inspections[0] if result.covered_inspections else None
    def state_at(time):
        state = initial_state
        for event in sorted(schedule, key=lambda row: row["time_s"]):
            if event["time_s"] > time:
                break
            state = event["current_state_id"]
        return state
    first_success = bool(first and first["state_id"] == state_at(first["time_s"]))
    transitions, previous = [], initial_state
    for event in sorted(schedule, key=lambda row: row["time_s"]):
        if 0 < event["time_s"] <= max_time_s and event["current_state_id"] != previous:
            transitions.append(event["time_s"])
        previous = event["current_state_id"]
    revisits = [visit for visit in result.visits if visit["round"] > 0]
    return {
        "success": success, "first_inspection_success": first_success,
        "recovery_eligible": first is not None and not first_success,
        "recovered": bool(success and first and not first_success
                          and result.elapsed_s >= first["time_s"]),
        "dynamic_eligible": bool(transitions), "online_recovery_eligible": bool(transitions),
        "online_recovered": bool(success and transitions and result.elapsed_s >= transitions[0]),
        "revisit_count": len(revisits),
        "revisit_success_count": int(success and bool(result.visits) and result.visits[-1]["round"] > 0),
        "distance": result.path_m, "reference_distance": reference_distance,
        "spl": float(success)*reference_distance/max(result.path_m, reference_distance, 1e-9),
        "time": result.elapsed_s, "inspection_count": len(result.covered_inspections),
    }


def evaluate_search(
    episode: dict,
    truth: dict,
    belief: dict[int, float],
    *,
    start,
    goals: dict,
    distance: Callable,
    inspect: Callable,
    task: str = "n2",
) -> dict:
    from evolvingnav_paper.policy import rank_candidates

    if task not in {"n1", "n2"}:
        raise ValueError(task)
    true_state = int(truth["current_state_id"])
    oracle_m = float(truth["oracle_shortest_path_m"])
    max_inspections = 1 if task == "n1" else int(episode["episode_budget"]["max_candidate_inspections"])
    max_path = float(episode["episode_budget"]["max_path_length_m"])
    current, travelled, inspected, actions, evidence = start, 0.0, [], [], []
    remaining = set(belief) & set(goals)
    success = False
    for _ in range(max_inspections):
        costs = {state: float(distance(current, goals[state])) for state in remaining}
        reachable = {state: belief[state] for state in remaining if math.isfinite(costs[state])}
        if not reachable:
            break
        state = rank_candidates(reachable, costs, task=task)[0]
        goal = goals[state]
        leg = costs[state]
        if not math.isfinite(leg) or travelled + leg > max_path:
            break
        travelled += leg
        inspected.append(state)
        actions.extend((f"NAVIGATE_TO({state})", f"INSPECT({state})"))
        current = goal
        observation = inspect(state, goal)
        evidence.append({"state_id": state, **observation})
        if observation["detected"]:
            actions.append("STOP")
            success = (
                float(observation["distance_to_valid_goal_m"])
                <= float(episode["success_spec"]["max_geodesic_distance_m"])
                and float(observation["visible_fraction"]) >= 0.20
            )
            break
        remaining.remove(state)
    return {
        "base_episode_id": episode["base_episode_id"],
        "task": task,
        "success": success,
        "inspections": len(inspected),
        "inspection_order": inspected,
        "inspection_evidence": evidence,
        "actions": actions,
        "path_m": round(travelled, 6),
        "spl": round(float(success) * oracle_m / max(travelled, oracle_m, 1e-9), 6),
        "true_state_id": true_state,
    }
