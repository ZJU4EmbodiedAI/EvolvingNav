"""Event-driven EvolvingNav controller (event-driven equations 11–15, 20–22)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from evolvingnav.filter import BeliefFilter, EvidenceLedger
from evolvingnav.policy import candidate_utility


@dataclass(frozen=True)
class ViewEvidence:
    evidence_id: str
    state_id: int
    surface_samples: frozenset[int]
    detection_probability: float
    covered_fraction: float = 1.0
    pose_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    features: dict = field(default_factory=dict)
    sample_features: dict[int, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    protocol: str = "n3"
    max_inspections: int = 10
    max_path_m: float = 100.0
    max_time_s: float = 3600.0
    max_steps: int = 500
    max_explorations: int = 10
    chunk_m: float = 2.0
    speed_mps: float = 1.0
    inspection_s: float = 1.0
    lambda_time: float = 0.05
    lambda_inspect: float = 0.25
    lambda_scan: float = 0.25
    explore_cost: float = 2.0
    discovery_probability: float = 0.5
    speed_ema_alpha: float = 0.8
    unknown_state: int | None = None


@dataclass
class AgentResult:
    found: bool = False
    actions: list[str] = field(default_factory=list)
    inspections: list[int] = field(default_factory=list)
    path_m: float = 0.0
    elapsed_s: float = 0.0
    posterior: dict[int, float] = field(default_factory=dict)
    termination: str = ""
    evidence_trace: list[dict] = field(default_factory=list)
    covered_inspections: list[dict] = field(default_factory=list)
    steps: int = 0
    explorations: int = 0
    visits: list[dict] = field(default_factory=list)


class World(Protocol):
    def distance(self, goal) -> float: ...
    def move_chunk(self, goal, max_distance: float) -> tuple[float, float]: ...
    def inspect(self, state: int) -> tuple[bool, list[ViewEvidence]]: ...
    def explore(self, budget_m: float) -> tuple[dict[int, tuple[object, float]], float, float]: ...


class Agent:
    def __init__(self, prior: dict[int, float], goals: dict[int, object], world: World,
                 transition, config: AgentConfig | None = None,
                 sample_count: dict[int, int] | None = None, controller=None,
                 memory=None, target_id: str | None = None,
                 time_origin_s: float = 0.0, target: dict | None = None) -> None:
        self.config = config or AgentConfig()
        self.filter = BeliefFilter(prior, transition)
        self.goals = {state: goal for state, goal in goals.items() if state in prior}
        self.world = world
        self.ledger = EvidenceLedger(sample_count=sample_count)
        self.speed = self.config.speed_mps
        self.controller = controller
        self.memory = memory
        self.target_id = target_id
        self.time_origin_s = time_origin_s
        self.target = {"entity_id": target_id, **(target or {})}
        self._counted_rounds: set[tuple[int, int]] = set()
        self._plans: dict[int, object] = {}
        if controller is not None and hasattr(controller, "bind_tools"):
            controller.bind_tools({
                "query_memory": lambda filters: self.memory.query(filters, cutoff=self.now_s)
                    if self.memory is not None else [],
                "predict_belief": lambda _args: self.filter.posterior.copy(),
            })
        self.now_s = time_origin_s

    def _return_probability(self, state: int, eta: float) -> float:
        if state not in self.filter.posterior:
            return 0.0
        forecast = self.filter.arrival(eta)
        # Equation 21 excludes probability already at this state.
        return max(0.0, forecast.get(state, 0.0) - self.filter.posterior.get(state, 0.0)
                   * self.filter.transition.matrix(list(self.filter.posterior), eta)[
                       list(self.filter.posterior).index(state), list(self.filter.posterior).index(state)])

    def _choose(self, result: AgentResult) -> int | str | None:
        scored: list[tuple[float, int | str]] = []
        for state, goal in self.goals.items():
            distance = self.world.distance(goal)
            if not math.isfinite(distance) or result.path_m + distance > self.config.max_path_m:
                continue
            eta = distance / self.speed + self.config.inspection_s
            arrival = self.filter.arrival(eta)
            return_probability = self._return_probability(state, eta)
            if not self.ledger.eligible(
                state, belief=self.filter.posterior.get(state, 0.0),
                return_probability=return_probability, new_coverage=0.0,
                dynamic=self.filter.transition.dynamic,
                now_s=result.elapsed_s,
            ):
                continue
            if hasattr(self.world, "plan_view"):
                plan = self.world.plan_view(state, self.ledger.covered_samples(state),
                                            round_id=self.ledger.round(state))
                if plan is None:
                    continue
                goal, new_detection = plan
                self._plans[state] = goal
                distance = self.world.distance(goal)
                if not math.isfinite(distance) or result.path_m + distance > self.config.max_path_m:
                    continue
                eta = distance / self.speed + self.config.inspection_s
                arrival = self.filter.arrival(eta)
            uncovered = max(0.0, 1.0 - self.ledger.coverage(state))
            if not hasattr(self.world, "plan_view"):
                new_detection = (
                    self.world.expected_new_detection(state, uncovered)
                    if hasattr(self.world, "expected_new_detection") else uncovered
                )
            utility = candidate_utility(
                arrival_probability=arrival.get(state, 0.0),
                new_detection_probability=new_detection,
                distance_m=distance, eta_s=eta,
                lambda_time=self.config.lambda_time,
                lambda_inspect=self.config.lambda_inspect,
            )
            scored.append((self.filter.posterior[state] if self.config.protocol == "n1" else utility, state))
        unknown = self.config.unknown_state
        if (unknown is not None and self.config.protocol != "n1"
                and result.explorations < self.config.max_explorations
                and self.filter.posterior.get(unknown, 0.0) > 0
                and getattr(self.world, "has_frontier", lambda: True)()):
            frontier_cost = getattr(self.world, "exploration_cost", lambda: self.config.explore_cost)()
            utility = (self.filter.posterior[unknown] * self.config.discovery_probability
                       / (frontier_cost + self.config.lambda_scan))
            scored.append((utility, "EXPLORE"))
        if not scored:
            return None
        preferred = max(scored, key=lambda row: (row[0], -row[1] if isinstance(row[1], int) else 0))
        if self.controller is not None:
            labels = [f"NAVIGATE_TO({action})" if isinstance(action, int) else action
                      for _, action in scored]
            chosen = self.controller.choose(labels, {
                "target": self.target,
                "memory": self.memory.query({"entity_id": self.target_id}, cutoff=self.now_s)
                    if self.memory is not None else [],
                "observation": getattr(self.world, "public_observation", {}),
                "belief": self.filter.posterior,
                "utilities": dict(zip(labels, [utility for utility, _ in scored], strict=True)),
                "elapsed_s": result.elapsed_s, "path_m": result.path_m,
            })
            for utility, action in scored:
                label = f"NAVIGATE_TO({action})" if isinstance(action, int) else action
                if label == chosen and utility >= preferred[0] - 1e-9:
                    return action
        return preferred[1]

    def _admit_negative(self, evidence: list[ViewEvidence], now_s: float,
                        result: AgentResult, *, negative: bool = True) -> None:
        for item in evidence:
            samples = self.ledger.new_samples(item)
            new_coverage = self.ledger.admit(item)
            key = (item.state_id, self.ledger.round(item.state_id))
            if self.ledger.coverage(item.state_id) >= self.ledger.sufficient_coverage and key not in self._counted_rounds:
                self._counted_rounds.add(key)
                self.ledger.mark_inspected(item.state_id, now_s=now_s)
                result.covered_inspections.append({
                    "state_id": item.state_id, "round": key[1], "time_s": now_s,
                    "evidence_id": item.evidence_id,
                })
            if negative and new_coverage > 0 and self.config.protocol in {"n1", "n3", "n4"}:
                calibrator = getattr(self.world, "calibrator", None)
                if calibrator is not None and item.features:
                    features = {**item.features, "coverage": new_coverage,
                                "projected_pixels": item.features["projected_pixels"]
                                * len(samples) / max(len(item.surface_samples), 1)}
                    if samples and all("projected_pixels" in item.sample_features.get(s, {}) for s in samples):
                        features["projected_pixels"] = sum(item.sample_features[s]["projected_pixels"] for s in samples)
                    for name in ("range_m", "angle_cos", "depth_quality"):
                        values = [item.sample_features[s][name] for s in samples if s in item.sample_features]
                        if values:
                            features[name] = sum(values) / len(values)
                    probability = calibrator.predict(features)
                else:
                    probability = min(1.0, item.detection_probability * new_coverage
                                      / max(item.covered_fraction, 1e-9))
                before = self.filter.posterior.get(item.state_id, 0.0)
                applied = self.filter.negative(
                    {item.state_id: probability},
                    f"{item.evidence_id}:{item.state_id}",
                )
                if applied:
                    result.evidence_trace.append({
                        "evidence_id": item.evidence_id,
                        "state_id": item.state_id,
                        "time_s": now_s,
                        "new_coverage": new_coverage,
                        "detection_probability": probability,
                        "prior": before,
                        "posterior": self.filter.posterior.get(item.state_id, 0.0),
                    })
                if applied and self.memory is not None:
                    self.memory.record_negative(
                        item.evidence_id, item.state_id,
                        self.time_origin_s + now_s, item.pose_xyz
                    )

    def _observe(self, observation, result: AgentResult) -> bool:
        detected, evidence = observation if isinstance(observation, tuple) else (False, observation)
        self.now_s = self.time_origin_s + result.elapsed_s
        self._admit_negative(evidence, result.elapsed_s, result, negative=not detected)
        if self.memory is not None and hasattr(self.world, "record_memory"):
            self.world.record_memory(self.memory, self.target_id, self.now_s)
        if detected:
            detection = getattr(self.world, "last_detection", None)
            if self.memory is not None and self.target_id is not None and detection is not None:
                self.memory.observe(
                    self.target_id, detection.get("state_id", result.inspections[-1] if result.inspections else -1),
                    self.now_s, detection["confidence"], detection["evidence_id"], detection["world_point"],
                    tuple(detection.get("feature", ())), category=self.target.get("category", ""),
                    attributes=self.target.get("attributes"), relations=detection.get("relations"),
                )
            if detection is not None:
                state = detection.get("state_id")
                if state not in self.goals:
                    return False
                if self.world.distance(self.goals[state]) > 1.:
                    confidence = float(detection["confidence"])
                    self.filter.posterior = {key: (1.-confidence)*mass + confidence*float(key == state)
                                             for key, mass in self.filter.posterior.items()}
                    return False
            result.found = True
            result.termination = "verified"
            result.actions.append("STOP")
        return detected

    def _budget(self, result: AgentResult) -> bool:
        for exhausted, reason in (
            (result.elapsed_s >= self.config.max_time_s, "time_budget_exhausted"),
            (result.steps >= self.config.max_steps, "step_budget_exhausted"),
            (len(result.covered_inspections) >= self.config.max_inspections, "inspection_budget_exhausted"),
        ):
            if exhausted:
                result.termination = reason
                return False
        return True

    def run(self) -> AgentResult:
        result = AgentResult()
        while self._budget(result):
            self.now_s = self.time_origin_s + result.elapsed_s
            selected = self._choose(result)
            if selected is None:
                result.termination = "search_exhausted"
                unknown = self.config.unknown_state
                unknown_mass = self.filter.posterior.get(unknown, 0.0) if unknown is not None else 0.0
                searchable = sum(self.filter.posterior.get(state, 0.0) for state in self.goals)
                no_frontier = not getattr(self.world, "has_frontier", lambda: False)()
                if (no_frontier or result.explorations >= self.config.max_explorations) and unknown_mass < 0.05 and searchable < 0.05:
                    result.actions.append("NOT_FOUND")
                break
            if selected == "EXPLORE":
                result.actions.append("EXPLORE")
                self.world.remaining_time_s = self.config.max_time_s - result.elapsed_s
                self.world.remaining_steps = self.config.max_steps - result.steps
                before_steps = getattr(self.world, "steps", 0)
                discovered, duration, displacement = self.world.explore(
                    min(self.config.chunk_m, self.config.max_path_m - result.path_m)
                )
                self.filter.advance(duration)
                result.elapsed_s += duration
                result.path_m += displacement
                result.steps += max(1, getattr(self.world, "steps", before_steps + 1) - before_steps)
                if not discovered:
                    if hasattr(self.world, "exploration_observation") and self._observe(self.world.exploration_observation, result):
                        break
                    if displacement > 0:
                        continue
                    result.termination = "exploration_exhausted"
                    break
                result.explorations += 1
                unknown = self.config.unknown_state
                mass = self.filter.posterior[unknown]
                total_weight = sum(weight for _, weight in discovered.values())
                allocated = min(mass, total_weight)
                self.filter.posterior[unknown] -= allocated
                for state, (goal, probability) in discovered.items():
                    self.goals[state] = goal
                    if hasattr(self.world, "sample_count"):
                        self.ledger.sample_count[state] = self.world.sample_count(state)
                    if hasattr(self.filter.transition, "add_candidate"):
                        self.filter.transition.add_candidate(state)
                    self.filter.posterior[state] = self.filter.posterior.get(state, 0.0) + (
                        allocated * probability / total_weight)
                if hasattr(self.world, "exploration_observation") and self._observe(self.world.exploration_observation, result):
                    break
                continue
            state = int(selected)
            goal = self._plans.get(state, self.goals[state])
            result.actions.append(f"NAVIGATE_TO({state})")
            result.visits.append({"state_id": state, "round": self.ledger.round(state),
                                  "time_s": result.elapsed_s})
            while self.world.distance(goal) > 1e-4:
                if not self._budget(result):
                    result.posterior = self.filter.posterior.copy()
                    return result
                remaining = self.config.max_path_m - result.path_m
                if remaining <= 0:
                    result.termination = "path_budget_exhausted"
                    result.posterior = self.filter.posterior.copy()
                    return result
                time_left = self.config.max_time_s - result.elapsed_s
                self.world.remaining_time_s = time_left
                self.world.remaining_steps = self.config.max_steps - result.steps
                before_steps = getattr(self.world, "steps", 0)
                displacement, duration = self.world.move_chunk(
                    goal, min(self.config.chunk_m, remaining, time_left * self.speed))
                if displacement <= 0 or duration <= 0:
                    result.termination = "navigation_blocked"
                    result.posterior = self.filter.posterior.copy()
                    return result
                self.filter.advance(duration)
                result.path_m += displacement
                result.elapsed_s += duration
                result.steps += max(1, getattr(self.world, "steps", before_steps + 1) - before_steps)
                self.speed = (self.config.speed_ema_alpha * self.speed
                              + (1 - self.config.speed_ema_alpha) * displacement / duration)
                if hasattr(self.world, "observe_chunk"):
                    if self._observe(self.world.observe_chunk(), result):
                        result.posterior = self.filter.posterior.copy()
                        return result
                if self._choose(result) != state:
                    break
            else:
                if not self._budget(result):
                    break
                if result.elapsed_s + self.config.inspection_s > self.config.max_time_s:
                    result.termination = "time_budget_exhausted"
                    break
                result.actions.append(f"INSPECT({state})")
                if hasattr(self.world, "advance_time"):
                    self.world.advance_time(self.config.inspection_s)
                self.filter.advance(self.config.inspection_s)
                result.elapsed_s += self.config.inspection_s
                result.steps += 1
                detected, evidence = self.world.inspect(state)
                result.inspections.append(state)
                self.ledger.mark_inspected(state, now_s=result.elapsed_s)
                if self._observe((detected, evidence), result):
                    break
        result.posterior = self.filter.posterior.copy()
        return result
