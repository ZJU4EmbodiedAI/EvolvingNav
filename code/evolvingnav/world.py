"""High-level Habitat action adapter; evaluator-private fields never enter Agent."""

from __future__ import annotations

import habitat_sim
import numpy as np
import base64
import io
import math
from PIL import Image

from evolvingnav.agent import ViewEvidence
from evolvingnav.coverage import (
    camera_transform, candidate_surface_samples,
    heading_quaternion,
    visible_sample_ids, view_features,
)
from evolvingnav.habitat_utils import set_agent
from evolvingnav.memory import backproject


class HabitatAgentWorld:
    def __init__(self, backend, viewpoints: dict[int, dict], state_centers: dict[int, list],
                 start_xyz, start_xyzw, *, speed_mps: float = 1.0,
                 calibrator=None, category_recall: float = 0.8,
                 known_states: set[int] | None = None,
                 motion_schedule: list[dict] | None = None,
                 surface_points: dict[int, list] | None = None) -> None:
        self.backend = backend
        self.viewpoints = viewpoints
        self.state_centers = state_centers
        self.position = np.asarray(start_xyz, dtype=float)
        self.rotation = list(start_xyzw)
        self.speed_mps = speed_mps
        self.calibrator = calibrator
        self.category_recall = (calibrator.category_recalls.get(getattr(backend, "target_category", ""), category_recall)
                                if calibrator is not None else category_recall)
        self.frame = 0
        self.private_inspections: list[dict] = []
        self.last_detection = None
        self.public_observation = {}
        self.steps = 0
        self.remaining_time_s = math.inf
        self.remaining_steps = 500
        self.active_views = {}
        self.used_views = {}
        self.exploration_observation = (False, [])
        surface_points = surface_points or {}
        self.samples = {state: candidate_surface_samples(
            center, place_points=surface_points.get(state))
                        for state, center in state_centers.items()}
        self.frontiers = set(viewpoints) - (known_states or set())
        self.known_states = set(known_states or viewpoints)
        self.motion_schedule = sorted(motion_schedule or [], key=lambda row: row["time_s"])
        self.world_time_s = 0.0
        self._next_motion = 0
        self.advance_time(0.)

    def advance_time(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("world time cannot move backwards")
        self.world_time_s += seconds
        while (self._next_motion < len(self.motion_schedule)
               and self.motion_schedule[self._next_motion]["time_s"] <= self.world_time_s):
            self.backend.move_target(self.motion_schedule[self._next_motion])
            self._next_motion += 1

    def distance(self, goal) -> float:
        return self.backend.distance(self.position, goal)

    def sample_count(self, state: int) -> int:
        return len(self.samples[state])

    def expected_new_detection(self, state: int, uncovered: float, viewpoint=None) -> float:
        if uncovered <= 0:
            return 0.
        viewpoint = viewpoint or self.viewpoints[state]
        if self.calibrator is None:
            return self.category_recall * uncovered
        features, _ = view_features(self.samples[state], frozenset(range(len(self.samples[state]))),
            viewpoint["position_xyz"], viewpoint["rotation_xyzw"], np.ones((240, 320)),
            category=getattr(self.backend, "target_category", ""), category_recall=self.category_recall)
        features["coverage"] = uncovered
        features["projected_pixels"] *= uncovered
        features["image_quality"] = .5
        return self.calibrator.predict(features)

    def plan_view(self, state: int, covered: frozenset[int], *, round_id: int = 0):
        from evolvingnav.coverage import _rotation
        options = []
        used = self.used_views.setdefault((state, round_id), set())
        for index, viewpoint in enumerate([self.viewpoints[state], *self.viewpoints[state].get("alternatives", [])]):
            if index in used:
                continue
            transform = camera_transform(viewpoint["position_xyz"], viewpoint["rotation_xyzw"])
            points = (self.samples[state]-transform[:3, 3]) @ _rotation(viewpoint["rotation_xyzw"])
            horizontal = np.tan(np.deg2rad(79.)/2)
            visible = {i for i, (x, y, z) in enumerate(points)
                       if z < -.05 and abs(x/-z) < horizontal and abs(y/-z) < horizontal*.75}
            new = len(visible-covered)/len(self.samples[state])
            distance = self.distance(viewpoint["position_xyz"])
            if new > .05 and np.isfinite(distance):
                probability = self.expected_new_detection(state, new, viewpoint)
                options.append((probability/(distance+.25), viewpoint, probability, index))
        if not options:
            return None
        _, viewpoint, probability, index = max(options, key=lambda row: row[0])
        self.active_views[state] = viewpoint
        self.active_views[(state, "round")] = round_id, index
        return viewpoint["position_xyz"], probability

    def has_frontier(self) -> bool:
        return bool(self.frontiers)

    def exploration_cost(self) -> float:
        return min((self.distance(self.viewpoints[state]["position_xyz"])
                    for state in self.frontiers), default=math.inf)

    def move_chunk(self, goal, max_distance: float) -> tuple[float, float]:
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(self.position, dtype=np.float32)
        path.requested_end = np.asarray(goal, dtype=np.float32)
        if not self.backend.real.pathfinder.find_path(path):
            return 0.0, 0.0
        max_distance = min(max_distance, self.remaining_time_s*self.speed_mps, self.remaining_steps*.25)
        remaining = min(float(max_distance), float(path.geodesic_distance))
        if remaining <= 0:
            return 0., 0.
        points = [np.asarray(point, dtype=float) for point in path.points]
        new_position = points[0]
        for point in points[1:]:
            length = float(np.linalg.norm(point - new_position))
            if length >= remaining:
                new_position = new_position + (point - new_position) * (remaining / length)
                break
            remaining -= length
            new_position = point
        displacement = min(float(max_distance), float(path.geodesic_distance))
        heading = new_position - self.position
        if np.linalg.norm(heading[[0, 2]]) > 1e-6:
            self.rotation = heading_quaternion(heading)
        self.position = new_position
        set_agent(self.backend.real.get_agent(0), self.position.tolist(), self.rotation)
        self.advance_time(displacement / self.speed_mps)
        self.steps += max(1, math.ceil(displacement/.25))
        return displacement, displacement / self.speed_mps

    def inspect(self, state: int) -> tuple[bool, list[ViewEvidence]]:
        viewpoint = self.active_views.get(state, self.viewpoints[state])
        self.position = np.asarray(viewpoint["position_xyz"], dtype=float)
        self.rotation = viewpoint["rotation_xyzw"]
        round_id, index = self.active_views.get((state, "round"), (0, 0))
        self.used_views.setdefault((state, round_id), set()).add(index)
        return self._sense(state)

    def _sense(self, state: int | None = None):
        private = self.backend.observe(self.position.tolist(), self.rotation)
        self.private_inspections.append({"state_id": state, "time_s": self.world_time_s,
                                        "position_xyz": self.position.tolist(),
                                        "rotation_xyzw": list(self.rotation), **private})
        observation = self.backend.last_observation
        if observation is None:
            raise RuntimeError("Habitat did not produce an RGB-D frame")
        self.last_detection = None
        if self.backend.last_detections:
            detection_index, strongest = max(enumerate(self.backend.last_detections), key=lambda row: row[1].confidence)
            depth = observation["depth"]
            pixels = np.argwhere(strongest.mask & np.isfinite(depth) & (depth > 0))
            if len(pixels):
                v, u = np.median(pixels, axis=0)
                depth_m = float(np.median(depth[pixels[:, 0], pixels[:, 1]]))
                height, width = depth.shape
                focal = width / (2 * np.tan(np.deg2rad(79.0) / 2))
                intrinsics = np.array([[focal, 0, width / 2],
                                       [0, focal, height / 2], [0, 0, 1]], dtype=float)
                point = backproject(u, v, depth_m, intrinsics,
                                    camera_transform(self.position, self.rotation, optical=True))
                observed_state = min(self.state_centers, key=lambda key: np.linalg.norm(
                    np.asarray(self.state_centers[key])-point))
                self.last_detection = {
                    "world_point": point,
                    "confidence": strongest.confidence,
                    "evidence_id": f"frame-{self.frame + 1}:{detection_index}",
                    "state_id": observed_state,
                }
        buffer = io.BytesIO()
        Image.fromarray(observation["rgb"][..., :3]).save(buffer, format="JPEG")
        self.public_observation = {
            "evidence_id": f"frame-{self.frame+1}", "time_s": self.world_time_s,
            "position_xyz": self.position.tolist(), "rotation_xyzw": list(self.rotation),
            "image_url": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(),
        }
        return bool(private["detected"]), self._evidence_from_depth(observation["depth"])

    def observe_chunk(self):
        return self._sense()

    def record_memory(self, memory, target_id, timestamp):
        observation = {**self.backend.last_observation, **self.public_observation,
                       "timestamp": timestamp}
        detections = [{"category": item.category, "confidence": item.confidence,
                       "mask": item.mask} for item in self.backend.last_detections]
        identities = {}
        if self.last_detection is not None and target_id is not None:
            identities[int(self.last_detection["evidence_id"].split(":")[-1])] = target_id
        memory.ingest(observation, detections, self.state_centers, entity_ids=identities)

    def _evidence_from_depth(self, depth: np.ndarray) -> list[ViewEvidence]:
        self.frame += 1
        evidence = []
        for candidate, samples in self.samples.items():
            if candidate not in self.known_states:
                continue
            covered = visible_sample_ids(
                samples, self.position, self.rotation, depth, 79.0
            )
            if covered:
                fraction = len(covered) / len(samples)
                features, sample_features = view_features(samples, covered, self.position, self.rotation,
                    depth, self.backend.last_observation["rgb"],
                    category=getattr(self.backend, "target_category", ""), category_recall=self.category_recall)
                if self.calibrator is None:
                    detection_probability = self.category_recall * fraction
                else:
                    detection_probability = self.calibrator.predict(features)
                evidence.append(ViewEvidence(
                    f"frame-{self.frame}:state-{candidate}", candidate, covered,
                    detection_probability, fraction, tuple(self.position.tolist()), features, sample_features,
                ))
        return evidence

    def explore(self, budget_m: float) -> tuple[dict[int, tuple[object, float]], float, float]:
        reachable = [
            (self.distance(self.viewpoints[state]["position_xyz"]), state)
            for state in self.frontiers
        ]
        reachable = [(distance, state) for distance, state in reachable
                     if np.isfinite(distance)]
        if not reachable:
            return {}, 0.0, 0.0
        distance, state = min(reachable)
        goal = self.viewpoints[state]["position_xyz"]
        displacement, duration = self.move_chunk(goal, min(distance, budget_m))
        if self.distance(goal) > 1e-4:
            self.exploration_observation = self._sense()
            return {}, duration, displacement
        self.frontiers.remove(state)
        self.known_states.add(state)
        self.rotation = self.viewpoints[state]["rotation_xyzw"]
        set_agent(self.backend.real.get_agent(0), self.position.tolist(), self.rotation)
        self.exploration_observation = self._sense(state)
        detected = self.exploration_observation[0]
        return {state: (goal, 1.0 if detected else 0.5)}, duration, displacement
