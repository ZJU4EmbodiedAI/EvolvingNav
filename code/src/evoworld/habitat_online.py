"""Persistent Habitat-Sim execution for one native target episode.

This is an evaluator-side adapter.  Public observations contain RGB-D and a
visibility bit only; hidden object state, event application, and semantic
masks stay in :meth:`private_result`.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .visibility import calibrated_visibility


class NativeHabitatSession:
    def __init__(self, catalog, target, world, *, query_time, start_view,
                 speed_mps=1.0, max_distance_m=100.0, max_steps=500,
                 max_inspections=10, max_seconds=3600.0, gpu_device=0):
        try:
            import habitat_sim
            import magnum as mn
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("NativeHabitatSession requires readyagent-m1-habitat") from exc
        self._habitat_sim = habitat_sim
        self._mn = mn
        self.catalog = catalog
        self.target = target
        self.world = world
        self.clock = float(query_time)
        self.query_time = float(query_time)
        self.speed_mps = float(speed_mps)
        self.max_distance_m = float(max_distance_m)
        self.max_steps = int(max_steps)
        self.max_inspections = int(max_inspections)
        self.max_seconds = float(max_seconds)
        self.distance = 0.0
        self.steps = 0
        self.inspections = 0
        self.replans = 0
        self._closed = False
        self._events = sorted(world.get("events", []), key=lambda event: event["timestamp"])
        self._event_index = 0
        self._hidden_state = world["initial_state"]
        self._views = {view["view_id"]: view for view in target["views"]}
        self._placements = {placement["state_id"]: placement for placement in target["placements"]}
        if start_view not in self._views:
            raise KeyError(f"unknown start view: {start_view}")
        if self.speed_mps <= 0:
            raise ValueError("speed_mps must be positive")
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = str(Path(catalog["habitat_scene"]).resolve())
        cfg.scene_dataset_config_file = str(Path(catalog["dataset_config"]).resolve())
        cfg.enable_physics = True
        cfg.gpu_device_id = int(gpu_device)
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        height, width = catalog.get("resolution", [240, 320])
        specs = []
        for name, kind in (("rgb", habitat_sim.SensorType.COLOR),
                           ("depth", habitat_sim.SensorType.DEPTH),
                           ("semantic", habitat_sim.SensorType.SEMANTIC)):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid = name
            spec.sensor_type = kind
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            spec.resolution = [int(height), int(width)]
            spec.position = mn.Vector3(0.0, float(catalog.get("sensor_height_m", 1.35)), 0.0)
            spec.hfov = float(catalog.get("hfov_degrees", 79))
            specs.append(spec)
        agent_cfg.sensor_specifications = specs
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
        if not self.sim.pathfinder.load_nav_mesh(str(Path(catalog["navmesh"]).resolve())):
            self.close()
            raise RuntimeError("failed to load native navmesh")
        self._rom = self.sim.get_rigid_object_manager()
        self._target_object = self._add_target_object()
        self.view_id = start_view
        self._trace = []
        self._set_view(start_view)
        self._set_target_state(self._hidden_state)
        self._apply_events()
        self._historical_event_count = self._event_index

    def _add_target_object(self):
        tm = self.sim.get_object_template_manager()
        tm.load_configs(str(Path(self.target["template_path"]).resolve()))
        handles = tm.get_template_handles(self.target["template_hash"])
        if not handles:
            raise RuntimeError("native target template could not be loaded")
        obj = self._rom.add_object_by_template_handle(handles[-1])
        obj.motion_type = self._habitat_sim.physics.MotionType.KINEMATIC
        obj.semantic_id = int(self.target.get("semantic_id", 60000))
        return obj

    def _set_target_state(self, state_id):
        placement = self._placements.get(state_id)
        if placement is None:
            raise KeyError(f"missing native placement for hidden state {state_id}")
        self._target_object.translation = self._mn.Vector3(*placement["position"])
        w, x, y, z = placement.get("rotation_wxyz", [1, 0, 0, 0])
        self._target_object.rotation = self._mn.Quaternion(self._mn.Vector3(x, y, z), w)

    def _apply_events(self):
        applied = []
        while self._event_index < len(self._events) and self._events[self._event_index]["timestamp"] <= self.clock:
            event = self._events[self._event_index]
            if event.get("previous_state") != self._hidden_state:
                raise ValueError("non-causal hidden event during Habitat execution")
            self._hidden_state = event["next_state"]
            self._set_target_state(self._hidden_state)
            self._event_index += 1
            applied.append(dict(event))
        return applied

    def _set_view(self, view_id):
        view = self._views[view_id]
        state = self.sim.get_agent(0).get_state()
        state.position = self._mn.Vector3(*view["position"])
        import quaternion
        state.rotation = np.quaternion(*view["rotation_wxyz"])
        self.sim.get_agent(0).set_state(state, reset_sensors=True)
        self.view_id = view_id

    def _path(self, source, destination):
        path = self._habitat_sim.ShortestPath()
        path.requested_start = np.asarray(self._views[source]["position"], dtype=np.float32)
        path.requested_end = np.asarray(self._views[destination]["position"], dtype=np.float32)
        if not self.sim.pathfinder.find_path(path):
            raise RuntimeError(f"no native NavMesh path: {source}->{destination}")
        distance = float(path.geodesic_distance)
        if not math.isfinite(distance) or distance < 0:
            raise RuntimeError("invalid native NavMesh distance")
        points = [np.asarray(point, dtype=np.float32) for point in path.points]
        if len(points) < 2:
            points = [np.asarray(self._views[source]["position"], dtype=np.float32),
                      np.asarray(self._views[destination]["position"], dtype=np.float32)]
        return distance, points

    def _path_distance(self, source, destination):
        return self._path(source, destination)[0]

    def move_to(self, view_id):
        if view_id not in self._views:
            raise KeyError(f"unknown view: {view_id}")
        distance, points = self._path(self.view_id, view_id)
        segment_lengths = [float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:])]
        route_length = sum(segment_lengths)
        route_steps = sum(max(1, math.ceil(length / 0.25)) for length in segment_lengths)
        if self.steps + route_steps > self.max_steps:
            raise RuntimeError("navigation step budget exceeded")
        if self.distance + distance > self.max_distance_m:
            raise RuntimeError("navigation distance budget exceeded")
        duration = distance / self.speed_mps
        if self.clock + duration - self.query_time > self.max_seconds:
            raise RuntimeError("navigation time budget exceeded")
        source = self.view_id
        for a, b, length in zip(points, points[1:], segment_lengths):
            n = max(1, math.ceil(length / 0.25))
            for index in range(1, n + 1):
                state = self.sim.get_agent(0).get_state()
                state.position = self._mn.Vector3(*(a + (b - a) * (index / n)))
                self.sim.get_agent(0).set_state(state, reset_sensors=False)
                increment = distance * (length / route_length) / n if route_length else distance / max(route_steps, 1)
                self.distance += increment
                self.clock += increment / self.speed_mps
                self.steps += 1
                self._apply_events()
        self._set_view(view_id)
        self._apply_events()
        self._trace.append({"type": "move", "from_view": source, "to_view": view_id,
                            "distance_m": distance, "timestamp": self.clock})
        return {"clock": self.clock, "distance_m": self.distance}

    def inspect(self, view_id=None):
        if view_id is not None and view_id != self.view_id:
            self.move_to(view_id)
        if self.inspections + 1 > self.max_inspections:
            raise RuntimeError("inspection budget exceeded")
        self._apply_events()
        observations = self.sim.get_sensor_observations()
        semantic = np.asarray(observations["semantic"])
        target_mask = semantic == int(self.target.get("semantic_id", 60000))
        target_pixels = int(np.count_nonzero(target_mask))
        isolated_mask = self._isolated_target_mask()
        isolated_pixels = int(np.count_nonzero(isolated_mask))
        overlap_pixels = int(np.count_nonzero(target_mask & isolated_mask))
        calibration = calibrated_visibility(overlap_pixels, isolated_pixels)
        self.inspections += 1
        view = self._views[self.view_id]
        observation = {"timestamp": self.clock, "view_id": self.view_id,
                       "camera_pose": {"position": view["position"],
                                       "rotation_wxyz": view["rotation_wxyz"]},
                       "rgb": np.asarray(observations["rgb"])[..., :3].copy(),
                       "depth": np.asarray(observations["depth"]).copy()}
        self._trace.append({"type": "observation", "timestamp": self.clock,
                            "view_id": self.view_id})
        self._last_target_pixels = target_pixels
        self._last_isolated_pixels = isolated_pixels
        self._last_overlap_pixels = overlap_pixels
        return observation

    def _isolated_target_mask(self):
        """Render the same target/pose without stage or other HSSD objects."""
        from .bake import look_at, simulator

        isolated = simulator("NONE", self.catalog["dataset_config"],
                             resolution=tuple(self.catalog.get("resolution", [240, 320])))
        try:
            tm = isolated.get_object_template_manager()
            tm.load_configs(str(Path(self.target["template_path"]).resolve()))
            handles = tm.get_template_handles(self.target["template_hash"])
            if not handles:
                raise RuntimeError("isolated target template could not be loaded")
            obj = isolated.get_rigid_object_manager().add_object_by_template_handle(handles[-1])
            obj.motion_type = self._habitat_sim.physics.MotionType.KINEMATIC
            obj.semantic_id = int(self.target.get("semantic_id", 60000))
            placement = self._placements[self._hidden_state]
            obj.translation = self._mn.Vector3(*placement["position"])
            w, x, y, z = placement.get("rotation_wxyz", [1, 0, 0, 0])
            obj.rotation = self._mn.Quaternion(self._mn.Vector3(x, y, z), w)
            view = self._views[self.view_id]
            look_at(isolated, view["position"], rotation=view["rotation_wxyz"])
            return np.asarray(isolated.get_sensor_observations()["semantic"]) == obj.semantic_id
        finally:
            isolated.close()

    def replan(self, next_view):
        if next_view not in self._views:
            raise KeyError(f"unknown view: {next_view}")
        self.replans += 1
        self._trace.append({"type": "replan", "next_view": next_view, "timestamp": self.clock})

    def public_trace(self):
        return [dict(item) for item in self._trace]

    def private_result(self):
        isolated_pixels = getattr(self, "_last_isolated_pixels", 0)
        calibration = calibrated_visibility(getattr(self, "_last_overlap_pixels", 0), isolated_pixels)
        return {"clock": self.clock, "distance_m": self.distance, "steps": self.steps,
                "inspections": self.inspections, "replans": self.replans,
                "hidden_state": self._hidden_state,
                "historical_event_count": self._historical_event_count,
                "applied_event_count": self._event_index - self._historical_event_count,
                "target_pixels": getattr(self, "_last_target_pixels", 0),
                "isolated_target_pixels": isolated_pixels,
                "overlap_target_pixels": getattr(self, "_last_overlap_pixels", 0),
                "visibility_fraction": calibration["fraction"],
                "visibility_threshold": calibration["threshold"],
                "visibility_passes": calibration["passes"],
                "visibility_calibration_kind": "isolated_silhouette"}

    def close(self):
        if not self._closed:
            self._closed = True
            if hasattr(self, "sim"):
                self.sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
