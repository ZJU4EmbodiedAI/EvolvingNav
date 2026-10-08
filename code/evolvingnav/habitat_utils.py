"""Habitat-Sim camera, NavMesh and agent-pose helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import habitat_sim
import numpy as np


def sensor_specs(width: int, height: int) -> list[Any]:
    specs = []
    for name, sensor_type in (
        ("rgb", habitat_sim.SensorType.COLOR),
        ("depth", habitat_sim.SensorType.DEPTH),
        ("semantic", habitat_sim.SensorType.SEMANTIC),
    ):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = name
        spec.sensor_type = sensor_type
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [height, width]
        spec.position = [0.0, 1.35, 0.0]
        spec.hfov = 79.0
        specs.append(spec)
    return specs


def make_simulator(
    hssd: Path, scene_id: str, navmesh: Path, gpu: int, width: int, height: int
) -> Any:
    sim_config = habitat_sim.SimulatorConfiguration()
    sim_config.scene_id = str(
        (hssd / f"scenes-uncluttered/{scene_id}.scene_instance.json").resolve()
    )
    sim_config.scene_dataset_config_file = str(
        (hssd / "hssd-hab-uncluttered.scene_dataset_config.json").resolve()
    )
    sim_config.enable_physics = True
    sim_config.create_renderer = True
    sim_config.load_semantic_mesh = True
    sim_config.gpu_device_id = gpu
    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = sensor_specs(width, height)
    simulator = habitat_sim.Simulator(
        habitat_sim.Configuration(sim_config, [agent_config])
    )
    if not simulator.pathfinder.load_nav_mesh(str(navmesh.resolve())):
        simulator.close()
        raise RuntimeError(f"cannot load navmesh {navmesh}")
    return simulator


def set_agent(agent: Any, position: list[float], xyzw: list[float]) -> None:
    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = np.asarray(xyzw, dtype=np.float32)
    agent.set_state(state)


def path_distance(pathfinder: Any, start: list[float], goal: list[float]) -> float:
    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_end = np.asarray(goal, dtype=np.float32)
    return float(path.geodesic_distance) if pathfinder.find_path(path) else math.inf
