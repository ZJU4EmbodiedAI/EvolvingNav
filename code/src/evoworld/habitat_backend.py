"""Optional Habitat-Sim bridge.

The generator remains usable without Habitat for schema tests, while this
module refuses to claim a scene is executable until scene config and navmesh
assets are present. RGB-D capture is deliberately isolated behind this API.
"""
from __future__ import annotations

from pathlib import Path


def validate_scene_assets(scene_file: str | Path, dataset_config: str | Path, navmesh: str | Path) -> dict[str, object]:
    paths = {"scene_file": Path(scene_file), "dataset_config": Path(dataset_config), "navmesh": Path(navmesh)}
    missing = [name for name, path in paths.items() if not path.is_file()]
    return {"executable": not missing, "missing": missing, "paths": {name: str(path) for name, path in paths.items()}}


def render_observation(*, scene_file: str | Path, dataset_config: str | Path, navmesh: str | Path, position: list[float], rotation: list[float], width: int = 320, height: int = 240, gpu_device: int = 0) -> dict[str, object]:
    assets = validate_scene_assets(scene_file, dataset_config, navmesh)
    if not assets["executable"]:
        raise FileNotFoundError(f"Habitat assets missing: {assets['missing']}")
    try:
        import habitat_sim
        import magnum as mn
    except ImportError as exc:
        raise RuntimeError("render_observation requires readyagent-m1-habitat") from exc
    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(Path(scene_file).resolve())
    config.scene_dataset_config_file = str(Path(dataset_config).resolve())
    config.create_renderer = True
    config.load_semantic_mesh = True
    config.gpu_device_id = gpu_device
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid, spec.sensor_type, spec.sensor_subtype = "rgb", habitat_sim.SensorType.COLOR, habitat_sim.SensorSubType.PINHOLE
    spec.resolution, spec.position, spec.hfov = [height, width], mn.Vector3(0.0, 1.35, 0.0), 86.0
    agent = habitat_sim.agent.AgentConfiguration(); agent.sensor_specifications = [spec]
    with habitat_sim.Simulator(habitat_sim.Configuration(config, [agent])) as sim:
        if not sim.pathfinder.load_nav_mesh(str(Path(navmesh).resolve())):
            raise RuntimeError(f"failed to load navmesh {navmesh}")
        state = sim.get_agent(0).get_state(); state.position = mn.Vector3(*position); state.rotation = mn.Quaternion(*rotation)
        sim.get_agent(0).set_state(state, reset_sensors=True)
        observation = sim.get_sensor_observations()
        return {"rgb": observation["rgb"], "camera_pose": list(position) + list(rotation), "timestamp": 0.0}
