"""Offline calibration of native catalog visibility using isolated target renders."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bake import simulator, look_at, visibility_record


def prune_unverifiable_states(catalog, *, threshold=0.20, min_states=4):
    """Remove legal placements that have no viewpoint meeting stop success."""
    import copy
    result = copy.deepcopy(catalog)
    for target in result.get("objects", []):
        good_states = {placement["state_id"] for placement in target.get("placements", [])
                       if max((float(frame.get("visibility_fraction", 0.0))
                               for key, frame in target.get("frames", {}).items()
                               if key.startswith(placement["state_id"] + "/")), default=0.0) >= threshold}
        if len(good_states) < min_states:
            raise ValueError(f"target {target.get('object_id')} has fewer than four calibrated states")
        target["placements"] = [p for p in target["placements"] if p["state_id"] in good_states]
        kept_views = {p["state_id"] for p in target["placements"]} | set(target.get("starts", []))
        target["views"] = [v for v in target.get("views", []) if v["view_id"] in kept_views]
        target["frames"] = {key: frame for key, frame in target.get("frames", {}).items()
                             if key.split("/", 1)[0] in good_states}
        target["paths"] = {key: path for key, path in target.get("paths", {}).items()
                            if key.split("/", 1)[0] in kept_views and key.split("/", 1)[1] in kept_views}
    result["state_pruning"] = {"method": "isolated_silhouette_threshold", "threshold": threshold}
    return result


def calibrated_frame_record(frame, isolated_target_pixels, *, visible_target_pixels=None, threshold=0.20):
    visible_pixels = max(0, int(frame.get("target_pixels", 0) if visible_target_pixels is None else visible_target_pixels))
    isolated_pixels = max(0, int(isolated_target_pixels))
    fraction = visible_pixels / isolated_pixels if isolated_pixels else 0.0
    return {**frame,
            "visible_target_pixels": visible_pixels,
            "isolated_target_pixels": isolated_pixels,
            "visibility_fraction": fraction,
            "visible": fraction >= threshold,
            "visibility_kind": "isolated_silhouette"}


def _isolated_pixels(catalog, target, state_id, view_id):
    import habitat_sim
    import magnum as mn

    sim = simulator("NONE", catalog["dataset_config"],
                    resolution=tuple(catalog.get("resolution", [240, 320])))
    try:
        tm = sim.get_object_template_manager()
        tm.load_configs(str(Path(target["template_path"]).resolve()))
        handles = tm.get_template_handles(target["template_hash"])
        if not handles:
            raise RuntimeError(f"template missing: {target['template_hash']}")
        obj = sim.get_rigid_object_manager().add_object_by_template_handle(handles[-1])
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.semantic_id = int(target.get("semantic_id", 60000))
        placement = next(p for p in target["placements"] if p["state_id"] == state_id)
        obj.translation = mn.Vector3(*placement["position"])
        w, x, y, z = placement.get("rotation_wxyz", [1, 0, 0, 0])
        obj.rotation = mn.Quaternion(mn.Vector3(x, y, z), w)
        view = next(v for v in target["views"] if v["view_id"] == view_id)
        look_at(sim, view["position"], rotation=view["rotation_wxyz"])
        semantic = np.asarray(sim.get_sensor_observations()["semantic"])
        return int(np.count_nonzero(semantic == obj.semantic_id))
    finally:
        sim.close()


def _isolated_renderer(catalog, target):
    import habitat_sim
    import magnum as mn

    sim = simulator("NONE", catalog["dataset_config"],
                    resolution=tuple(catalog.get("resolution", [240, 320])))
    tm = sim.get_object_template_manager()
    tm.load_configs(str(Path(target["template_path"]).resolve()))
    handles = tm.get_template_handles(target["template_hash"])
    if not handles:
        sim.close()
        raise RuntimeError(f"template missing: {target['template_hash']}")
    obj = sim.get_rigid_object_manager().add_object_by_template_handle(handles[-1])
    obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    obj.semantic_id = int(target.get("semantic_id", 60000))
    return sim, obj


def calibrate_catalog(input_path, output_path):
    """Copy one catalog and attach isolated-silhouette visibility metadata."""
    input_path, output_path = Path(input_path), Path(output_path)
    catalog = json.loads(input_path.read_text(encoding="utf-8"))
    import magnum as mn
    scene_sim = simulator(catalog["habitat_scene"], catalog["dataset_config"],
                          resolution=tuple(catalog.get("resolution", [240, 320])))
    try:
        for target in catalog["objects"]:
            scene_tm = scene_sim.get_object_template_manager()
            scene_tm.load_configs(str(Path(target["template_path"]).resolve()))
            scene_handles = scene_tm.get_template_handles(target["template_hash"])
            if not scene_handles:
                raise RuntimeError(f"template missing: {target['template_hash']}")
            scene_obj = scene_sim.get_rigid_object_manager().add_object_by_template_handle(scene_handles[-1])
            import habitat_sim
            scene_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            scene_obj.semantic_id = int(target.get("semantic_id", 60000))
            isolated_sim, isolated_obj = _isolated_renderer(catalog, target)
            try:
                placements = {p["state_id"]: p for p in target["placements"]}
                views = {v["view_id"]: v for v in target["views"]}
                for key, frame in target["frames"].items():
                    state_id, view_id = key.split("/", 1)
                    placement = placements[state_id]
                    w, x, y, z = placement.get("rotation_wxyz", [1, 0, 0, 0])
                    for obj in (scene_obj, isolated_obj):
                        obj.translation = mn.Vector3(*placement["position"])
                        obj.rotation = mn.Quaternion(mn.Vector3(x, y, z), w)
                    view = views[view_id]
                    look_at(scene_sim, view["position"], rotation=view["rotation_wxyz"])
                    look_at(isolated_sim, view["position"], rotation=view["rotation_wxyz"])
                    scene_mask = np.asarray(scene_sim.get_sensor_observations()["semantic"])
                    isolated_mask = np.asarray(isolated_sim.get_sensor_observations()["semantic"])
                    exact = visibility_record(scene_mask, isolated_mask, semantic_id=scene_obj.semantic_id)
                    if exact["target_pixels"] != int(frame["target_pixels"]):
                        raise ValueError(f"catalog render drift: {catalog['scene_id']} {target['object_id']} {key}")
                    target["frames"][key] = {**frame, **exact}
            finally:
                isolated_sim.close()
                scene_sim.get_rigid_object_manager().remove_object_by_id(scene_obj.object_id)
    finally:
        scene_sim.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return catalog


def calibrate_catalog_root(input_root, output_root, scene_ids=None):
    input_root, output_root = Path(input_root), Path(output_root)
    if scene_ids is None:
        scene_ids = sorted(path.parent.name for path in input_root.glob("*/catalog.json"))
    for scene_id in scene_ids:
        source = input_root / scene_id / "catalog.json"
        destination = output_root / scene_id / "catalog.json"
        calibrate_catalog(source, destination)
    return {"scenes": list(scene_ids), "input_root": str(input_root), "output_root": str(output_root),
            "visibility_kind": "isolated_silhouette", "calibration_method": "scene_isolated_mask_overlap",
            "threshold": 0.20}
