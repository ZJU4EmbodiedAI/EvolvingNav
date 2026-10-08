"""Real HSSD candidate construction, per-object placement checks and RGB-D cache.

Each portable target is replayed in an independent world with the original HSSD
background. No geometry, depth, detections, or reachable points are fabricated.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np

from .io import dump, read_jsonl, uid

HEIGHT = 1.35
SEMANTIC_ID = 60000


def visibility_record(scene_semantic, isolated_semantic, semantic_id=SEMANTIC_ID, threshold=0.20):
    """Create private visibility fields from scene and isolated masks."""
    scene_mask = np.asarray(scene_semantic) == semantic_id
    isolated_mask = np.asarray(isolated_semantic) == semantic_id
    target_pixels = int(scene_mask.sum())
    isolated_pixels = int(isolated_mask.sum())
    visible_pixels = int((scene_mask & isolated_mask).sum())
    fraction = visible_pixels / isolated_pixels if isolated_pixels else 0.0
    return {"target_pixels": target_pixels,
            "isolated_target_pixels": isolated_pixels,
            "visible_target_pixels": visible_pixels,
            "visibility_fraction": fraction,
            "visible": fraction >= threshold,
            "visibility_kind": "isolated_silhouette"}


def simulator(scene, dataset, resolution=(240, 320)):
    import habitat_sim
    import magnum as mn
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(scene)
    cfg.scene_dataset_config_file = str(dataset)
    cfg.enable_physics = True
    cfg.gpu_device_id = 0
    agent = habitat_sim.agent.AgentConfiguration()
    sensors = []
    for name, kind in [("rgb", habitat_sim.SensorType.COLOR),
                       ("depth", habitat_sim.SensorType.DEPTH),
                       ("semantic", habitat_sim.SensorType.SEMANTIC)]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = name, kind
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution, spec.position, spec.hfov = list(resolution), mn.Vector3(0, HEIGHT, 0), 79
        sensors.append(spec)
    agent.sensor_specifications = sensors
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))


def overlay(hssd, sid, out):
    """Private overlay fixes inconsistent source descriptor references."""
    payload = json.loads((hssd / "scenes-uncluttered" / (sid + ".scene_instance.json")).read_text())
    # HSSD ships two valid registries. The unmodified scene instance must be
    # passed to Habitat because descriptor lookup uses the registry key, not an
    # arbitrary filename synthesized by a benchmark overlay.
    scene = hssd / "scenes-uncluttered" / (sid + ".scene_instance.json")
    dataset = hssd / ("hssd-hab.scene_dataset_config.json" if payload.get("semantic_scene_instance") == "hssd_ssd_map" else "hssd-hab-uncluttered.scene_dataset_config.json")
    return scene, dataset


def look_at(sim, position, target=None, rotation=None):
    from habitat_sim.utils.common import quat_from_two_vectors
    state = sim.get_agent(0).get_state()
    state.position = np.asarray(position, dtype=np.float32)
    if rotation is not None:
        import quaternion
        state.rotation = np.quaternion(*rotation)
    else:
        direction = np.asarray(target) - (state.position + [0, HEIGHT, 0])
        state.rotation = quat_from_two_vectors(np.array([0, 0, -1.0]), direction)
    sim.get_agent(0).set_state(state, reset_sensors=True)
    q = state.rotation
    return [float(q.real), *map(float, q.imag)]


def path_between(sim, a, b):
    import habitat_sim
    path = habitat_sim.ShortestPath()
    path.requested_start, path.requested_end = np.array(a), np.array(b)
    if not sim.pathfinder.find_path(path):
        return None
    return {"distance": float(path.geodesic_distance),
            "points": [list(map(float, p)) for p in path.points]}


def category(name):
    name = name.lower()
    for key, tokens in [
        ("desk", ["desk"]), ("dresser", ["dresser", "sideboard"]),
        ("counter", ["counter", "island", "cabinet"]), ("table", ["table", "nightstand"]),
        ("shelf", ["shelf", "shelving", "bookcase"])
    ]:
        if any(token in name for token in tokens):
            return key
    return None


def save_frame(out, ident, obs):
    from PIL import Image
    rgb = out / "frames" / (ident + ".png")
    depth = out / "frames" / (ident + ".npz")
    rgb.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(obs["rgb"][..., :3]).save(rgb)
    np.savez_compressed(depth, depth_m=obs["depth"].astype(np.float32))
    mask = obs["semantic"] == SEMANTIC_ID
    return {"rgb": str(rgb.resolve()), "depth": str(depth.resolve()),
            "target_pixels": int(mask.sum()),
            "depth_min_m": float(np.min(obs["depth"])),
            "depth_max_m": float(np.max(obs["depth"]))}


def bake_scene(hssd, sid, out, pool, seed=20260926, max_states=6, max_objects=4):
    import habitat_sim
    import magnum as mn
    from habitat.datasets.rearrange.samplers.receptacle import find_receptacles
    from habitat.sims.habitat_simulator import sim_utilities as su

    out.mkdir(parents=True, exist_ok=True)
    scene, dataset = overlay(hssd, sid, out)
    rng = random.Random(int(uid(seed, sid), 16))
    np.random.seed(int(uid(seed, sid), 16) % (2**32))
    sim = simulator(scene, dataset)
    sim.seed(seed)
    try:
        nav = out / "scene.navmesh"
        if nav.exists():
            assert sim.pathfinder.load_nav_mesh(str(nav))
        else:
            settings = habitat_sim.NavMeshSettings()
            settings.set_defaults()
            settings.agent_radius, settings.agent_height = .2, 1.5
            settings.include_static_objects = True
            assert sim.recompute_navmesh(sim.pathfinder, settings)
            sim.pathfinder.save_nav_mesh(str(nav))
        sim.pathfinder.seed(seed)
        island = max(range(sim.pathfinder.num_islands), key=sim.pathfinder.island_area)
        names = json.loads((hssd / "metadata/objects.json").read_text())
        region_rows = json.loads((hssd / "semantics/scenes" / (sid + ".semantic_config.json")).read_text())["region_annotations"]
        from matplotlib.path import Path as Polygon
        regions = [(r["label"], Polygon([[p[0], p[2]] for p in r["poly_loop"]])) for r in region_rows]
        active = set(json.loads((hssd / "scene_filter_files" / (sid + ".rec_filter.json")).read_text())["active"])
        recs = []
        rom = sim.get_rigid_object_manager()
        for rec in sorted(find_receptacles(sim), key=lambda r: r.unique_name):
            if rec.unique_name not in active or not rec.parent_object_handle:
                continue
            match = re.search(r"[0-9a-f]{40}", rec.parent_object_handle)
            name = names.get(match.group(0), {}).get("name", "") if match else ""
            kind = category(name)
            if not kind:
                continue
            points = [np.asarray(rec.sample_uniform_global(sim, .7), dtype=float) for _ in range(4)]
            center = np.mean(points, axis=0)
            nearest = np.asarray(sim.pathfinder.snap_point(center, island))
            if not np.isfinite(nearest).all() or np.linalg.norm(nearest[[0, 2]] - center[[0, 2]]) > 3:
                continue
            room = next((label for label, poly in regions if poly.contains_point(center[[0, 2]])), "unassigned")
            if room == "unassigned" or any(x in room.lower() for x in ["outdoor", "porch", "balcony", "garage"]):
                continue
            recs.append({"rec": rec, "points": points, "center": center, "category": kind,
                         "room": room, "name": name, "parent": rom.get_object_by_handle(rec.parent_object_handle)})
        # Round-robin rooms for candidate diversity; ordering does not use target state.
        grouped = {}
        rng.shuffle(recs)
        for rec in recs:
            grouped.setdefault(rec["room"], []).append(rec)
        recs = []
        while any(grouped.values()):
            for room in sorted(grouped):
                if grouped[room]:
                    recs.append(grouped[room].pop())
        pool_rows = list(read_jsonl(pool))
        chosen = []
        for cat in ("mug", "bottle", "book", "phone", "bowl", "plate"):
            matches = [p for p in pool_rows if p["category_canonical"] == cat]
            if matches:
                chosen.append(matches[0])
        objects, reasons = [], []
        for item in chosen[:max_objects]:
            tm = sim.get_object_template_manager()
            tm.load_configs(str(hssd / item["template_handle"]))
            handles = tm.get_template_handles(item["template_hash"])
            obj = rom.add_object_by_template_handle(handles[-1])
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            obj.semantic_id = SEMANTIC_ID
            obj.rotation = mn.Quaternion()
            accepted = []
            for rec in recs:
                if len(accepted) >= max_states:
                    break
                good = None
                for point in rec["points"]:
                    obj.translation = mn.Vector3(*(point + [0, .4, 0]))
                    if not su.snap_down(sim, obj, support_obj_ids=[rec["parent"].object_id], max_collision_depth=.005):
                        continue
                    pose = np.asarray(obj.translation, dtype=float).copy()
                    viewpoints = []
                    for radius in (1.1, 1.6, 2.2):
                        for angle in np.linspace(0, 2 * math.pi, 12, endpoint=False):
                            request = pose + [radius * math.cos(angle), -pose[1] + point[1] - .8, radius * math.sin(angle)]
                            vp = np.asarray(sim.pathfinder.snap_point(request, island), dtype=float)
                            if not np.isfinite(vp).all() or np.linalg.norm(vp[[0, 2]] - pose[[0, 2]]) < .6:
                                continue
                            rotation = look_at(sim, vp, pose)
                            obs = sim.get_sensor_observations()
                            pixels = int((obs["semantic"] == SEMANTIC_ID).sum())
                            if pixels >= 16:
                                viewpoints.append((pixels, vp.copy(), rotation))
                        if viewpoints:
                            break
                    if viewpoints:
                        pixels, vp, rotation = max(viewpoints, key=lambda v: v[0])
                        good = {
                            "state_id": uid("receptacle", sid, rec["rec"].unique_name),
                            "relation": "on", "receptacle_category": rec["category"], "room": rec["room"],
                            "support_point": point.tolist(), "position": pose.tolist(),
                            "rotation_wxyz": [1., 0., 0., 0.], "viewpoint": vp.tolist(),
                            "view_rotation_wxyz": rotation, "support_handle": rec["rec"].parent_object_handle,
                            "placement_check": {"snap_down": True, "max_collision_depth_m": .005},
                            "target_pixels_at_viewpoint": pixels,
                        }
                        break
                if good:
                    accepted.append(good)
            if len(accepted) < 4:
                reasons.append({"category": item["category_canonical"], "reason": "fewer than four collision/visibility checked states", "count": len(accepted)})
                rom.remove_object_by_id(obj.object_id)
                continue
            # All placements are individually checked with this exact mesh.
            # Last legal receptacle is hidden from the known candidate set.
            views = [{"view_id": p["state_id"], "position": p["viewpoint"],
                      "rotation_wxyz": p["view_rotation_wxyz"]} for p in accepted]
            starts = []
            for _ in range(250):
                start = sim.pathfinder.get_random_navigable_point(island_index=island)
                paths = [path_between(sim, start, p["viewpoint"]) for p in accepted]
                if any(p is None or p["distance"] < 3 for p in paths):
                    continue
                yaw = rng.uniform(-math.pi, math.pi)
                rotation = [math.cos(yaw / 2), 0, math.sin(yaw / 2), 0]
                view = {"view_id": uid("start", sid, len(objects), len(starts)),
                        "position": list(map(float, start)), "rotation_wxyz": rotation}
                # Admission checks all possible target positions, not current state.
                visible = False
                for p in accepted:
                    obj.translation = mn.Vector3(*p["position"])
                    look_at(sim, start, rotation=rotation)
                    if np.any(sim.get_sensor_observations()["semantic"] == SEMANTIC_ID):
                        visible = True
                        break
                if not visible:
                    starts.append(view)
                if len(starts) == 4:
                    break
            if not starts:
                reasons.append({"category": item["category_canonical"], "reason": "no start >=3m away and hidden for every candidate"})
                rom.remove_object_by_id(obj.object_id)
                continue
            views += starts
            object_id = uid("object", sid, item["template_hash"])
            frames = {}
            for p in accepted:
                obj.translation = mn.Vector3(*p["position"])
                for view in views:
                    look_at(sim, view["position"], rotation=view["rotation_wxyz"])
                    obs = sim.get_sensor_observations()
                    ident = uid("frame", seed, sid, object_id, p["state_id"], view["view_id"])
                    record = save_frame(out, ident, obs)
                    # Semantics are private evaluator annotations, never agent inputs.
                    record["visible"] = record["target_pixels"] >= 16
                    record["visibility_kind"] = "pixel_count; isolated-silhouette fraction not yet calibrated"
                    frames[p["state_id"] + "/" + view["view_id"]] = record
            paths = {}
            for a in views:
                for b in views:
                    path = path_between(sim, a["position"], b["position"])
                    if path is None:
                        raise ValueError("disconnected retained viewpoint")
                    paths[a["view_id"] + "/" + b["view_id"]] = path
            objects.append({"object_id": object_id, "category": item["category_canonical"],
                            "template_path": str(hssd / item["template_handle"]),
                            "template_hash": item["template_hash"], "semantic_id": SEMANTIC_ID,
                            "placements": accepted, "views": views, "starts": [s["view_id"] for s in starts],
                            "frames": frames, "paths": paths})
            rom.remove_object_by_id(obj.object_id)
            print(json.dumps({"scene": sid, "object": item["category_canonical"], "states": len(accepted), "frames": len(frames)}), flush=True)
        if len(objects) < 2:
            raise ValueError("scene has fewer than two validated target categories: " + str(reasons))
        result = {"scene_id": sid, "habitat_scene": str(scene.resolve()), "dataset_config": str(dataset.resolve()),
                  "navmesh": str(nav.resolve()), "habitat_sim_version": habitat_sim.__version__,
                  "island_id": island, "island_area_m2": sim.pathfinder.island_area(island),
                  "resolution": [240, 320], "sensor_height_m": HEIGHT, "hfov_degrees": 79,
                  "seed": seed, "objects": objects, "rejections": reasons,
                  "observation_backend": "habitat_rgbd", "world_scope": "one movable target per independent replay; HSSD background fixed"}
        dump(out / "catalog.json", result)
        return result
    finally:
        sim.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hssd", type=Path, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260926)
    a = p.parse_args()
    bake_scene(a.hssd.resolve(), a.scene, a.out.resolve(), a.pool, a.seed)


if __name__ == "__main__":
    main()
