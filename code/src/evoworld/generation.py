"""Deterministic benchmark materialization from synthetic household worlds."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .events import Event
from .evolution import REGIMES, generate_events, replay_state
from .observations import Observation, history_before, write_observation_stub

TASKS = ("N1", "N2", "N3", "N4", "N5")
TASK_WEIGHTS = (0.1351, 0.1337, 0.1327, 0.2652, 0.0334)
OBJECTS = ("mug", "keys", "book", "bottle")
LOCATIONS = ("kitchen_table", "kitchen_cabinet", "dining_table", "entry_drawer", "living_desk", "unknown")


def _split(day: int, days: int) -> str:
    if days == 90:
        return "train" if day < 80 else "val" if day < 85 else "test"
    train_end = max(1, int(days * 0.60))
    val_end = max(train_end + 1, int(days * 0.80))
    return "train" if day < train_end else "val" if day < val_end else "test"


def _stable_id(*parts: object) -> str:
    raw = "|".join(map(str, parts)).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def _candidate_states(scene_id: str) -> list[dict[str, Any]]:
    states = []
    for index, name in enumerate(LOCATIONS[:-1]):
        states.append({"state_id": f"{scene_id}:state_{index}", "semantic_relation": name.split("_", 1)[1], "room": name.split("_", 1)[0], "geometric_location": [float(index), 0.8, float(index % 2)], "inspection_viewpoint": [float(index) + 0.6, 1.35, float(index % 2) + 0.5], "reachable": True})
    states.append({"state_id": f"{scene_id}:unknown", "semantic_relation": "unknown", "room": None, "geometric_location": None, "inspection_viewpoint": None, "reachable": True})
    return states


def _source_scene(scene_id: str, asset_root: Path | None) -> dict[str, Any]:
    if asset_root is None:
        return {"scene_id": scene_id, "asset_status": "synthetic_stub", "scene_file": None, "dataset_config": None, "navmesh": None}
    scene = asset_root / "scenes-uncluttered" / f"{scene_id}.scene_instance.json"
    config = asset_root / "hssd-hab-uncluttered.scene_dataset_config.json"
    nav = asset_root.parents[1] / "task_datasets" / "HSSD" / "cache" / "navmeshes" / f"{scene_id}.navmesh"
    return {"scene_id": scene_id, "asset_status": "habitat_scene_present" if scene.is_file() else "missing_scene", "scene_file": str(scene), "dataset_config": str(config), "navmesh": str(nav) if nav.is_file() else None}


def _episode(
    *, scene_id: str, day: int, episode_index: int, query_time: float, task: str,
    regime: str, events: list[Event], output_root: Path, rng: random.Random,
    observation_cache: dict[tuple[str, str, int], tuple[str, str]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    object_id = f"{OBJECTS[episode_index % len(OBJECTS)]}_0"
    initial = {f"{category}_0": LOCATIONS[index % (len(LOCATIONS) - 1)] for index, category in enumerate(OBJECTS)}
    candidates = {key: list(LOCATIONS[:-1]) for key in initial}
    current = replay_state(initial, events, query_time)
    candidate_rows = _candidate_states(scene_id)
    observations: list[Observation] = []
    for obs_index, stamp in enumerate((max(0.0, query_time - 86400.0), max(0.0, query_time - 3600.0))):
        state = replay_state(initial, events, stamp)[object_id]
        cache_key = (scene_id, state, obs_index)
        if observation_cache is not None and cache_key in observation_cache:
            rgb_path, depth_path = observation_cache[cache_key]
        else:
            stem = output_root / "observations" / "cache" / f"{scene_id}_{state}_{obs_index}"
            write_observation_stub(stem, label=f"{scene_id}:{state}:{obs_index}")
            rgb_path, depth_path = str(stem.with_suffix(".ppm")), str(stem.with_name(stem.stem + "_depth.pgm"))
            if observation_cache is not None:
                observation_cache[cache_key] = (rgb_path, depth_path)
        observations.append(Observation(timestamp=stamp, location=state, rgb=rgb_path, depth=depth_path, camera_pose=[0.0, 1.35, 0.0, 0.0, 0.0, 0.0, 1.0], visible_objects=[object_id] if rng.random() > 0.35 else [], visibility=0.85 if rng.random() > 0.25 else 0.35))
    public = {
        "id": f"{scene_id}:{day:02d}:{episode_index:08d}",
        "scene": scene_id,
        "target": {"object_id": object_id, "category": object_id.split("_", 1)[0]},
        "timestamp_history": [item["timestamp"] for item in history_before(observations, query_time)],
        "query_time": query_time,
        "history": history_before(observations, query_time),
        "candidate_states": candidate_rows,
        "navigation_budget": {"max_distance_m": 100.0, "max_steps": 500, "max_inspections": 10},
        "task_type": task,
        "mobility_regime": "hidden",
        "public_fields": ["id", "scene", "target", "timestamp_history", "query_time", "history", "candidate_states", "navigation_budget", "task_type"],
    }
    private = {
        "id": public["id"], "scene": scene_id, "target_object": object_id,
        "current_state": current[object_id], "mobility_regime": regime,
        "world_id": f"{scene_id}:{regime}",
        "query_window": [query_time, query_time + 86400.0],
        "task_type": task,
    }
    return public, private


def generate_dataset(
    *, output_root: str | Path, scene_ids: list[str], days: int, episodes: int,
    seed: int = 20260926, asset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a public release and evaluator-only state for a fixed seed."""
    if not scene_ids or days <= 0 or episodes <= 0:
        raise ValueError("scene_ids, days and episodes must be positive")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    asset = Path(asset_root) if asset_root else None
    rng = random.Random(seed)
    scene_assets = {}
    for scene_id in scene_ids:
        scene_assets[scene_id] = _source_scene(scene_id, asset)
        scene_dir = root / "scenes" / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / "habitat_config.json").write_text(json.dumps({"scene_id": scene_id, **scene_assets[scene_id]}, indent=2), encoding="utf-8")
        (scene_dir / "objects.json").write_text(json.dumps([{"object_id": f"{category}_0", "category": category, "identity_persistent": True} for category in OBJECTS], indent=2), encoding="utf-8")
        (scene_dir / "navmesh").write_text(json.dumps({"status": "referenced_external_asset", "path": scene_assets[scene_id]["navmesh"]}), encoding="utf-8")
    counts = Counter()
    episode_paths = {split: (root / "episodes" / f"{split}.jsonl") for split in ("train", "val", "test")}
    private_paths = {split: (root / "private" / f"{split}.jsonl") for split in ("train", "val", "test")}
    for path in (*episode_paths.values(), *private_paths.values()):
        path.parent.mkdir(parents=True, exist_ok=True)
    public_handles = {split: path.open("w", encoding="utf-8") for split, path in episode_paths.items()}
    private_handles = {split: path.open("w", encoding="utf-8") for split, path in private_paths.items()}
    observation_cache: dict[tuple[str, str, int], tuple[str, str]] = {}
    world_cache: dict[tuple[str, str], tuple[dict[str, str], list[Event]]] = {}
    world_dir = root / "private" / "worlds"
    world_dir.mkdir(parents=True, exist_ok=True)
    for index in range(episodes):
        scene_id = scene_ids[index % len(scene_ids)]
        day = (index // len(scene_ids)) % days
        # Iterate complete scene blocks inside each regime so every scene has
        # all four paired worlds, rather than coupling 54 scenes to a 4-way
        # modulo pattern (gcd(54, 4) would otherwise drop half the pairs).
        regime = REGIMES[(index // len(scene_ids)) % len(REGIMES)]
        task = rng.choices(TASKS, weights=TASK_WEIGHTS, k=1)[0]
        world_key = (scene_id, regime)
        if world_key not in world_cache:
            initial = {f"{category}_0": LOCATIONS[object_index % (len(LOCATIONS) - 1)] for object_index, category in enumerate(OBJECTS)}
            legal = {key: list(LOCATIONS[:-1]) for key in initial}
            generated_events = generate_events(initial, legal, days=days, regime=regime, seed=seed + len(world_cache), actor_profiles={"keys_0": {"preferred_state": "entry_drawer"}})
            world_cache[world_key] = (initial, generated_events)
            world_path = world_dir / f"{scene_id}__{regime}.json"
            world_path.write_text(json.dumps({"world_id": f"{scene_id}:{regime}", "scene": scene_id, "mobility_regime": regime, "initial_state": initial, "events": [event.to_dict() for event in generated_events]}, sort_keys=True), encoding="utf-8")
        initial, events = world_cache[world_key]
        query_time = day * 86400.0 + 12.0 * 3600.0
        public, private = _episode(scene_id=scene_id, day=day, episode_index=index, query_time=query_time, task=task, regime=regime, events=events, output_root=root, rng=rng, observation_cache=observation_cache)
        split = _split(day, days)
        public_handles[split].write(json.dumps(public, sort_keys=True) + "\n")
        private_handles[split].write(json.dumps(private, sort_keys=True) + "\n")
        counts[task] += 1
    for handle in (*public_handles.values(), *private_handles.values()):
        handle.close()
    split_counts = {split: sum(1 for _ in open(path, encoding="utf-8")) for split, path in episode_paths.items()}
    manifest = {"name": "EVOWORLD-BENCH", "version": "0.1.0", "seed": seed, "scene_count": len(scene_ids), "days": days, "scene_day_count": len(scene_ids) * days, "episode_count": episodes, "task_counts": dict(counts), "splits": split_counts, "scene_assets": scene_assets, "observation_cache_count": len(observation_cache), "privacy": {"public_excludes": ["current_state", "hidden_future_events", "all_events", "mobility_regime"], "private_evaluator": "private/*.jsonl"}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"episode_count": episodes, "scene_day_count": len(scene_ids) * days, "splits": manifest["splits"], "task_counts": dict(counts)}
