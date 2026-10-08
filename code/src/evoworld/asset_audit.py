"""Local asset audit and 54-scene selection without silent proxying."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_assets(hssd_root: str | Path, sources_root: str | Path, output: str | Path) -> dict[str, Any]:
    hssd = Path(hssd_root)
    sources = Path(sources_root)
    scene_files = sorted((hssd / "scenes-uncluttered").glob("*.scene_instance.json"))
    source_paths = {"CASAS": sources / "casas/data.zip", "HD_EPIC": sources / "hd_epic_annotations", "ParaHome": sources / "parahome_data", "HOMER_PLUS": sources / "HOMER_PLUS"}
    recommended = hssd.parent / "task_datasets/HSSD/scene-selection-p4d/recommended_scenes.json"
    ranked_ids: list[str] = []
    if recommended.is_file():
        ranked_ids = [row["scene_id"] for row in json.loads(recommended.read_text()).get("scenes", [])]
    if not ranked_ids:
        ranked_ids = [path.name.removesuffix(".scene_instance.json") for path in scene_files]
    selected = []
    for scene_id in ranked_ids:
        if len(selected) >= 54:
            break
        scene = hssd / "scenes-uncluttered" / f"{scene_id}.scene_instance.json"
        sem = hssd / "semantics/scenes" / f"{scene_id}.semantic_config.json"
        if scene.is_file():
            nav = hssd.parents[1] / f"task_datasets/HSSD/cache/navmeshes/{scene_id}.navmesh"
            missing = [str(path) for path in (scene, sem, hssd / "hssd-hab-uncluttered.scene_dataset_config.json") if not path.is_file()]
            selected.append({"scene_id": scene_id, "scene_file": str(scene), "semantic_file": str(sem), "navmesh": str(nav), "navmesh_ready": nav.is_file(), "missing_assets": missing})
    report = {"benchmark": "EVOWORLD-BENCH", "hssd_scene_files": len(scene_files), "selected_scene_count": len(selected), "selected_scenes": selected, "scenes": selected, "complete_scene_count": sum(not item["missing_assets"] and item["navmesh_ready"] for item in selected), "sources": {name: {"path": str(path), "present": path.exists()} for name, path in source_paths.items()}, "download_required": [name for name, path in source_paths.items() if not path.exists()], "navmesh_ready_count": sum(item["navmesh_ready"] for item in selected), "notes": ["Scene files and source traces are local; missing navmeshes must be generated and runtime-probed before a full release.", "No missing human trace is replaced with copied episodes or zero-valued results."]}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
