#!/usr/bin/env python3
"""Export the streaming release into the paper's scene/episode layout."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from evoworld.agent_bridge import public_scene_map


def public_objects(catalog):
    return [{'object_id': target['object_id'], 'category': target['category'],
             'candidate_states': public_scene_map(target)['states']}
            for target in catalog['objects']]


def link_or_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        return
    os.symlink(source.resolve(), destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('export requires a new directory; original releases are immutable')
    args.output.mkdir(parents=True)
    manifest=json.loads((args.dataset_root/'manifest.json').read_text())
    for split in manifest['split_counts']:
        link_or_copy(args.dataset_root / "episodes" / f"{split}.jsonl",
                     args.output / "episodes" / f"{split}.jsonl")
        link_or_copy(args.dataset_root / "private" / f"{split}.jsonl",
                     args.output / "private" / f"{split}.jsonl")
    link_or_copy(args.dataset_root / "manifest.json", args.output / "manifest.json")
    scene_ids = []
    for source in sorted(args.catalog_root.glob("*/catalog.json")):
        if source.parent.name not in manifest['scenes']:
            continue
        catalog = json.loads(source.read_text(encoding="utf-8"))
        scene = args.output / "scenes" / source.parent.name
        scene.mkdir(parents=True, exist_ok=True)
        (scene / "habitat_config.json").write_text(json.dumps({
            "scene_id": catalog["habitat_scene"],
            "dataset_config": catalog["dataset_config"],
            "resolution": catalog.get("resolution"),
            "sensor_height_m": catalog.get("sensor_height_m"),
            "hfov_degrees": catalog.get("hfov_degrees"),
            "observation_backend": catalog.get("observation_backend"),
        }, indent=2) + "\n", encoding="utf-8")
        (scene / "objects.json").write_text(json.dumps(public_objects(catalog), indent=2) + "\n", encoding="utf-8")
        link_or_copy(source, args.output/'private'/'catalogs'/source.parent.name/'catalog.json')
        link_or_copy(Path(catalog["navmesh"]), scene / "navmesh" / "scene.navmesh")
        scene_ids.append(source.parent.name)
    (args.output / "layout_manifest.json").write_text(json.dumps({
        "name": "EVOWORLD-BENCH", "format": "streaming_jsonl",
        "scene_count": len(scene_ids), "scene_ids": scene_ids,
        "source_dataset": str(args.dataset_root), "source_catalogs": str(args.catalog_root),
        "privacy": "public objects contain static support geometry only; mesh poses and frames are evaluator-private",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "scenes": len(scene_ids)}))


if __name__ == "__main__":
    main()
