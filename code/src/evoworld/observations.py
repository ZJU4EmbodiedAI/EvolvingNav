"""Past-only observation records and optional lightweight RGB-D stubs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Observation:
    timestamp: float
    location: str
    rgb: str
    depth: str
    camera_pose: list[float]
    visible_objects: list[str]
    visibility: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def write_observation_stub(path: Path, *, label: str, width: int = 96, height: int = 64) -> None:
    """Write a deterministic PPM/PGM pair without requiring Pillow.

    Habitat rendering can replace these files in-place. The stub is useful for
    deterministic contract checks and preserves the exact observation contract.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = path.with_suffix(".ppm")
    depth = path.with_name(path.stem + "_depth.pgm")
    color = (24 + (hash(label) & 31), 72, 86)
    with rgb.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode())
        handle.write(bytes(color) * (width * height))
    with depth.open("wb") as handle:
        handle.write(f"P5\n{width} {height}\n65535\n".encode())
        handle.write((1200).to_bytes(2, "big") * (width * height))


def history_before(observations: Iterable[Observation], query_time: float) -> list[dict[str, object]]:
    return [asdict(observation) for observation in observations if observation.timestamp <= query_time]


def write_history(path: Path, observations: Iterable[Observation]) -> None:
    path.write_text(json.dumps([asdict(item) for item in observations], indent=2), encoding="utf-8")
