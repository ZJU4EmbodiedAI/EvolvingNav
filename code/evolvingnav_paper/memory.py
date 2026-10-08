"""Causal, time-valid 4D entity memory (paper Equations 3, 17 and 18)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np


def backproject(u: float, v: float, depth_m: float, intrinsics: np.ndarray,
                camera_to_world: np.ndarray) -> np.ndarray:
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("depth must be positive and finite")
    ray = np.linalg.solve(np.asarray(intrinsics, dtype=float), [u, v, 1.0])
    camera_point = np.r_[depth_m * ray, 1.0]
    world = np.asarray(camera_to_world, dtype=float) @ camera_point
    return world[:3] / world[3]


@dataclass
class EntityVersion:
    valid_from: float
    valid_to: float | None
    state_id: int
    confidence: float
    feature: tuple[float, ...] = ()
    evidence_handles: list[str] = field(default_factory=list)
    world_points: list[tuple[float, float, float]] = field(default_factory=list)
    evidence_times: list[float] = field(default_factory=list)
    evidence_confidences: list[float] = field(default_factory=list)
    category: str = ""
    attributes: dict[str, str] = field(default_factory=dict)
    relations: dict[str, str] = field(default_factory=dict)
    point_clouds: dict[str, list] = field(default_factory=dict)


@dataclass(frozen=True)
class NegativeFrame:
    evidence_id: str
    state_id: int
    timestamp: float
    pose_xyz: tuple[float, float, float]


class VersionedMemory:
    def __init__(self) -> None:
        self.entities: dict[str, list[EntityVersion]] = {}
        self.used_evidence: set[str] = set()
        self.negative_frames: list[NegativeFrame] = []
        self.frames: dict[str, dict] = {}

    def record_negative(self, evidence_id: str, state_id: int, timestamp: float,
                        pose_xyz) -> None:
        if evidence_id in self.used_evidence:
            return
        self.negative_frames.append(NegativeFrame(
            evidence_id, state_id, timestamp, tuple(float(x) for x in pose_xyz)
        ))
        self.used_evidence.add(evidence_id)

    def evidence_at(self, cutoff: float) -> list[NegativeFrame]:
        return [frame for frame in self.negative_frames if frame.timestamp <= cutoff]

    def observe(self, entity_id: str, state_id: int, timestamp: float,
                confidence: float, evidence_id: str, world_point: np.ndarray,
                feature: tuple[float, ...] = (), *, category: str = "",
                attributes: dict | None = None, relations: dict | None = None) -> None:
        if evidence_id in self.used_evidence:
            return
        versions = self.entities.setdefault(entity_id, [])
        if versions and timestamp < versions[-1].evidence_times[-1]:
            raise ValueError("causal memory cannot accept an older observation")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be a probability")
        point = tuple(float(x) for x in world_point)
        attributes, relations = attributes or {}, relations or {}
        if (versions and versions[-1].state_id == state_id
                and versions[-1].attributes == attributes
                and versions[-1].relations == relations):
            current = versions[-1]
            current.confidence = max(current.confidence, confidence)
            current.evidence_handles.append(evidence_id)
            current.world_points.append(point)
            current.evidence_times.append(timestamp)
            current.evidence_confidences.append(confidence)
        else:
            if versions:
                versions[-1].valid_to = timestamp
            versions.append(EntityVersion(timestamp, None, state_id, confidence,
                                          feature, [evidence_id], [point],
                                          [timestamp], [confidence], category,
                                          dict(attributes), dict(relations)))
        self.used_evidence.add(evidence_id)

    @staticmethod
    def _causal(version: EntityVersion, cutoff: float) -> EntityVersion:
        indices = [i for i, time in enumerate(version.evidence_times) if time <= cutoff]
        return replace(
            version,
            valid_to=version.valid_to if version.valid_to is not None and version.valid_to <= cutoff else None,
            confidence=max(version.evidence_confidences[i] for i in indices),
            evidence_handles=[version.evidence_handles[i] for i in indices],
            world_points=[version.world_points[i] for i in indices],
            evidence_times=[version.evidence_times[i] for i in indices],
            evidence_confidences=[version.evidence_confidences[i] for i in indices],
            point_clouds={key: value for key, value in version.point_clouds.items()
                          if key in [version.evidence_handles[i] for i in indices]},
        )

    def at(self, entity_id: str, timestamp: float) -> EntityVersion | None:
        for version in reversed(self.entities.get(entity_id, [])):
            if version.valid_from <= timestamp and (version.valid_to is None or timestamp < version.valid_to):
                return self._causal(version, timestamp)
        return None

    def history(self, entity_id: str, cutoff: float) -> list[EntityVersion]:
        return [self._causal(v, cutoff)
                for v in self.entities.get(entity_id, []) if v.valid_from <= cutoff]

    def query(self, filters: dict, *, cutoff: float, limit: int = 64) -> list[dict]:
        upper = min(cutoff, float(filters.get("before_s", cutoff)))
        lower = float(filters.get("after_s", -np.inf))
        found = []
        for entity, versions in self.entities.items():
            if filters.get("entity_id", entity) != entity:
                continue
            for version in versions:
                if not lower <= version.valid_from <= upper:
                    continue
                value = self._causal(version, upper)
                if filters.get("category", value.category) != value.category:
                    continue
                if filters.get("state_id", value.state_id) != value.state_id:
                    continue
                if any(getattr(value, key).get(k) != v
                       for key in ("attributes", "relations")
                       for k, v in filters.get(key, {}).items()):
                    continue
                found.append({"entity_id": entity, **asdict(value)})
        return sorted(found, key=lambda v: v["valid_from"], reverse=True)[:limit]

    def register_frame(self, evidence_id: str, timestamp: float, *, rgb=None,
                       depth=None, pose=None) -> None:
        self.frames.setdefault(evidence_id, dict(timestamp=timestamp, rgb=rgb, depth=depth, pose=pose))

    def associate(self, category: str, point, feature, timestamp: float,
                  *, max_distance_m: float = 0.5, min_similarity: float = 0.8) -> str | None:
        candidates = []
        vector = np.asarray(feature, dtype=float)
        for entity in self.entities:
            version = self.at(entity, timestamp)
            if version is None or version.category != category:
                continue
            distance = np.linalg.norm(np.asarray(version.world_points[-1]) - point)
            previous = np.asarray(version.feature, dtype=float)
            similarity = (float(vector @ previous / max(np.linalg.norm(vector)*np.linalg.norm(previous), 1e-9))
                          if vector.shape == previous.shape and vector.size else 0.)
            if distance <= max_distance_m or similarity >= min_similarity:
                candidates.append((similarity + np.exp(-distance), entity))
        return max(candidates)[1] if candidates else None

    def ingest(self, observation: dict, detections: list[dict], state_centers: dict,
               *, entity_ids: dict[int, str] | None = None) -> list[str]:
        from evolvingnav_paper.coverage import camera_transform
        rgb, depth = np.asarray(observation["rgb"]), np.asarray(observation["depth"])
        timestamp = float(observation["timestamp"])
        pose = {key: observation[key] for key in ("position_xyz", "rotation_xyzw")}
        self.register_frame(observation["evidence_id"], timestamp, rgb=rgb.copy(), depth=depth.copy(), pose=pose)
        height, width = depth.shape
        focal = width / (2*np.tan(np.deg2rad(observation.get("hfov_degrees", 79.))/2))
        intrinsics = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
        transform = camera_transform(pose["position_xyz"], pose["rotation_xyzw"], optical=True)
        entities = []
        for index, detection in enumerate(detections):
            mask = np.asarray(detection["mask"], dtype=bool) & np.isfinite(depth) & (depth > 0)
            pixels = np.argwhere(mask)
            if not len(pixels):
                continue
            selected = pixels[np.linspace(0, len(pixels)-1, min(len(pixels), 128)).astype(int)]
            points = np.asarray([backproject(u, v, depth[v, u], intrinsics, transform) for v, u in selected])
            center = np.median(points, axis=0)
            feature = tuple(np.concatenate([np.histogram(rgb[..., c][mask], bins=16,
                                                        range=(0, 256))[0]/len(pixels) for c in range(3)]))
            entity = (entity_ids or {}).get(index) or self.associate(
                detection["category"], center, feature, timestamp)
            entity = entity or f"entity-{len(self.entities)}"
            state = min(state_centers, key=lambda key: np.linalg.norm(np.asarray(state_centers[key])-center))
            evidence = f"{observation['evidence_id']}:{index}"
            self.observe(entity, state, timestamp, float(detection["confidence"]), evidence, center,
                         feature, category=detection["category"], attributes=detection.get("attributes"),
                         relations=detection.get("relations", {"at": str(state)}))
            self.entities[entity][-1].point_clouds[evidence] = points.tolist()
            entities.append(entity)
        return entities
