"""Project public candidate geometry into an online depth frame (Appendix A.4)."""

from __future__ import annotations

import math

import numpy as np


def candidate_surface_samples(center_xyz, radius_m: float = 0.15,
                              place_points=None) -> np.ndarray:
    """A 5x5 surface grid from public receptacle slots or a center fallback."""
    center = np.asarray(center_xyz, dtype=float)
    if place_points:
        points = np.asarray(place_points, dtype=float)
        xs = np.linspace(points[:, 0].min(), points[:, 0].max(), 5)
        zs = np.linspace(points[:, 2].min(), points[:, 2].max(), 5)
        height = float(np.median(points[:, 1]))
        return np.asarray([[x, height, z] for x in xs for z in zs])
    offsets = np.linspace(-radius_m, radius_m, 5)
    return np.asarray([center + [x, 0.0, z] for x in offsets for z in offsets])


def _rotation(xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(xyzw, dtype=float)
    norm = x*x + y*y + z*z + w*w
    if norm <= 0:
        raise ValueError("invalid camera orientation")
    s = 2.0 / norm
    return np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)],
    ])


def camera_forward(xyzw) -> np.ndarray:
    return _rotation(xyzw) @ np.array([0.0, 0.0, -1.0])


def heading_quaternion(displacement_xyz) -> list[float]:
    direction = np.asarray(displacement_xyz, dtype=float)
    if math.hypot(direction[0], direction[2]) <= 1e-9:
        raise ValueError("heading needs horizontal displacement")
    yaw = math.atan2(-direction[0], -direction[2])
    return [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]


def depth_quality(depth: np.ndarray) -> float:
    values = np.asarray(depth)
    return float(np.mean(np.isfinite(values) & (values > 0)))


def view_features(samples, covered, position, rotation, depth, rgb=None,
                  *, category="", category_recall=.8) -> tuple[dict, dict]:
    transform = camera_transform(position, rotation)
    rays = np.asarray(samples)-transform[:3, 3]
    camera = rays @ transform[:3, :3]
    height, width = depth.shape
    focal = width/(2*np.tan(np.deg2rad(79.)/2))
    indices = sorted(covered)
    lengths = np.linalg.norm(rays, axis=1)
    cosines = np.maximum(0., rays @ camera_forward(rotation)/np.maximum(lengths, 1e-9))
    projected = camera[:, :2]*focal/np.maximum(-camera[:, 2:3], .05)
    in_front = camera[:, 2] < -.05
    extent = np.ptp(projected[in_front], axis=0) if in_front.any() else np.zeros(2)
    area_per_sample = float(np.clip(np.prod(np.maximum(extent, 1.)), 0., width*height)/len(samples))
    quality = depth_quality(depth)
    feature = {"coverage": len(indices)/len(samples),
               "range_m": float(np.mean(lengths[indices])) if indices else 0.,
               "angle_cos": float(np.mean(cosines[indices])) if indices else 0.,
               "projected_pixels": area_per_sample*len(indices), "depth_quality": quality,
               "image_quality": float(np.std(np.asarray(rgb)[..., :3])/128.) if rgb is not None else 0.,
               "category": category, "category_recall": category_recall}
    per_sample = {index: {"range_m": float(lengths[index]), "angle_cos": float(cosines[index]),
                          "depth_quality": quality, "projected_pixels": area_per_sample} for index in indices}
    return feature, per_sample


def camera_transform(agent_xyz, agent_xyzw, sensor_height_m: float = 1.35,
                     *, optical: bool = False) -> np.ndarray:
    rotation = _rotation(agent_xyzw)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(agent_xyz, dtype=float) + rotation @ [0, sensor_height_m, 0]
    if optical:
        transform[:3, :3] = rotation @ np.diag([1., -1., -1.])
    return transform


def visible_sample_ids(samples: np.ndarray, agent_xyz, agent_xyzw, depth: np.ndarray,
                       hfov_degrees: float, *, sensor_height_m: float = 1.35,
                       depth_tolerance_m: float = 0.25) -> frozenset[int]:
    height, width = depth.shape
    transform = camera_transform(agent_xyz, agent_xyzw, sensor_height_m)
    rotation = transform[:3, :3]
    camera_xyz = transform[:3, 3]
    camera_points = (np.asarray(samples, dtype=float) - camera_xyz) @ rotation
    focal = width / (2.0 * math.tan(math.radians(hfov_degrees) / 2.0))
    visible: set[int] = set()
    for index, (x, y, z) in enumerate(camera_points):
        if z >= -0.05:
            continue
        u = int(round(width / 2 + focal * x / -z))
        v = int(round(height / 2 - focal * y / -z))
        if 0 <= u < width and 0 <= v < height:
            measured = float(depth[v, u])
            if math.isfinite(measured) and measured > 0 and abs(measured + z) <= depth_tolerance_m:
                visible.add(index)
    return frozenset(visible)
