"""World-blind patrol planning with private replay to obtain actual RGB-D."""
from __future__ import annotations

import random

from .io import uid


def planned_views(target, *, seed, observation_count):
    if observation_count <= 0:
        raise ValueError('observation_count must be positive')
    legal = sorted(p['state_id'] for p in target['placements'])
    if not legal:
        raise ValueError('target has no candidate views')
    rng = random.Random(int(uid(seed, target['object_id'], 'patrol'), 16))
    order = list(legal)
    rng.shuffle(order)
    return [order[i % len(order)] for i in range(observation_count)]


def _state_at(world, timestamp):
    state = world['initial_state']
    for event in sorted(world.get('events', []), key=lambda e: e['timestamp']):
        if event['timestamp'] > timestamp:
            break
        if event['previous_state'] != state:
            raise ValueError('non-causal patrol world')
        state = event['next_state']
    return state


def collect_patrol(target, world, *, seed, timestamps):
    if any(t < 0 for t in timestamps) or list(timestamps) != sorted(set(timestamps)):
        raise ValueError('timestamps must be unique, ordered and nonnegative')
    views = {v['view_id']: v for v in target['views']}
    schedule = planned_views(target, seed=seed, observation_count=len(timestamps))
    observations = []
    for timestamp, view_id in zip(timestamps, schedule):
        state = _state_at(world, timestamp)
        frame = target['frames'][state+'/'+view_id]
        visibility_kind = frame.get('visibility_kind', 'binary_pixel_threshold')
        visibility = (float(frame.get('visibility_fraction', 1.0 if frame.get('visible') else 0.0))
                      if visibility_kind == 'isolated_silhouette' else (1.0 if frame.get('visible') else 0.0))
        view = views[view_id]
        observations.append(dict(timestamp=timestamp, view_id=view_id,
                                 camera_pose=dict(position=view['position'],
                                                  rotation_wxyz=view['rotation_wxyz']),
                                 rgb=frame['rgb'], depth=frame['depth'],
                                 visible=bool(frame['visible']), visibility=visibility,
                                 visibility_kind=visibility_kind))
    return observations


def admit_patrol(observations, *, query_time):
    return bool(observations) and all(0 <= o['timestamp'] < query_time for o in observations) and any(
        o['visible'] for o in observations)
