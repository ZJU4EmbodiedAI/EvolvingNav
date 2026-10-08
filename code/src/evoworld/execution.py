"""Deterministic route-level N4 execution over a validated native catalog.

The agent-facing trace contains only actions and RGB-D evidence. Hidden state
and applied events remain on the evaluator-side executor object.
"""
from __future__ import annotations

import math


class DynamicEpisodeExecutor:
    def __init__(self, target, world, *, query_time, start_view,
                 speed_mps=1.0, max_distance_m=100.0, max_steps=500,
                 max_inspections=10, max_seconds=3600.0):
        self.target = target
        self.world = world
        self.clock = float(query_time)
        self.speed_mps = float(speed_mps)
        self.max_distance_m = float(max_distance_m)
        self.max_steps = int(max_steps)
        self.max_inspections = int(max_inspections)
        self.distance = 0.0
        self.steps = 0
        self.inspections = 0
        self.replans = 0
        self.view_id = None
        self._hidden_state = world.get('initial_state')
        self._events = sorted(world.get('events', []), key=lambda x: x['timestamp'])
        self._event_index = 0
        self._trace = []
        self._views = {v['view_id']: v for v in target.get('views', [])}
        self._paths = target.get('paths', {})
        self._set_view(start_view)
        self._apply_events()

    def _set_view(self, view_id):
        if view_id not in self._views:
            raise KeyError(f'unknown view: {view_id}')
        self.view_id = view_id

    def _apply_events(self):
        applied = []
        while self._event_index < len(self._events) and self._events[self._event_index]['timestamp'] <= self.clock:
            event = self._events[self._event_index]
            if event.get('previous_state') not in (None, self._hidden_state):
                raise ValueError('non-causal hidden event during execution')
            self._hidden_state = event['next_state']
            self._event_index += 1
            applied.append(event)
        return applied

    def _path_distance(self, source, destination):
        path = self._paths.get(f'{source}/{destination}')
        if path is None:
            raise KeyError(f'missing validated path: {source}/{destination}')
        distance = float(path['distance'])
        if not math.isfinite(distance) or distance < 0:
            raise ValueError('invalid path distance')
        return distance

    def move_to(self, view_id):
        distance = self._path_distance(self.view_id, view_id)
        if self.steps + 1 > self.max_steps or self.distance + distance > self.max_distance_m:
            raise RuntimeError('navigation budget exceeded')
        if self.speed_mps <= 0:
            raise ValueError('speed_mps must be positive')
        source = self.view_id
        self.distance += distance
        self.clock += distance / self.speed_mps
        self.steps += 1
        self._set_view(view_id)
        applied = self._apply_events()
        self._trace.append({'type': 'move', 'from_view': source, 'to_view': view_id,
                            'distance_m': distance, 'timestamp': self.clock})
        return {'clock': self.clock, 'distance_m': self.distance,
                'hidden_state': self._hidden_state, 'applied_events': applied}

    def inspect(self, view_id=None):
        if view_id is not None and view_id != self.view_id:
            self.move_to(view_id)
        if self.inspections + 1 > self.max_inspections:
            raise RuntimeError('inspection budget exceeded')
        self._apply_events()
        frame = self.target['frames'].get(f'{self._hidden_state}/{self.view_id}')
        if frame is None:
            raise KeyError(f'missing frame for hidden state/view {self._hidden_state}/{self.view_id}')
        self.inspections += 1
        observation = {'timestamp': self.clock, 'view_id': self.view_id,
                       'rgb': frame['rgb'], 'depth': frame['depth'],
                       'visible': bool(frame.get('visible', False))}
        self._trace.append({'type': 'observation', **observation})
        return observation

    def replan(self, next_view):
        if next_view not in self._views:
            raise KeyError(f'unknown view: {next_view}')
        self.replans += 1
        self._trace.append({'type': 'replan', 'next_view': next_view,
                            'timestamp': self.clock})

    def public_trace(self):
        return [dict(item) for item in self._trace]

    def evaluator_state(self):
        return {'clock': self.clock, 'distance_m': self.distance,
                'steps': self.steps, 'inspections': self.inspections,
                'replans': self.replans, 'hidden_state': self._hidden_state,
                'applied_event_count': self._event_index}
