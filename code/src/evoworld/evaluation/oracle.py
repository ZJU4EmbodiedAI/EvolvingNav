"""Evaluator-only time-feasible reference paths for dynamic episodes."""
from __future__ import annotations

import heapq
import math

from ..visibility import visibility_fraction


def _state_and_index(world, timestamp):
    state = world["initial_state"]
    index = 0
    for event in sorted(world.get("events", []), key=lambda item: item["timestamp"]):
        if event["timestamp"] > timestamp:
            break
        state = event["next_state"]
        index += 1
    return state, index


def dynamic_oracle_distance(target, world, *, query_time, start_view,
                            speed_mps=1.0, max_seconds=3600.0,
                            max_distance_m=100.0, max_steps=500,
                            threshold=0.20):
    """Find minimum travel distance to a viewpoint valid at arrival time.

    The oracle sees the private event schedule.  It does not use a static
    target shortcut: each candidate route is evaluated at its arrival state.
    ``None`` means no time/distance-feasible verification viewpoint exists.
    """
    if speed_mps <= 0 or max_seconds < 0 or max_distance_m < 0:
        raise ValueError("invalid oracle budget")
    views = {view["view_id"]: view for view in target.get("views", [])}
    if start_view not in views:
        raise KeyError(f"unknown start view: {start_view}")
    events = sorted(world.get("events", []), key=lambda item: item["timestamp"])
    start_state, start_index = _state_and_index(world, float(query_time))

    def valid(state, view):
        try:
            return visibility_fraction(target, state, view) >= float(threshold)
        except KeyError:
            return False

    heap = [(0.0, float(query_time), start_view, start_index, 0)]
    labels = {(start_view, start_index): [(0.0, float(query_time), 0)]}

    def push(distance, timestamp, view, index, steps):
        key = (view, index)
        candidate = (distance, timestamp, steps)
        current = labels.setdefault(key, [])
        if any(d <= distance and t <= timestamp and s <= steps
               for d, t, s in current):
            return
        current[:] = [(d, t, s) for d, t, s in current
                      if not (distance <= d and timestamp <= t and steps <= s)]
        current.append(candidate)
        heapq.heappush(heap, (distance, timestamp, view, index, steps))

    while heap:
        distance, timestamp, current, event_index, steps = heapq.heappop(heap)
        if (distance, timestamp, steps) not in labels.get((current, event_index), []):
            continue
        state = world["initial_state"] if not event_index else events[event_index - 1]["next_state"]
        if valid(state, current):
            return distance
        if event_index < len(events):
            next_time = float(events[event_index]["timestamp"])
            if next_time >= timestamp and next_time - query_time <= max_seconds:
                _, next_index = _state_and_index(world, next_time)
                push(distance, next_time, current, next_index, steps)
        if steps >= max_steps:
            continue
        for destination in views:
            if destination == current:
                continue
            path = target.get("paths", {}).get(f"{current}/{destination}")
            if path is None:
                continue
            edge = float(path.get("distance", math.inf))
            if not math.isfinite(edge) or edge < 0:
                continue
            new_distance = distance + edge
            new_timestamp = timestamp + edge / speed_mps
            if new_distance > max_distance_m or new_timestamp - query_time > max_seconds:
                continue
            _, new_index = _state_and_index(world, new_timestamp)
            push(new_distance, new_timestamp, destination, new_index, steps + 1)
    return None
