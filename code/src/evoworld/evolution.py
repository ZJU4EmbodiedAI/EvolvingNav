"""Causal household transition models and private-state replay."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Mapping, Sequence

from .events import Event

REGIMES = ("static", "stable_activity_routine", "personal_habit", "random")
ACTIVITIES = ("breakfast_cleanup", "meal_preparation", "tidying", "bedtime_return")


def _event_time(day: int, object_index: int, rng: random.Random) -> float:
    base = day * 86400.0 + (7.0 + (object_index % 3) * 5.0) * 3600.0
    return base + rng.uniform(0.0, 1800.0)


def generate_events(
    initial_state: Mapping[str, str],
    candidates: Mapping[str, Sequence[str]],
    *,
    days: int,
    regime: str,
    seed: int,
    actor_profiles: Mapping[str, Mapping[str, str]] | None = None,
) -> list[Event]:
    """Generate synthetic transitions constrained by legal candidate states.

    Source statistics only parameterize timing/activity choices; no source
    trajectory is copied. Random worlds use the same object/event rate as the
    routine world but sample destinations independently of time-of-day.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown mobility regime: {regime}")
    rng = random.Random(seed)
    events: list[Event] = []
    for object_index, (object_id, initial) in enumerate(sorted(initial_state.items())):
        legal = list(candidates[object_id])
        current = initial
        if regime == "static":
            continue
        scheduled_days = [day for day in range(days) if day % 3 != 2]
        planned = []
        for event_index, _day in enumerate(scheduled_days):
            choices = [state for state in legal if state != current]
            if not choices:
                continue
            destination = choices[event_index % len(choices)]
            planned.append(destination)
            current = destination
        if regime == "random":
            original = list(planned)
            for _ in range(100):
                rng.shuffle(planned)
                probe = initial
                valid = True
                for destination in planned:
                    if destination == probe:
                        valid = False
                        break
                    probe = destination
                if valid:
                    break
            else:
                planned = original
        current = initial
        plan_index = 0
        for day in range(days):
            if day % 3 == 2:
                continue
            if regime == "personal_habit" and day % 4 == 3:
                continue
            # Paired random worlds retain the routine event-rate mask. Only
            # temporal destination dependence is removed.
            if regime == "random" and day % 3 == 2:
                continue
            if regime == "personal_habit" and actor_profiles:
                profile = actor_profiles.get(object_id, {})
                preferred = profile.get("preferred_state")
                if preferred in legal and preferred != current:
                    destination = preferred
                else:
                    choices = [state for state in legal if state != current]
                    if not choices:
                        continue
                    destination = rng.choice(choices)
            elif regime in ("stable_activity_routine", "random"):
                if plan_index >= len(planned):
                    continue
                destination = planned[plan_index]
                plan_index += 1
            else:
                choices = [state for state in legal if state != current]
                if not choices:
                    continue
                destination = rng.choice(choices)
            events.append(
                Event(
                    timestamp=_event_time(day, object_index, rng),
                    actor=f"resident_{object_index % 2}",
                    object_id=object_id,
                    object_category=object_id.split("_", 1)[0],
                    previous_state=current,
                    next_state=destination,
                    activity_type=ACTIVITIES[day % len(ACTIVITIES)],
                    source="synthetic_trace_aligned",
                )
            )
            current = destination
    return sorted(events, key=lambda event: (event.timestamp, event.object_id))


def replay_state(initial: Mapping[str, str], events: Sequence[Event], timestamp: float) -> dict[str, str]:
    state = dict(initial)
    for event in sorted(events, key=lambda item: (item.timestamp, item.object_id)):
        if event.timestamp > timestamp:
            break
        if state.get(event.object_id) != event.previous_state:
            # A malformed schedule must never silently overwrite causal state.
            raise ValueError(f"non-causal event for {event.object_id} at {event.timestamp}")
        state[event.object_id] = event.next_state
    return state


def events_by_object(events: Sequence[Event]) -> dict[str, list[Event]]:
    grouped: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        grouped[event.object_id].append(event)
    return {key: sorted(value, key=lambda item: item.timestamp) for key, value in grouped.items()}
