"""Trace-grounded, causal household timelines."""
from __future__ import annotations
import random

def make_world(obj, *, days, regime, seed, stats):
    if regime == "static": return {"initial_state": obj["placements"][0]["state_id"], "events": [], "regime": regime}
    rng = random.Random(seed); states = [p["state_id"] for p in obj["placements"]]
    count = min(days, max(1, int(stats.get("move_rate", 1) * days / 90)))
    if count < 1: count = min(days, 4)
    # Use the same scheduled days for paired routine/random worlds; random only
    # permutes destination sequence and timestamp assignment.
    days_used = list(range(0, days, max(1, days // count)))[:count]
    destinations = [states[(i + 1) % len(states)] for i in range(count)]
    activity_counts = stats.get('category_activity_counts', {}).get(obj['category'], {})
    activity = (rng.choices(list(activity_counts), weights=list(activity_counts.values()), k=1)[0]
                if activity_counts else "prepare_meal")
    if regime == "personal":
        profile = stats.get("owner_profile", {})
        preferred = profile.get("preferred_state")
        if preferred not in states:
            preferred = states[-1]
        destinations = []
        current_profile = states[0]
        for i in range(count):
            if current_profile != preferred:
                destination = preferred
            else:
                destination = states[(states.index(current_profile) + 1) % len(states)]
            destinations.append(destination)
            current_profile = destination
        activity = "owner_habit_return"
    if regime == "random":
        # Shuffle by legal swaps, never repair by changing a destination: that
        # would silently change the routine control's destination histogram.
        # Finite swap randomization is not claimed to be uniform sampling.
        for _ in range(20 * count):
            a, b = rng.randrange(count), rng.randrange(count)
            destinations[a], destinations[b] = destinations[b], destinations[a]
            legal = destinations[0] != states[0]
            for i in {a, a + 1, b, b + 1}:
                if 0 < i < count and destinations[i] == destinations[i - 1]:
                    legal = False
            if not legal:
                destinations[a], destinations[b] = destinations[b], destinations[a]
    current = states[0]; events = []
    for index, day in enumerate(days_used):
        dest = destinations[index]
        hours = stats.get('routine_hours', [8, 11, 14, 17])
        if not hours or any(not 0 <= h < 23 for h in hours):
            raise ValueError('routine_hours must be nonempty hours in [0,23)')
        anchor = stats.get('query_anchor_s')
        if regime == "random" and anchor is not None and day == int(float(anchor) // 86400):
            # N4 predeclares a common post-query window. The random control
            # changes only the event on the query day, never reorders a day-0
            # transition across a months-long causal state chain. The window
            # is clipped to this day to preserve matched per-day frequencies.
            remaining = (day + 1) * 86400 - float(anchor) - 1e-6
            if remaining <= 0:
                raise ValueError('dynamic anchor outside scheduled event day')
            timestamp = float(anchor) + rng.uniform(1e-6, min(3600., remaining))
        elif regime == "random":
            # Match event counts and destination marginals while removing the
            # routine's clock phase; this is an independent time-of-day draw.
            timestamp = day * 86400 + rng.uniform(0, 86400 - 1e-6)
        else:
            timestamp = day * 86400 + hours[index % len(hours)] * 3600
        event = {"timestamp": timestamp, "actor": "resident_0", "object_id": obj["object_id"], "object_category": obj["category"], "previous_state": current, "next_state": dest, "activity_type": activity, "source": stats.get("event_source", "synthetic_trace_statistics"),
                 "source_refs": list(stats.get("source_refs", []))}
        events.append(event); current = dest
    world = {"initial_state": states[0], "events": sorted(events, key=lambda x: x["timestamp"]), "regime": regime}
    if regime == "personal":
        # The profile is private evaluator metadata; it is never serialized
        # into the public episode row.
        world["owner_profile"] = dict(stats.get("owner_profile", {}))
    return world
