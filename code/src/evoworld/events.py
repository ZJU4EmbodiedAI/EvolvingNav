"""The public, source-agnostic event schema used by the generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Event:
    timestamp: float
    actor: str
    object_id: str
    object_category: str
    previous_state: str
    next_state: str
    activity_type: str
    source: str

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("event timestamps must be non-negative")
        if not self.object_id or not self.previous_state or not self.next_state:
            raise ValueError("events require object identity and both states")
        if self.previous_state == self.next_state:
            raise ValueError("events must represent a state transition")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> "Event":
        return cls(**row)  # type: ignore[arg-type]


class EventLog:
    def __init__(self, events: Iterable[Event] = ()) -> None:
        self._events = list(events)

    def append(self, event: Event) -> None:
        self._events.append(event)

    def ordered(self) -> list[Event]:
        return sorted(self._events, key=lambda event: (event.timestamp, event.object_id))

    def for_object(self, object_id: str) -> list[Event]:
        return [event for event in self.ordered() if event.object_id == object_id]

    def to_dicts(self) -> list[dict[str, object]]:
        return [event.to_dict() for event in self.ordered()]
