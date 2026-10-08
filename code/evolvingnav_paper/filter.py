"""Current-time filtering, counterfactual arrival forecasts, and evidence rounds."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


class BeliefFilter:
    def __init__(self, prior: dict[int, float], transition) -> None:
        if not prior or any(v < 0 for v in prior.values()) or sum(prior.values()) <= 0:
            raise ValueError("prior must be a nonempty probability distribution")
        total = sum(prior.values())
        self.posterior = {int(k): float(v / total) for k, v in prior.items()}
        self.transition = transition
        self.applied_evidence: set[str] = set()

    def arrival(self, elapsed_s: float) -> dict[int, float]:
        if elapsed_s < 0:
            raise ValueError("time cannot go backwards")
        states = list(self.posterior)
        kernel = self.transition.matrix(states, elapsed_s)
        prediction = np.asarray(list(self.posterior.values())) @ kernel
        return dict(zip(states, prediction.tolist(), strict=True))

    def advance(self, elapsed_s: float) -> None:
        self.posterior = self.arrival(elapsed_s)
        if hasattr(self.transition, "advance_clock"):
            self.transition.advance_clock(elapsed_s)

    def negative(self, detection_probability: dict[int, float], evidence_id: str) -> bool:
        if evidence_id in self.applied_evidence:
            return False
        unnormalized = {
            state: probability * (1.0 - float(detection_probability.get(state, 0.0)))
            for state, probability in self.posterior.items()
        }
        if any(value < 0 for value in unnormalized.values()):
            raise ValueError("detection probability must be in [0, 1]")
        total = sum(unnormalized.values())
        if total <= 0:
            raise ValueError("measurement removes all belief mass")
        self.posterior = {state: value / total for state, value in unnormalized.items()}
        self.applied_evidence.add(evidence_id)
        return True


class EvidenceLedger:
    def __init__(self, *, min_new_coverage: float = 0.05, sufficient_coverage: float = 0.70,
                 sample_count: dict[int, int] | None = None) -> None:
        self.min_new_coverage = min_new_coverage
        self.sufficient_coverage = sufficient_coverage
        self.sample_count = sample_count or {}
        self._round = defaultdict(int)
        self._covered: dict[tuple[int, int], set[int]] = defaultdict(set)
        self._measured: dict[tuple[int, int], set[int]] = defaultdict(set)
        self._used: set[str] = set()
        self._inspected: set[int] = set()
        self._last_round_time: dict[int, float] = {}

    def round(self, state: int) -> int:
        return self._round[state]

    def coverage(self, state: int) -> float:
        samples = self.sample_count.get(state, 1)
        return len(self._covered[(state, self.round(state))]) / samples

    def covered_samples(self, state: int) -> frozenset[int]:
        return frozenset(self._covered[(state, self.round(state))])

    def new_samples(self, evidence) -> frozenset[int]:
        return evidence.surface_samples - self._measured[(evidence.state_id, self.round(evidence.state_id))]

    def admit(self, evidence) -> float:
        if evidence.evidence_id in self._used:
            return 0.0
        state = evidence.state_id
        previous = self._covered[(state, self.round(state))]
        new = self.new_samples(evidence)
        fraction = len(new) / self.sample_count.get(state, max(len(new), 1))
        previous.update(evidence.surface_samples)
        self._used.add(evidence.evidence_id)
        if fraction <= self.min_new_coverage:
            return 0.0
        self._measured[(state, self.round(state))].update(new)
        return min(1.0, fraction)

    def mark_inspected(self, state: int, now_s: float | None = None) -> None:
        self._inspected.add(state)
        if now_s is not None:
            self._last_round_time[state] = now_s

    def eligible(self, state: int, *, belief: float, return_probability: float,
                 new_coverage: float, dynamic: bool = True,
                 now_s: float | None = None) -> bool:
        if self.coverage(state) < self.sufficient_coverage:
            return True
        if new_coverage > self.min_new_coverage:
            return True
        time_passed = now_s is None or now_s > self._last_round_time.get(state, -math.inf)
        if dynamic and time_passed and (belief > 0.10 or return_probability > 0.05):
            self._round[state] += 1
            self._inspected.discard(state)
            if now_s is not None:
                self._last_round_time[state] = now_s
            return True
        return False
