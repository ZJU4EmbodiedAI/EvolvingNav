"""Online belief updates used by N3/N4 evaluators."""
from __future__ import annotations

class BeliefFilter:
    def __init__(self, belief):
        self.belief = dict(belief)
        self._evidence = set()
        self._normalize()

    def _normalize(self):
        total = sum(max(0.0, float(v)) for v in self.belief.values())
        if total <= 0: raise ValueError("belief must have positive mass")
        self.belief = {k: max(0.0, float(v)) / total for k, v in self.belief.items()}

    def predict(self, transition, elapsed_seconds):
        updated = {key: 0.0 for key in self.belief}
        for source, mass in self.belief.items():
            for target, probability in transition.get(source, {source: 1.0}).items():
                updated[target] = updated.get(target, 0.0) + mass * probability
        self.belief = updated; self._normalize(); return dict(self.belief)

    def negative(self, state, detection_probability, evidence_id):
        if evidence_id in self._evidence: return dict(self.belief)
        self._evidence.add(evidence_id)
        visibility = max(0.0, min(1.0, float(detection_probability)))
        self.belief[state] = self.belief.get(state, 0.0) * (1.0 - visibility)
        self._normalize(); return dict(self.belief)

    def positive(self, state, evidence_id):
        if evidence_id in self._evidence: return dict(self.belief)
        self._evidence.add(evidence_id); self.belief = {key: (1.0 if key == state else 0.0) for key in self.belief}; self._normalize(); return dict(self.belief)
