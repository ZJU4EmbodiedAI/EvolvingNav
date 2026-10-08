from __future__ import annotations

from typing import Sequence


def expected_calibration_error(confidences: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> float:
    if len(confidences) != len(outcomes):
        raise ValueError("confidence and outcome arrays must have equal length")
    if not confidences:
        return 0.0
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        members = [i for i, value in enumerate(confidences) if lo <= value < hi or (index == bins - 1 and value == hi)]
        if members:
            error += len(members) / total * abs(sum(confidences[i] for i in members) / len(members) - sum(outcomes[i] for i in members) / len(members))
    return float(error)
