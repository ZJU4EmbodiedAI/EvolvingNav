from __future__ import annotations

import math
from typing import Iterable, Mapping

from .calibration import expected_calibration_error


def prediction_metrics(rows: Iterable[Mapping[str, object]], top_k: int = 3) -> dict[str, float]:
    records = list(rows)
    if not records:
        return {"top1_accuracy": 0.0, "top_k_accuracy": 0.0, "mrr": 0.0, "nll": 0.0, "ece": 0.0}
    top1, topk, reciprocal, nll, conf, correct = [], [], [], [], [], []
    for row in records:
        probabilities = {str(k): float(v) for k, v in dict(row["probabilities"]).items()}  # type: ignore[arg-type]
        if not probabilities or any(value < 0 or value > 1 for value in probabilities.values()) or abs(sum(probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("probabilities must be non-empty, in [0, 1], and sum to one")
        label = str(row["label"])
        ranked = sorted(probabilities, key=probabilities.get, reverse=True)
        top1.append(int(ranked[0] == label))
        topk.append(int(label in ranked[:top_k]))
        reciprocal.append(1.0 / (ranked.index(label) + 1) if label in ranked else 0.0)
        nll.append(-math.log(max(probabilities.get(label, 0.0), 1e-12)))
        conf.append(probabilities[ranked[0]])
        correct.append(int(ranked[0] == label))
    return {"top1_accuracy": sum(top1) / len(top1), "top_k_accuracy": sum(topk) / len(topk), "mrr": sum(reciprocal) / len(reciprocal), "nll": sum(nll) / len(nll), "ece": expected_calibration_error(conf, correct)}
