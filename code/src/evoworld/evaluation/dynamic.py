from __future__ import annotations

from typing import Iterable, Mapping


def dynamic_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, float]:
    records = list(rows)
    dynamic = [row for row in records if bool(row.get("dynamic_denominator_member", row.get("scheduled_transition", False)))]
    invalidated = [row for row in dynamic if bool(row.get("transition_invalidated", False))]
    recovery_eligible = [row for row in dynamic if bool(row.get("recovery_eligible", False))]
    recovered = [row for row in recovery_eligible if bool(row.get("recovered", False))]
    revisits = [row for row in dynamic if bool(row.get("revisited", False))]
    return {"dynamic_sr": sum(bool(row.get("success", False)) for row in dynamic) / len(dynamic) if dynamic else 0.0,
            "recovery_sr": len(recovered) / len(recovery_eligible) if recovery_eligible else 0.0,
            "online_recovery_sr": sum(bool(row.get("recovered", False)) for row in invalidated) / len(invalidated) if invalidated else 0.0,
            "revisit_success": sum(bool(row.get("success", False)) for row in revisits) / len(revisits) if revisits else 0.0,
            "excess_distance": sum(float(row.get("distance", 0.0)) - float(row.get("reference_distance", 0.0)) for row in dynamic) / len(dynamic) if dynamic else 0.0,
            "recovery_count": float(len(recovered))}
