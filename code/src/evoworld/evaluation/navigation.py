from __future__ import annotations

from typing import Iterable, Mapping


def navigation_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, float]:
    records = list(rows)
    if not records:
        return {"success_rate": 0.0, "spl": 0.0, "distance": 0.0, "time": 0.0,
                "inspection_count": 0.0, "first_inspection_sr": 0.0, "recovery_sr": 0.0}
    success = [int(row["success"]) for row in records]
    distances = [float(row.get("distance", 0.0)) for row in records]
    references = [max(float(row.get("reference_distance", 0.0)), 1e-9) for row in records]
    spl = [s * ref / max(distance, ref) for s, ref, distance in zip(success, references, distances)]
    eligible = [row for row in records if bool(row.get("recovery_eligible", False))]
    return {"success_rate": sum(success) / len(success), "spl": sum(spl) / len(spl),
            "distance": sum(distances) / len(distances),
            "time": sum(float(row.get("time", 0.0)) for row in records) / len(records),
            "inspection_count": sum(float(row.get("inspection_count", 0.0)) for row in records) / len(records),
            "first_inspection_sr": sum(bool(row.get("first_inspection_success", False)) for row in records) / len(records),
            "recovery_sr": sum(bool(row.get("recovered", False)) for row in eligible) / len(eligible) if eligible else 0.0}
