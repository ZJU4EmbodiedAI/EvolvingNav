"""Validation-only logistic calibration of online detector recall."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


FEATURES = (
    "coverage", "range_m", "angle_cos", "projected_pixels",
    "depth_quality", "category_recall", "image_quality",
)


@dataclass(frozen=True)
class DetectionCalibrator:
    intercept: float
    coefficients: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    category_coefficients: dict[str, float] = field(default_factory=dict)
    category_recalls: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, validation_rows: list[dict]) -> "DetectionCalibrator":
        if not validation_rows or {int(row["detected"]) for row in validation_rows} != {0, 1}:
            raise ValueError("calibration requires positive and negative validation observations")
        features = np.asarray(
            [[float(row.get(key, 0.)) for key in FEATURES] for row in validation_rows], dtype=float
        )
        if not np.isfinite(features).all():
            raise ValueError("calibration features must be finite")
        scaler = StandardScaler().fit(features)
        categories = sorted({str(row.get("category", "")) for row in validation_rows})
        one_hot = np.asarray([[row.get("category", "") == category for category in categories]
                              for row in validation_rows], dtype=float)
        classifier = LogisticRegression(max_iter=1000).fit(
            np.concatenate((scaler.transform(features), one_hot), axis=1),
            [int(row["detected"]) for row in validation_rows]
        )
        return cls(
            float(classifier.intercept_[0]), tuple(classifier.coef_[0, :len(FEATURES)].tolist()),
            tuple(scaler.mean_.tolist()), tuple(scaler.scale_.tolist()),
            dict(zip(categories, classifier.coef_[0, len(FEATURES):].tolist(), strict=True)),
            {category: float(np.mean([int(row["detected"]) for row in validation_rows
                                     if row.get("category", "") == category])) for category in categories},
        )

    def predict(self, online_features: dict) -> float:
        values = np.asarray([float(online_features.get(key, 0.)) for key in FEATURES[:len(self.coefficients)]])
        standardized = (values - self.means) / self.scales
        logit = (self.intercept + float(np.dot(self.coefficients, standardized))
                 + self.category_coefficients.get(online_features.get("category", ""), 0.))
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30))))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({
            "intercept": self.intercept, "coefficients": self.coefficients,
            "means": self.means, "scales": self.scales,
            "features": FEATURES[:len(self.coefficients)],
            "category_coefficients": self.category_coefficients,
            "category_recalls": self.category_recalls,
        }, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DetectionCalibrator":
        data = json.loads(path.read_text(encoding="utf-8"))
        if tuple(data["features"]) != FEATURES[:len(data["coefficients"])]:
            raise ValueError("calibrator feature schema mismatch")
        return cls(
            float(data["intercept"]), tuple(data["coefficients"]),
            tuple(data["means"]), tuple(data["scales"]),
            data.get("category_coefficients", {}), data.get("category_recalls", {}),
        )
