"""Private evaluator visibility calibration helpers."""
from __future__ import annotations


def _pixels(frame):
    if "target_pixels" in frame:
        return max(0.0, float(frame["target_pixels"]))
    return 1.0 if frame.get("visible", False) else 0.0


def state_view_max_pixels(target, state_id):
    values = [_pixels(frame) for key, frame in target.get("frames", {}).items()
              if key.startswith(f"{state_id}/")]
    return max(values, default=0.0)


def visibility_fraction(target, state_id, view_id):
    """Return normalized target pixels for private diagnostics.

    The denominator is the maximum rendered target pixels for the same state
    across validated catalog views.  This is deliberately labelled a
    ``max_viewpoint_pixel_normalization`` proxy until an isolated silhouette
    mask is stored by the bake pipeline; it must not be exposed publicly.
    """
    frame = target.get("frames", {}).get(f"{state_id}/{view_id}")
    if frame is None:
        raise KeyError(f"missing frame {state_id}/{view_id}")
    if frame.get("visibility_kind") == "isolated_silhouette" and "visibility_fraction" in frame:
        return max(0.0, min(1.0, float(frame["visibility_fraction"])))
    denominator = state_view_max_pixels(target, state_id)
    return _pixels(frame) / denominator if denominator else 0.0


def calibrated_visibility(frame_pixels, state_max_pixels, *, threshold=0.20):
    """Return private proxy fraction and threshold decision."""
    denominator = max(float(state_max_pixels), 0.0)
    fraction = max(float(frame_pixels), 0.0) / denominator if denominator else 0.0
    return {"fraction": fraction, "threshold": float(threshold),
            "passes": fraction >= float(threshold),
            "calibration_kind": "max_viewpoint_pixel_normalization"}
