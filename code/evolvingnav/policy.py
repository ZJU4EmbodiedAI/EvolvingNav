"""Public-input belief and candidate selection for the N1/N2 high-level track."""

from __future__ import annotations

import math

import numpy as np

from pack_p4d_hssd_records import pack_split
from readyagent.p4d_belief.data import MODEL_INPUT_KEYS, OPTIONAL_MODEL_INPUT_KEYS, derive_last_state


def pack_public_query(query: dict, schema: dict, candidate_features: dict) -> dict[str, np.ndarray]:
    """Reuse the frozen feature schema without passing evaluator labels to the model."""
    query_time = float(query["input"]["query"]["query_time_s"])
    for field in ("target_history", "observable_context_history"):
        timestamps = [float(row["timestamp_s"]) for row in query["input"].get(field, [])]
        if any(timestamp > query_time for timestamp in timestamps):
            raise ValueError(f"future observation in {field}")
        if timestamps != sorted(timestamps):
            raise ValueError(f"out-of-order observation in {field}")
    record = {
        "record_id": query["query_id"],
        "world_variant": "routine",  # packer metadata only; discarded below
        "input": query["input"],
        "supervision": {"current_state_id": 0, "moved_since_last_positive": False},
    }
    packed = pack_split(
        [record],
        max_history=64,
        category_to_id=schema["category_to_id"],
        state_count=int(schema["state_count_including_unknown"]),
        candidate_features=candidate_features,
    )
    allowed = (*MODEL_INPUT_KEYS, *OPTIONAL_MODEL_INPUT_KEYS, "instance_uuid",
               "candidate_region_category_id", "candidate_receptacle_category_id",
               "candidate_center_xyz", "candidate_is_unknown")
    return {key: packed[key] for key in allowed if key in packed}


def candidate_utility(
    *, arrival_probability: float, new_detection_probability: float,
    distance_m: float, eta_s: float, lambda_time: float, lambda_inspect: float,
) -> float:
    """Equation (12): arrival belief times new detection chance per action cost."""
    denominator = distance_m + lambda_time * eta_s + lambda_inspect
    if not math.isfinite(denominator) or denominator <= 0:
        return -math.inf
    return arrival_probability * new_detection_probability / denominator


def rank_candidates(
    belief: dict[int, float], public_costs: dict[int, float], *, task: str = "n2",
    detection_probabilities: dict[int, float] | None = None,
    speed_mps: float = 1.0, inspection_s: float = 1.0,
    lambda_time: float = 0.05, lambda_inspect: float = 0.25,
) -> list[int]:
    """N1 takes Top-1; fixed-target N2 applies Equation (12) to public viewpoints."""
    if not belief:
        return []
    if task == "n1":
        return [max(belief, key=lambda state: (belief[state], -state))]
    if task != "n2":
        raise ValueError(task)
    if speed_mps <= 0:
        raise ValueError("speed_mps must be positive")
    detection_probabilities = detection_probabilities or {}
    return sorted(
        belief,
        key=lambda state: (
            -candidate_utility(
                arrival_probability=belief[state],
                new_detection_probability=detection_probabilities.get(state, 1.0),
                distance_m=public_costs[state],
                eta_s=public_costs[state] / speed_mps + inspection_s,
                lambda_time=lambda_time,
                lambda_inspect=lambda_inspect,
            ),
            state,
        ),
    )


def load_belief(checkpoint_path, dataset_root):
    import torch
    from readyagent.p4d_belief.data import load_catalog
    from readyagent.p4d_belief.models import ModelConfig
    from readyagent.p4d_belief.training import build_model

    catalog = load_catalog(dataset_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name", "p4d" if "transition_head" in checkpoint else None) != "p4d":
        raise ValueError("checkpoint is not a P4D belief model")
    model = build_model("p4d", catalog, ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.eval(), catalog.schema


def model_input_batch(arrays: dict[str, np.ndarray], schema: dict):
    """Strict public-input allowlist for belief and transition inference."""
    import torch

    last_state = int(derive_last_state(arrays)[0])
    candidates = arrays["candidate_state_ids"][0].tolist()
    last_index = candidates.index(last_state)
    instance = str(arrays["instance_uuid"][0])
    vocabulary = schema.get("instance_uuid_to_id", {})
    batch = {
        key: torch.as_tensor(arrays[key])
        for key in (*MODEL_INPUT_KEYS, *OPTIONAL_MODEL_INPUT_KEYS)
        if key in arrays
    }
    batch["target_instance_id"] = torch.tensor([vocabulary.get(instance, 0)])
    batch["target_instance_known"] = torch.tensor([instance in vocabulary])
    batch["last_state"] = torch.tensor([last_state])
    batch["last_candidate_index"] = torch.tensor([last_index])
    for source, destination in (
        ("candidate_region_category_id", "location_region_category"),
        ("candidate_receptacle_category_id", "location_receptacle_category"),
        ("candidate_center_xyz", "location_center_xyz"),
        ("candidate_is_unknown", "location_is_unknown"),
    ):
        if source in arrays:
            batch[destination] = torch.as_tensor(arrays[source])[None]
    return batch


def predict_public(model, schema: dict, arrays: dict[str, np.ndarray]) -> dict[int, float]:
    batch = model_input_batch(arrays, schema)
    candidates = arrays["candidate_state_ids"][0].tolist()
    import torch

    with torch.inference_mode():
        probabilities = model(batch)["probabilities"][0].cpu().numpy()
    return {int(state): float(probabilities[index]) for index, state in enumerate(candidates) if state >= 0}
