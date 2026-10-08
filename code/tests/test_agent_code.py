from __future__ import annotations

import numpy as np

from evolvingnav.policy import candidate_utility, pack_public_query, predict_public, rank_candidates
from evolvingnav.evaluate import evaluate_search
from evolvingnav.backend import visible_fraction_from_masks


def test_public_packing_never_exposes_labels_or_world_metadata() -> None:
    query = {
        "query_id": "q1", "world_variant": "routine",
        "input": {
            "target": {"instance_uuid": "mug_1", "category": "mug"},
            "query": {"query_time_s": 100000.0},
            "history_summary": {"elapsed_since_last_positive_s": 10.0},
            "target_history": [{
                "timestamp_s": 99990.0, "delta_t_from_previous_s": 0.0,
                "event_type": "positive_observation", "observed_state_id": 1,
                "detector_confidence": 1.0, "instance_match_confidence": 1.0,
                "visible_fraction_estimate": 1.0,
            }],
            "candidate_state_ids": [1, 2],
        },
    }
    schema = {"category_to_id": {"mug": 0}, "state_count_including_unknown": 3}
    features = {
        "candidate_region_category_id": np.zeros(3, dtype=np.int16),
        "candidate_receptacle_category_id": np.zeros(3, dtype=np.int16),
        "candidate_center_xyz": np.zeros((3, 3), dtype=np.float32),
        "candidate_is_unknown": np.zeros(3, dtype=bool),
    }
    arrays = pack_public_query(query, schema, features)
    assert "y_current_state" not in arrays
    assert "meta_world_variant_id" not in arrays
    assert arrays["observed_state_id"][0, -1] == 1


def test_search_uses_public_cost_and_evaluator_scores_private_goal() -> None:
    belief = {1: 0.75, 2: 0.25}
    order = rank_candidates(belief, {1: 3.0, 2: 1.0})
    assert order == [1, 2]
    truth = {"current_state_id": 2, "oracle_shortest_path_m": 1.0}
    public = {
        "base_episode_id": "ep",
        "episode_budget": {"max_candidate_inspections": 2, "max_path_length_m": 8.0},
        "success_spec": {"max_geodesic_distance_m": 1.0},
    }
    def route(a, b):
        return abs(a - b)

    def inspect(state, _viewpoint):
        return {
            "detected": state == 2, "visible_fraction": 1.0,
            "distance_to_valid_goal_m": 0.0,
        }

    result = evaluate_search(
        public, truth, belief, start=0.0, goals={1: 3.0, 2: 4.0},
        distance=route, inspect=inspect,
    )
    assert result["success"] is True
    assert result["inspections"] == 2
    assert result["path_m"] == 4.0
    assert result["spl"] == 0.25


def test_candidate_utility_matches_equation_twelve() -> None:
    assert candidate_utility(
        arrival_probability=0.6, new_detection_probability=0.8,
        distance_m=2.0, eta_s=5.0, lambda_time=0.2, lambda_inspect=1.0,
    ) == 0.12


def test_n1_selects_belief_top1_even_when_other_goal_is_nearer() -> None:
    belief = {1: 0.7, 2: 0.3}
    costs = {1: 9.0, 2: 1.0}
    assert rank_candidates(belief, costs, task="n1") == [1]


def test_model_receives_last_state_derived_from_public_history() -> None:
    import torch

    class RecordingModel:
        def __call__(self, batch):
            assert batch["last_state"].tolist() == [1]
            assert batch["last_candidate_index"].tolist() == [0]
            return {"probabilities": torch.tensor([[0.8, 0.2]])}

    arrays = {
        "event_type": np.array([[1]], dtype=np.int8),
        "history_mask": np.array([[True]]),
        "observed_state_id": np.array([[1]], dtype=np.int16),
        "candidate_state_ids": np.array([[1, 2]], dtype=np.int16),
        "candidate_mask": np.array([[True, True]]),
        "instance_uuid": np.array(["mug_1"]),
    }
    schema = {"instance_uuid_to_id": {"mug_1": 0}}
    assert predict_public(RecordingModel(), schema, arrays) == {1: 0.800000011920929, 2: 0.20000000298023224}


def test_transition_model_batch_contains_no_evaluator_fields() -> None:
    from evolvingnav.policy import model_input_batch

    arrays = {
        "event_type": np.array([[1]], dtype=np.int8),
        "history_mask": np.array([[True]]),
        "observed_state_id": np.array([[1]], dtype=np.int16),
        "candidate_state_ids": np.array([[1, 2]], dtype=np.int16),
        "candidate_mask": np.array([[True, True]]),
        "instance_uuid": np.array(["mug_1"]),
        "evaluation_private": np.array([123]),
    }
    result = model_input_batch(arrays, {"instance_uuid_to_id": {"mug_1": 0}})
    assert "evaluation_private" not in result
    assert result["last_state"].item() == 1


def test_public_packing_rejects_future_observation() -> None:
    import pytest

    query = {
        "query_id": "future", "input": {
            "target": {"instance_uuid": "mug_1", "category": "mug"},
            "query": {"query_time_s": 100.0},
            "history_summary": {"elapsed_since_last_positive_s": 0.0},
            "target_history": [{
                "timestamp_s": 101.0, "delta_t_from_previous_s": 0.0,
                "event_type": "positive_observation", "observed_state_id": 1,
                "detector_confidence": 1.0, "instance_match_confidence": 1.0,
                "visible_fraction_estimate": 1.0,
            }],
            "candidate_state_ids": [1, 2],
        },
    }
    schema = {"category_to_id": {"mug": 0}, "state_count_including_unknown": 3}
    features = {
        "candidate_region_category_id": np.zeros(3, dtype=np.int16),
        "candidate_receptacle_category_id": np.zeros(3, dtype=np.int16),
        "candidate_center_xyz": np.zeros((3, 3), dtype=np.float32),
        "candidate_is_unknown": np.zeros(3, dtype=bool),
    }
    with pytest.raises(ValueError, match="future"):
        pack_public_query(query, schema, features)


def test_n2_replans_from_current_viewpoint_and_stops_at_verified_goal() -> None:
    episode = {
        "base_episode_id": "ep", "episode_budget": {
            "max_candidate_inspections": 3, "max_path_length_m": 30.0,
        },
        "success_spec": {"max_geodesic_distance_m": 1.0, "require_stop_action": True},
    }
    truth = {"current_state_id": 3, "oracle_shortest_path_m": 5.0}
    belief = {1: 0.8, 2: 0.12, 3: 0.08}
    goals = {1: 4.0, 2: -3.0, 3: 5.0}

    def inspect(state, viewpoint):
        return {
            "detected": state == 3,
            "visible_fraction": 0.9 if state == 3 else 0.0,
            "distance_to_valid_goal_m": 0.0 if state == 3 else 5.0,
        }

    result = evaluate_search(
        episode, truth, belief, start=0.0, goals=goals,
        distance=lambda a, b: abs(a - b), inspect=inspect,
    )
    assert result["inspection_order"] == [1, 3]
    assert result["actions"][-1] == "STOP"
    assert result["success"] is True
    assert result["inspection_evidence"][-1]["state_id"] == 3
    assert result["inspection_evidence"][-1]["visible_fraction"] == 0.9


def test_correct_state_without_valid_view_is_not_success() -> None:
    episode = {
        "base_episode_id": "ep", "episode_budget": {
            "max_candidate_inspections": 1, "max_path_length_m": 10.0,
        },
        "success_spec": {"max_geodesic_distance_m": 1.0, "require_stop_action": True},
    }
    truth = {"current_state_id": 1, "oracle_shortest_path_m": 2.0}

    def inspect(_state, _viewpoint):
        return {"detected": True, "visible_fraction": 0.8, "distance_to_valid_goal_m": 1.5}

    result = evaluate_search(
        episode, truth, {1: 1.0}, start=0.0, goals={1: 2.0},
        distance=lambda a, b: abs(a - b), inspect=inspect, task="n1",
    )
    assert result["success"] is False
    assert result["actions"][-1] == "STOP"


def test_verified_instance_at_neighbor_state_viewpoint_can_succeed() -> None:
    episode = {
        "base_episode_id": "ep", "episode_budget": {
            "max_candidate_inspections": 1, "max_path_length_m": 10.0,
        },
        "success_spec": {"max_geodesic_distance_m": 1.0},
    }
    truth = {"current_state_id": 2, "oracle_shortest_path_m": 2.0}

    result = evaluate_search(
        episode, truth, {1: 1.0}, start=0.0, goals={1: 2.0},
        distance=lambda a, b: abs(a - b),
        inspect=lambda _state, _viewpoint: {
            "detected": True, "visible_fraction": 0.8,
            "distance_to_valid_goal_m": 0.4,
        },
        task="n1",
    )
    assert result["success"] is True


def test_success_requires_twenty_percent_visibility() -> None:
    episode = {
        "base_episode_id": "ep", "episode_budget": {
            "max_candidate_inspections": 1, "max_path_length_m": 10.0,
        },
        "success_spec": {"max_geodesic_distance_m": 1.0, "min_visible_fraction": 0.05},
    }
    truth = {"current_state_id": 1, "oracle_shortest_path_m": 2.0}

    def inspect(_state, _viewpoint):
        return {"detected": True, "visible_fraction": 0.19, "distance_to_valid_goal_m": 0.0}

    result = evaluate_search(
        episode, truth, {1: 1.0}, start=0.0, goals={1: 2.0},
        distance=lambda a, b: abs(a - b), inspect=inspect, task="n1",
    )
    assert result["success"] is False


def test_visible_fraction_uses_target_only_projection_as_denominator() -> None:
    actual = np.array([[7, 7, 0], [0, 7, 0]])
    target_only = np.array([[7, 7, 7], [7, 7, 0]])
    assert visible_fraction_from_masks(actual, target_only, semantic_id=7) == 0.6


def test_grounded_sam_policy_detection_uses_rgb_depth_and_category_only() -> None:
    from evolvingnav.perception import detect_category

    def detector(rgb, categories):
        assert rgb.shape == (2, 2, 3)
        assert categories == ("bottle",)
        return (("bottle", 0.9, (0.0, 0.0, 1.0, 1.0)),)

    def segmenter(rgb, box):
        assert rgb.shape == (2, 2, 3)
        assert box == (0.0, 0.0, 1.0, 1.0)
        return np.ones((2, 2), dtype=bool)

    assert detect_category(
        detector, segmenter, np.zeros((2, 2, 4), dtype=np.uint8), "bottle"
    ) is True


def test_grounded_sam_returns_mask_for_rgbd_memory_backprojection() -> None:
    from evolvingnav.perception import detect_instances

    found = detect_instances(
        lambda _rgb, _categories: (("bottle", 0.9, (0., 0., 1., 1.)),),
        lambda _rgb, _box: np.ones((2, 2), dtype=bool),
        np.zeros((2, 2, 4), dtype=np.uint8), "bottle",
    )
    assert len(found) == 1
    assert found[0].confidence == 0.9
    assert found[0].mask.sum() == 4


def test_grounded_sam_requires_a_nonempty_target_category_mask() -> None:
    from evolvingnav.perception import detect_category

    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    def detector(_rgb, _categories):
        return (
            ("book", 0.9, (0.0, 0.0, 1.0, 1.0)),
            ("bottle", 0.8, (0.0, 0.0, 1.0, 1.0)),
        )

    def empty_mask(_rgb, _box):
        return np.zeros((2, 2), dtype=bool)
    assert detect_category(detector, empty_mask, rgb, "bottle") is False


def test_training_cli_exposes_optimizer_parameters(monkeypatch) -> None:
    import sys
    from train_p4d_belief import arguments

    monkeypatch.setattr(sys, "argv", [
        "train_p4d_belief.py", "--dataset-root", "/data/p4d", "--output", "/tmp/p4d",
        "--weight-decay", "0.01", "--gradient-clip", "1.0",
    ])
    args = arguments()
    assert args.weight_decay == 0.01
    assert args.gradient_clip == 1.0


def test_training_cli_defaults_match_benchmark_configuration(monkeypatch) -> None:
    import sys
    from train_p4d_belief import arguments

    monkeypatch.setattr(sys, "argv", [
        "train_p4d_belief.py", "--dataset-root", "/data/p4d", "--output", "/tmp/p4d",
    ])
    args = arguments()
    assert args.models == ["p4d"]
    assert args.seeds == [0, 1, 2, 3, 4]
    assert (args.layers, args.hidden_dim, args.heads, args.dropout) == (3, 128, 4, 0.1)
    assert (args.batch_size, args.patience, args.weight_decay) == (64, 10, 0.01)
    assert args.eval_split == "val"
    assert args.use_instance_identity is True
    assert args.use_compatibility is True


def test_compatibility_head_receives_state_loss_gradient() -> None:
    import torch

    from readyagent.p4d_belief.data import Catalog
    from readyagent.p4d_belief.models import ModelConfig, P4DBelief, state_nll

    catalog = Catalog(
        region_category=torch.tensor([0, 0, 0]),
        receptacle_category=torch.tensor([0, 0, 0]),
        center_xyz=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        is_unknown=torch.tensor([False, False, True]),
        region_instance_ids=("room", "room", "unknown"),
        schema={
            "region_category_to_id": {"room": 0},
            "receptacle_category_to_id": {"shelf": 0},
            "category_to_id": {"mug": 0},
        },
    )
    batch = {
        "event_type": torch.tensor([[1, 0]]),
        "event_time_days": torch.tensor([[0.0, 0.0]]),
        "observed_state_id": torch.tensor([[0, -1]]),
        "candidate_state_id": torch.tensor([[-1, -1]]),
        "evidence_features": torch.zeros((1, 2, 6)),
        "history_mask": torch.tensor([[True, False]]),
        "candidate_state_ids": torch.tensor([[0, 1, 2]]),
        "candidate_mask": torch.tensor([[True, True, True]]),
        "target_category_id": torch.tensor([0]),
        "query_time_days": torch.tensor([0.1]),
        "query_time_of_day_sin_cos": torch.tensor([[0.0, 1.0]]),
        "query_weekday_id": torch.tensor([0]),
        "elapsed_since_last_positive_days": torch.tensor([0.1]),
        "last_state": torch.tensor([0]),
        "last_candidate_index": torch.tensor([0]),
        "target_candidate_index": torch.tensor([1]),
    }
    model = P4DBelief(catalog, ModelConfig(
        hidden_dim=32, layers=1, heads=4, dropout=0.0, use_compatibility=True,
    ))
    loss = state_nll(model(batch)["probabilities"], batch["target_candidate_index"])
    loss.backward()
    gradients = [parameter.grad for parameter in model.compatibility.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
