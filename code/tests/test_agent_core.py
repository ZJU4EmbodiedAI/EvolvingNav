from __future__ import annotations

import numpy as np
import pytest

from evolvingnav.agent import Agent, AgentConfig, ViewEvidence
from evolvingnav.filter import BeliefFilter, EvidenceLedger
from evolvingnav.memory import VersionedMemory, backproject
from evolvingnav.transition import IdentityTransition, MatrixTransition
from evolvingnav.coverage import (
    camera_forward, candidate_surface_samples, depth_quality, heading_quaternion,
    visible_sample_ids,
)


def test_backprojection_and_causal_versions() -> None:
    point = backproject(1, 1, 2.0, np.diag([2., 2., 1.]), np.eye(4))
    np.testing.assert_allclose(point, [1., 1., 2.])
    memory = VersionedMemory()
    memory.observe("cup", 2, 4.0, 0.8, "frame-1", point)
    memory.observe("cup", 2, 8.0, 0.95, "frame-later", point)
    memory.observe("cup", 3, 9.0, 0.9, "frame-2", point)
    assert memory.at("cup", 7.0).state_id == 2
    assert memory.at("cup", 7.0).valid_to is None
    assert memory.at("cup", 7.0).evidence_handles == ["frame-1"]
    assert memory.at("cup", 7.0).confidence == 0.8
    assert memory.at("cup", 10.0).state_id == 3
    assert memory.history("cup", 10.0)[0].valid_to == 9.0
    memory.record_negative("frame-neg", 2, 10.5, [1, 2, 3])
    assert memory.evidence_at(10.0) == []
    assert memory.evidence_at(11.0)[0].state_id == 2
    with pytest.raises(ValueError, match="causal"):
        memory.observe("cup", 1, 8.0, 0.8, "old", point)


def test_filter_propagation_then_new_evidence_once_and_arrival_independent() -> None:
    kernel = MatrixTransition(np.array([[0.8, 0.2], [0.1, 0.9]]))
    belief = BeliefFilter({1: 0.8, 2: 0.2}, kernel)
    arrival = belief.arrival(5.0)
    assert arrival[2] > 0.2
    assert belief.posterior == {1: 0.8, 2: 0.2}
    belief.advance(5.0)
    assert belief.posterior[2] == pytest.approx(arrival[2])
    evidence = ViewEvidence("frame-1", 1, frozenset({1, 2, 3, 4}), 0.8)
    ledger = EvidenceLedger(min_new_coverage=0.05, sample_count={1: 10, 2: 10})
    admitted = ledger.admit(evidence)
    assert admitted == pytest.approx(0.4)
    prior = belief.posterior[1]
    belief.negative({1: 0.8 * admitted}, "frame-1:1")
    assert belief.posterior[1] < prior
    unchanged = belief.posterior.copy()
    assert ledger.admit(evidence) == 0.0
    assert belief.negative({1: 0.3}, "frame-1:1") is False
    assert belief.posterior == unchanged


def test_filter_advances_transition_context_only_after_elapsed_chunk() -> None:
    class Clocked(IdentityTransition):
        def __init__(self):
            self.clock = 0.0

        def advance_clock(self, seconds):
            self.clock += seconds

    transition = Clocked()
    belief = BeliefFilter({1: 1.0}, transition)
    belief.arrival(10.0)
    assert transition.clock == 0.0
    belief.advance(4.0)
    assert transition.clock == 4.0


def test_dynamic_reopening_and_static_no_return() -> None:
    ledger = EvidenceLedger(sample_count={1: 10})
    ledger.admit(ViewEvidence("a", 1, frozenset(range(8)), 0.9))
    assert not ledger.eligible(1, belief=0.01, return_probability=0.0, new_coverage=0.0)
    assert ledger.eligible(1, belief=0.01, return_probability=0.06, new_coverage=0.0)
    assert ledger.round(1) == 1
    assert ledger.admit(ViewEvidence("b", 1, frozenset(range(8)), 0.9)) == 0.8
    fixed = BeliefFilter({1: 0.8, 2: 0.2}, IdentityTransition())
    fixed.advance(100.0)
    assert fixed.posterior == {1: 0.8, 2: 0.2}


def test_dynamic_round_does_not_reopen_without_elapsed_time() -> None:
    ledger = EvidenceLedger(sample_count={1: 10})
    ledger.admit(ViewEvidence("a", 1, frozenset(range(8)), 0.9))
    ledger.mark_inspected(1, now_s=5.0)
    assert not ledger.eligible(1, belief=0.8, return_probability=0.0,
                               new_coverage=0.0, now_s=5.0)
    assert ledger.eligible(1, belief=0.8, return_probability=0.0,
                           new_coverage=0.0, now_s=6.0)
    assert ledger.round(1) == 1
    assert ledger.eligible(1, belief=0.8, return_probability=0.0,
                           new_coverage=0.0, now_s=6.0)
    assert ledger.round(1) == 1


def test_agent_replans_after_negative_and_stops_on_verified_detection() -> None:
    class World:
        def __init__(self):
            self.position = 0.0

        def distance(self, goal):
            return abs(goal - self.position)

        def move_chunk(self, goal, max_distance):
            delta = min(abs(goal - self.position), max_distance)
            self.position += np.sign(goal - self.position) * delta
            return delta, delta / 1.0

        def inspect(self, state):
            return (state == 2, [ViewEvidence(f"frame-{state}", state,
                frozenset(range(10)), 0.9)])

        def explore(self):
            return {}, 0.0

    world = World()
    agent = Agent(
        {1: 0.8, 2: 0.2}, {1: 1.0, 2: 3.0, 3: 0.1}, world,
        IdentityTransition(), AgentConfig(max_inspections=2, max_path_m=10.0,
            chunk_m=1.0), sample_count={1: 10, 2: 10},
    )
    result = agent.run()
    assert result.found
    assert result.inspections == [1, 2]
    assert result.actions[-1] == "STOP"
    assert agent.filter.posterior[1] < 0.8


def test_agent_uses_new_rgbd_evidence_before_arrival() -> None:
    class World:
        def __init__(self):
            self.position = 0.0
            self.frames = 0

        def distance(self, goal):
            return abs(goal - self.position)

        def move_chunk(self, goal, max_distance):
            displacement = min(abs(goal - self.position), max_distance)
            self.position += np.sign(goal - self.position) * displacement
            return displacement, displacement

        def observe_chunk(self):
            self.frames += 1
            return [ViewEvidence("en-route-1", 1, frozenset({0}), 1.0)] if self.frames == 1 else []

        def inspect(self, state):
            return state == 2, []

    agent = Agent({1: 0.8, 2: 0.2}, {1: 2.0, 2: 4.0}, World(),
                  IdentityTransition(), AgentConfig(max_inspections=2, chunk_m=1.0),
                  sample_count={1: 1, 2: 1})
    result = agent.run()
    assert result.found
    assert result.inspections == [2]
    assert agent.filter.posterior[1] == 0.0


def test_unknown_mass_executes_explore_and_adds_a_searchable_state() -> None:
    class World:
        def __init__(self):
            self.position = 0.0

        def distance(self, goal):
            return abs(goal - self.position)

        def move_chunk(self, goal, max_distance):
            distance = min(abs(goal - self.position), max_distance)
            self.position += distance
            return distance, distance

        def explore(self, _budget_m):
            self.position = 1.0
            return {2: (2.0, 0.8)}, 1.0, 1.0

        def inspect(self, state):
            return state == 2, []

    agent = Agent(
        {1: 0.05, 99: 0.95}, {1: 10.0}, World(), IdentityTransition(),
        AgentConfig(unknown_state=99, max_inspections=2),
    )
    result = agent.run()
    assert result.found
    assert result.actions[0] == "EXPLORE"
    assert result.inspections == [2]
    assert result.posterior[99] < 0.95


def test_online_depth_coverage_uses_public_geometry_not_target_mask() -> None:
    samples = candidate_surface_samples([0, 0, -2], radius_m=0.0)
    depth = np.full((100, 100), 2.0, dtype=float)
    seen = visible_sample_ids(samples, [0, 0, 0], [0, 0, 0, 1], depth, 90.0,
                              sensor_height_m=0.0)
    assert seen == frozenset(range(len(samples)))
    depth[:] = 1.0
    assert not visible_sample_ids(samples, [0, 0, 0], [0, 0, 0, 1], depth, 90.0,
                                  sensor_height_m=0.0)
    np.testing.assert_allclose(camera_forward(heading_quaternion([1, 0, 0])),
                               [1, 0, 0], atol=1e-6)
    assert depth_quality(np.array([[1., 2.], [0., np.nan]])) == 0.5
    slots = [[-1., 0.9, -1.], [1., 0.9, -1.], [-1., 0.9, 1.], [1., 0.9, 1.]]
    surface = candidate_surface_samples([0., 0.9, 0.], place_points=slots)
    assert len(surface) == 25
    np.testing.assert_allclose(surface.min(axis=0), [-1., 0.9, -1.])
    np.testing.assert_allclose(surface.max(axis=0), [1., 0.9, 1.])


def test_learned_transition_rows_are_normalized_and_chronological_loss() -> None:
    import torch
    from evolvingnav.transition_model import TransitionHead, transition_nll

    head = TransitionHead(hidden_dim=8)
    context = torch.randn(2, 8)
    candidates = torch.randn(2, 3, 8)
    probabilities = head(context, candidates, torch.tensor([2.0, 4.0]),
                         torch.ones(2, 3, dtype=torch.bool))
    assert probabilities.shape == (2, 3, 3)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 3))
    loss = transition_nll(probabilities, torch.tensor([0, 2]), torch.tensor([1, 0]))
    assert torch.isfinite(loss)
    loss.backward()
    assert head.mlp[0].weight.grad is not None


def test_transition_pairs_use_only_same_world_instance_and_later_times() -> None:
    from evolvingnav.transition_model import chronological_pairs

    pairs = chronological_pairs(
        instance_ids=np.array(["a", "a", "b", "a"]),
        world_ids=np.array([0, 0, 0, 1]),
        times_s=np.array([1., 4., 2., 5.]),
        states=np.array([1, 2, 3, 4]), max_horizon_s=10,
    )
    assert pairs == [(0, 1, 3.0, 1, 2)]


def test_event_horizon_pairs_are_causal_and_cover_short_target_motion() -> None:
    from evolvingnav.transition_model import event_horizon_pairs

    pairs = event_horizon_pairs(
        instance_ids=np.array(["a", "a", "a"]),
        world_ids=np.array([0, 0, 0]),
        query_times_s=np.array([10., 40., 100.]),
        query_states=np.array([1, 1, 2]),
        events=[{"instance_uuid": "a", "event_time_s": 60.,
                 "source_state_id": 1, "destination_state_id": 2}],
        horizons_s=(20.,),
    )
    assert (1, 50.0, 20.0, 1, 2) in pairs
    assert all(source_time >= [10., 40., 100.][source] for source, source_time, *_ in pairs)


def test_validation_fitted_detection_probability_uses_online_features() -> None:
    from evolvingnav.calibration import DetectionCalibrator

    rows = [
        {"coverage": float(i) / 20, "range_m": 1.0, "angle_cos": 1.0,
         "projected_pixels": 100, "depth_quality": 1.0, "category_recall": 0.9,
         "detected": i >= 10}
        for i in range(21)
    ]
    calibrator = DetectionCalibrator.fit(rows)
    low = calibrator.predict({key: value for key, value in rows[0].items() if key != "detected"})
    high = calibrator.predict({key: value for key, value in rows[-1].items() if key != "detected"})
    assert 0 < low < high < 1


def test_frozen_vlm_controller_can_only_select_legal_public_action() -> None:
    from evolvingnav.controller import LunaToolController

    def requester(payload):
        assert payload["model"] == "gpt-5.6-luna"
        assert "evaluation_private" not in str(payload)
        assert payload["tools"][0]["parameters"]["properties"]["action"]["enum"] == [
            "NAVIGATE_TO(1)", "EXPLORE"
        ]
        return {"output": [{"type": "function_call", "name": "select_action",
                            "arguments": '{"action":"NAVIGATE_TO(1)"}'}]}

    controller = LunaToolController(requester=requester)
    assert controller.choose(["NAVIGATE_TO(1)", "EXPLORE"],
                             {"belief": {1: 0.7}}) == "NAVIGATE_TO(1)"
