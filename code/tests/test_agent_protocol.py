import numpy as np
import pytest

from evolvingnav.agent import Agent, AgentConfig, AgentResult, ViewEvidence
from evolvingnav.calibration import DetectionCalibrator
from evolvingnav.coverage import camera_transform, visible_sample_ids
from evolvingnav.filter import EvidenceLedger
from evolvingnav.memory import VersionedMemory, backproject
from evolvingnav.transition import IdentityTransition


class RouteWorld:
    def __init__(self):
        self.position = 0.0
        self.scans = 0

    def distance(self, goal):
        return abs(goal - self.position)

    def move_chunk(self, goal, max_distance):
        distance = min(self.distance(goal), max_distance)
        self.position += np.sign(goal - self.position) * distance
        return distance, distance

    def inspect(self, state):
        self.scans += 1
        return True, []


def test_habitat_optical_backprojection_round_trip():
    intrinsics = np.array([[50., 0, 50.], [0, 50., 50.], [0, 0, 1.]])
    point = backproject(50, 50, 2., intrinsics,
                        camera_transform([0, 0, 0], [0, 0, 0, 1], optical=True))
    np.testing.assert_allclose(point, [0., 1.35, -2.])
    assert visible_sample_ids(np.array([point]), [0, 0, 0], [0, 0, 0, 1],
                              np.full((100, 100), 2.), 90.) == {0}


def test_weak_static_inspection_can_receive_more_geometry():
    ledger = EvidenceLedger(sample_count={1: 10})
    ledger.admit(ViewEvidence('weak', 1, frozenset({0, 1}), .2, .2))
    ledger.mark_inspected(1, now_s=1.)
    assert ledger.eligible(1, belief=.8, return_probability=0., new_coverage=0.,
                           dynamic=False, now_s=2.)


def test_likelihood_is_calibrated_on_new_samples():
    cal = DetectionCalibrator(-5., (10., 0., 0., 0., 0., 0.), (0.,)*6, (1.,)*6)
    features = dict(coverage=1., range_m=1., angle_cos=1., projected_pixels=25.,
                    depth_quality=1., category_recall=.8)
    world = RouteWorld()
    world.calibrator = cal
    agent = Agent({1: .5, 2: .5}, {1: 1.}, world, IdentityTransition(),
                  sample_count={1: 25})
    agent.ledger.admit(ViewEvidence('old', 1, frozenset(range(20)), .9))
    evidence = ViewEvidence('new', 1, frozenset(range(25)), cal.predict(features),
                            features=features)
    result = AgentResult()
    agent._admit_negative([evidence], 1., result)
    expected = cal.predict({**features, 'coverage': .2, 'projected_pixels': 5.})
    assert result.evidence_trace[0]['detection_probability'] == pytest.approx(expected)


def test_calibration_conditions_on_category_and_image_quality(tmp_path):
    common = dict(coverage=.8, range_m=2., angle_cos=1., projected_pixels=100.,
                  depth_quality=1., category_recall=.5)
    rows = [{**common, 'category': category, 'image_quality': quality,
             'detected': int(category == 'mug' and quality > .5)}
            for category in ('mug', 'book') for quality in (0., .2, .8, 1.) for _ in range(4)]
    cal = DetectionCalibrator.fit(rows)
    assert cal.predict({**common, 'category': 'mug', 'image_quality': 1.}) > cal.predict(
        {**common, 'category': 'book', 'image_quality': 1.})
    assert cal.predict({**common, 'category': 'mug', 'image_quality': 1.}) > cal.predict(
        {**common, 'category': 'mug', 'image_quality': 0.})
    cal.save(tmp_path / 'calibration.json')
    assert DetectionCalibrator.load(tmp_path / 'calibration.json') == cal


def test_positive_enroute_view_stops_before_destination():
    class World(RouteWorld):
        def observe_chunk(self):
            return True, []
    world = World()
    result = Agent({1: 1.}, {1: 10.}, world, IdentityTransition(),
                   AgentConfig(chunk_m=1.)).run()
    assert result.found
    assert result.path_m == 1.
    assert world.scans == 0
    assert result.actions[-1] == 'STOP'


def test_distant_positive_guides_agent_to_public_verification_view():
    class World(RouteWorld):
        last_detection = {'state_id': 1, 'confidence': .9, 'evidence_id': 'positive',
                          'world_point': [0., 0., 0.]}
        def observe_chunk(self):
            return True, []
    result = Agent({1: 1.}, {1: 4.}, World(), IdentityTransition(),
                   AgentConfig(chunk_m=1.)).run()
    assert result.found and result.path_m == 3.


def test_enroute_coverage_counts_as_first_inspection():
    class World(RouteWorld):
        def observe_chunk(self):
            return [ViewEvidence('enroute', 1, frozenset(range(8)), .8, .8)]
    result = Agent({1: .8, 2: .2}, {1: 2., 2: 4.}, World(), IdentityTransition(),
                   AgentConfig(chunk_m=1.), sample_count={1: 10, 2: 10}).run()
    assert result.covered_inspections[0]['state_id'] == 1
    assert result.covered_inspections[0]['time_s'] == 1.


def test_enroute_negative_starts_no_new_round_at_same_decision_time():
    from evolvingnav.transition import MatrixTransition
    agent = Agent({1: .8, 2: .2}, {1: 1., 2: 2.}, RouteWorld(),
                  MatrixTransition(np.eye(2)), sample_count={1: 10})
    agent._admit_negative([ViewEvidence('enroute', 1, frozenset(range(8)), .8, .8)],
                          2., AgentResult())
    assert not agent.ledger.eligible(1, belief=.5, return_probability=0., new_coverage=0.,
                                     dynamic=True, now_s=2.)
    assert agent.ledger.eligible(1, belief=.5, return_probability=0., new_coverage=0.,
                                 dynamic=True, now_s=3.)


def test_time_budget_limits_action_execution():
    world = RouteWorld()
    result = Agent({1: 1.}, {1: 10.}, world, IdentityTransition(),
                   AgentConfig(max_time_s=2., chunk_m=1.)).run()
    assert result.elapsed_s <= 2.
    assert not result.found
    assert result.termination == 'time_budget_exhausted'


def test_n1_event_agent_uses_top_one_not_cost_ranking():
    agent = Agent({1: .8, 2: .2}, {1: 10., 2: 1.}, RouteWorld(),
                  IdentityTransition(), AgentConfig(protocol='n1'))
    assert agent._choose(AgentResult()) == 1


def test_n2_keeps_prior_after_negative_observation():
    agent = Agent({1: .8, 2: .2}, {1: 1., 2: 2.}, RouteWorld(),
                  IdentityTransition(), AgentConfig(protocol='n2'), sample_count={1: 10})
    agent._admit_negative([ViewEvidence('neg', 1, frozenset(range(10)), .9)],
                          1., AgentResult())
    assert agent.filter.posterior == {1: .8, 2: .2}


def test_controller_receives_target_and_causal_memory():
    memory = VersionedMemory()
    memory.observe('cup', 1, 5., .9, 'past', np.zeros(3))
    memory.observe('cup', 2, 20., .9, 'future', np.ones(3))
    class Controller:
        def choose(self, legal, context):
            assert context['target']['entity_id'] == 'cup'
            assert context['memory'][0]['state_id'] == 1
            assert 'future' not in str(context)
            return legal[0]
    agent = Agent({1: 1.}, {1: 1.}, RouteWorld(), IdentityTransition(),
                  memory=memory, target_id='cup', time_origin_s=10., controller=Controller())
    assert agent._choose(AgentResult()) == 1


def test_temporal_memory_query_selects_matching_entity_version():
    memory = VersionedMemory()
    memory.observe('cup', 1, 5., .9, 'past', np.zeros(3),
                   category='mug', attributes={'color': 'red'}, relations={'on': 'table'})
    memory.observe('cup', 2, 20., .9, 'later', np.ones(3),
                   category='mug', attributes={'color': 'red'}, relations={'on': 'shelf'})
    result = memory.query({'category': 'mug', 'relations': {'on': 'table'},
                           'before_s': 10.}, cutoff=30.)
    assert result[0]['entity_id'] == 'cup'
    assert result[0]['state_id'] == 1


def test_unseen_public_instance_uses_semantics_without_known_identity():
    from evolvingnav.policy import model_input_batch
    arrays = {'event_type': np.array([[1]]), 'history_mask': np.array([[True]]),
              'observed_state_id': np.array([[1]]), 'candidate_state_ids': np.array([[1, 2]]),
              'candidate_mask': np.array([[True, True]]), 'instance_uuid': np.array(['new'])}
    batch = model_input_batch(arrays, {'instance_uuid_to_id': {'known': 0}})
    assert not batch['target_instance_known'].item()


def test_n5_cli_selects_underlying_protocol():
    from evolvingnav.run import arguments
    args = arguments(['--task', 'n5', '--protocol', 'n3', '--dataset', '/dataset',
                      '--tasks', '/tasks', '--checkpoint', '/model.pt', '--hssd-root', '/hssd',
                      '--navmesh-root', '/nav', '--output', '/out'])
    assert args.task == 'n5' and args.protocol == 'n3'


def test_runner_uses_public_scene_geometry_and_all_episode_budgets():
    from evolvingnav.run import public_features, episode_config
    schema = {'region_category_to_id': {'room': 0, 'unknown': 1},
              'receptacle_category_to_id': {'table': 0, 'unknown': 1}}
    states = [{'state_id': 0, 'region_category': 'room', 'receptacle_category': 'table',
               'state_center': [8., 0., 9.], 'is_unknown': False},
              {'state_id': 1, 'region_category': 'unknown', 'receptacle_category': 'unknown',
               'state_center': None, 'is_unknown': True}]
    features = public_features(states, schema)
    np.testing.assert_allclose(features['candidate_center_xyz'][0], [8., 0., 9.])
    episode = {'episode_budget': {'max_candidate_inspections': 3, 'max_path_length_m': 15.,
                                 'max_time_s': 7., 'max_steps': 21, 'max_explorations': 2},
               'public_refs': {'action_space': ['EXPLORE']}}
    config = episode_config(episode, 'n2', 1)
    assert (config.protocol, config.max_time_s, config.max_steps, config.max_explorations) == ('n2', 7., 21, 2)


def test_runner_rejects_nonbenchmark_success_contract():
    from evolvingnav.run import validate_episode_contract
    episode = {'success_spec': {'max_geodesic_distance_m': 1., 'min_visible_fraction': .05,
                                'require_target_visible': True, 'require_stop_action': True},
               'episode_budget': {'max_steps': 500, 'max_path_length_m': 100.,
                                  'max_candidate_inspections': 10}}
    with pytest.raises(ValueError, match='success'):
        validate_episode_contract(episode, 'n3')
    episode['success_spec']['min_visible_fraction'] = .20
    validate_episode_contract(episode, 'n3')
    with pytest.raises(ValueError, match='window'):
        validate_episode_contract(episode, 'n4')


def test_new_scene_semantic_labels_use_unknown_vocab_not_new_weights():
    from evolvingnav.run import public_features
    states = [{'state_id': 0, 'region_category': 'garage', 'receptacle_category': 'new_shelf',
               'state_center': [1., 2., 3.]}]
    schema = {'region_category_to_id': {'unknown': 2},
              'receptacle_category_to_id': {'unknown': 3}}
    features = public_features(states, schema)
    assert features['candidate_region_category_id'].tolist() == [2]
    assert features['candidate_receptacle_category_id'].tolist() == [3]
    assert not features['candidate_is_unknown'][0]


def test_exploration_continues_after_partial_navigation_chunk():
    class World(RouteWorld):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.exploration_observation = (False, [])
        def explore(self, budget):
            self.calls += 1
            assert budget <= 2.
            self.position += 1.
            if self.calls == 2:
                self.exploration_observation = (True, [])
            return {}, 1., 1.
    world = World()
    result = Agent({99: 1.}, {}, world, IdentityTransition(),
                   AgentConfig(unknown_state=99, chunk_m=2.)).run()
    assert result.found and world.calls == 2


def test_exploration_utility_uses_current_frontier_cost():
    world = RouteWorld()
    world.exploration_cost = lambda: 100.
    agent = Agent({1: .3, 99: .7}, {1: 2.}, world, IdentityTransition(),
                  AgentConfig(unknown_state=99))
    assert agent._choose(AgentResult()) == 1


def test_controller_executes_memory_tool_before_selecting_action():
    from evolvingnav.controller import LunaToolController
    responses = iter([
        {'output': [{'type': 'function_call', 'call_id': 'memory-1', 'name': 'query_memory',
                     'arguments': '{"filters":{"entity_id":"cup"}}'}]},
        {'output': [{'type': 'function_call', 'call_id': 'select-1', 'name': 'select_action',
                     'arguments': '{"action":"NAVIGATE_TO(1)"}'}]},
    ])
    payloads = []
    def request(payload):
        payloads.append(__import__('copy').deepcopy(payload))
        return next(responses)
    controller = LunaToolController(requester=request)
    controller.bind_tools({'query_memory': lambda filters: [{'entity_id': filters['entity_id'], 'state_id': 1}]})
    assert controller.choose(['NAVIGATE_TO(1)'], {'target': {'entity_id': 'cup'}}) == 'NAVIGATE_TO(1)'
    outputs = [item for item in payloads[1]['input'] if item.get('type') == 'function_call_output']
    assert outputs[0]['call_id'] == 'memory-1' and 'cup' in outputs[0]['output']


def test_mask_grounding_materializes_causal_entity_memory():
    memory = VersionedMemory()
    observation = {'evidence_id': 'rgbd-1', 'timestamp': 10.,
                   'rgb': np.full((2, 2, 3), 200, dtype=np.uint8),
                   'depth': np.full((2, 2), 2.), 'position_xyz': [0, 0, 0],
                   'rotation_xyzw': [0, 0, 0, 1]}
    detection = {'category': 'mug', 'confidence': .9, 'mask': np.ones((2, 2), dtype=bool)}
    entities = memory.ingest(observation, [detection], {1: [0., 1.35, -2.]})
    version = memory.at(entities[0], 10.)
    assert version.category == 'mug'
    assert version.state_id == 1
    assert version.feature and len(version.point_clouds['rgbd-1:0']) > 1
    assert memory.frames['rgbd-1']['pose']['rotation_xyzw'] == [0, 0, 0, 1]
    assert memory.query({'category': 'mug'}, cutoff=9.) == []


def test_dynamic_metrics_use_predeclared_subset_and_verification_time():
    from evolvingnav.evaluate import aggregate_metrics
    rows = [
        {'success': True, 'spl': .8, 'first_inspection_success': True,
         'recovery_eligible': False, 'recovered': False, 'dynamic_eligible': True,
         'online_recovery_eligible': True, 'online_recovered': False,
         'distance': 3., 'reference_distance': 2., 'revisit_count': 0, 'revisit_success_count': 0},
        {'success': True, 'spl': .6, 'first_inspection_success': False,
         'recovery_eligible': True, 'recovered': True, 'dynamic_eligible': True,
         'online_recovery_eligible': True, 'online_recovered': True,
         'distance': 4., 'reference_distance': 2., 'revisit_count': 1, 'revisit_success_count': 1},
    ]
    result = aggregate_metrics(rows)
    assert result['dynamic_sr'] == 1.
    assert result['online_recovery_sr'] == .5
    assert result['recovery_sr'] == 1.
    assert result['first_inspection_sr'] == .5


def test_dynamic_oracle_waits_without_changing_preselected_window():
    from evolvingnav.evaluate import oracle_distance
    phases = [(0., [3.]), (4., [0.])]
    distance = oracle_distance(start=0., phases=phases, distance=lambda a, b: abs(a-b),
                               max_time_s=10., max_path_m=10., inspection_s=1.)
    assert distance == 0.


def test_live_world_returns_positive_chunk_and_registers_frame():
    from types import SimpleNamespace
    from evolvingnav.world import HabitatAgentWorld
    mask = np.zeros((100, 100), dtype=bool)
    mask[45:55, 45:55] = True
    detection = SimpleNamespace(category='mug', confidence=.9, mask=mask)
    class Backend:
        target_category = 'mug'
        detector = object()
        def observe(self, position, rotation):
            self.last_observation = {'rgb': np.full((100, 100, 3), 200, dtype=np.uint8),
                                     'depth': np.full((100, 100), 2.)}
            self.last_detections = [SimpleNamespace(category='mug', confidence=.3, mask=mask.copy()), detection]
            return {'detected': True, 'visible_fraction': 1., 'distance_to_valid_goal_m': 0.}
    world = HabitatAgentWorld(Backend(), {1: {'position_xyz': [0, 0, 0],
                                               'rotation_xyzw': [0, 0, 0, 1]}},
                              {1: [0, 1.35, -2.]}, [0, 0, 0], [0, 0, 0, 1])
    positive, evidence = world.observe_chunk()
    assert positive
    assert world.last_detection['world_point'][2] < 0
    memory = VersionedMemory()
    world.record_memory(memory, 'cup', 10.)
    assert memory.at('cup', 10.).state_id == 1
    assert memory.frames and evidence


def test_multiview_plan_offers_remaining_surface_after_weak_view():
    from types import SimpleNamespace
    from evolvingnav.world import HabitatAgentWorld
    viewpoints = {1: {'position_xyz': [0, 0, 0], 'rotation_xyzw': [0, 0, 0, 1],
                      'alternatives': [{'position_xyz': [1, 0, 0], 'rotation_xyzw': [0, 0, 0, 1]}]}}
    backend = SimpleNamespace(distance=lambda start, end: float(np.linalg.norm(np.asarray(end)-start)))
    world = HabitatAgentWorld(backend, viewpoints, {1: [0, 1.35, -2.]}, [0, 0, 0], [0, 0, 0, 1])
    goal, probability = world.plan_view(1, frozenset(range(5)))
    assert probability > 0 and goal in [[0, 0, 0], [1, 0, 0]]
    assert world.plan_view(1, frozenset(range(25))) is None


def test_weak_view_uses_an_alternative_before_repeating_same_pose():
    from evolvingnav.world import HabitatAgentWorld
    viewpoints = {1: {'position_xyz': [0, 0, 0], 'rotation_xyzw': [0, 0, 0, 1],
                      'alternatives': [{'position_xyz': [1, 0, 0], 'rotation_xyzw': [0, 0, 0, 1]}]}}
    class Backend:
        def distance(self, start, end):
            return float(np.linalg.norm(np.asarray(end)-start))
        def observe(self, position, rotation):
            self.last_observation = {'rgb': np.zeros((100, 100, 3), dtype=np.uint8),
                                     'depth': np.ones((100, 100))}
            self.last_detections = []
            return {'detected': False}
    world = HabitatAgentWorld(Backend(), viewpoints, {1: [0, 1.35, -2.]},
                              [0, 0, 0], [0, 0, 0, 1])
    assert world.plan_view(1, frozenset())[0] == [0, 0, 0]
    world.inspect(1)
    assert world.plan_view(1, frozenset())[0] == [1, 0, 0]
    world.inspect(1)
    assert world.plan_view(1, frozenset()) is None
    assert world.plan_view(1, frozenset(), round_id=1) is not None


def test_transition_subset_conserves_mass_and_advances_calendar():
    import torch
    from evolvingnav.transition_model import NeuralTransition, TransitionHead
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
        def backbone(self, batch):
            assert batch['query_weekday_id'].item() == 1
            return torch.zeros(1, 8), torch.zeros(1, 3, 8)
    batch = dict(candidate_state_ids=torch.tensor([[0, 1, 2]]),
                 candidate_mask=torch.ones(1, 3, dtype=torch.bool),
                 query_time_days=torch.tensor([.99]),
                 query_weekday_id=torch.tensor([0]),
                 elapsed_since_last_positive_days=torch.tensor([.1]))
    transition = NeuralTransition(Backbone(), TransitionHead(8), batch)
    transition.advance_clock(864.)
    matrix = transition.matrix([0, 2], 2.)
    np.testing.assert_allclose(matrix.sum(-1), [1., 1.], atol=1e-6)


def test_shared_encoder_accepts_new_scene_public_geometry():
    import torch
    from readyagent.p4d_belief.data import Catalog
    from readyagent.p4d_belief.models import ModelConfig, P4DBelief
    catalog = Catalog(torch.tensor([0, 0]), torch.tensor([0, 0]),
                      torch.tensor([[0., 0., 0.], [1., 1., 1.]]), torch.tensor([False, True]),
                      ('room', 'unknown'), {'region_category_to_id': {'room': 0},
                      'receptacle_category_to_id': {'table': 0}, 'category_to_id': {'mug': 0},
                      'instance_uuid_to_id': {'known': 0}})
    model = P4DBelief(catalog, ModelConfig(hidden_dim=16, layers=1, heads=4, dropout=0.,
                                         use_instance_identity=True))
    batch = dict(event_type=torch.tensor([[1]]), event_time_days=torch.tensor([[0.]]),
                 observed_state_id=torch.tensor([[2]]), candidate_state_id=torch.tensor([[-1]]),
                 evidence_features=torch.zeros(1, 1, 6), history_mask=torch.tensor([[True]]),
                 candidate_state_ids=torch.tensor([[0, 1, 2, 3]]), candidate_mask=torch.ones(1, 4, dtype=torch.bool),
                 target_category_id=torch.tensor([0]), query_time_days=torch.tensor([1.]),
                 query_time_of_day_sin_cos=torch.tensor([[0., 1.]]), query_weekday_id=torch.tensor([1]),
                 elapsed_since_last_positive_days=torch.tensor([1.]), last_state=torch.tensor([2]),
                 last_candidate_index=torch.tensor([2]), target_instance_id=torch.tensor([0]),
                 target_instance_known=torch.tensor([False]),
                 location_region_category=torch.zeros(1, 4, dtype=torch.long),
                 location_receptacle_category=torch.zeros(1, 4, dtype=torch.long),
                 location_center_xyz=torch.tensor([[[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],[0.,0.,0.]]]),
                 location_is_unknown=torch.tensor([[False, False, False, True]]))
    probabilities = model(batch)['probabilities']
    assert probabilities.shape == (1, 4)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))


def test_early_dynamic_stop_stays_in_online_recovery_denominator():
    from evolvingnav.evaluate import score_agent
    result = AgentResult(found=True, actions=['STOP'], path_m=3., elapsed_s=2.)
    result.covered_inspections = [{'state_id': 1, 'round': 0, 'time_s': 1., 'evidence_id': 'f'}]
    frames = [{'state_id': 1, 'time_s': 1., 'true_state_id': 1},
              {'state_id': 1, 'time_s': 2., 'true_state_id': 1, 'detected': True,
               'identified_target': True, 'visible_fraction': .8, 'distance_to_valid_goal_m': .4}]
    schedule = [{'time_s': 5., 'current_state_id': 2}]
    row = score_agent(result, frames, reference_distance=3., initial_state=1,
                      schedule=schedule, max_time_s=10.)
    assert row['success'] and row['dynamic_eligible'] and row['online_recovery_eligible']
    assert not row['online_recovered']
