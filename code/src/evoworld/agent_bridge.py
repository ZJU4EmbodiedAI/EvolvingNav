"""Public-input and private-runtime adapters for the local EvolvingNav Agent.

The agent package is loaded explicitly from --agent-root. Its policy receives
public receptacle geometry and RGB-D; mesh poses and semantic masks remain on
the evaluator backend.
"""
from __future__ import annotations

from pathlib import Path
import sys
import os
import re


def load_api_key_file(path):
    """Load an API key into the process only; never return or persist its value."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = None
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        match = re.match(r'^(?:export\s+)?(?:OPENAI_API_KEY|chatgpt)\s*(?:=|:)\s*["\']?([^"\'\s]+)', line, re.I)
        if match:
            value = match.group(1)
            break
        if line.startswith('sk-'):
            value = line.split()[0]
            break
    if not value:
        raise ValueError('API key file contains no OPENAI_API_KEY assignment')
    os.environ['OPENAI_API_KEY'] = value
    return 'OPENAI_API_KEY'


def build_controller(name='utility', api_key_file=None):
    if name == 'utility':
        return None
    if name != 'luna':
        raise ValueError(f'unknown controller: {name}')
    if api_key_file is None:
        raise ValueError('luna controller requires --api-key-file')
    load_api_key_file(api_key_file)
    from evolvingnav.controller import LunaToolController
    return LunaToolController()


def select_public_episodes(rows, tasks, quota):
    """Freeze hash-ranked episode IDs using public data before truth is read."""
    import hashlib
    chosen = {task: [] for task in tasks}
    for index, row in enumerate(rows):
        task = row.get('task_type')
        if task not in chosen:
            continue
        key = hashlib.sha256(str(row['id']).encode()).hexdigest()
        chosen[task].append((key, index, row))
        chosen[task].sort(key=lambda value: (value[0],value[1]))
        del chosen[task][quota:]
    if any(len(values) != quota for values in chosen.values()):
        raise ValueError('insufficient public episodes for fixed quota')
    return sorted([(index,row) for values in chosen.values() for _,index,row in values])


def load_agent_package(root):
    root = Path(root).resolve()
    if not (root/'evolvingnav/agent.py').is_file():
        raise FileNotFoundError(f'EvolvingNav Agent package missing: {root}')
    for path in (root, root/'src', root/'scripts'):
        sys.path.insert(0, str(path))
    return root


def xyzw(wxyz):
    return [*wxyz[1:], wxyz[0]]


def public_scene_map(target):
    """Export fixed receptacle support geometry, never movable-target centers."""
    views = {view['view_id']: view for view in target['views']}
    states = []
    for placement in target['placements']:
        view = views[placement['state_id']]
        # support_point is static surface geometry admitted during native bake.
        # position is the movable mesh pose and is deliberately not consulted.
        center = placement['support_point']
        states.append({'state_id': placement['state_id'], 'room': placement.get('room', 'unknown'),
                       'receptacle_category': placement.get('receptacle_category', 'unknown'),
                       'relation': placement.get('relation', 'on'), 'surface_center': list(center),
                       'viewpoint': list(view['position']),
                       'view_rotation_xyzw': xyzw(view['rotation_wxyz'])})
    return {'states': states, 'provenance': 'fixed_receptacle_support_geometry'}


def public_query(row, state_ids, *, scene_map=None):
    """Translate supported sightings; evaluator silhouette values are discarded."""
    candidates = [state_ids[c['state_id']] for c in row['candidate_states']]
    unknown = state_ids['unknown']
    history = []
    previous = None
    states = (scene_map or {}).get('states', [])
    candidate_state_ids = set(candidates)
    def nearest_candidate(item):
        pose = item.get('camera_pose', {}).get('position') if isinstance(item.get('camera_pose'), dict) else None
        if pose is None or not states:
            return None
        best = min(states, key=lambda state: sum((float(a)-float(b))**2 for a,b in zip(pose,state.get('viewpoint',[0,0,0]))))
        state = state_ids.get(best.get('state_id'))
        return state if state in candidate_state_ids else None
    for item in row['history']:
        timestamp = float(item['timestamp'])
        if timestamp > row['query_time'] or (previous is not None and timestamp < previous):
            raise ValueError('history is not an ordered causal prefix')
        if item.get('visible') and 'location' in item:
            state = state_ids.get(item['location'], unknown)
            if state not in candidates:
                state = unknown
            history.append({'timestamp_s': timestamp, 'event_type': 'positive_observation',
                            'observed_state_id': state,
                            'delta_t_from_previous_s': 0. if previous is None else timestamp-previous,
                            'detector_confidence': float(item.get('detector_confidence', 1.)),
                            'instance_match_confidence': float(item.get('identity_confidence', 1.)),
                            'visible_fraction_estimate': 1.})
            previous = timestamp
        elif not item.get('visible'):
            state = nearest_candidate(item)
            if state is not None:
                history.append({'timestamp_s': timestamp, 'event_type': 'candidate_inspection',
                    'candidate_state_id': state,
                    'surface_in_frustum_fraction': float(item.get('surface_in_frustum_fraction', 0.0)),
                    'surface_unoccluded_fraction': float(item.get('surface_unoccluded_fraction', 0.0)),
                    'negative_evidence_strength': float(item.get('negative_evidence_strength', 0.0)),
                    'delta_t_from_previous_s': 0. if previous is None else timestamp-previous})
            previous = timestamp
    context = []
    for item in row.get('context_history', []):
        timestamp = float(item['timestamp'])
        if timestamp > row['query_time']:
            raise ValueError('future context observation')
        context.append({'timestamp_s': timestamp,
            'observed_category_counts': {str(k): int(v) for k,v in item.get('observed_category_counts', {}).items()},
            'detected_object_count': int(item.get('detected_object_count', 0)),
            'observed_change_count': int(item.get('observed_change_count', 0)),
            'inspected_candidate_state_ids': [state_ids[s] for s in item.get('inspected_candidate_state_ids', []) if s in state_ids],
            'observed_region_ids': [str(s) for s in item.get('observed_region_ids', [])]})
    if not history:
        raise ValueError('query has no supported positive history')
    return {'query_id': row['id'], 'input': {
        'target': {'instance_uuid': row['target']['object_id'], 'category': row['target']['category']},
        'query': {'query_time_s': float(row['query_time'])},
        'target_history': history, 'observable_context_history': sorted(context, key=lambda x:x['timestamp_s']),
        'candidate_state_ids': candidates,
        'history_summary': {'elapsed_since_last_positive_s': row['query_time']-history[-1]['timestamp_s']}}}


def scheduled_motion(row, truth, target, state_ids):
    """Evaluator-only, fixed-window N4 events; other protocols remain frozen."""
    if row['task_type'] != 'N4':
        return []
    placements = {p['state_id']: p for p in target['placements']}
    views = {v['view_id']: v for v in target['views']}
    query, end = row['query_time'], row['query_time']+row['navigation_budget']['max_seconds']
    schedule = []
    for event in truth['world']['events']:
        if not query < event['timestamp'] <= end or event['previous_state'] == event['next_state']:
            continue
        state = event['next_state']
        schedule.append({'time_s': event['timestamp']-query, 'current_state_id': state_ids[state],
                         'target_position_xyz': placements[state]['position'],
                         'target_rotation_wxyz': placements[state].get('rotation_wxyz', [1, 0, 0, 0]),
                         'valid_goal_viewpoints': [{'position_xyz': views[state]['position']}]})
    return schedule


def native_backend(catalog, target, truth, state_ids, inspector):
    """Instantiate two Habitat worlds behind the Agent's public World interface."""
    from evolvingnav.backend import HabitatInspectionBackend
    from .bake import simulator
    import habitat_sim
    import magnum as mn
    import numpy as np

    class Backend(HabitatInspectionBackend):
        def observe(self, position, rotation):
            result = super().observe(position, rotation)
            projected = np.asarray(self.target_only.get_sensor_observations()['semantic']) == self.semantic_id
            visible = self.last_semantic == self.semantic_id
            denominator = int(projected.sum())
            result['visible_fraction'] = float((visible & projected).sum()/denominator) if denominator else 0.
            sink = getattr(self, 'observation_sink', None)
            if sink is not None:
                sink({'rgb': self.last_observation['rgb'], 'depth': self.last_observation['depth'],
                      'position_xyz': list(position), 'rotation_xyzw': list(rotation),
                      'detections': [{'category': d.category, 'confidence': d.confidence}
                                     for d in self.last_detections]})
            return result

        def move_target(self, event):
            super().move_target(event)
            w, x, y, z = event.get('target_rotation_wxyz', [1, 0, 0, 0])
            for sim, oid in ((self.real, self.object_id), (self.target_only, self.target_only_object_id)):
                sim.get_rigid_object_manager().get_object_by_id(oid).rotation = mn.Quaternion(mn.Vector3(x,y,z), w)

    backend = Backend(Path(catalog['habitat_scene']).parent, catalog['scene_id'],
                      Path(catalog['navmesh']), {}, detector=inspector)
    backend.semantic_id = int(target.get('semantic_id', 60000))
    backend.target_category = target['category']
    placement = next(p for p in target['placements'] if p['state_id'] == truth['current_state'])
    view = next(v for v in target['views'] if v['view_id'] == truth['current_state'])
    backend.truth = {'current_state_id': state_ids[truth['current_state']],
                     'target_position_xyz': placement['position'],
                     'valid_goal_viewpoints': [{'position_xyz': view['position']}]}
    try:
        for isolated in (False, True):
            sim = simulator('NONE' if isolated else catalog['habitat_scene'], catalog['dataset_config'],
                            resolution=tuple(catalog.get('resolution', [240, 320])))
            if isolated:
                backend.target_only = sim
            else:
                backend.real = sim
                if not sim.pathfinder.load_nav_mesh(catalog['navmesh']):
                    raise RuntimeError('native navmesh failed to load')
            tm = sim.get_object_template_manager()
            tm.load_configs(target['template_path'])
            handles = tm.get_template_handles(target['template_hash'])
            if not handles:
                raise RuntimeError('native target template failed to load')
            obj = sim.get_rigid_object_manager().add_object_by_template_handle(handles[-1])
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            obj.semantic_id = backend.semantic_id
            obj.translation = mn.Vector3(*placement['position'])
            w, x, y, z = placement.get('rotation_wxyz', [1, 0, 0, 0])
            obj.rotation = mn.Quaternion(mn.Vector3(x,y,z), w)
            if isolated:
                backend.target_only_object_id = obj.object_id
            else:
                backend.object_id = obj.object_id
        return backend
    except Exception:
        backend.close()
        raise


def pack_query_for_model(row, scene_map, state_ids, schema):
    import numpy as np
    from evolvingnav.policy import pack_public_query
    regions = schema['region_category_to_id']
    receptacles = schema['receptacle_category_to_id']
    states = scene_map['states'] + [{'room': 'unknown', 'receptacle_category': 'unknown',
                                   'surface_center': [0, 0, 0]}]
    features = {
        'candidate_region_category_id': np.asarray([regions.get(s['room'], regions['unknown']) for s in states]),
        'candidate_receptacle_category_id': np.asarray([receptacles.get(s['receptacle_category'], receptacles['unknown']) for s in states]),
        'candidate_center_xyz': np.asarray([s['surface_center'] for s in states], dtype=np.float32),
        'candidate_is_unknown': np.asarray([False]*(len(states)-1)+[True])}
    return pack_public_query(public_query(row, state_ids, scene_map=scene_map),
                             {**schema, 'state_count_including_unknown': len(states)}, features)
