"""Materialize temporally consistent episodes from native HSSD catalogs."""
from __future__ import annotations
import json
import random
from contextlib import ExitStack
from pathlib import Path
from .io import dump, uid
from .timeline import make_world
from .protocols import make_eqa_task, make_vln_task
from .patrol import collect_patrol


def _frame_visibility(frame):
    """Return the private calibrated fraction, or legacy binary fallback."""
    if frame.get('visibility_kind') == 'isolated_silhouette' and 'visibility_fraction' in frame:
        return max(0.0, min(1.0, float(frame['visibility_fraction'])))
    return 1.0 if frame.get('visible', False) else 0.0


def state_at(world, timestamp):
    state = world['initial_state']
    for event in world['events']:
        if event['timestamp'] > timestamp:
            break
        state = event['next_state']
    return state


def generate(config):
    root = Path(config['catalog_root']); scenes = [str(x) for x in config['scene_ids']]
    days = int(config.get('days', 5)); count = int(config.get('episodes', 1000)); seed = int(config.get('seed', 1))
    if not scenes or days <= 0 or count <= 0: raise ValueError('positive scenes, days and episodes required')
    if config.get('temporal_splits') and days != 90:
        raise ValueError('paper temporal_splits require exactly 90 days')
    catalogs = {}
    source_refs = []
    behavior_model = {}
    source_stats_path = config.get('source_statistics')
    if source_stats_path and Path(source_stats_path).is_file():
        source_stats = json.loads(Path(source_stats_path).read_text(encoding='utf-8'))
        source_refs = [f"{name}:aggregate" for name, item in source_stats.get('sources', {}).items()
                       if item.get('status') == 'available']
        behavior_model = source_stats.get('behavior_model', {})
    routine_hours = config.get('statistics', {}).get('routine_hours', [8,11,14,17])
    hour_weights = behavior_model.get('hour_weights', [])
    if len(hour_weights) == 24 and sum(hour_weights[1:23]) > 0:
        # Controlled four-phase synthetic routine fitted to source marginals.
        # Exclude midnight endpoints so pre-query patrol and N4 window fit.
        routine_hours = random.Random(int(uid(seed,'aggregate-routine-hours'),16)).choices(
            list(range(1,23)),weights=hour_weights[1:23],k=4)
    balanced = str(config.get('protocol_revision', '')).startswith('balanced_v')
    extended_tasks = config.get('protocol_revision') == 'balanced_v5'
    strict_pairing = bool(config.get('strict_pairing'))
    if strict_pairing and not balanced:
        raise ValueError('strict_pairing requires a balanced protocol revision')
    for sid in scenes:
        path = root / sid / 'catalog.json'; catalog = json.loads(path.read_text())
        if catalog.get('observation_backend') != 'habitat_rgbd': raise ValueError(f'catalog is not native RGB-D: {sid}')
        if not catalog.get('objects'): raise ValueError(f'empty native catalog: {sid}')
        catalogs[sid] = catalog
    held_out = set(config.get('held_out_scene_ids', [])) if extended_tasks else set()
    if extended_tasks:
        if not held_out or not held_out < set(scenes):
            raise ValueError('balanced_v5 requires a nonempty proper held-out scene subset')
        train_scenes = [scene for scene in scenes if scene not in held_out]
        held_out_scenes = [scene for scene in scenes if scene in held_out]
    explicit_splits = config.get('split_counts') if config.get('temporal_splits') else None
    task_counts = config.get('task_counts')
    split_task_counts = config.get('split_task_counts')
    split_task_sequences = None
    if split_task_counts is not None:
        if not explicit_splits or set(split_task_counts) != set(explicit_splits):
            raise ValueError('split_task_counts requires train/val/test split_counts')
        split_task_sequences = {}
        for split, counts in split_task_counts.items():
            if sum(int(v) for v in counts.values()) != explicit_splits[split]:
                raise ValueError(f'split task counts do not sum for {split}')
            if strict_pairing:
                if any(int(amount) % 4 for amount in counts.values()):
                    raise ValueError(f'strict pairing requires task quotas divisible by four for {split}')
                # A task is assigned to a complete four-regime block.  The
                # sequence is shuffled at block granularity, never per row.
                blocks = [[task] * 4 for task, amount in sorted(counts.items())
                          for _ in range(int(amount) // 4)]
                random.Random(int(uid(seed, 'task-sequence', split), 16)).shuffle(blocks)
                seq = [task for block in blocks for task in block]
            else:
                seq = [task for task, amount in sorted(counts.items()) for _ in range(int(amount))]
                random.Random(int(uid(seed, 'task-sequence', split), 16)).shuffle(seq)
            split_task_sequences[split] = seq
    if task_counts is not None:
        allowed_tasks = {'N1', 'N2', 'N3', 'N4', 'N5', 'VLN', 'EQA'}
        if set(task_counts) - allowed_tasks or any(int(v) < 0 for v in task_counts.values()) or sum(task_counts.values()) != count:
            raise ValueError('task_counts must be nonnegative, known tasks summing to episodes')
        task_sequence = [task for task, amount in sorted(task_counts.items()) for _ in range(int(amount))]
        random.Random(int(uid(seed, 'task-sequence'), 16)).shuffle(task_sequence)
    else:
        task_sequence = None
    requests = []
    if explicit_splits:
        if sum(explicit_splits.values()) != count or set(explicit_splits) != {'train', 'val', 'test'}:
            raise ValueError('split_counts must sum to episodes and contain train/val/test')
        split_ranges = {'train': (0, 80), 'val': (80, 85), 'test': (85, 90)}
        indexed = [(split, j) for split in ('train', 'val', 'test') for j in range(explicit_splits[split])]
    else:
        indexed = [(None, index) for index in range(count)]
    for index, (requested_split, split_index) in enumerate(indexed):
        # In strict mode each four-row block is one scene/task counterfactual
        # group.  The legacy order remains unchanged for prototype tests.
        strict_pair_slot = split_index // 4 if strict_pairing and requested_split else None
        local_index = ((strict_pair_slot * 4 + split_index % 4)
                       if strict_pair_slot is not None else index // len(scenes))
        # One local-index block is one regime per day. Explicit paper splits
        # use their own day cycle so requested counts are exact.
        if requested_split:
            lo, hi = split_ranges[requested_split]
            day = lo + ((split_index // 4) % (hi - lo))
        else:
            day = ((local_index // 4) if config.get('calendar_coverage') else local_index) % days
        selection = None
        if balanced:
            local = local_index
            lo, hi = split_ranges[requested_split] if requested_split else (0, days)
            width = hi - lo
            day = lo + (local // 4) % width
            task_cycle = tuple(config.get('task_cycle', ('N1', 'N2', 'N3', 'N4')))
            if not task_cycle or any(task not in {'N1', 'N2', 'N3', 'N4', 'N5', 'VLN', 'EQA'} for task in task_cycle):
                raise ValueError('invalid task_cycle')
            task_index = (strict_pair_slot * 4 if strict_pair_slot is not None else split_index)
            task = (split_task_sequences[requested_split][task_index] if split_task_sequences else
                    task_sequence[index] if task_sequence is not None else task_cycle[(local // 4) % len(task_cycle)])
            if extended_tasks and requested_split in ('train', 'val') and task == 'N5':
                task = 'N1'
            if extended_tasks:
                eligible_scenes = held_out_scenes if task == 'N5' else train_scenes
                scene_id = eligible_scenes[(strict_pair_slot if strict_pair_slot is not None else split_index) % len(eligible_scenes)]
            else:
                scene_id = scenes[(strict_pair_slot if strict_pair_slot is not None else split_index) % len(scenes)]
            scene_index = scenes.index(scene_id)
            objects = catalogs[scenes[scene_index]]['objects']
            target_index = (local // (len(task_cycle) * width)) % len(objects)
            rng = random.Random(int(uid(seed, scenes[scene_index], requested_split, local // 4), 16))
            query = (day * 86400 + routine_hours[day % len(routine_hours)] * 3600 - rng.uniform(30, 180)
                     if task == 'N4' else day * 86400 + rng.uniform(18*3600, 22*3600))
            morning_end = min(6*3600, (query-day*86400)*.6)
            times = sorted(set([max(0., query-3*86400), max(0., query-86400),
                                day*86400+rng.uniform(morning_end*.5, morning_end)]))
            selection = dict(scene_index=scene_index, target_index=target_index,
                             local_index=local, task=task, start_draw=rng.randrange(1000000))
        else:
            query = float(config.get('query_time', day * 86400 + 18 * 3600))
            times = config.get('history_timestamps', [max(0., query - 12 * 3600)])
        if not 0 < query < days * 86400: raise ValueError('query_time outside world horizon')
        if not times or any(not 0 <= t < query for t in times) or list(times) != sorted(set(times)):
            raise ValueError('history timestamps must be unique, ordered, nonnegative and before query')
        requests.append((query, times, requested_split, selection))
    out = Path(config['output'])
    if (out / 'manifest.json').exists(): raise FileExistsError(f'refusing to overwrite existing generation: {out}')
    (out / 'episodes').mkdir(parents=True, exist_ok=True); (out / 'private').mkdir(exist_ok=True)
    worlds = {}
    splits = ('train', 'val', 'test') if config.get('temporal_splits') else ('prototype',)
    split_counts = {split: 0 for split in splits}
    with ExitStack() as stack:
        handles = {(kind, split): stack.enter_context((out/kind/(split+'.jsonl')).open('w'))
                   for kind in ('episodes', 'private') for split in splits}
        for index, (query, times, requested_split, selection) in enumerate(requests):
            sid = scenes[selection['scene_index'] if selection else index % len(scenes)]
            catalog = catalogs[sid]
            local_index = selection['local_index'] if selection else index // len(scenes)
            target = catalog['objects'][selection['target_index'] if selection else (local_index // 4) % len(catalog['objects'])]
            regime = ('static', 'routine', 'personal', 'random')[local_index % 4]
            # Dynamic N4 windows are part of the hidden schedule contract; do
            # not reuse a cached world whose first event was anchored to a
            # different query time.
            anchor_key = (round(query, 3) if selection is not None and
                          selection['task'] == 'N4' and regime != 'static' else None)
            key = (sid, target['object_id'], regime, anchor_key)
            if key not in worlds:
                stats = dict(config.get('statistics', {'move_rate': 90}))
                stats.setdefault('source_refs', source_refs)
                stats['routine_hours'] = routine_hours
                if behavior_model:
                    stats['category_activity_counts'] = behavior_model.get('category_activity_counts', {})
                if selection is not None and selection['task'] == 'N4' and regime != 'static':
                    stats['query_anchor_s'] = query
                if regime == 'personal':
                    states = [placement['state_id'] for placement in target['placements']]
                    profile_cfg = config.get('owner_profiles', {})
                    configured = profile_cfg.get(target['object_id']) if isinstance(profile_cfg, dict) else None
                    if isinstance(configured, dict):
                        preferred = configured.get('preferred_state')
                        owner_id = str(configured.get('owner_id', 'resident_0'))
                    else:
                        # Deterministic synthetic resident profiles are chosen
                        # from legal receptacles and are kept evaluator-only.
                        owner_index = int(uid(seed, sid, target['object_id'], 'owner-profile'), 16)
                        owner_id = f'resident_{owner_index % 3}'
                        preferred = states[1 + (owner_index % max(1, len(states) - 1))] if len(states) > 1 else states[0]
                    if preferred not in states:
                        raise ValueError(f'owner profile preferred_state is not legal for {target["object_id"]}')
                    stats['owner_profile'] = {'owner_id': owner_id, 'preferred_state': preferred}
                worlds[key] = make_world(target, days=days, regime=regime,
                                         seed=int(uid(seed, sid, target['object_id']), 16),
                                         stats=stats)
            world = worlds[key]; current = state_at(world, query)
            pair_group_id = None
            history_world = world
            if balanced and selection is not None:
                pair_group_id = uid('paired', seed, requested_split, sid,
                                    target['object_id'], selection['local_index'] // 4,
                                    selection['task'])
                history_key = (sid, target['object_id'], 'paired_history')
                if history_key not in worlds:
                    worlds[history_key] = make_world(
                        target, days=days, regime='static',
                        seed=int(uid(seed, sid, target['object_id'], 'paired-history'), 16),
                        stats={'move_rate': 0, 'source_refs': source_refs})
                history_world = worlds[history_key]
            views = {v['view_id']: v for v in target['views']}; history = []
            if config.get('patrol_mode'):
                patrol_seed = int(uid(seed, sid, pair_group_id or index), 16)
                # Plan one complete receptacle sweep per observation window.
                # Timestamps and viewpoints depend on public geometry/time only;
                # private replay may admit/reject, but NEVER changes the plan.
                sweep_times = []
                previous = 0.
                for timestamp in times:
                    n = len(target['placements'])
                    if timestamp == 0:
                        continue  # A full past sweep cannot fit before time zero.
                    span = min(10., timestamp-previous)
                    sweep_times.extend(timestamp-span*(n-i-1)/n for i in range(n))
                    previous = timestamp
                patrol = collect_patrol(target, history_world, seed=patrol_seed, timestamps=sweep_times)
                if not any(item['visible'] for item in patrol):
                    raise ValueError('fixed patrol failed admission; no history sighting (no seed retry permitted)')
                for item in patrol:
                    history_item = dict(timestamp=item['timestamp'], rgb=item['rgb'], depth=item['depth'],
                                         camera_pose=item['camera_pose'], visible=item['visible'])
                    if item['visible']:
                        history_item['location'] = state_at(history_world, item['timestamp'])
                    history.append(history_item)
            else:
                for timestamp in times:
                    observed = state_at(history_world, timestamp); view = views[observed]; frame = target['frames'][observed+'/'+view['view_id']]
                    if not frame['visible']: raise ValueError('history requires a validated positive sighting')
                    history.append(dict(timestamp=timestamp, location=observed, rgb=frame['rgb'], depth=frame['depth'],
                                        camera_pose=dict(position=view['position'], rotation_wxyz=view['rotation_wxyz']),
                                        visible=True))
            candidates = [dict(state_id=p['state_id'], room=p['room'], receptacle_category=p['receptacle_category'],
                               relation=p.get('relation','on'), viewpoint=p['viewpoint']) for p in target['placements'][:-1]]
            candidates.append(dict(state_id='unknown', relation='unobserved'))
            start_draw = selection['start_draw'] if selection else local_index
            start = views[target['starts'][start_draw % len(target['starts'])]]
            task = selection['task'] if selection else 'N4' if regime == 'random' else 'N2'
            if extended_tasks and task == 'N5' and sid not in set(config.get('held_out_scene_ids', [])):
                raise ValueError(f'N5 scene is not held out: {sid}')
            if balanced and task == 'N4' and regime != 'static':
                if not any(query < e['timestamp'] <= query+3600 for e in world['events']):
                    raise ValueError('N4 dynamic admission failed: no transition in execution budget')
            row = dict(id=uid('native_v5' if extended_tasks else 'native_v4' if balanced else 'native_v3' if config.get('calendar_coverage') else 'native_v2', seed, sid, index), scene=sid,
                       target=dict(object_id=target['object_id'], category=target['category']), query_time=query,
                       history=history, candidate_states=candidates, start=start,
                       navigation_budget=dict(max_steps=500, max_distance_m=100, max_inspections=10, max_seconds=3600),
                       task_type=task)
            if pair_group_id is not None:
                row['pair_group_id'] = pair_group_id
            if not extended_tasks:
                row['mobility_regime'] = 'hidden'
            if task == 'VLN':
                row.update(make_vln_task(target['category'], target['placements'][0]['room'], sid))
            elif task == 'EQA':
                row.update({k: v for k, v in make_eqa_task('where', target['category'], current, sid).items() if k != 'private_answer'})
            elif task == 'N5':
                row['generalization_split'] = 'held_out'
            truth = dict(id=row['id'], scene=sid, target_object=target['object_id'], current_state=current,
                         mobility_regime=regime, world=world, history_world=history_world,
                         evolution_during_execution=task == 'N4',
                         dynamic_denominator_member=bool(task == 'N4' and regime != 'static'),
                         hidden_future_events=[e for e in world['events'] if e['timestamp'] > query] if task == 'N4' else [])
            if task == 'EQA':
                truth['private_answer'] = current
            split = requested_split or (('train' if query < 80*86400 else 'val' if query < 85*86400 else 'test') if config.get('temporal_splits') else 'prototype')
            handles['episodes', split].write(json.dumps(row, allow_nan=False)+'\n')
            handles['private', split].write(json.dumps(truth, allow_nan=False)+'\n')
            split_counts[split] += 1
    calibrated = all(frame.get('visibility_kind') == 'isolated_silhouette'
                     for catalog in catalogs.values()
                     for target_row in catalog['objects']
                     for frame in target_row['frames'].values())
    limitations = ['N1-N5 task construction and external-agent score aggregation are separate evaluation stages',
                   'N4 online execution uses balanced admission with scheduled transitions inside the evaluation budget',
                   'source calibration and human review are separate release checks',
                   'single movable target per replay']
    if not calibrated:
        limitations.append('binary visibility; isolated-silhouette calibration is required for calibrated evidence')
    source_statistics = None
    if config.get('source_statistics'):
        source_path = Path(config['source_statistics'])
        if source_path.is_file():
            source_statistics = json.loads(source_path.read_text(encoding='utf-8'))
    dump(out/'manifest.json', dict(name='EVOWORLD-BENCH', schema_version='native_v5' if extended_tasks else 'native_v4' if balanced else 'native_v3' if config.get('calendar_coverage') else 'native_v2', observation_backend='habitat_rgbd',
        episodes=count, scenes=scenes, days=days, seed=seed, native=True, release_status='construction',
        calendar_coverage=bool(config.get('calendar_coverage')), generator_config=config,
        split_counts=split_counts,
        temporal_contract='history replayed at observation time', visibility_calibration='isolated_silhouette' if calibrated else 'binary_pixel_threshold',
        patrol_contract='single world-blind full sweep; admission only; no truth-dependent seed retry',
        public_visibility_contract='private silhouette fractions excluded; historical identity annotations retained',
        source_statistics=source_statistics,
        behavior_parameterization=dict(routine_hours=routine_hours, source_distributions_consumed=bool(behavior_model),
            movement_frequency='synthetic controlled move_rate; not inferred from sensor counts'),
        limitations=limitations))
    return dict(episodes=count, scenes=len(scenes), output=str(out))
