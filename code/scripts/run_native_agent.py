#!/usr/bin/env python3
"""Run the existing EvolvingNav Agent on native EVOWORLD episodes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from evoworld.agent_bridge import (load_agent_package, public_scene_map, public_query,
    pack_query_for_model, scheduled_motion, native_backend, xyzw, select_public_episodes)
from evoworld.agent_bridge import build_controller
from evoworld.io import read_jsonl


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--catalog-root', type=Path, required=True)
    parser.add_argument('--agent-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['prototype','val','test'], default='test')
    parser.add_argument('--tasks', nargs='+', default=['N1','N2','N3','N4','N5'])
    parser.add_argument('--limit-per-task', type=int, default=15)
    parser.add_argument('--prior', choices=['last_seen','uniform','checkpoint'], default='checkpoint')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--training-dataset', type=Path)
    parser.add_argument('--transition-checkpoint', type=Path)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--controller', choices=['utility','luna'], default='utility')
    parser.add_argument('--api-key-file', type=Path)
    args = parser.parse_args()
    if args.limit_per_task < 1 or args.output.exists():
        raise ValueError('positive episode quota and new output directory required')
    load_agent_package(args.agent_root)
    import numpy as np
    from evolvingnav.agent import Agent, AgentConfig
    from evolvingnav.calibration import DetectionCalibrator
    from evolvingnav.evaluate import score_agent, aggregate_metrics, oracle_distance
    from evolvingnav.memory import VersionedMemory
    from evolvingnav.perception import GroundedSAMInspector
    from evolvingnav.policy import load_belief, predict_public, model_input_batch
    from evolvingnav.transition import IdentityTransition
    from evolvingnav.transition_model import NeuralTransition, TransitionHead
    from evolvingnav.world import HabitatAgentWorld

    public_selection = select_public_episodes(read_jsonl(args.dataset/'episodes'/f'{args.split}.jsonl'),
                                             args.tasks,args.limit_per_task)
    wanted = dict(public_selection)
    selected, counts = [], Counter()
    with (args.dataset/'private'/f'{args.split}.jsonl').open() as handle:
        for index,line in enumerate(handle):
            if index not in wanted:
                continue
            row,truth = wanted[index],json.loads(line)
            if row['id'] != truth['id']:
                raise ValueError('unaligned public/private episode streams')
            selected.append((row,truth,index)); counts[row['task_type']] += 1
    if any(counts[t] != args.limit_per_task for t in args.tasks):
        raise ValueError(f'episode quota incomplete: {counts}')
    model = schema = head = None
    if args.prior == 'checkpoint':
        if args.checkpoint is None or args.training_dataset is None:
            raise ValueError('checkpoint prior requires checkpoint and training dataset')
        model, schema = load_belief(args.transition_checkpoint or args.checkpoint, args.training_dataset)
    if args.transition_checkpoint is not None:
        import torch
        payload = torch.load(args.transition_checkpoint, map_location='cpu', weights_only=True)
        head = TransitionHead(payload['model_config']['hidden_dim'])
        head.load_state_dict(payload['transition_head']); head.eval()
    args.output.mkdir(parents=True)
    manifest = {'dataset': str(args.dataset.resolve()), 'split': args.split,
        'requested_episodes': len(selected), 'episode_ids': [r['id'] for r,_,_ in selected],
        'source_indices': [i for _,_,i in selected], 'tasks': dict(counts),
        'selection': 'SHA256 episode ID, public-only selection before truth loading',
        'prior': args.prior, 'controller': args.controller,
        'agent_root': str(args.agent_root.resolve()),
        'checkpoint_sha256': digest(args.checkpoint) if args.checkpoint else None,
        'transition_checkpoint_sha256': digest(args.transition_checkpoint) if args.transition_checkpoint else None,
        'calibration_sha256': digest(args.calibration), 'score_status': 'native_adapter',
        'limitations': ['belief checkpoint is transferred from P4D unless separately trained on EVOWORLD',
            'Luna API key is process-only and never written to artifacts',
            'N4 forecasting is frozen without a transition checkpoint',
            'cross-split RGB-D frame reuse remains a construction audit gap; do not report formal scores until resolved'],
        'agent_files_sha256': {str(p.relative_to(args.agent_root)): digest(p)
            for p in sorted((args.agent_root/'evolvingnav').glob('*.py'))}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    calibrator = DetectionCalibrator.load(args.calibration)
    inspector = GroundedSAMInspector(args.agent_root/'configs/perception.yaml',
        dino_model='IDEA-Research/grounding-dino-tiny', sam_model='facebook/sam2.1-hiera-tiny')
    controller = build_controller(args.controller, args.api_key_file)
    catalogs, scores, invalid = {}, [], []
    try:
        with (args.output/'scores.private.jsonl').open('w') as private_log, (args.output/'policy.jsonl').open('w') as public_log:
            for episode_index, (row,truth,source_index) in enumerate(selected):
                if row['scene'] not in catalogs:
                    catalogs[row['scene']] = json.loads((args.catalog_root/row['scene']/'catalog.json').read_text())
                catalog = catalogs[row['scene']]
                target = next(o for o in catalog['objects'] if o['object_id'] == row['target']['object_id'])
                scene_map = public_scene_map(target)
                states = scene_map['states']
                ids = {s['state_id']: i for i,s in enumerate(states)}; ids['unknown'] = len(states)
                query = public_query(row, ids, scene_map=scene_map)
                if model is not None:
                    arrays = pack_query_for_model(row, scene_map, ids, schema)
                    prior = predict_public(model,schema,arrays)
                else:
                    candidates = query['input']['candidate_state_ids']
                    last = query['input']['target_history'][-1]['observed_state_id']
                    # Retain a small alternative mass to allow evidence updates.
                    prior = {s: 1/len(candidates) if args.prior == 'uniform' else
                             (.95 if s == last else .05/(len(candidates)-1)) for s in candidates}
                viewpoints = {ids[s['state_id']]: {'position_xyz': s['viewpoint'],
                             'rotation_xyzw': s['view_rotation_xyzw']} for s in states}
                centers = {ids[s['state_id']]: s['surface_center'] for s in states}
                known = set(query['input']['candidate_state_ids']) - {ids['unknown']}
                goals = {s: viewpoints[s]['position_xyz'] for s in known}
                protocol = row['task_type'].lower() if row['task_type'] != 'N5' else 'n3'
                budget = row['navigation_budget']
                config = AgentConfig(protocol=protocol, max_inspections=1 if protocol=='n1' else budget['max_inspections'],
                    max_path_m=budget['max_distance_m'], max_steps=budget['max_steps'],
                    max_time_s=budget['max_seconds'], unknown_state=ids['unknown'])
                backend = None
                try:
                    backend = native_backend(catalog,target,truth,ids,inspector)
                    frame_records = []
                    frame_dir = args.output/'frames'/row['id']
                    frame_dir.mkdir(parents=True)
                    def save_frame(observation):
                        from PIL import Image
                        number = len(frame_records)
                        rgb_path, depth_path = frame_dir/f'{number:05d}.png', frame_dir/f'{number:05d}.npz'
                        Image.fromarray(observation['rgb'][..., :3]).save(rgb_path)
                        np.savez_compressed(depth_path, depth_m=observation['depth'].astype(np.float32))
                        frame_records.append({'frame_id': number, 'rgb': str(rgb_path), 'depth': str(depth_path),
                            'position_xyz': observation['position_xyz'], 'rotation_xyzw': observation['rotation_xyzw'],
                            'detections': observation['detections']})
                    backend.observation_sink = save_frame
                    schedule = scheduled_motion(row,truth,target,ids)
                    world = HabitatAgentWorld(backend,viewpoints,centers,row['start']['position'],
                        xyzw(row['start']['rotation_wxyz']), known_states=known,
                        motion_schedule=schedule,calibrator=calibrator)
                    start_state = ids[truth['current_state']]
                    phases = [(0.,[viewpoints[start_state]['position_xyz']])] + [
                        (e['time_s'],[v['position_xyz'] for v in e['valid_goal_viewpoints']]) for e in schedule]
                    reference = oracle_distance(start=row['start']['position'],phases=phases,
                        distance=backend.distance,max_time_s=config.max_time_s,
                        max_path_m=config.max_path_m,max_steps=config.max_steps)
                    transition = NeuralTransition(model,head,model_input_batch(arrays,schema)) if protocol=='n4' and head is not None else IdentityTransition()
                    memory = VersionedMemory()
                    for i,item in enumerate(query['input']['target_history']):
                        state = item['observed_state_id']
                        memory.observe(row['target']['object_id'], state,item['timestamp_s'],
                            item['detector_confidence'],f'{row["id"]}:history:{i}',
                            np.asarray(centers.get(state,[0,0,0])),category=target['category'])
                    result = Agent(prior,goals,world,transition,config,
                        sample_count={s:world.sample_count(s) for s in known}, memory=memory,
                        target_id=row['target']['object_id'],time_origin_s=row['query_time'],
                        target={'category':target['category']}, controller=controller).run()
                    score = score_agent(result,world.private_inspections,reference_distance=reference,
                        initial_state=start_state,schedule=schedule,max_time_s=config.max_time_s)
                    score.update(episode_id=row['id'],task_type=row['task_type'],scene=row['scene'],
                        regime=truth['mobility_regime'],termination=result.termination,
                        applied_transitions=world._next_motion,
                        inspection_order=result.inspections,covered_inspections=result.covered_inspections,
                        visits=result.visits,private_inspections=world.private_inspections)
                    scores.append(score)
                    private_log.write(json.dumps(score)+'\n'); private_log.flush()
                    public_record = {'episode_id':row['id'],'task_type':row['task_type'],
                        'actions':result.actions,'prior':prior,'posterior':result.posterior,
                        'evidence':result.evidence_trace,'distance_m':result.path_m,
                        'elapsed_s':result.elapsed_s,'steps':result.steps,'termination':result.termination,
                        'observations':frame_records}
                    public_log.write(json.dumps(public_record)+'\n'); public_log.flush()
                    print(json.dumps({'completed':len(scores),'requested':len(selected),'task':row['task_type'],
                                      'success':score['success'],'termination':result.termination}),flush=True)
                except (RuntimeError,ValueError,KeyError) as exc:
                    invalid.append({'episode_id':row['id'],'error_type':type(exc).__name__,'reason':str(exc)})
                    print(json.dumps({'invalid':len(invalid),'episode_id':row['id'],'error_type':type(exc).__name__}),flush=True)
                finally:
                    if backend is not None:
                        backend.close()
    finally:
        inspector.close()
    summary = {**aggregate_metrics(scores), 'requested':len(selected),'invalid_count':len(invalid),
        'quota_complete':len(scores)==len(selected),'score_status':'native_adapter',
        'by_task':{task:aggregate_metrics([s for s in scores if s['task_type']==task]) for task in args.tasks}}
    (args.output/'invalid.json').write_text(json.dumps(invalid,indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return 0 if summary['quota_complete'] else 1


if __name__=='__main__':
    raise SystemExit(main())
