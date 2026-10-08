"""Task contracts shared by construction, split, and evaluator code."""
from __future__ import annotations

import hashlib
import math


def held_out_scene_split(scene_ids, held_out_fraction=0.2):
    scenes = sorted({str(scene) for scene in scene_ids})
    if not scenes or not 0 < held_out_fraction < 1:
        raise ValueError('scene_ids and held_out_fraction are invalid')
    count = max(1, math.ceil(len(scenes) * held_out_fraction))
    ranked = sorted(scenes, key=lambda scene: hashlib.sha256(scene.encode()).hexdigest())
    held_out = ranked[:count]
    return {'train': [scene for scene in scenes if scene not in held_out],
            'held_out': held_out, 'policy': 'sha256 deterministic scene holdout'}


def make_vln_task(target_category, room, scene_id):
    return {'task_type': 'VLN', 'scene': str(scene_id),
            'language_goal': f'Find the {target_category} and verify it.',
            'goal_constraints': {'room_hint': room, 'requires_visual_verification': True}}


def make_eqa_task(question_type, target_category, private_answer, scene_id):
    prompts = {
        'where': f'Where is the {target_category} at the query time?',
        'what': f'What object was being tracked?',
        'when': f'When was the {target_category} last observed?'
    }
    if question_type not in prompts:
        raise ValueError(f'unknown EQA question type: {question_type}')
    return {'task_type': 'EQA', 'scene': str(scene_id), 'question': prompts[question_type],
            'private_answer': private_answer, 'answer_type': question_type}


def serialize_public_task(task):
    forbidden = {'private_answer', 'answer', 'current_state', 'hidden_events',
                 'mobility_regime', 'oracle_target_location'}
    return {key: value for key, value in task.items() if key not in forbidden}


def validate_public_task(task):
    forbidden = {'private_answer', 'answer', 'current_state', 'hidden_events',
                 'hidden_future_events', 'mobility_regime', 'oracle_target_location'}
    leaked = forbidden & set(task)
    if leaked:
        raise ValueError(f'public task exposes private fields: {sorted(leaked)}')
    if task.get('task_type') not in {'N1', 'N2', 'N3', 'N4', 'N5', 'VLN', 'EQA'}:
        raise ValueError('unknown task type')
    return True
