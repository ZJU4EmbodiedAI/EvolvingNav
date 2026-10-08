"""Execution-trace scoring for native N4 episodes."""
from __future__ import annotations


def evaluate_actions(executor, actions, *, dynamic_eligible=None, transition_invalidated=None):
    if not actions:
        raise ValueError('actions cannot be empty')
    observations = []
    transition_seen = False
    observations_before_transition = []
    observations_after_transition = []
    for action in actions:
        if not action or action[0] not in {'move', 'inspect', 'replan'}:
            raise ValueError(f'unknown action: {action[0] if action else None}')
        before = executor.evaluator_state()['applied_event_count']
        if action[0] == 'move':
            executor.move_to(action[1])
        elif action[0] == 'inspect':
            observations.append(executor.inspect(action[1] if len(action) > 1 else None))
        else:
            executor.replan(action[1])
        after_state = executor.evaluator_state()
        if after_state['applied_event_count'] > before:
            transition_seen = True
        if observations:
            (observations_after_transition if transition_seen else observations_before_transition).append(observations[-1])
    final = executor.evaluator_state()
    success = any(item['visible'] for item in observations)
    first_inspection_success = bool(observations[0]['visible']) if observations else False
    recovery_eligible = bool(transition_invalidated) and not first_inspection_success
    recovered = bool(recovery_eligible and any(item['visible'] for item in observations_after_transition))
    revisited = bool(recovery_eligible and observations_after_transition)
    start_view = executor.public_trace()[0]['from_view'] if executor.public_trace() and executor.public_trace()[0]['type'] == 'move' else executor.view_id
    reference = executor._path_distance(start_view, executor.view_id)
    return {
        'success': success,
        'scheduled_transition': transition_seen,
        'dynamic_denominator_member': bool(dynamic_eligible),
        'transition_invalidated': bool(transition_invalidated),
        'first_inspection_success': first_inspection_success,
        'recovery_eligible': recovery_eligible,
        'recovered': recovered,
        'revisited': revisited,
        'distance': final['distance_m'],
        'reference_distance': reference,
        'time': final['clock'],
        'inspection_count': final['inspections'],
        'replans': final['replans'],
        'excess_distance': final['distance_m'] - reference,
    }
