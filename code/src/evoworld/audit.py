from __future__ import annotations
import json
from pathlib import Path
from .io import read_jsonl
from .stream_audit import _forbidden


def audit_release(root, catalog_root=None):
    """Audit temporal/privacy integrity, not protocol release readiness.

    Deliberately replay states independently of the generation implementation.
    Catalog checking additionally validates the exact historical frame and pose.
    Public/private files must have aligned order, as written by the generator.
    Records are streamed: private world timelines are not all held in memory.
    """
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    public = sorted((root/'episodes').glob('*.jsonl'))
    private = sorted((root/'private').glob('*.jsonl'))
    errors, private_ids, ids, catalogs = [], set(), set(), {}
    leaks = count = error_count = 0

    def error(ident, reason):
        nonlocal error_count
        error_count += 1
        if len(errors) < 100:
            errors.append(dict(id=ident, reason=reason))

    def forbidden(value):
        return _forbidden(value)

    def replay(world, timestamp):
        state = world['initial_state']
        for event in world['events']:
            if event['timestamp'] > timestamp:
                break
            state = event['next_state']
        return state

    private_rows = (truth for path in private for truth in read_jsonl(path))
    for path in public:
        for row in read_jsonl(path):
            count += 1
            ident = row['id']
            if ident in ids:
                error(ident, 'duplicate public id')
            ids.add(ident)
            if forbidden(row):
                leaks += 1
                error(ident, 'private field in public record')
            truth = next(private_rows, None)
            if truth is None:
                error(ident, 'missing private record')
                continue
            if truth['id'] in private_ids:
                error(truth['id'], 'duplicate private id')
            private_ids.add(truth['id'])
            if truth['id'] != ident:
                error(ident, 'public/private record order or id mismatch')
                continue
            try:
                world = truth['world']
                history_world = truth.get('history_world', world)
                state, previous_time = world['initial_state'], -1
                for event in world['events']:
                    if (event['previous_state'] != state or event['next_state'] == state
                            or event['timestamp'] <= previous_time):
                        error(ident, 'invalid causal event chain')
                        break
                    state, previous_time = event['next_state'], event['timestamp']
                if truth['current_state'] != replay(world, row['query_time']):
                    error(ident, 'query state mismatch')
                if truth['scene'] != row['scene'] or truth['target_object'] != row['target']['object_id']:
                    error(ident, 'public/private identity mismatch')
                target = None
                if catalog_root is not None:
                    sid = row['scene']
                    if sid not in catalogs:
                        catalog = json.loads((Path(catalog_root)/sid/'catalog.json').read_text())
                        catalogs[sid] = {o['object_id']: o for o in catalog['objects']}
                    target = catalogs[sid][row['target']['object_id']]
                if not row['history']:
                    error(ident, 'empty history')
                last_time = -1
                for history in row['history']:
                    timestamp = history['timestamp']
                    if not last_time < timestamp < row['query_time'] or timestamp < 0:
                        error(ident, 'history timestamp outside ordered past')
                    last_time = timestamp
                    observed = replay(history_world, timestamp)
                    visible = bool(history.get('visible', True))
                    if visible and history.get('location') != observed:
                        error(ident, 'history state mismatch')
                    if not visible and 'location' in history:
                        error(ident, 'negative observation exposes location')
                    if target is not None:
                        matching = [v for v in target['views']
                                    if history['camera_pose'] == dict(position=v['position'],
                                                                     rotation_wxyz=v['rotation_wxyz'])]
                        if not any(target['frames'][observed+'/'+v['view_id']]['rgb'] == history['rgb']
                                   and target['frames'][observed+'/'+v['view_id']]['depth'] == history['depth']
                                   and target['frames'][observed+'/'+v['view_id']]['visible'] == visible for v in matching):
                            error(ident, 'history frame/pose not matched to replayed catalog state')
                expected = ([e for e in world['events'] if e['timestamp'] > row['query_time']]
                            if truth['evolution_during_execution'] else [])
                if truth['hidden_future_events'] != expected:
                    error(ident, 'execution transition mismatch')
            except (KeyError, TypeError, ValueError, OSError) as exc:
                error(ident, 'missing or invalid audit evidence: '+str(exc))
    for truth in private_rows:
        error(truth['id'], 'extra private record')
        private_ids.add(truth['id'])
    if private_ids != ids:
        error(None, 'public/private id sets differ')
    if count != manifest.get('episodes') or not count:
        error(None, 'manifest episode count mismatch or empty dataset')
    return dict(manifest=manifest, episode_count=count, public_files=len(public),
                private_files=len(private), leak_count=leaks, error_count=error_count,
                errors=errors, **{'pass': error_count == 0}, release_eligible=False,
                scope='temporal and privacy integrity only; protocol and human review not certified',
                catalog_checked=catalog_root is not None)
