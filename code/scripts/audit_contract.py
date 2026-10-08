#!/usr/bin/env python3
"""Evidence-based contract audit, beyond the construction integrity checks."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evoworld.io import read_jsonl


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--catalog-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths, counts, silhouettes, scene_days, task_scenes = {}, Counter(), Counter(), set(), {}
    # key -> [member_count, first_public_signature_hash, mismatch]
    pair_rows = {}
    unpaired = Counter()
    split_names = [p.stem for p in sorted((args.dataset/'episodes').glob('*.jsonl'))]
    for split in split_names:
        frame_set=set()
        # Keep the full-scale audit bounded-memory.  The JSONL streams are
        # intentionally not materialized as a list (803k records is several
        # gigabytes once Python objects are allocated).
        for row in read_jsonl(args.dataset/'episodes'/f'{split}.jsonl'):
            counts[split]+=1; scene_days.add((row['scene'],int(row['query_time']//86400)))
            task_scenes.setdefault((split,row['task_type']),set()).add(row['scene'])
            for h in row['history']:
                frame_set.add(h['rgb'])
                if h.get('visibility_kind')=='isolated_silhouette':
                    silhouettes[split]+=1
            if row.get('pair_group_id'):
                key = (split, row['pair_group_id'])
                signature = {k: v for k, v in row.items() if k not in {'id', 'pair_group_id'}}
                signature_hash = hashlib.sha256(
                    json.dumps(signature, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                if key not in pair_rows:
                    pair_rows[key] = [0, signature_hash, False]
                meta = pair_rows[key]
                meta[2] = bool(meta[2] or meta[1] != signature_hash)
                meta[0] += 1
            else:
                unpaired[split] += 1
        paths[split]=frame_set
    shared={f'{a}_{b}':len(paths.get(a,set())&paths.get(b,set())) for a,b in [('train','val'),('train','test'),('val','test')]}
    unverifiable=[]
    for p in args.catalog_root.glob('*/catalog.json'):
        catalog=json.loads(p.read_text())
        for o in catalog['objects']:
            for placement in o['placements']:
                prefix=placement['state_id']+'/'
                if not any(f.get('visibility_fraction',0)>=.2 for k,f in o['frames'].items() if k.startswith(prefix)):
                    unverifiable.append([p.parent.name,o['object_id'],placement['state_id']])
    generator=Path(__file__).resolve().parents[1]/'src/evoworld/native_generation.py'
    text=generator.read_text()
    patrol_depends_on_truth=('for attempt in range(64)' in text and "any(item['visible']" in text)
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    pair_counts=Counter(meta[0] for meta in pair_rows.values())
    pair_mismatches=[]
    for (split,group), meta in pair_rows.items():
        if meta[0] != 4 or meta[2]:
            pair_mismatches.append({'split':split,'pair_group_id':group,'members':meta[0],
                                    'public_signatures':2 if meta[2] else 1})
    behavior=manifest.get('behavior_parameterization', {})
    source_consumed=bool(behavior.get('source_distributions_consumed'))
    strict_pairing=bool(manifest.get('generator_config', {}).get('strict_pairing'))
    report={'dataset':str(args.dataset.resolve()),'episodes':sum(counts.values()),'split_counts':dict(counts),
        'scene_days':len(scene_days),'unique_rgb_paths':{s:len(v) for s,v in paths.items()},
        'shared_rgb_paths_between_splits':shared,'public_private_silhouette_records':dict(silhouettes),
        'unverifiable_placements':unverifiable,'patrol_seed_retry_depends_on_private_sighting':patrol_depends_on_truth,
        'paired_group_size_counts':dict(pair_counts),'paired_public_mismatches':pair_mismatches,
        'unpaired_episodes':dict(unpaired), 'strict_pairing': strict_pairing,
        'source_statistics_parameterize_schedule':source_consumed,
        'source_evidence':'aggregate behavior model is consumed when manifest records source_distributions_consumed=true',
        'release_eligible':False,'gaps':[]}
    if any(shared.values()): report['gaps'].append('RGB-D frame identifiers repeat across chronological splits; quantify train/val/test observation dependence before formal scores')
    if sum(silhouettes.values()): report['gaps'].append('Public history visibility currently contains evaluator isolated-target fractions')
    if patrol_depends_on_truth: report['gaps'].append('Generator changes patrol seed after hidden-world sighting checks; implement a fixed world-blind patrol followed by admission only')
    if pair_mismatches or not pair_rows or (strict_pairing and sum(unpaired.values())):
        report['gaps'].append('Audit matched static/routine/random query groups and atomic split assignments')
    if not source_consumed:
        report['gaps'].append('Fit household schedule parameters from actual aggregate source statistics')
    report['gaps']+=['Train EVOWORLD belief/transition heads on train only, fit calibration on val only',
        'Complete fixed-manifest RGB-D Agent evaluation with coverage-defined inspections and actual scheduled N4 changes']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
