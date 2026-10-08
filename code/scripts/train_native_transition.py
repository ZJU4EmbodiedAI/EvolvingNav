#!/usr/bin/env python3
"""Train the EVOWORLD belief and transition heads with train-only labels.

Test records are never opened. Shared P4D semantic weights initialize the
existing Agent.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evoworld.agent_bridge import (load_agent_package, public_scene_map,
    pack_query_for_model, select_public_episodes)
from evoworld.io import read_jsonl
from evoworld.native_training import supervised_pairs, collate_queries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','catalog-root','agent-root','checkpoint','training-dataset','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--train-per-task',type=int,default=250)
    parser.add_argument('--val-per-task',type=int,default=100)
    parser.add_argument('--horizons-s',type=float,nargs='+',default=[2,10,30,60,120,300])
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    load_agent_package(args.agent_root)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from evolvingnav.policy import load_belief, model_input_batch
    from evolvingnav.transition_model import TransitionHead
    from train_transition import epoch
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    model,schema=load_belief(args.checkpoint,args.training_dataset)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    head=TransitionHead(payload['model_config']['hidden_dim']).to(device)
    catalogs={}; manifests={}; sets={}
    for split,quota in (('train',args.train_per_task),('val',args.val_per_task)):
        selected=select_public_episodes(read_jsonl(args.dataset/'episodes'/f'{split}.jsonl'),
            ['N1','N2','N3','N4'],quota)
        wanted=dict(selected); samples=[]; changes=0
        with (args.dataset/'private'/f'{split}.jsonl').open() as handle:
            for index,line in enumerate(handle):
                if index not in wanted: continue
                row=wanted[index];truth=json.loads(line)
                if row['id']!=truth['id']: raise ValueError('unaligned supervision streams')
                sid=row['scene']
                if sid not in catalogs:
                    catalogs[sid]=json.loads((args.catalog_root/sid/'catalog.json').read_text())
                target=next(o for o in catalogs[sid]['objects'] if o['object_id']==row['target']['object_id'])
                scene_map=public_scene_map(target)
                ids={s['state_id']:i for i,s in enumerate(scene_map['states'])}
                ids['unknown']=len(scene_map['states'])
                for pair in supervised_pairs(row,truth,split=split,horizons=args.horizons_s):
                    anchored={**row,'query_time':pair['anchor']}
                    arrays=pack_query_for_model(anchored,scene_map,ids,schema)
                    batch={k:v.squeeze(0) for k,v in model_input_batch(arrays,schema).items()}
                    candidates=batch['candidate_state_ids'].tolist()
                    source,destination=ids[pair['source']],ids[pair['destination']]
                    if source not in candidates: source=ids['unknown']
                    if destination not in candidates: destination=ids['unknown']
                    # Labels are separate tensors, never keys accepted by the backbone.
                    batch['target_candidate_index']=torch.tensor(candidates.index(source))
                    batch['transition_source_index']=torch.tensor(candidates.index(source))
                    batch['transition_destination_index']=torch.tensor(candidates.index(destination))
                    batch['transition_horizon_s']=torch.tensor(pair['horizon'],dtype=torch.float32)
                    samples.append(batch);changes+=source!=destination
        if not samples: raise ValueError(f'no supervised pairs in {split}')
        sets[split]=samples
        manifests[split]={'episode_ids':[r['id'] for _,r in selected],
            'source_indices':[i for i,_ in selected],'pairs':len(samples),'changed_pairs':changes}
    args.output.mkdir(parents=True)
    manifest={'status':'native_training','test_used':False,
        'dataset':str(args.dataset.resolve()),'seed':args.seed,'device':str(device),
        'initial_checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'splits':manifests,'horizons_s':args.horizons_s,
        'hyperparameters':{'epochs':args.epochs,'patience':args.patience,'batch_size':args.batch_size,
            'learning_rate':3e-4,'weight_decay':1e-2,'gradient_clip':1.},
        'limitations':['v7 observation reuse and unpaired-world construction remain disqualifying for formal release',
            'initialization is transferred P4D; historical identity annotations are supplied by the benchmark',
            'synthetic one-target schedules, not a full multi-resident lifecycle simulation']}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    loaders={s:DataLoader(v,batch_size=args.batch_size,shuffle=s=='train',collate_fn=collate_queries)
             for s,v in sets.items()}
    optimizer=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=3e-4,weight_decay=1e-2)
    best,stale,state=float('inf'),0,None
    with (args.output/'learning_curve.jsonl').open('w') as log:
        for number in range(1,args.epochs+1):
            train=epoch(model,head,loaders['train'],optimizer,device,1.)
            val=epoch(model,head,loaders['val'],None,device,1.)
            record={'epoch':number,'train_joint_nll':train,'val_joint_nll':val}
            log.write(json.dumps(record)+'\n');log.flush();print(json.dumps(record),flush=True)
            if val<best-1e-5:
                best,stale=val,0
                state=(copy.deepcopy(model.state_dict()),copy.deepcopy(head.state_dict()))
            else: stale+=1
            if stale>=args.patience: break
    if state is None: raise RuntimeError('no checkpoint produced')
    torch.save({'model':state[0],'transition_head':state[1],'model_config':payload['model_config'],
        'model_name':'p4d','horizons_s':args.horizons_s,'best_validation_loss':best,
        'provenance':'EVOWORLD native training; train/val only; no test access'},args.output/'best.pt')
    summary={'epochs_completed':number,'best_val_joint_nll':best,'test_used':False,
        'train_pairs':len(sets['train']),'val_pairs':len(sets['val']),'status':'native_training'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
