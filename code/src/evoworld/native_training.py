"""Split-bounded evaluator supervision, kept outside all Agent model inputs."""
from .native_generation import state_at


def supervised_pairs(row, truth, *, split, horizons):
    if split not in {'train','val'}:
        raise ValueError('supervision is restricted to train/val; test is never used')
    lo,hi = (0.,80*86400.) if split=='train' else (80*86400.,85*86400.)
    query=float(row['query_time'])
    if not lo<=query<hi or not horizons or any(h<=0 for h in horizons):
        raise ValueError('invalid split query or transition horizons')
    world=truth['world']
    anchors=[(query,float(h)) for h in horizons]
    next_event=next((e for e in world['events'] if query<e['timestamp']<hi),None)
    if next_event is not None:
        anchors.extend((max(query,float(next_event['timestamp'])-h/2),float(h)) for h in horizons)
    pairs=[]
    for anchor,horizon in sorted(set(anchors)):
        if lo<=anchor<anchor+horizon<hi:
            pairs.append({'anchor':anchor,'horizon':horizon,
                'source':state_at(world,anchor),'destination':state_at(world,anchor+horizon)})
    return pairs


def collate_queries(records):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    variable={'candidate_state_ids','candidate_mask','location_region_category',
              'location_receptacle_category','location_center_xyz','location_is_unknown'}
    result={}
    for key in records[0]:
        values=[r[key] for r in records]
        result[key]=(pad_sequence(values,batch_first=True,padding_value=0)
                     if key in variable else torch.stack(values))
    return result
