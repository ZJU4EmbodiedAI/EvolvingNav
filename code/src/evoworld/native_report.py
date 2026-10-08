"""Independent native rollout aggregation; absent denominators remain null."""
from .evaluation.predict import prediction_metrics


def summarize_scores(rows):
    def mean(values):
        return sum(values)/len(values) if values else None
    def rate(records,key): return mean([bool(r[key]) for r in records])
    dynamic=[r for r in rows if r['dynamic_eligible']]
    recovery=[r for r in rows if r['recovery_eligible']]
    online=[r for r in rows if r['online_recovery_eligible']]
    revisits=sum(r['revisit_count'] for r in dynamic)
    return {'episodes':len(rows),'successes':sum(bool(r['success']) for r in rows),
        'sr':rate(rows,'success'),'spl':mean([r['spl'] for r in rows]),
        'first_inspection_sr':rate(rows,'first_inspection_success'),
        'recovery_sr':rate(recovery,'recovered'),'recovery_denominator':len(recovery),
        'dynamic_sr':rate(dynamic,'success'),'dynamic_denominator':len(dynamic),
        'online_recovery_sr':rate(online,'online_recovered'),'online_recovery_denominator':len(online),
        'revisit_success':sum(r['revisit_success_count'] for r in dynamic)/revisits if revisits else None,
        'revisit_denominator':revisits,
        'excess_distance':mean([r['distance']-r['reference_distance'] for r in dynamic]),
        'distance_m':mean([r['distance'] for r in rows]),'time_s':mean([r['time'] for r in rows]),
        'inspection_count':mean([r['inspection_count'] for r in rows]),
        'applied_transitions':sum(r.get('applied_transitions',0) for r in rows),
        'false_positive_stops':sum(r.get('termination')=='verified' and not r['success'] for r in rows)}


def belief_metrics(rows):
    if not rows: return None
    result=prediction_metrics(rows)
    result['brier']=sum(sum((float(p)-float(str(s)==str(r['label'])))**2
        for s,p in r['probabilities'].items()) for r in rows)/len(rows)
    return result
