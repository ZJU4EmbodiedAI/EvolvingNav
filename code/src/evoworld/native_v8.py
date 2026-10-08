"""Non-destructive v8 construction configurations.

The full configuration deliberately rounds split/task quotas to complete
four-regime counterfactual groups.  The paper does not prescribe exact task
quotas, while an incomplete group would make a paired mobility comparison
ambiguous.
"""
from copy import deepcopy


def _round_to_four(counts, total):
    """Return deterministic task quotas divisible by four and summing total."""
    if total % 4:
        raise ValueError('strict paired split total must be divisible by four')
    names = sorted(counts)
    result = {name: int(counts[name]) // 4 * 4 for name in names}
    remaining = total - sum(result.values())
    # Largest remainders receive the missing complete groups.  Stable name
    # order makes the manifest reproducible across Python versions.
    order = sorted(names, key=lambda name: (-int(counts[name]) % 4, name))
    for name in order[: remaining // 4]:
        result[name] += 4
    if sum(result.values()) != total or any(value % 4 for value in result.values()):
        raise AssertionError('failed to normalize paired task quotas')
    return result


def _strict_pair_quotas(config):
    """Normalize temporal split and task counts for atomic four-regime groups."""
    split_counts = {str(k): int(v) for k, v in config['split_counts'].items()}
    # Move only the two-episode remainders between val/test; total scale stays
    # unchanged and each chronological split remains non-empty.
    remainders = {split: value % 4 for split, value in split_counts.items()}
    if any(remainders.values()):
        if set(split_counts) != {'train', 'val', 'test'}:
            raise ValueError('strict pairing requires train/val/test split counts')
        total = sum(split_counts.values())
        base = {split: value // 4 * 4 for split, value in split_counts.items()}
        # Allocate the global remainder to train first, then val, then test;
        # this changes only the split boundary bookkeeping, not total episodes.
        left = total - sum(base.values())
        for split in ('train', 'val', 'test'):
            if left >= 4:
                base[split] += 4
                left -= 4
        if left:
            raise ValueError('strict pairing cannot normalize split counts')
        split_counts = base
    task_counts = config.get('split_task_counts')
    if task_counts is None:
        raise ValueError('strict pairing requires split_task_counts')
    normalized = {}
    for split, counts in task_counts.items():
        normalized[split] = _round_to_four(counts, split_counts[split])
    result = deepcopy(config)
    result['split_counts'] = split_counts
    result['split_task_counts'] = normalized
    result['strict_pairing'] = True
    result['pairing_group_size'] = 4
    result['pairing_quota_normalization'] = 'deterministic_round_to_four_preserving_total'
    result['episodes'] = sum(split_counts.values())
    return result


def prototype_config(source, *, catalog_root, statistics, output):
    config = deepcopy(source)
    for key in ('temporal_splits','split_counts','split_task_counts','task_counts'):
        config.pop(key, None)
    config.update(scene_ids=config['scene_ids'][:5],days=5,episodes=1000,
        catalog_root=str(catalog_root),source_statistics=str(statistics),output=str(output),
        patrol_mode=True,protocol_revision='balanced_v4',held_out_scene_ids=[],
        statistics={'move_rate':90},seed=20261007)
    return config


def full_config(source, *, catalog_root, statistics, output):
    config = deepcopy(source)
    config.update(catalog_root=str(catalog_root), source_statistics=str(statistics),
        output=str(output), patrol_mode=True, protocol_revision='balanced_v5',
        seed=20261008, strict_pairing=True)
    return _strict_pair_quotas(config)
