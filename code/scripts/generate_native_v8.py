#!/usr/bin/env python3
"""Generate the reproducible EVOWORLD-BENCH native dataset."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from evoworld.native_generation import generate
from evoworld.native_v8 import prototype_config, full_config
from evoworld.source_statistics import write_source_statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',type=Path,default=Path('data/native_full_v7_release/manifest.json'))
    parser.add_argument('--catalog-root',type=Path,default=Path('artifacts/catalog_calibrated_v3'))
    parser.add_argument('--sources-root',type=Path,default=Path('/home/gaomingjian/Desktop/VLA/data/task_datasets'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scale',choices=['prototype','full'],default='prototype')
    parser.add_argument('--dry-run',action='store_true',
                        help='validate and print the resolved configuration without materializing episodes')
    args=parser.parse_args()
    if args.output.exists() and not args.dry_run:
        raise FileExistsError(args.output)
    source=json.loads(args.source_manifest.read_text())
    if args.dry_run:
        statistics = args.output/'source_statistics.json'
    else:
        args.output.mkdir(parents=True)
        statistics=args.output/'source_statistics.json'
        write_source_statistics(args.sources_root,statistics)
    builder = full_config if args.scale == 'full' else prototype_config
    config=builder(source['generator_config'],catalog_root=args.catalog_root,
        statistics=statistics,output=args.output)
    if args.dry_run:
        print(json.dumps({'scale': args.scale, 'episodes': config['episodes'],
                          'scenes': len(config['scene_ids']), 'days': config['days'],
                          'split_counts': config.get('split_counts'),
                          'split_task_counts': config.get('split_task_counts'),
                          'strict_pairing': bool(config.get('strict_pairing')),
                          'pairing_group_size': config.get('pairing_group_size')}, indent=2))
        return
    print(json.dumps(generate(config),indent=2))


if __name__=='__main__': main()
