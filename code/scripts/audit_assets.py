#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evoworld.asset_audit import audit_assets

parser = argparse.ArgumentParser(); parser.add_argument("--hssd-root", required=True, type=Path); parser.add_argument("--sources-root", required=True, type=Path); parser.add_argument("--output", required=True, type=Path); a = parser.parse_args(); print(json.dumps(audit_assets(a.hssd_root, a.sources_root, a.output), indent=2))
