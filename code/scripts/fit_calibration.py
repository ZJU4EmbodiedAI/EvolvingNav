#!/usr/bin/env python3
"""Fit detection recall from held-out RGB-D validation observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolvingnav.calibration import DetectionCalibrator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.validation_jsonl.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    DetectionCalibrator.fit(rows).save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
