#!/usr/bin/env python3
"""Run the two independent construction audits for a native dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evoworld.audit import audit_release
from evoworld.stream_audit import audit_stream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--catalog-root")
    args = parser.parse_args()
    root = Path(args.root)
    temporal = audit_release(root, catalog_root=args.catalog_root)
    stream = audit_stream(root)
    (root / "independent_audit.json").write_text(json.dumps(temporal, indent=2) + "\n", encoding="utf-8")
    (root / "stream_integrity_audit.json").write_text(json.dumps(stream, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"temporal": temporal, "stream": stream}, indent=2))
    raise SystemExit(0 if temporal["pass"] and stream["pass"] else 1)


if __name__ == "__main__":
    main()
