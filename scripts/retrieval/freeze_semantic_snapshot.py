"""Freeze semantic records from a full graph export without accessing a database."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from citygml_sg.retrieval.snapshot_freeze import freeze_semantic_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = freeze_semantic_snapshot(args.source, args.output_dir)
        print(json.dumps({"snapshot": result["snapshot"], "room_count": result["room_count"]}))
        return 0
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        print(f"Snapshot freeze failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
