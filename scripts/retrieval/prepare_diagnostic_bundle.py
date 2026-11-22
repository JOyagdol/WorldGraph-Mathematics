"""Copy previously frozen observations and a reference into a GT-free diagnostic bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from citygml_sg.retrieval.experiment_bundle import prepare_diagnostic_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-json", type=Path, required=True)
    parser.add_argument("--observation-freeze", type=Path, required=True,
                        help="JSON with observations entries {query_id, sha256}, frozen before GT access")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        frozen = json.loads(args.observation_freeze.read_text(encoding="utf-8-sig"))
        entries = frozen["observations"]
        expected = {entry["query_id"]: entry["sha256"] for entry in entries}
        if len(expected) != len(entries):
            raise ValueError("Duplicate query IDs in observation freeze")
        result = prepare_diagnostic_bundle(args.manifest, args.graph_json, args.output_dir,
                                           expected_hashes=expected)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Bundle preparation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
