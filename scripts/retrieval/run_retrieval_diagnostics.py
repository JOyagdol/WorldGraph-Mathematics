"""Run bounded, GT-free diagnostics; never produce accuracy/Oracle results."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from citygml_sg.retrieval.diagnostic_runner import run_diagnostics
from citygml_sg.retrieval.oracle_runner import content_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--graph-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, required=True)
    parser.add_argument("--budget-seconds", type=float, required=True)
    parser.add_argument("--per-query-timeout-seconds", type=float, required=True)
    args = parser.parse_args()
    try:
        manifest_bytes = args.manifest.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        original_manifest_hash = content_hash(manifest)
        # Public manifests use paths relative to their own directory, not the shell cwd.
        for case in manifest.get("cases", []):
            path = Path(case["ovg_path"])
            if not path.is_absolute():
                case["ovg_path"] = str((args.manifest.resolve().parent / path).resolve())
        manifest["source_file_identity"] = {
            "file_sha256": sha256(manifest_bytes).hexdigest(),
            "content_sha256_before_path_resolution": original_manifest_hash,
            "path_resolution": "relative_to_manifest_directory",
        }
        result = run_diagnostics(
            manifest, yaml.safe_load(args.config.read_text(encoding="utf-8-sig")),
            workspace_root=ROOT, output=args.output_dir,
            snapshot=json.loads(args.graph_json.read_text(encoding="utf-8-sig")),
            max_queries=args.max_queries, budget_seconds=args.budget_seconds,
            per_query_timeout_seconds=args.per_query_timeout_seconds,
        )
        print(json.dumps({key: result.get(key) for key in (
            "status", "completed_attempt_count", "expected_attempt_count", "failures")},
            ensure_ascii=False))
        return 0 if result["status"] == "diagnostic_completed" else 2
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Diagnostic setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
