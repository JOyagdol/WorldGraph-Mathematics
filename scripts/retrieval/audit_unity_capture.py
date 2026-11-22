"""Audit a stopped Unity schema-4 session into a new ignored evaluation-only JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from citygml_sg.retrieval.capture_audit import audit_capture_session, write_capture_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output.resolve()
        if output.suffix.lower() != ".json":
            raise ValueError("Output must be a new JSON file")
        if output.is_relative_to(args.session_dir.resolve()):
            raise ValueError("Output must be outside the raw capture session")
        if not output.is_relative_to(ROOT):
            raise ValueError("Output must be inside this checkout at a Git-ignored path")
        ignored = subprocess.run(["git", "check-ignore", "-q", "--", str(output)],
                                 cwd=ROOT, capture_output=True, check=False)
        if ignored.returncode != 0:
            raise ValueError("Output must be Git-ignored, e.g. private/ or data/output/")
        if output.exists():
            raise ValueError("Output already exists; choose a new file")
        report = audit_capture_session(args.session_dir, args.catalog)
        write_capture_audit(report, output)
        print(json.dumps({"status": report["status"], "capture_count": report["capture_count"],
                          "independent_spatial_review": "required",
                          "independent_annotation_review": "required"}))
        return 0 if report["status"] == "label_contract_valid" else 1
    except (OSError, ValueError) as exc:
        print(f"Capture audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
