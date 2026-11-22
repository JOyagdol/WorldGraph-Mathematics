"""Prepare input components from existing local JSON files; never run retrieval."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citygml_sg.retrieval.oracle_runner import PROVENANCE_FIELDS, git_provenance
from citygml_sg.retrieval.reference_preparation import SAFE_ID, prepare_reference_inputs


def _read(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value, sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="cases with unique query_id and ovg_path; paths relative to manifest")
    parser.add_argument("--graph-json", type=Path, required=True,
                        help="Existing semantic snapshot; unsupported/geometry nodes are rejected")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty directory; no existing artifact is overwritten")
    parser.add_argument("--anchor-threshold", type=float, required=True,
                        help="Explicit inclusive node and edge threshold in [0,1]")
    parser.add_argument("--hop-radius", type=int, default=2)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    try:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("output directory must be new or empty")
        manifest, manifest_hash = _read(args.manifest)
        snapshot, snapshot_hash = _read(args.graph_json)
        cases = manifest.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("manifest requires a nonempty cases list")
        observations, sources = {}, {}
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("manifest cases must be objects")
            query_id, name = case.get("query_id"), case.get("ovg_path")
            if not isinstance(query_id, str) or not SAFE_ID.fullmatch(query_id):
                raise ValueError("query_id must be a safe, nonempty identifier")
            if query_id in observations:
                raise ValueError(f"duplicate query_id: {query_id}")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{query_id} requires ovg_path")
            path = Path(name)
            if not path.is_absolute():
                path = args.manifest.parent / path
            raw, file_hash = _read(path)
            observations[query_id] = raw
            # Declarations are kept separately and are never promoted to verification.
            by_source = {name: {key: value for key, value in source.items()
                               if key in PROVENANCE_FIELDS}
                         for name, source in (("manifest", manifest), ("observation", raw), ("case", case))}
            declarations = {key: value for source in by_source.values() for key, value in source.items()}
            conflicts = sorted(key for key in declarations if any(
                source[key] != declarations[key] for source in by_source.values() if key in source))
            sources[query_id] = {"observation_file_sha256": file_hash,
                                 "declared_annotation_metadata": declarations,
                                 "annotation_declarations_by_source": by_source,
                                 "annotation_declaration_precedence": ["manifest", "observation", "case"],
                                 "conflicting_annotation_fields": conflicts,
                                 "annotation_verification_performed": False}
        prepared = prepare_reference_inputs(snapshot, observations,
            anchor_threshold=args.anchor_threshold, hop_radius=args.hop_radius)
        prepared.update({
            "manifest_file_sha256": manifest_hash,
            "graph_snapshot_file_sha256": snapshot_hash,
            "source_metadata": sources, "code_provenance": git_provenance(PROJECT_ROOT),
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        # Validate every input and serialize before creating any output artifact.
        serialized = json.dumps(prepared, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        output.mkdir(parents=True, exist_ok=True)
        with (output / "preparation.json").open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        print("Input preparation completed; retrieval status: not_run")
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"Input preparation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
