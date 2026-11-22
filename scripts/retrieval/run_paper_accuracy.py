"""Run source-backed accuracy with the original map adapter and exact solver.

The default uses the original semantic map; an explicit HOSTED_BY support ledger
can select a separately named relation-only intervention. Geometric membership
and optimized solver dispatch are excluded. Each query/variant is inferred once;
no latency is collected or reported. Withdrawn fast-solver runs require their
historical source archive and cannot execute through the current matcher.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citygml_sg.ovg.validation.oracle_observation import normalize_observation, select_anchors
from citygml_sg.retrieval.evaluation.oracle_metrics import (
    aggregate_evaluations, candidate_ids_hash, evaluate_retrieval,
)
from citygml_sg.retrieval.graph_matching.joint_matcher import rank_rooms
from citygml_sg.retrieval.oracle_runner import (
    DEFAULT_PROVENANCE, PROVENANCE_FIELDS, content_hash, git_provenance,
    load_run_config, validate_manifest,
)
from citygml_sg.retrieval.reference_graph import build_room_candidates
from citygml_sg.retrieval.space_groups import group_room_candidates

ENTRYPOINT_VERSION = "paper_accuracy_exact_reference_v1"
REFERENCE_POLICY = "semantic_room_adapter_with_explicit_space_union"


def _path(value: Path | str) -> Path:
    value = Path(value)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _json_input(path: Path) -> tuple[Any, str]:
    payload = path.read_bytes()
    return json.loads(payload.decode("utf-8-sig")), sha256(payload).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def build_parser(*, table8=False) -> argparse.ArgumentParser:
    description = ("Run the adopted Table 8 Full, objects-only and relations-only methods "
                   "on the source-supported reference with the original exact solver.")
    parser = argparse.ArgumentParser(description=description if table8 else __doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--graph-json", type=Path, required=True,
                        help="Frozen semantic snapshot; the original Room adapter is always used")
    parser.add_argument("--space-groups", type=Path,
                        help="Optional JSON list of explicit space unions, declared before evaluation")
    parser.add_argument("--hosted-support-ledger", type=Path, required=table8,
                        help="Source-derived room support for existing HOSTED_BY edges")
    parser.add_argument("--method-config", type=Path,
                        default=Path("configs/paper_table8_opening.yaml" if table8 else
                                     "configs/paper_accuracy_reference.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty directory; previous artifacts are never overwritten")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Validate inputs and freeze identities without inference")
    return parser


def main(argv: list[str] | None = None, *, table8=False) -> int:
    parser = build_parser(table8=table8)
    args = parser.parse_args(argv)
    output = _path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output-dir must be a new or empty directory")
    config_path = _path(args.method_config)
    try:
        config = load_run_config(config_path)
    except ValueError as exc:
        parser.error(str(exc))
    if config["method"]["solver_mode"] != "exact_reference":
        parser.error("Paper accuracy requires solver_mode exact_reference; optimized runs are separate")
    expected_stage = "paper_table8_shared_wall_or_single_opening"
    if (config["protocol"]["experiment_stage"] == expected_stage) != table8:
        parser.error("Use run_paper_table8.py and its Table 8 config together")
    if config["timing"]["warmup"] != 0 or config["timing"]["repeat"] != 1:
        parser.error("Paper accuracy performs one inference: timing warmup=0 and repeat=1 are required")
    manifest, manifest_sha = _json_input(_path(args.manifest))
    validate_manifest(manifest)
    snapshot, snapshot_sha = _json_input(_path(args.graph_json))
    groups, groups_sha = [], None
    if args.space_groups:
        groups, groups_sha = _json_input(_path(args.space_groups))
        if isinstance(groups, dict) and set(groups) == {"groups"}:
            groups = groups["groups"]
        if not isinstance(groups, list) or not groups:
            parser.error("--space-groups requires a nonempty list of groups or an object containing that list under 'groups'")
    # The adapter is fixed here; neither a config nor a snapshot flag selects
    # geometric membership. Grouping never reads query evidence or GT.
    candidates = group_room_candidates(build_room_candidates(snapshot), groups)
    baseline_candidate_hash = content_hash(candidates)
    ledger_sha = ledger_content_sha = None
    reference_policy = REFERENCE_POLICY
    if args.hosted_support_ledger:
        from citygml_sg.retrieval.hosted_reference import POLICY, scope_hosted_by
        ledger, ledger_sha = _json_input(_path(args.hosted_support_ledger))
        candidates = scope_hosted_by(candidates, ledger, snapshot)
        ledger_content_sha = content_hash(ledger)
        reference_policy = POLICY
    method_version = config["method"]["version"]
    inference = rank_rooms
    if table8:
        from functools import partial
        from citygml_sg.retrieval.opening_pattern_ranker import (
            METHOD_VERSION, VARIANTS, prepare_opening_patterns, rank_opening_evidence,
        )
        if any(variant not in VARIANTS for variant in config["variants"]):
            parser.error("Table 8 supports full, objects_only and relations_only")
        inference = partial(rank_opening_evidence,
                            reference_patterns=prepare_opening_patterns(candidates))
        method_version = METHOD_VERSION
    pool = sorted(candidate["room_id"] for candidate in candidates)
    if not pool:
        raise ValueError("The snapshot must contain Room candidates")
    if "candidate_ids" in manifest and sorted(manifest["candidate_ids"]) != pool:
        raise ValueError("Manifest candidate_ids differ from the full declared space pool")
    prepared = []
    for case in manifest["cases"]:
        if case["target_room_id"] not in pool:
            raise ValueError(f"GT for {case['query_id']} is absent from the declared space pool")
        raw, observation_sha = _json_input(_path(case["ovg_path"]))
        normalized = normalize_observation(raw, config["method"])
        prepared.append((case, select_anchors(normalized, config["method"]["anchor_threshold"]), observation_sha))
    identity = {
        "entrypoint_version": "paper_table8_opening_v1" if table8 else ENTRYPOINT_VERSION,
        "method_version": method_version, "status": "not_run",
        "reference_policy": reference_policy, "solver_mode": "exact_reference",
        "baseline_candidate_content_sha256": baseline_candidate_hash,
        "hosted_support_ledger_file_sha256": ledger_sha,
        "hosted_support_ledger_content_sha256": ledger_content_sha,
        "latency_status": "not_collected", "inferences_per_query_variant": 1,
        "query_count": len(prepared), "variants": config["variants"],
        "candidate_ids": pool, "candidate_ids_hash": candidate_ids_hash(pool),
        "candidate_content_sha256": content_hash(candidates),
        "manifest_file_sha256": manifest_sha, "graph_snapshot_file_sha256": snapshot_sha,
        "graph_snapshot_sha256": content_hash(snapshot), "space_groups_file_sha256": groups_sha,
        "config_file_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "config_sha256": content_hash(config), "git": git_provenance(PROJECT_ROOT),
        "observations": [{"query_id": c["query_id"], "sha256": h} for c, _, h in prepared],
    }
    _write(output / "run_manifest.json", identity)
    _write(output / "resolved_config.json", config)
    _write(output / "candidate_pool.json", {"candidates": candidates, "groups": groups})
    if args.prepare_only:
        print("Paper accuracy: prepared; no inference or latency measurement")
        return 0
    records = []
    for case, anchors, observation_sha in prepared:
        for variant in config["variants"]:
            try:
                report = inference(anchors, candidates, config=config["method"], variant=variant)
            except Exception as exc:
                # Preserve the query in strict aggregates instead of dropping it.
                report = {"status": "execution_failed", "ranking": [], "full_ranking": False,
                          "candidate_ids": pool, "candidate_count": len(pool),
                          "candidate_ids_hash": identity["candidate_ids_hash"],
                          "error_type": type(exc).__name__}
            report.pop("timings", None)
            report.update({**DEFAULT_PROVENANCE,
                           **{k: config["protocol"][k] for k in PROVENANCE_FIELDS if k in config["protocol"]},
                           **{k: manifest[k] for k in PROVENANCE_FIELDS if k in manifest},
                           **{k: case[k] for k in PROVENANCE_FIELDS if k in case},
                           "query_id": case["query_id"], "variant_id": variant,
                           "run_id": output.name, "graph_snapshot_id": manifest["graph_snapshot_id"],
                           "graph_snapshot_sha256": identity["graph_snapshot_sha256"],
                           "observation_sha256": observation_sha, "config_sha256": identity["config_sha256"],
                           "method_version": method_version, "code_provenance": identity["git"],
                           "latency_status": "not_collected", "reference_policy": reference_policy})
            # Evaluation target is attached only after inference returns.
            evaluation = evaluate_retrieval(report, case["target_room_id"])
            records.append(evaluation)
            report["evaluation"] = evaluation
            _write(output / "per_query" / case["query_id"] / f"{variant}.json", report)
    metrics = aggregate_evaluations(records)
    _write(output / "evaluated_queries.json", {"records": records})
    _write(output / "accuracy_metrics.json", metrics)
    identity["status"] = "incomplete" if any(r["evaluation_status"] == "incomplete" for r in records) else "completed"
    identity["case_variant_count"] = len(records)
    _write(output / "run_manifest.json", identity)
    print(f"Paper accuracy: {identity['status']}; {len(records)} outcomes retained; latency not collected")
    return 2 if identity["status"] == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
