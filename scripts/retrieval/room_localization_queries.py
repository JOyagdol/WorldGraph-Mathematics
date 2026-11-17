"""Run room-localization Cypher scenarios against the Neo4j world graph."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citygml_sg.config.settings import load_project_config
from citygml_sg.retrieval.query_generator import SCENARIOS
from citygml_sg.retrieval.scoring import (
    DEFAULT_ROOM_RETRIEVAL_PARAMS,
    build_room_retrieval_params,
)
from citygml_sg.retrieval.reporting import to_json_safe as _json_safe
from citygml_sg.storage.neo4j.client import Neo4jClient
from citygml_sg.utils.io import ensure_dir


DEFAULT_PARAMS = DEFAULT_ROOM_RETRIEVAL_PARAMS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run room-localization Cypher scenario tests")
    parser.add_argument(
        "--method", choices=["legacy_heuristic", "joint_evidence_v1_proposed"],
        default="legacy_heuristic", help="Retrieval method; legacy behavior remains the default",
    )
    parser.add_argument(
        "--method-config", default="configs/joint_evidence_v1.yaml",
        help="Method-only YAML for joint evidence (separate from Neo4j credentials)",
    )
    parser.add_argument(
        "--variant", default="full",
        choices=["full", "objects_only", "relations_only", "uniform_category", "boolean_metadata"],
        help="Joint-evidence intervention; legacy scenarios are unchanged",
    )
    parser.add_argument(
        "--reference-graph", default=None,
        help="Joint method: local graph snapshot JSON; otherwise prefetch the complete Neo4j Room pool",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Project config path containing Neo4j connection info",
    )
    parser.add_argument(
        "--output",
        default="data/output/room_localization_query_report.json",
        help="Room-localization scenario report JSON path",
    )
    parser.add_argument(
        "--scenario",
        choices=["all", *SCENARIOS.keys()],
        default="all",
        help="Scenario to run",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Legacy: max rows per scenario. Joint: console preview only; JSON keeps every candidate",
    )
    parser.add_argument(
        "--furniture-keywords",
        nargs="*",
        default=None,
        help="Furniture category/name keywords observed from an image view graph",
    )
    parser.add_argument(
        "--source-keywords",
        nargs="*",
        default=None,
        help="Source object keywords for furniture-pair scenario",
    )
    parser.add_argument(
        "--target-keywords",
        nargs="*",
        default=None,
        help="Target object keywords for furniture-pair scenario",
    )
    parser.add_argument(
        "--view-graph",
        default=None,
        help="Observed view graph JSON (required for joint). Legacy uses weight*confidence*visibility; joint uses confidence*visibility.",
    )
    return parser


def _build_params(args: argparse.Namespace) -> dict[str, object]:
    return build_room_retrieval_params(
        view_graph_path=args.view_graph,
        limit=args.limit,
        furniture_keywords=args.furniture_keywords,
        source_keywords=args.source_keywords,
        target_keywords=args.target_keywords,
    )


def _selected_scenarios(name: str) -> list[tuple[str, dict[str, object]]]:
    if name == "all":
        return list(SCENARIOS.items())
    return [(name, SCENARIOS[name])]


def _run_joint(args: argparse.Namespace) -> int:
    """Prepare a complete reference cache and write full inference-only evidence."""
    from citygml_sg.ovg.validation.oracle_observation import normalize_observation
    from citygml_sg.retrieval.oracle_runner import (
        content_hash, load_run_config, read_json, retrieve_normalized,
    )
    from citygml_sg.retrieval.reference_graph import (
        build_room_candidates, fetch_reference_snapshot, reference_snapshot_stats,
    )

    if not args.view_graph:
        raise ValueError("--method joint_evidence_v1_proposed requires --view-graph")
    if args.limit is not None and args.limit < 1:
        raise ValueError("Joint console --limit must be >= 1")
    if args.scenario != "all" or any(value is not None for value in
                                    (args.furniture_keywords, args.source_keywords, args.target_keywords)):
        raise ValueError("--scenario and keyword overrides are legacy-only; the joint method uses the observation graph")
    run_start = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    run_config = load_run_config(Path(args.method_config))
    schema_start = perf_counter()
    raw_observation = read_json(Path(args.view_graph))
    schema_load_ms = (perf_counter() - schema_start) * 1000
    normalization_start = perf_counter()
    normalized = normalize_observation(raw_observation, run_config["method"])
    normalization_ms = (perf_counter() - normalization_start) * 1000

    setup_timing = {"connection_preflight_ms": 0.0, "reference_fetch_ms": 0.0, "reference_load_ms": 0.0}
    if args.reference_graph:
        reference_start = perf_counter()
        snapshot = read_json(Path(args.reference_graph))
        setup_timing["reference_load_ms"] = (perf_counter() - reference_start) * 1000
        reference_source = "local_snapshot"
    else:
        project = load_project_config(args.config)
        neo4j = project.neo4j
        client = Neo4jClient(neo4j.uri, neo4j.username, neo4j.password, database=neo4j.database)
        try:
            with client.session() as session:
                preflight_start = perf_counter()
                session.run("RETURN 1 AS ready").consume()
                setup_timing["connection_preflight_ms"] = (perf_counter() - preflight_start) * 1000
                reference_start = perf_counter()
                snapshot = fetch_reference_snapshot(session)
                setup_timing["reference_fetch_ms"] = (perf_counter() - reference_start) * 1000
        finally:
            client.close()
        reference_source = "neo4j_semantic_prefetch"
    prepare_start = perf_counter()
    candidates = build_room_candidates(snapshot)
    snapshot_stats = reference_snapshot_stats(snapshot)
    setup_timing["reference_prepare_ms"] = (perf_counter() - prepare_start) * 1000
    report = retrieve_normalized(normalized, candidates, method=run_config["method"], variant=args.variant)
    report.update({
        "method": args.method, "report_kind": "single_observation_inference",
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
        "observation_sha256": content_hash(raw_observation),
        "reference_snapshot_sha256": content_hash(snapshot),
        "reference_source": reference_source, "reference_stats": snapshot_stats,
        "reference_cache": "preload", "setup_timing": setup_timing,
        "display_limit": args.limit if args.limit is not None else 10,
        "candidate_filter_policy": "none; all reference Rooms retained",
        "legacy_constraint_policy": "min_object_matches, candidate_limit, and other legacy room/matching constraints do not filter the joint candidate pool",
        "experiment_stage": "oracle_preparation", "input_condition": "researcher_curated",
        "oracle_validation_status": "pending", "annotation_protocol_version": None,
        "evaluation_status": "not_evaluated_no_ground_truth",
    })
    report["timings"].update({"schema_load_ms": schema_load_ms,
                               "observation_normalization_ms": normalization_ms,
                               "total_before_write_ms": (perf_counter() - run_start) * 1000})
    output = Path(args.output)
    ensure_dir(output.parent)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Joint-evidence full ranking written: {output}")
    print(f"status={report['status']} variant={args.variant} candidates={report['candidate_count']} "
          f"online_retrieval_ms={report['timings']['online_retrieval_ms']:.3f}")
    print("--limit controls this preview only; min_object_matches and legacy candidate limits are ignored.")
    print("Reference snapshot preparation is reported separately from online retrieval with the preloaded cache.")
    for row in report["ranking"][:report["display_limit"]]:
        print(f"  {row['rank']}. {row['room_id']} score={row['score']:.8f} "
              f"object={row['object_numerator']:.6f} relation={row['relation_numerator']:.6f} "
              f"optimal={row['solver']['optimality_proven']}")
    return 2 if report["status"] == "solver_incomplete" else 0


def main() -> int:
    run_t0 = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    args = build_parser().parse_args()
    if args.method == "joint_evidence_v1_proposed":
        return _run_joint(args)
    params = _build_params(args)
    config = load_project_config(args.config)
    neo4j = config.neo4j
    client = Neo4jClient(neo4j.uri, neo4j.username, neo4j.password, database=neo4j.database)

    results: list[dict[str, object]] = []
    try:
        with client.session() as session:
            for scenario_id, scenario in _selected_scenarios(args.scenario):
                query = str(scenario["query"])
                query_started_at = datetime.now(timezone.utc).isoformat()
                query_t0 = perf_counter()
                records = session.run(query, **params).data()
                query_elapsed_ms = (perf_counter() - query_t0) * 1000.0
                query_finished_at = datetime.now(timezone.utc).isoformat()
                rows = [_json_safe(record) for record in records]
                top_room = rows[0] if rows else None
                results.append(
                    {
                        "id": scenario_id,
                        "description": scenario["description"],
                        "view_graph_schema": scenario["view_graph_schema"],
                        "query": query.strip(),
                        "params": params,
                        "row_count": len(rows),
                        "elapsed_ms": round(query_elapsed_ms, 3),
                        "query_elapsed_ms": round(query_elapsed_ms, 3),
                        "query_started_at": query_started_at,
                        "query_finished_at": query_finished_at,
                        "top_room": top_room,
                        "rows": rows,
                    }
                )
    finally:
        client.close()

    finished_at = datetime.now(timezone.utc).isoformat()
    query_elapsed_ms_total = sum(float(result.get("query_elapsed_ms", 0.0)) for result in results)
    total_elapsed_ms = (perf_counter() - run_t0) * 1000.0
    report = {
        "summary": {
            "started_at": started_at,
            "finished_at": finished_at,
            "total_elapsed_ms": round(total_elapsed_ms, 3),
            "query_elapsed_ms_total": round(query_elapsed_ms_total, 3),
            "config_path": str(args.config),
            "output_path": str(args.output),
            "neo4j_uri": neo4j.uri,
            "neo4j_database": neo4j.database,
            "scenario": args.scenario,
            "scenario_total": len(results),
            "limit": int(params["limit"]),
            "view_graph_path": params.get("view_graph_path"),
            "observed_object_count": len(params.get("observed_objects", [])),
            "observed_relation_count": len(params.get("observed_relations", [])),
        },
        "scenarios": results,
    }

    output = Path(args.output)
    ensure_dir(output.parent)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Room-localization query report written: {output}")
    print(
        "Timing: "
        f"total={round((perf_counter() - run_t0) * 1000.0, 3)}ms "
        f"query_total={round(query_elapsed_ms_total, 3)}ms "
        f"scenarios={len(results)}"
    )
    for result in results:
        print(
            "  "
            f"{result['id']}: query_elapsed_ms={result['query_elapsed_ms']} "
            f"rows={result['row_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
