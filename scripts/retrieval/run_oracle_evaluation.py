"""Prepare or run the Oracle retrieval component; never certify observations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from citygml_sg.retrieval.evaluation.oracle_metrics import candidate_ids_hash
from citygml_sg.retrieval.oracle_runner import (
    content_hash, git_provenance, load_run_config, read_json, run_experiment,
)
from citygml_sg.retrieval.reference_graph import (
    build_room_candidates, fetch_reference_snapshot, reference_snapshot_stats,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("docs/examples/oracle_evaluation_manifest.template.json"))
    parser.add_argument("--method-config", type=Path,
                        default=Path("configs/joint_evidence_v1.yaml"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--graph-json", type=Path, help="Existing CityGML graph export or fixture")
    source.add_argument("--config", type=Path, help="Ignored local Neo4j connection config")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true",
                      help="Validate inputs/config and write not_run artifacts without a graph")
    mode.add_argument("--inventory-only", action="store_true",
                      help="Export canonical room IDs and graph inventory without queries")
    parser.add_argument("--output-dir", type=Path,
                        help="New or empty directory; existing results are never overwritten")
    parser.add_argument("--warmup", type=int, help="Warmup repeats per query/variant (default 3)")
    parser.add_argument("--repeat", type=int, help="Measured warm repeats (default 30)")
    return parser


def _path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _snapshot_loader(args: argparse.Namespace, timing: dict[str, float]):
    def load() -> dict[str, Any]:
        if args.graph_json:
            start = perf_counter()
            snapshot = read_json(_path(args.graph_json))
            timing["graph_file_load_ms"] = (perf_counter() - start) * 1000
            return snapshot

        # Lazy imports keep offline fixture/preparation independent of a running DB.
        from citygml_sg.config.settings import load_project_config
        from citygml_sg.storage.neo4j.client import Neo4jClient

        config = load_project_config(_path(args.config)).neo4j
        start = perf_counter()
        client = Neo4jClient(config.uri, config.username, config.password, config.database)
        timing["driver_creation_ms"] = (perf_counter() - start) * 1000
        try:
            with client.session() as session:
                start = perf_counter()
                session.run("RETURN 1 AS ok").consume()
                timing["connection_preflight_ms"] = (perf_counter() - start) * 1000
                start = perf_counter()
                # One read transaction for inventory, semantic nodes, and semantic edges.
                # A transaction is not an immutable snapshot under concurrent DB writes:
                # freeze the source during collection or use an exported JSON for comparisons.
                with session.begin_transaction() as transaction:
                    snapshot = fetch_reference_snapshot(transaction)
                timing["graph_query_fetch_ms"] = (perf_counter() - start) * 1000
                return snapshot
        finally:
            client.close()
    return load


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _inventory(output: Path, loader, timing: dict[str, float]) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "not_run", "mode": "inventory_only", **git_provenance(PROJECT_ROOT)}
    _write(output / "run_manifest.json", manifest)
    snapshot = loader()
    start = perf_counter()
    candidates = build_room_candidates(snapshot)
    timing["candidate_build_ms"] = (perf_counter() - start) * 1000
    ids = sorted(candidate["room_id"] for candidate in candidates)
    pool = {
        "candidate_count": len(ids), "candidate_ids": ids,
        "candidate_ids_hash": candidate_ids_hash(ids),
        "rooms": [{"room_id": c["room_id"], "room_name": c.get("room_name"),
                   "node_count": len(c["nodes"]), "edge_count": len(c["edges"])}
                  for c in candidates],
    }
    _write(output / "candidate_pool.json", pool)
    _write(output / "graph_inventory.json", reference_snapshot_stats(snapshot))
    manifest.update({"inventory_status": "completed", "graph_hash": content_hash(snapshot),
                     "candidate_ids_hash": pool["candidate_ids_hash"], "setup_timing": timing})
    _write(output / "run_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.prepare_only and not (args.graph_json or args.config):
        parser.error("Supply --graph-json or --config, or choose --prepare-only")
    if args.warmup is not None and args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.repeat is not None and args.repeat < 1:
        parser.error("--repeat must be >= 1")
    output = _path(args.output_dir) if args.output_dir else PROJECT_ROOT / (
        "data/output/oracle_runs/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    timing: dict[str, float] = {}
    # Refuse reused directories before entering the failure-artifact handler.
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output-dir must be a new or empty directory")
    try:
        if args.inventory_only:
            result = _inventory(output, _snapshot_loader(args, timing), timing)
        else:
            start = perf_counter()
            config = load_run_config(_path(args.method_config))
            manifest = read_json(_path(args.manifest))
            timing["config_manifest_load_ms"] = (perf_counter() - start) * 1000
            if args.warmup is not None:
                config["timing"]["warmup"] = args.warmup
            if args.repeat is not None:
                config["timing"]["repeat"] = args.repeat
            result = run_experiment(
                manifest, config, workspace_root=PROJECT_ROOT, output=output,
                snapshot_loader=None if args.prepare_only else _snapshot_loader(args, timing),
                prepare_only=args.prepare_only, setup_timing=timing,
            )
    except Exception as exc:
        # DB exceptions can contain URI/account details. Keep them out of artifacts.
        uses_neo4j = args.config is not None and not args.prepare_only
        message = str(exc) if isinstance(exc, ValueError) and not uses_neo4j else type(exc).__name__
        output.mkdir(parents=True, exist_ok=True)
        previous = read_json(output / "run_manifest.json") if (output / "run_manifest.json").exists() else {}
        previous.update({"status": "incomplete", "error_type": type(exc).__name__,
                         "error": message, "setup_timing": timing})
        _write(output / "run_manifest.json", previous)
        print(f"Oracle retrieval component: incomplete ({message})", file=sys.stderr)
        print(f"Artifacts: {output}", file=sys.stderr)
        return 2
    print(f"Oracle retrieval component: {result['status']}")
    print(f"Artifacts: {output}")
    if args.inventory_only:
        print("Use candidate_pool.json canonical room_id values in a local manifest.")
    elif args.prepare_only:
        print("No retrieval, metrics, timing, or annotation verification was performed.")
    return 2 if result["status"] == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
