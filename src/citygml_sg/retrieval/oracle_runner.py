"""Run the proposed retrieval component with explicit experiment provenance.

This is an Oracle preparation/retrieval building block. It neither certifies
annotations nor implements image perception or the entire Oracle protocol.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import platform
import random
import re
from statistics import mean, median, stdev
import subprocess
from time import perf_counter
from typing import Any, Callable

import yaml

from citygml_sg.ovg.validation.oracle_observation import (
    NORMALIZATION_VERSION as OBSERVATION_NORMALIZATION_VERSION,
    normalize_observation, select_anchors,
)
from citygml_sg.retrieval.evaluation.oracle_metrics import (
    MTR_POLICY, VARIANT_IDS, aggregate_evaluations, candidate_ids_hash, evaluate_retrieval,
)
from citygml_sg.retrieval.graph_matching.joint_matcher import rank_rooms
from citygml_sg.retrieval.reference_graph import (
    NORMALIZATION_VERSION as REFERENCE_NORMALIZATION_VERSION,
    build_room_candidates, reference_snapshot_stats,
)
from citygml_sg.retrieval.reporting.ablation_table import build_ablation_table
from citygml_sg.retrieval.scoring.joint_score import DEFAULT_CONFIG, effective_config

PROVENANCE_FIELDS = (
    "experiment_stage", "input_condition", "oracle_validation_status",
    "annotation_protocol_version", "validation_evidence", "view_id", "parent_query_id",
    "source_type", "perturbation", "parent_input_condition", "parent_oracle_validation_status",
)
DEFAULT_PROVENANCE = {
    "experiment_stage": "oracle_preparation", "input_condition": "researcher_curated",
    "oracle_validation_status": "pending", "annotation_protocol_version": None,
}
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def content_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def load_run_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    return _resolve_run_config(raw)


def _resolve_run_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Resolve all executed defaults, including method defaults, before hashing."""
    if not isinstance(raw, dict) or set(raw) - {"method", "variants", "timing", "protocol"}:
        raise ValueError("Method config accepts only method, variants, timing and protocol")
    result = deepcopy(raw)
    for key in ("method", "timing", "protocol"):
        if not isinstance(result.setdefault(key, {}), dict):
            raise ValueError(f"{key} config must be an object")
    variants = result.setdefault("variants", list(VARIANT_IDS))
    if (not isinstance(variants, list) or not variants
            or any(not isinstance(variant, str) for variant in variants)
            or len(set(variants)) != len(variants)):
        raise ValueError("variants must be a nonempty unique list")
    if any(variant not in VARIANT_IDS for variant in variants):
        raise ValueError("Unknown joint-evidence variant")
    timing = result["timing"]
    if set(timing) - {"warmup", "repeat", "seed", "reference_cache", "primary"}:
        raise ValueError("Unsupported timing configuration field")
    for key, default, minimum in (("warmup", 3, 0), ("repeat", 30, 1), ("seed", 0, 0)):
        value = timing.setdefault(key, default)
        if type(value) is not int or value < minimum:
            raise ValueError(f"timing.{key} must be an integer >= {minimum}")
    if timing.setdefault("reference_cache", "preload") != "preload":
        raise ValueError("This version supports explicit preload reference caching only")
    if timing.setdefault("primary", "warm_online_retrieval") != "warm_online_retrieval":
        raise ValueError("Primary latency must be warm_online_retrieval")
    protocol = result["protocol"]
    if set(protocol) - {*PROVENANCE_FIELDS, "task", "require_full_ranking", "mtr_policy"}:
        raise ValueError("Unsupported protocol configuration field")
    if protocol.setdefault("task", "closed_set_single_gt_room") != "closed_set_single_gt_room":
        raise ValueError("This evaluator implements closed_set_single_gt_room")
    if protocol.setdefault("require_full_ranking", True) is not True:
        raise ValueError("Full ranking is required")
    if protocol.setdefault("mtr_policy", MTR_POLICY) != MTR_POLICY:
        raise ValueError(f"The implemented MTR policy is {MTR_POLICY}")
    for key, value in DEFAULT_PROVENANCE.items():
        protocol.setdefault(key, value)
    # Validate even prepare-only configurations before reporting ready-to-run status.
    allowed_method_keys = set(DEFAULT_CONFIG) | {"anchor_threshold"}
    if set(result["method"]) - allowed_method_keys:
        raise ValueError("Unsupported method configuration field; connection settings are separate")
    result["method"] = {key: value for key, value in effective_config(result["method"]).items()
                        if key in allowed_method_keys}
    result["method"].setdefault("anchor_threshold", 0.0)
    select_anchors({"nodes": [], "edges": []}, result["method"]["anchor_threshold"])
    normalize_observation({"objects": [], "relations": []}, result["method"])
    content_hash(result)  # Reject non-finite config values; never serialize credentials.
    return result


def validate_manifest(manifest: dict[str, Any], *, prepare_only: bool = False) -> list[str]:
    """Validate benchmark identity before any query; return missing preparation fields."""
    if manifest.get("protocol", "closed_set_single_gt_room") != "closed_set_single_gt_room":
        raise ValueError("Manifest protocol must be closed_set_single_gt_room")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest requires a nonempty cases list")
    missing: list[str] = []
    for key in ("graph_snapshot_id", "candidate_pool_id"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            missing.append(key)
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each manifest case must be an object")
        query_id = case.get("query_id")
        if not isinstance(query_id, str) or not SAFE_ID.fullmatch(query_id):
            raise ValueError("query_id must be a safe, nonempty file identifier")
        if query_id in seen:
            raise ValueError(f"Duplicate query_id: {query_id}")
        seen.add(query_id)
        if not isinstance(case.get("ovg_path"), str) or not case["ovg_path"]:
            raise ValueError(f"{query_id} requires ovg_path")
        target = case.get("target_room_id")
        if not isinstance(target, str) or not target.strip() or target.strip() != target:
            missing.append(f"{query_id}.target_room_id")
        provenance = {**DEFAULT_PROVENANCE,
                      **{k: manifest[k] for k in PROVENANCE_FIELDS if k in manifest},
                      **{k: case[k] for k in PROVENANCE_FIELDS if k in case}}
        for key in ("experiment_stage", "input_condition", "oracle_validation_status"):
            if not isinstance(provenance[key], str) or not provenance[key]:
                raise ValueError(f"{query_id}.{key} must be explicit")
        status = provenance["oracle_validation_status"]
        if status not in {"pending", "oracle_verified", "rejected", "not_applicable"}:
            raise ValueError(f"Unsupported oracle_validation_status: {status}")
        if status == "oracle_verified":
            for key in ("annotation_protocol_version", "validation_evidence", "view_id"):
                if not provenance.get(key):
                    raise ValueError(
                        f"oracle_verified requires {key}; annotation is not auto-certified"
                    )
            if provenance.get("source_type") in {None, "to_verify"}:
                raise ValueError("oracle_verified requires an identified observation source")
            if provenance.get("input_condition") in {
                "automatic_pipeline", "complete_room_diagnostic", "gt_correspondence_diagnostic",
            }:
                raise ValueError(
                    "Diagnostic/automatic inputs cannot be labelled clean Oracle retrieval"
                )
        if provenance["input_condition"] == "controlled_perturbation":
            for key in ("parent_query_id", "parent_input_condition",
                        "parent_oracle_validation_status", "perturbation"):
                if not provenance.get(key):
                    raise ValueError(f"controlled_perturbation requires {key}")
            if status == "oracle_verified":
                raise ValueError("Perturbed observations must retain parent validation separately")
    if missing and not prepare_only:
        raise ValueError("Manifest is not ready; fill canonical inventory fields: "
                         + ", ".join(missing))
    return missing


def git_provenance(root: Path) -> dict[str, Any]:
    """Fingerprint relevant code, including untracked sources, without copying secrets."""
    try:
        def git(*args: str) -> bytes:
            return subprocess.check_output(["git", "-C", str(root), *args],
                                           stderr=subprocess.DEVNULL)
        commit = git("rev-parse", "HEAD").decode().strip()
        scopes = ("src", "scripts", "tests", "configs/joint_evidence_v1.yaml")
        diff = git("diff", "--binary", "HEAD", "--", *scopes)
        untracked = git("ls-files", "--others", "--exclude-standard", "--", *scopes)
        untracked = untracked.decode().splitlines()
        digest = sha256(diff)
        for name in sorted(untracked):
            digest.update(name.encode("utf-8"))
            digest.update((root / name).read_bytes())
        return {"commit": commit, "dirty": bool(git("status", "--porcelain")),
                "relevant_diff_sha256": digest.hexdigest(),
                "untracked_source_count": len(untracked)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "relevant_diff_sha256": None,
                "status": "git_unavailable"}


def retrieve_normalized(
    normalized: dict[str, Any], candidates: list[dict[str, Any]], *,
    method: dict[str, Any], variant: str, clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Time normalized in-memory observation through complete ranking/evidence.

    Reference snapshot preparation is explicitly offline for this cache policy.
    IDF computation and joint matching/scoring remain inside the online interval.
    """
    online_start = clock()
    anchor_start = clock()
    anchors = select_anchors(normalized, method.get("anchor_threshold", 0.0))
    anchor_ms = (clock() - anchor_start) * 1000
    report = rank_rooms(anchors, candidates, config=method, variant=variant)
    report["anchors"] = anchors
    report["variant_id"] = variant
    report["timing_scope"] = "normalized_observation_to_complete_ranking_and_evidence"
    report["reference_cache"] = "preload"
    report.setdefault("timings", {}).update({
        "anchor_build_ms": anchor_ms, "db_query_fetch_ms": 0.0,
    })
    report["timings"]["online_retrieval_ms"] = (clock() - online_start) * 1000
    return report


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _write_tables(output: Path, metrics: dict[str, Any]) -> dict[str, str]:
    _write_json(output / "metrics_by_variant.json", metrics)
    groups = (metrics.get("by_condition", {}) if metrics.get("status") == "separated_conditions"
              else {"": metrics})
    statuses = {}
    for key, group in groups.items():
        target = output if not key else output / "conditions" / content_hash(key)[:12]
        target.mkdir(parents=True, exist_ok=True)
        table = build_ablation_table(group)
        statuses[key or "primary"] = table["status"]
        (target / "ablation_table.csv").write_text(table["csv"], encoding="utf-8")
        (target / "ablation_table.md").write_text(table["markdown"], encoding="utf-8")
        if key:
            _write_json(target / "condition.json", group.get("provenance"))
    return statuses


def stable_inference_signature(report: dict[str, Any]) -> str:
    """Fingerprint inference/evidence while excluding elapsed measurements anywhere."""
    time_keys = {
        "timing", "timings", "timing_stats", "elapsed", "elapsed_seconds", "elapsed_s",
        "duration", "duration_seconds", "duration_s", "wall_time", "wall_time_seconds",
    }

    def strip_times(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip_times(child) for key, child in value.items()
                    if key not in time_keys and not key.endswith("_ms")}
        if isinstance(value, list):
            return [strip_times(child) for child in value]
        return value

    return content_hash(strip_times(report))


def _timing_stats(values: list[float]) -> dict[str, Any]:
    return {"repeat": len(values), "mean_ms": mean(values) if values else None,
            "std_ms": stdev(values) if len(values) > 1 else None,
            "median_ms": median(values) if values else None,
            "p95_ms": sorted(values)[ceil(len(values) * 0.95) - 1] if values else None,
            "p95_method": "nearest_rank", "std_definition": "sample_standard_deviation_n_minus_1",
            "variation_scope": "within_query_variant_warm_repeats", "raw_ms": values}


def run_experiment(
    manifest: dict[str, Any], config: dict[str, Any], *, workspace_root: Path,
    output: Path, snapshot: dict[str, Any] | None = None,
    snapshot_loader: Callable[[], dict[str, Any]] | None = None,
    prepare_only: bool = False, setup_timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute every manifest case/variant, or write a truthful not_run preparation."""
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory for each run")
    output.mkdir(parents=True, exist_ok=True)
    run_id = output.resolve().name
    if not run_id:
        raise ValueError("A run output must be a named directory, not a filesystem root")
    run_manifest = {
        "run_id": run_id, "status": "not_run",
        "git": git_provenance(workspace_root), "environment": {
            "python": platform.python_version(), "platform": platform.platform()},
        "component_scope": "Oracle matching/evaluation component; not full Oracle validation",
        "timing_scope": "normalized_observation_to_complete_ranking_and_evidence",
        "reference_cache_policy": "preload; rebuild per run when snapshot/config changes",
        "normalization_versions": {"observation": OBSERVATION_NORMALIZATION_VERSION,
                                   "reference": REFERENCE_NORMALIZATION_VERSION},
        "setup_timing": setup_timing if setup_timing is not None else {},
    }
    write_ms = 0.0
    stage = "config_validation"
    active_query: str | None = None
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    retrieval_diagnostics: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []

    def write_json(path: Path, value: Any) -> None:
        nonlocal write_ms
        started = perf_counter()
        _write_json(path, value)
        write_ms += (perf_counter() - started) * 1000

    def write_text(path: Path, value: str) -> None:
        nonlocal write_ms
        started = perf_counter()
        path.write_text(value, encoding="utf-8")
        write_ms += (perf_counter() - started) * 1000

    def write_tables(metrics: dict[str, Any]) -> None:
        nonlocal write_ms
        started = perf_counter()
        statuses = _write_tables(output, metrics)
        write_ms += (perf_counter() - started) * 1000
        run_manifest["ablation_table_status_by_condition"] = statuses
        run_manifest["ablation_table_status"] = (
            "completed" if statuses and all(s == "completed" for s in statuses.values())
            else "not_run" if statuses and all(s == "not_run" for s in statuses.values())
            else "incomplete"
        )

    def finish(status: str) -> dict[str, Any]:
        run_manifest.update({
            "status": status, "case_variant_count": len(records), "execution_order": schedule,
            "report_write_ms": write_ms,
            "report_write_scope": (
                "artifact serialization and writes, including per-query and timing exports; "
                "excludes this final manifest self-write; all outside online retrieval"
            ),
        })
        _write_json(output / "run_manifest.json", run_manifest)
        return run_manifest

    def safe_reason(exc: Exception, *, source_stage: str) -> str:
        # Loader errors can embed Bolt connection details; never copy their message.
        if isinstance(exc, ValueError) and source_stage != "snapshot_load":
            return str(exc)
        return type(exc).__name__

    try:
        config = _resolve_run_config(config)
        source_manifest_hash = content_hash(manifest)
        manifest = deepcopy(manifest)
        for key in PROVENANCE_FIELDS:
            if key in config["protocol"]:
                manifest.setdefault(key, config["protocol"][key])
        stage = "manifest_validation"
        missing = validate_manifest(manifest, prepare_only=prepare_only)
        provenance = {**DEFAULT_PROVENANCE,
                      **{k: manifest[k] for k in PROVENANCE_FIELDS if k in manifest}}
        run_manifest.update({
            "manifest": manifest, "manifest_sha256": content_hash(manifest),
            "source_manifest_sha256": source_manifest_hash, "config_sha256": content_hash(config),
            "method_version": config["method"]["version"], "missing_preparation_fields": missing,
            "query_count": len(manifest["cases"]), "variant_order_seed": config["timing"]["seed"],
            "requested_variants": config["variants"],
            "effective_methods_by_variant": {
                variant: effective_config(config["method"], variant)
                for variant in config["variants"]
            },
        })
        write_json(output / "run_manifest.json", run_manifest)
        write_json(output / "resolved_config.json", config)
        stage = "observation_preparation"
        prepared = []
        for case in manifest["cases"]:
            active_query = case["query_id"]
            path = workspace_root / case["ovg_path"]
            started = perf_counter()
            # Parse the exact bytes that are hashed, avoiding a second read race.
            payload = path.read_bytes()
            raw = json.loads(payload.decode("utf-8-sig"))
            file_sha = sha256(payload).hexdigest()
            schema_load_ms = (perf_counter() - started) * 1000
            started = perf_counter()
            normalized = normalize_observation(raw, config["method"])
            normalization_ms = (perf_counter() - started) * 1000
            prepared.append((case, normalized, file_sha, schema_load_ms, normalization_ms))
        active_query = None
        if prepare_only:
            write_tables({"status": "not_run", "by_variant": {}, "by_condition": {}, "paired": {}})
            write_json(output / "preparation.json", {
                "status": "not_run", "missing_fields": missing,
                "observations": [{"query_id": c["query_id"], "sha256": h, "schema_load_ms": t,
                                  "observation_normalization_ms": nt, "warnings": n["warnings"]}
                                 for c, n, h, t, nt in prepared],
                "requires": ["verified canonical GT/pool",
                             "annotation protocol and view provenance",
                             "graph snapshot and stored metadata",
                             "tests and experiment execution"],
            })
            return finish("not_run")
        reference_start = perf_counter()
        stage = "snapshot_load"
        if snapshot is None:
            if snapshot_loader is None:
                raise ValueError("Provide graph JSON or a reference snapshot loader")
            snapshot = snapshot_loader()
        stage = "reference_preparation"
        candidates = build_room_candidates(snapshot)
        pool = sorted(candidate["room_id"] for candidate in candidates)
        if not pool:
            raise ValueError("Reference snapshot contains no Room candidates")
        fingerprint = candidate_ids_hash(pool)
        inventory = reference_snapshot_stats(snapshot)
        run_manifest.update({"graph_snapshot_sha256": content_hash(snapshot),
                             "candidate_ids_hash": fingerprint,
                             "reference_prepare_ms": (perf_counter() - reference_start) * 1000})
        write_json(output / "graph_inventory.json", inventory)
        write_json(output / "candidate_pool.json", {
            "candidate_ids": pool, "candidate_count": len(pool),
            "candidate_ids_hash": fingerprint, "candidate_pool_id": manifest["candidate_pool_id"],
            "rooms": [{"room_id": candidate["room_id"],
                       "node_count": len(candidate["nodes"]),
                       "edge_count": len(candidate["edges"]),
                       "nodes": candidate["nodes"], "edges": candidate["edges"],
                       "membership": candidate["membership"]} for candidate in candidates],
        })
        stage = "candidate_pool_validation"
        for case, _, _, _, _ in prepared:
            if case["target_room_id"] not in pool:
                active_query = case["query_id"]
                raise ValueError(f"GT Room for {case['query_id']} is absent from candidate pool")
        if "candidate_ids" in manifest and sorted(manifest["candidate_ids"]) != pool:
            raise ValueError("Manifest candidate_ids differ from the full reference Room inventory")

        def failure_report(exc: Exception, failure_stage: str) -> dict[str, Any]:
            return {"status": "execution_failed", "ranking": [], "full_ranking": False,
                    "candidate_ids": pool, "candidate_count": len(pool),
                    "candidate_ids_hash": fingerprint, "failure_stage": failure_stage,
                    "failure_reason": safe_reason(exc, source_stage=failure_stage),
                    "error_type": type(exc).__name__}

        stage = "retrieval"
        rng = random.Random(config["timing"]["seed"])
        with (output / "timing_runs.jsonl").open("w", encoding="utf-8") as timing_file:
            for case, normalized, file_sha, schema_load_ms, normalization_ms in prepared:
                active_query = case["query_id"]
                order = list(config["variants"])
                rng.shuffle(order)
                schedule.append({"query_id": case["query_id"], "variants": order})
                for variant in order:
                    context = {**provenance,
                               **{k: case[k] for k in PROVENANCE_FIELDS if k in case},
                               "query_id": case["query_id"], "variant_id": variant,
                               "run_id": run_id,
                               "graph_snapshot_id": manifest["graph_snapshot_id"]}
                    durations: list[float] = []
                    report: dict[str, Any] | None = None
                    last_attempt: dict[str, Any] | None = None
                    expected_signature = None
                    phase = "first_query"
                    try:
                        phases = [("first_query", 1), ("warmup", config["timing"]["warmup"]),
                                  ("warm", config["timing"]["repeat"])]
                        for phase, repetitions in phases:
                            for repeat in range(repetitions):
                                result = retrieve_normalized(
                                    normalized, candidates, method=config["method"],
                                    variant=variant,
                                )
                                last_attempt = result
                                started = perf_counter()
                                timing_file.write(json.dumps({
                                    **context, "phase": phase, "repeat": repeat,
                                    "status": result["status"], **result["timings"],
                                }, ensure_ascii=False, allow_nan=False) + "\n")
                                write_ms += (perf_counter() - started) * 1000
                                signature = stable_inference_signature(result)
                                if (expected_signature is not None
                                        and signature != expected_signature):
                                    raise ValueError(
                                        "Repeated inference produced different ranking/evidence"
                                    )
                                expected_signature, report = signature, result
                                if phase == "warm":
                                    durations.append(result["timings"]["online_retrieval_ms"])
                                if result["status"] in {"solver_incomplete", "execution_failed"}:
                                    break
                            if (report and report["status"]
                                    in {"solver_incomplete", "execution_failed"}):
                                break
                    except Exception as exc:
                        prior_report = report
                        report = failure_report(exc, phase)
                        if prior_report is not None:
                            report["last_successful_repeat"] = prior_report
                        if last_attempt is not None and last_attempt is not prior_report:
                            report["last_attempt_result"] = last_attempt
                        started = perf_counter()
                        timing_file.write(json.dumps({
                            **context, "phase": phase, "status": "execution_failed",
                            "online_retrieval_ms": None, "error_type": type(exc).__name__,
                        }, ensure_ascii=False, allow_nan=False) + "\n")
                        write_ms += (perf_counter() - started) * 1000
                    assert report is not None
                    report.update(context)
                    report.update({"observation_sha256": file_sha, "schema_load_ms": schema_load_ms,
                                   "observation_normalization_ms": normalization_ms,
                                   "schema_load_scope": "file_read_json_parse_and_sha256",
                                   "config_sha256": run_manifest["config_sha256"],
                                   "graph_snapshot_sha256": run_manifest["graph_snapshot_sha256"]})
                    stats = _timing_stats(durations)
                    timing_complete = len(durations) == config["timing"]["repeat"]
                    stats.update({"expected_repeat": config["timing"]["repeat"],
                                  "status": "completed" if timing_complete else "incomplete"})
                    try:
                        evaluation = evaluate_retrieval(report, case["target_room_id"])
                    except ValueError as exc:
                        invalid_report = report
                        report = {**failure_report(exc, "metric_validation"), **context,
                                  "invalid_report": invalid_report,
                                  "observation_sha256": file_sha,
                                  "schema_load_ms": schema_load_ms,
                                  "observation_normalization_ms": normalization_ms,
                                  "schema_load_scope": "file_read_json_parse_and_sha256",
                                  "config_sha256": run_manifest["config_sha256"],
                                  "graph_snapshot_sha256": run_manifest["graph_snapshot_sha256"]}
                        evaluation = evaluate_retrieval(report, case["target_room_id"])
                    if report["status"] in {"solver_incomplete", "execution_failed"}:
                        stats["status"] = "incomplete"
                    report["timing_stats"] = evaluation["timing_stats"] = stats
                    report["evaluation"] = evaluation  # GT is attached only after inference.
                    records.append(evaluation)
                    if report["status"] != "completed":
                        failures.append({"query_id": case["query_id"], "variant_id": variant,
                                         "status": report["status"],
                                         "stage": report.get("failure_stage"),
                                         "reason": report.get("failure_reason")})
                    elif evaluation["target_rank"] > 1 or evaluation["top_score_tie"]:
                        def candidate_diagnostic(row: dict[str, Any]) -> dict[str, Any]:
                            return {**{key: row.get(key) for key in (
                                "room_id", "rank", "score", "object_numerator",
                                "relation_numerator", "denominator", "tie_size",
                            )}, "unmatched_node_count": len(row.get("unmatched_nodes", [])),
                                "unmatched_edge_count": len(row.get("unmatched_edges", []))}

                        target_row = next(row for row in report["ranking"]
                                          if row["room_id"] == case["target_room_id"])
                        retrieval_diagnostics.append({
                            "query_id": case["query_id"], "variant_id": variant,
                            "target_rank": evaluation["target_rank"],
                            "wrong_top1": evaluation["target_rank"] > 1,
                            "top_score_tie": evaluation["top_score_tie"],
                            "top_candidate": candidate_diagnostic(report["ranking"][0]),
                            "target_candidate": candidate_diagnostic(target_row),
                            "evidence_path": f"per_query/{case['query_id']}/{variant}.json",
                        })
                    write_json(output / "per_query" / case["query_id"] / f"{variant}.json", report)
            started = perf_counter()
            timing_file.flush()
            write_ms += (perf_counter() - started) * 1000
        active_query = None
        stage = "aggregation"
        metrics = aggregate_evaluations(records)
        write_json(output / "evaluated_queries.json", {"records": records})
        write_tables(metrics)
        write_json(output / "failures.json", failures)
        write_json(output / "failure_analysis.json", {
            "execution_failures_and_abstentions": failures,
            "retrieval_diagnostics": retrieval_diagnostics,
        })
        failure_text = "# Retrieval outcomes and diagnostic evidence\n\n"
        if not failures:
            failure_text += "No execution failures or retrieval abstentions.\n"
        else:
            failure_text += "\n".join(
                f"- {item['query_id']} / {item['variant_id']}: {item['status']} "
                f"({item['reason'] or 'see matching evidence'})" for item in failures
            ) + "\n"
        if retrieval_diagnostics:
            failure_text += "\nCompleted retrievals with wrong top-1 or top-score ties:\n\n"
            failure_text += "\n".join(
                f"- {item['query_id']} / {item['variant_id']}: target rank {item['target_rank']}; "
                f"top-score tie={item['top_score_tie']}. "
                f"[Matching evidence]({item['evidence_path']})."
                for item in retrieval_diagnostics
            ) + "\n\nScore components and unmatched counts are in failure_analysis.json.\n"
        write_text(output / "failure_analysis.md", failure_text)
        run_manifest["metrics_status"] = metrics["status"]
        complete = all(summary["status"] == "completed"
                       for group in metrics["by_condition"].values()
                       for summary in group["by_variant"].values())
        return finish(metrics["status"] if complete else "incomplete")
    except Exception as exc:
        failure = {"status": "execution_failed", "stage": stage, "query_id": active_query,
                   "reason": safe_reason(exc, source_stage=stage), "error_type": type(exc).__name__}
        failures.append(failure)
        write_json(output / "failures.json", failures)
        write_json(output / "failure_analysis.json", {
            "execution_failures_and_abstentions": failures,
            "retrieval_diagnostics": retrieval_diagnostics,
        })
        write_json(output / "evaluated_queries.json", {"records": records})
        write_tables({"status": "incomplete", "by_variant": {
            variant: {"status": "incomplete"} for variant in VARIANT_IDS
        }, "by_condition": {}, "paired": {}})
        write_text(output / "failure_analysis.md",
                   f"# Run incomplete\n\nStage: {stage}. Error: {failure['reason']}\n")
        run_manifest["failure"] = failure
        finish("incomplete")
        raise
