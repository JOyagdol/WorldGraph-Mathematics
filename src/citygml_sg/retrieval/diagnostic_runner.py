"""Bounded, observation-only diagnostics for the proposed joint matcher.

No GT is accepted and no accuracy metrics or Oracle certification are produced.
Each query/variant executes once in an owned child process. Its hard timeout
includes child startup and I/O; worker retrieval timings exclude those costs and
are diagnostic measurements, not the repeated warm-latency experiment.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

from citygml_sg.ovg.validation.oracle_observation import normalize_observation, _reject_map_identifiers
from citygml_sg.retrieval.evaluation.oracle_metrics import candidate_ids_hash
from citygml_sg.retrieval.oracle_runner import (
    PROVENANCE_FIELDS, SAFE_ID, _resolve_run_config, content_hash, git_provenance,
    retrieve_normalized,
)
from citygml_sg.retrieval.reference_graph import build_room_candidates, reference_snapshot_stats


DIAGNOSTIC_VERSION = "no_gt_joint_diagnostic_v1"
_GT_KEYS = {"target_room_id", "target_room_ids", "gt_room_id", "gt_room_ids", "ground_truth"}
_GOOD_STATUSES = {"completed", "abstained_no_anchor", "abstained_no_evidence"}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _reject_gt(value: Any) -> None:
    if isinstance(value, dict):
        if _GT_KEYS.intersection(value):
            raise ValueError("Diagnostic manifests must not contain GT/target Room fields")
        for child in value.values():
            _reject_gt(child)
    elif isinstance(value, list):
        for child in value:
            _reject_gt(child)


def _hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


def _positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{field} must be finite and positive")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return value


def _cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise ValueError("Diagnostic manifest must be an object")
    _reject_gt(manifest)
    for field in ("graph_snapshot_id", "candidate_pool_id"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"{field} is required")
    _hash(manifest.get("graph_snapshot_sha256"), "graph_snapshot_sha256")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Diagnostic manifest requires nonempty cases")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each diagnostic case must be an object")
        query = case.get("query_id")
        if not isinstance(query, str) or not SAFE_ID.fullmatch(query) or query in seen:
            raise ValueError("query_id must be a unique safe file identifier")
        seen.add(query)
        if not isinstance(case.get("ovg_path"), str) or not case["ovg_path"].strip():
            raise ValueError("Each diagnostic case requires ovg_path")
        _hash(case.get("observation_sha256"), "observation_sha256")
    return cases


def _run_worker(job_path: Path, result_path: Path, timeout_seconds: float) -> dict[str, Any]:
    """Run/terminate only the child created here; no other application is touched."""
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    command = [sys.executable, "-m", "citygml_sg.retrieval.diagnostic_runner",
               "--worker", str(job_path), str(result_path)]
    started = perf_counter()
    process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        try:
            process.wait(timeout=max(0.001, timeout_seconds - (perf_counter() - started)))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            return {"status": "execution_failed", "failure_reason": "worker_timeout",
                    "failure_stage": "diagnostic_worker", "error_type": "TimeoutExpired",
                    "worker_wall_ms": (perf_counter() - started) * 1000}
        if process.returncode != 0 or not result_path.is_file():
            return {"status": "execution_failed", "failure_reason": "worker_process_failed",
                    "failure_stage": "diagnostic_worker", "worker_exit_code": process.returncode,
                    "worker_wall_ms": (perf_counter() - started) * 1000}
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["worker_wall_ms"] = (perf_counter() - started) * 1000
        return result
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)


def run_diagnostics(
    manifest: dict[str, Any], config: dict[str, Any], *, workspace_root: Path,
    output: Path, snapshot: dict[str, Any], max_queries: int,
    budget_seconds: float, per_query_timeout_seconds: float,
) -> dict[str, Any]:
    """Freeze first N manifest cases and run one bounded attempt per variant.

    The overall budget begins at entry and prevents new work after expiry. Child
    work has a hard deadline; synchronous validation/reference preparation and
    durable final writes are not preempted. They are included in elapsed time.
    The source snapshot is supplied by the caller, so DB loading is outside this
    function. Neither candidate selection nor query selection uses GT/outcomes.
    """
    started = perf_counter()
    output, workspace_root = Path(output), Path(workspace_root)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "status": "diagnostic_incomplete", "diagnostic_version": DIAGNOSTIC_VERSION,
        "mode": "no_gt_diagnostic", "metrics_status": "not_applicable_no_gt",
        "annotation_certification": "not_performed", "run_id": output.name,
        "stage": "validation", "completed_attempt_count": 0,
        "results": [], "failures": [], "active_attempt": None,
        "timing_claim": "single_attempt_diagnostic_only_not_warm_benchmark",
        "budget_scope": "entry_to_completion; preparation/final_writes_not_preemptible",
        "worker_timeout_scope": "query_variant_process_startup_load_inference_and_result_write",
        "worker_cleanup_grace_seconds": 4,
    }

    def checkpoint() -> None:
        state["elapsed_seconds"] = perf_counter() - started
        _atomic_json(output / "run_manifest.json", state)

    checkpoint()
    try:
        if type(max_queries) is not int or max_queries < 1:
            raise ValueError("max_queries must be a positive integer")
        budget = _positive(budget_seconds, "budget_seconds")
        per_query = _positive(per_query_timeout_seconds, "per_query_timeout_seconds")
        _reject_gt(config)
        resolved = _resolve_run_config(config)
        cases = _cases(manifest)
        selected = deepcopy(cases[:max_queries])
        source_hash = content_hash(snapshot)
        if source_hash != manifest["graph_snapshot_sha256"]:
            raise ValueError("graph_snapshot_sha256 does not match supplied snapshot")
        executed = {
            "method": resolved["method"], "variants": resolved["variants"],
            "diagnostic": {"max_queries": max_queries, "budget_seconds": budget,
                           "per_query_timeout_seconds": per_query,
                           "attempts_per_query_variant": 1,
                           "selection_policy": "first_N_in_manifest_order_before_outcomes"},
        }
        provenance = {"experiment_stage": "diagnostic_pending_input",
                      "input_condition": "researcher_curated", "oracle_validation_status": "pending"}
        provenance.update({k: resolved["protocol"][k] for k in PROVENANCE_FIELDS
                           if k in resolved["protocol"]})
        provenance.update({k: manifest[k] for k in PROVENANCE_FIELDS if k in manifest})
        prepared = []
        for case in selected:
            declared = {**provenance, **{k: case[k] for k in PROVENANCE_FIELDS if k in case}}
            if declared.get("oracle_validation_status") not in {"pending", "not_applicable"}:
                raise ValueError("Diagnostic runner accepts pending or not_applicable declarations only")
            payload = (workspace_root / case["ovg_path"]).read_bytes()
            actual = sha256(payload).hexdigest()
            if actual != case["observation_sha256"]:
                raise ValueError(f"observation_sha256 mismatch: {case['query_id']}")
            raw = json.loads(payload.decode("utf-8-sig"))
            _reject_map_identifiers(raw, "observation")
            normalized = normalize_observation(raw, resolved["method"])
            prepared.append((case, normalized, declared))
        state.update({
            "source_manifest_sha256": content_hash(manifest),
            "source_file_identity": deepcopy(manifest.get("source_file_identity")),
            "source_config_sha256": content_hash(resolved), "config_sha256": content_hash(executed),
            "graph_snapshot_id": manifest["graph_snapshot_id"],
            "graph_snapshot_sha256": source_hash, "candidate_pool_id": manifest["candidate_pool_id"],
            "method_version": resolved["method"]["version"], "git": git_provenance(workspace_root),
            "source_query_count": len(cases), "selected_query_count": len(selected),
            "selected_query_ids": [c["query_id"] for c in selected],
            "selection_policy": executed["diagnostic"]["selection_policy"],
            "expected_attempt_count": len(selected) * len(resolved["variants"]),
            "execution_schedule": [{"query_id": c["query_id"], "variant_id": v}
                                   for c in selected for v in resolved["variants"]],
            "selected_observation_hashes_verified": True,
            "ignored_benchmark_timing_config": resolved["timing"],
            "limits": executed["diagnostic"], "stage": "reference_preparation",
        })
        _atomic_json(output / "resolved_config.json", executed)
        _atomic_json(output / "selected_manifest.json", {
            "source_manifest_sha256": state["source_manifest_sha256"],
            "source_file_identity": deepcopy(state["source_file_identity"]), "cases": selected,
            "selection_policy": state["selection_policy"],
        })
        checkpoint()
        if perf_counter() - started >= budget:
            raise TimeoutError("overall_budget_exhausted_before_reference_preparation")
        candidates = build_room_candidates(snapshot)
        pool = sorted(candidate["room_id"] for candidate in candidates)
        if not pool:
            raise ValueError("Reference snapshot contains no Room candidates")
        fingerprint = candidate_ids_hash(pool)
        if "candidate_ids_hash" in manifest and manifest["candidate_ids_hash"] != fingerprint:
            raise ValueError("candidate_ids_hash does not match the full reference pool")
        reference_path = (output / "reference_candidates.json").resolve()
        _atomic_json(reference_path, candidates)
        _atomic_json(output / "graph_inventory.json", reference_snapshot_stats(snapshot))
        _atomic_json(output / "candidate_pool.json", {
            "candidate_ids": pool, "candidate_count": len(pool), "candidate_ids_hash": fingerprint,
        })
        state.update(candidate_count=len(pool), candidate_ids_hash=fingerprint, stage="retrieval")
        checkpoint()
        for case, normalized, declared in prepared:
            for variant in resolved["variants"]:
                remaining = budget - (perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("overall_budget_exhausted")
                query = case["query_id"]
                state["active_attempt"] = {"query_id": query, "variant_id": variant}
                checkpoint()
                job_path = (output / "worker_jobs" / query / f"{variant}.json").resolve()
                result_path = (output / "worker_results" / query / f"{variant}.json").resolve()
                _atomic_json(job_path, {"normalized": normalized, "method": resolved["method"],
                                       "variant": variant, "reference_path": str(reference_path)})
                remaining = budget - (perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("overall_budget_exhausted")
                report = _run_worker(job_path, result_path, min(per_query, remaining))
                if report.get("status") != "execution_failed":
                    ranking = report.get("ranking", [])
                    ranked_ids = [row.get("room_id") for row in ranking
                                  if isinstance(row, dict)] if isinstance(ranking, list) else []
                    valid = (report.get("status") in _GOOD_STATUSES | {"solver_incomplete"}
                             and report.get("full_ranking") is True
                             and len(ranked_ids) == len(pool) and set(ranked_ids) == set(pool)
                             and report.get("candidate_ids_hash") == fingerprint
                             and report.get("candidate_count") == len(pool)
                             and report.get("candidate_ids") == pool)
                    if report.get("status") in _GOOD_STATUSES:
                        valid = valid and all(row.get("solver", {}).get("optimality_proven") is True
                                              for row in ranking)
                    if not valid:
                        report = {"status": "execution_failed", "failure_stage": "worker_contract",
                                  "failure_reason": "invalid_full_pool_worker_report",
                                  "invalid_report": report}
                if report.get("status") == "execution_failed":
                    report.update(ranking=[], full_ranking=False, candidate_ids=pool,
                                  candidate_count=len(pool), candidate_ids_hash=fingerprint)
                report.update({
                    **declared, "query_id": query, "variant_id": variant,
                    "diagnostic_version": DIAGNOSTIC_VERSION, "mode": "no_gt_diagnostic",
                    "annotation_certification": "not_performed", "metrics_status": "not_applicable_no_gt",
                    "observation_sha256": case["observation_sha256"],
                    "normalized_observation_sha256": content_hash(normalized),
                    "graph_snapshot_sha256": source_hash, "config_sha256": state["config_sha256"],
                    "method_version": state["method_version"], "code_provenance": state["git"],
                    "timing_claim": state["timing_claim"],
                    "worker_timeout_seconds": min(per_query, remaining),
                })
                report_path = Path("per_query") / query / f"{variant}.json"
                _atomic_json(output / report_path, report)
                row = {"query_id": query, "variant_id": variant, "status": report["status"],
                       "full_ranking": report["full_ranking"], "path": report_path.as_posix()}
                state["results"].append(row)
                state["completed_attempt_count"] += 1
                if report["status"] not in _GOOD_STATUSES:
                    state["failures"].append({**row, "reason": report.get("failure_reason", report["status"])})
                state["active_attempt"] = None
                checkpoint()
        state["stage"] = "finished"
        if not state["failures"]:
            state["status"] = "diagnostic_completed"
    except (Exception, KeyboardInterrupt) as exc:
        state["failures"].append({"stage": state["stage"], "active_attempt": state["active_attempt"],
                                  "error_type": type(exc).__name__,
                                  "reason": str(exc) if isinstance(exc, (ValueError, TimeoutError))
                                  else type(exc).__name__})
    finally:
        state["unattempted"] = state.get("execution_schedule", [])[state["completed_attempt_count"]:]
        _atomic_json(output / "failures.json", state["failures"])
        checkpoint()
    return state


def _worker(job_path: Path, result_path: Path) -> None:
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        candidates = json.loads(Path(job["reference_path"]).read_text(encoding="utf-8"))
        report = retrieve_normalized(job["normalized"], candidates, method=job["method"],
                                     variant=job["variant"])
    except Exception as exc:
        report = {"status": "execution_failed", "failure_stage": "worker_inference",
                  "error_type": type(exc).__name__,
                  "failure_reason": str(exc) if isinstance(exc, ValueError) else type(exc).__name__}
    _atomic_json(result_path, report)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("This module is an internal diagnostic worker; use the project CLI")
    _worker(Path(sys.argv[2]), Path(sys.argv[3]))
