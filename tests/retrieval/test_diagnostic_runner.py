"""Synthetic diagnostic-runner contracts, not research measurements."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pytest

from citygml_sg.retrieval import diagnostic_runner as runner


def _fixture(tmp_path: Path, count: int = 1) -> tuple[dict, dict, dict]:
    snapshot = {
        "nodes": [{"id": "A", "type": "Room"}, {"id": "B", "type": "Room"},
                  {"id": "a", "type": "Furniture", "category": "chair"}],
        "edges": [{"id": "e", "source": "A", "target": "a", "type": "CONTAINS"}],
    }
    cases = []
    for index in range(count):
        path = tmp_path / f"view{index}.json"
        path.write_text(json.dumps({
            "objects": [{"alias": "seen", "label": "Furniture", "category": "chair"}],
            "relations": [["Room", "CONTAINS", "seen"]],
        }), encoding="utf-8")
        cases.append({"query_id": f"q{index}", "ovg_path": path.name,
                      "observation_sha256": sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "graph_snapshot_id": "synthetic-v1", "candidate_pool_id": "synthetic-two-rooms",
        "graph_snapshot_sha256": runner.content_hash(snapshot), "cases": cases,
        "input_condition": "synthetic_fixture", "oracle_validation_status": "not_applicable",
    }
    config = {"method": {}, "variants": ["full"], "timing": {"warmup": 3, "repeat": 30}}
    return manifest, config, snapshot


def _run(tmp_path: Path, manifest: dict, config: dict, snapshot: dict, **kwargs) -> dict:
    return runner.run_diagnostics(
        manifest, config, snapshot=snapshot, workspace_root=tmp_path, output=tmp_path / "out",
        max_queries=kwargs.pop("max_queries", 1), budget_seconds=kwargs.pop("budget_seconds", 20),
        per_query_timeout_seconds=kwargs.pop("per_query_timeout_seconds", 10), **kwargs,
    )


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _no_git(monkeypatch):
    monkeypatch.setattr(runner, "git_provenance", lambda root: {"commit": "synthetic", "dirty": True})


def test_owned_worker_runs_full_pool_without_gt_or_metric_artifacts(tmp_path):
    manifest, config, snapshot = _fixture(tmp_path)
    result = _run(tmp_path, manifest, config, snapshot)
    assert result["status"] == "diagnostic_completed"
    assert result["completed_attempt_count"] == 1
    assert result["metrics_status"] == "not_applicable_no_gt"
    report = _json(tmp_path / "out/per_query/q0/full.json")
    assert report["full_ranking"] is True
    assert {r["room_id"] for r in report["ranking"]} == {"A", "B"}
    assert report["method_version"] == "joint_evidence_v1_proposed"
    assert report["annotation_certification"] == "not_performed"
    assert report["oracle_validation_status"] == "not_applicable"
    assert report["timings"]["online_retrieval_ms"] >= 0
    assert report["worker_wall_ms"] >= report["timings"]["online_retrieval_ms"]
    assert "evaluation" not in report and "target_room_id" not in report
    assert not (tmp_path / "out/metrics_by_variant.json").exists()
    assert not (tmp_path / "out/ablation_table.csv").exists()
    assert _json(tmp_path / "out/run_manifest.json")["status"] == "diagnostic_completed"


@pytest.mark.parametrize("change,reason", [
    ("gt", "GT/target"), ("source", "observation_sha256 mismatch"),
    ("snapshot", "graph_snapshot_sha256"), ("verified", "pending or not_applicable"),
    ("duplicate", "unique safe"), ("pool", "candidate_ids_hash"),
])
def test_preflight_rejects_unsafe_identity_before_worker(tmp_path, monkeypatch, change, reason):
    manifest, config, snapshot = _fixture(tmp_path)
    if change == "gt":
        manifest["cases"][0]["target_room_id"] = "A"
    elif change == "source":
        (tmp_path / "view0.json").write_text("{}", encoding="utf-8")
    elif change == "snapshot":
        snapshot["nodes"].append({"id": "C", "type": "Room"})
    elif change == "verified":
        manifest["oracle_validation_status"] = "oracle_verified"
    elif change == "duplicate":
        manifest["cases"].append(deepcopy(manifest["cases"][0]))
    else:
        manifest["candidate_ids_hash"] = "0" * 64
    monkeypatch.setattr(runner, "_run_worker", lambda *args: pytest.fail("worker must not start"))
    result = _run(tmp_path, manifest, config, snapshot)
    assert result["status"] == "diagnostic_incomplete"
    assert result["completed_attempt_count"] == 0
    assert reason in result["failures"][0]["reason"]


def test_deterministic_selection_and_progress_are_written_before_each_attempt(tmp_path, monkeypatch):
    manifest, config, snapshot = _fixture(tmp_path, 3)
    config["variants"] = ["objects_only", "full"]
    seen = []

    def worker(job, output, timeout):
        progress = _json(tmp_path / "out/run_manifest.json")
        seen.append(progress["active_attempt"])
        assert progress["selected_query_ids"] == ["q0", "q1"]
        assert progress["completed_attempt_count"] == len(seen) - 1
        assert progress["selected_observation_hashes_verified"]
        return {"status": "execution_failed", "failure_reason": "synthetic_failure"}

    monkeypatch.setattr(runner, "_run_worker", worker)
    result = _run(tmp_path, manifest, config, snapshot, max_queries=2)
    assert seen == [{"query_id": q, "variant_id": v}
                    for q in ("q0", "q1") for v in ("objects_only", "full")]
    assert result["status"] == "diagnostic_incomplete"
    assert result["completed_attempt_count"] == 4
    assert len(result["failures"]) == 4 and result["unattempted"] == []
    assert not (tmp_path / "out/per_query/q2").exists()


def test_solver_incomplete_retains_feasible_full_ranking(tmp_path, monkeypatch):
    manifest, config, snapshot = _fixture(tmp_path)
    ranking = [{"room_id": "A", "score": 0.5, "solver": {
        "optimality_proven": False, "upper_bound": 1.0, "termination_reason": "max_states_exceeded"}},
        {"room_id": "B", "score": 0, "solver": {"optimality_proven": True}}]
    monkeypatch.setattr(runner, "_run_worker", lambda *args: {
        "status": "solver_incomplete", "full_ranking": True, "ranking": ranking,
        "candidate_ids": ["A", "B"], "candidate_count": 2,
        "candidate_ids_hash": runner.candidate_ids_hash(["A", "B"]),
    })
    result = _run(tmp_path, manifest, config, snapshot)
    assert result["status"] == "diagnostic_incomplete"
    report = _json(tmp_path / "out/per_query/q0/full.json")
    assert report["ranking"] == ranking and report["full_ranking"] is True
    assert "evaluation" not in report


def test_truncated_worker_ranking_is_failure_and_preserves_invalid_evidence(tmp_path, monkeypatch):
    manifest, config, snapshot = _fixture(tmp_path)
    monkeypatch.setattr(runner, "_run_worker", lambda *args: {
        "status": "completed", "full_ranking": True,
        "ranking": [{"room_id": "A", "solver": {"optimality_proven": True}}],
        "candidate_ids": ["A", "B"], "candidate_count": 2,
        "candidate_ids_hash": runner.candidate_ids_hash(["A", "B"]),
    })
    result = _run(tmp_path, manifest, config, snapshot)
    report = _json(tmp_path / "out/per_query/q0/full.json")
    assert result["status"] == "diagnostic_incomplete"
    assert report["full_ranking"] is False
    assert report["failure_reason"] == "invalid_full_pool_worker_report"
    assert report["invalid_report"]["ranking"][0]["room_id"] == "A"


def test_no_anchor_abstention_is_successful_diagnostic_not_accuracy(tmp_path):
    manifest, config, snapshot = _fixture(tmp_path)
    path = tmp_path / "view0.json"
    path.write_text('{"objects": [], "relations": []}', encoding="utf-8")
    manifest["cases"][0]["observation_sha256"] = sha256(path.read_bytes()).hexdigest()
    result = _run(tmp_path, manifest, config, snapshot)
    report = _json(tmp_path / "out/per_query/q0/full.json")
    assert result["status"] == "diagnostic_completed"
    assert report["status"] == "abstained_no_anchor" and report["full_ranking"]
    assert report["denominator"] == 0


def test_overall_budget_stops_new_queries_and_preserves_unattempted(tmp_path, monkeypatch):
    manifest, config, snapshot = _fixture(tmp_path, 2)
    clock = [0.0]
    monkeypatch.setattr(runner, "perf_counter", lambda: clock[0])

    def worker(*args):
        clock[0] = 11
        return {"status": "execution_failed", "failure_reason": "worker_timeout"}

    monkeypatch.setattr(runner, "_run_worker", worker)
    result = _run(tmp_path, manifest, config, snapshot, max_queries=2, budget_seconds=10)
    assert result["status"] == "diagnostic_incomplete"
    assert result["completed_attempt_count"] == 1
    assert result["unattempted"] == [{"query_id": "q1", "variant_id": "full"}]
    assert result["failures"][-1]["reason"] == "overall_budget_exhausted"


def test_worker_timeout_terminates_only_owned_process(tmp_path, monkeypatch):
    class Process:
        returncode = None
        terminated = False
        calls = 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("synthetic child", timeout)
            self.returncode = -1

        def terminate(self):
            self.terminated = True

        def poll(self):
            return self.returncode

    process = Process()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: process)
    report = runner._run_worker(tmp_path / "job.json", tmp_path / "result.json", 0.01)
    assert report["failure_reason"] == "worker_timeout"
    assert process.terminated and process.calls == 2


def test_keyboard_interrupt_leaves_durable_incomplete_state(tmp_path, monkeypatch):
    manifest, config, snapshot = _fixture(tmp_path)

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_run_worker", interrupted)
    result = _run(tmp_path, manifest, config, snapshot)
    assert result["status"] == "diagnostic_incomplete"
    assert result["failures"][0]["error_type"] == "KeyboardInterrupt"
    assert result["active_attempt"] == {"query_id": "q0", "variant_id": "full"}
    assert len(result["unattempted"]) == 1
    assert _json(tmp_path / "out/run_manifest.json")["status"] == "diagnostic_incomplete"


def test_existing_output_is_never_overwritten(tmp_path):
    manifest, config, snapshot = _fixture(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        _run(tmp_path, manifest, config, snapshot)
    assert sentinel.read_text(encoding="utf-8") == "keep"
