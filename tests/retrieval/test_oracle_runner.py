"""Synthetic runner integration tests; these fixtures are not research results."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from citygml_sg.retrieval import oracle_runner as runner
from citygml_sg.retrieval.evaluation.oracle_metrics import VARIANT_IDS


@pytest.fixture(autouse=True)
def _isolate_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "git_provenance", lambda root: {"commit": "fixture", "dirty": True})


def _snapshot() -> dict:
    return {
        "nodes": [
            {"id": "A", "type": "Room"}, {"id": "B", "type": "Room"},
            {"id": "chair_a", "type": "Furniture", "category": "chair"},
            {"id": "bed_b", "type": "Furniture", "category": "bed"},
        ],
        "edges": [
            {"id": "contains_a", "source": "A", "target": "chair_a", "type": "CONTAINS"},
            {"id": "contains_b", "source": "B", "target": "bed_b", "type": "CONTAINS"},
        ],
    }


def _inputs(tmp_path: Path, *, variants: list[str] | None = None) -> tuple[dict, dict]:
    observation = {
        "objects": [{"alias": "seen_chair", "label": "Furniture", "category": "chair"}],
        "relations": [["Room", "CONTAINS", "seen_chair"]],
    }
    (tmp_path / "view.json").write_text(json.dumps(observation), encoding="utf-8")
    manifest = {
        "graph_snapshot_id": "synthetic-v1", "candidate_pool_id": "two-rooms-v1",
        "cases": [{"query_id": "q1", "ovg_path": "view.json", "target_room_id": "A"}],
    }
    config = {
        "method": {}, "variants": list(VARIANT_IDS) if variants is None else variants,
        "timing": {"warmup": 1, "repeat": 2, "seed": 7}, "protocol": {},
    }
    return manifest, config


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_load_config_resolves_defaults_and_rejects_connection_settings(tmp_path: Path) -> None:
    path = tmp_path / "method.yaml"
    path.write_text("method: {}\n", encoding="utf-8")
    resolved = runner.load_run_config(path)
    assert resolved["method"]["alpha"] == resolved["method"]["beta"] == 1
    assert resolved["method"]["solver_mode"] == "exact_reference"
    assert resolved["method"]["anchor_threshold"] == 0
    assert "category_aliases" not in resolved["method"]
    assert "variant" not in resolved["method"]
    assert resolved["timing"]["repeat"] == 30
    assert resolved["protocol"]["input_condition"] == "researcher_curated"
    path.write_text("method:\n  password: should-never-be-copied\n", encoding="utf-8")
    with pytest.raises(ValueError, match="connection settings are separate"):
        runner.load_run_config(path)


def test_signature_ignores_nested_time_but_detects_changed_evidence() -> None:
    first = {"status": "completed", "timings": {"matching_ms": 10}, "ranking": [
        {"room_id": "A", "score": 0.5, "solver": {"elapsed_ms": 1, "optimality_proven": True},
         "node_matches": [{"observation_id": "a", "stored_node_id": "one"}]},
    ]}
    second = deepcopy(first)
    second["timings"]["matching_ms"] = 20
    second["ranking"][0]["solver"]["elapsed_ms"] = 9
    assert runner.stable_inference_signature(first) == runner.stable_inference_signature(second)
    second["ranking"][0]["node_matches"][0]["stored_node_id"] = "two"
    assert runner.stable_inference_signature(first) != runner.stable_inference_signature(second)


def test_online_timer_contains_anchor_matching_scoring_and_evidence(monkeypatch) -> None:
    current = [0.0]

    def anchors(observation, threshold):
        current[0] += 2.0
        return {"nodes": [], "edges": []}

    def rank(observation, candidates, *, config, variant):
        current[0] += 5.0
        return {"ranking": [{"evidence": "built"}],
                "timings": {"matching_ms": 3000, "scoring_sorting_ms": 2000}}

    monkeypatch.setattr(runner, "select_anchors", anchors)
    monkeypatch.setattr(runner, "rank_rooms", rank)
    report = runner.retrieve_normalized({}, [], method={}, variant="full", clock=lambda: current[0])
    assert report["timings"]["anchor_build_ms"] == 2000
    assert report["timings"]["online_retrieval_ms"] == 7000
    assert report["ranking"][0]["evidence"] == "built"


def test_synthetic_five_variant_run_exports_complete_metrics_and_raw_repeats(tmp_path) -> None:
    manifest, config = _inputs(tmp_path)
    output = tmp_path / "run"
    result = runner.run_experiment(manifest, config, workspace_root=tmp_path,
                                   output=output, snapshot=_snapshot())
    assert result["status"] == result["ablation_table_status"] == "completed"
    assert result["case_variant_count"] == 5
    resolved = _json(output / "resolved_config.json")
    assert resolved["method"]["max_states"] == 100000
    assert result["config_sha256"] == runner.content_hash(resolved)
    assert result["normalization_versions"]["observation"] == "semantic_observation_v1"
    assert result["normalization_versions"]["reference"] == "semantic_reference_v1"
    pool = _json(output / "candidate_pool.json")
    room = pool["rooms"][0]
    assert room["room_id"] == "A"
    assert room["membership"]["node_sources"]["chair_a"][0]["source"] == "room_contains"
    assert room["node_count"] == 2 and room["edge_count"] == 1
    assert result["effective_methods_by_variant"]["objects_only"]["beta"] == 0
    assert result["effective_methods_by_variant"]["relations_only"]["alpha"] == 0
    metrics = _json(output / "metrics_by_variant.json")
    for variant in VARIANT_IDS:
        assert metrics["by_variant"][variant]["query_count"] == 1
        assert metrics["by_variant"][variant]["recall_at_1"] == 1
        report = _json(output / "per_query" / "q1" / f"{variant}.json")
        assert report["timing_stats"]["repeat"] == 2
        assert report["timing_stats"]["status"] == "completed"
        assert report["observation_normalization_ms"] >= 0
        assert report["schema_load_scope"] == "file_read_json_parse_and_sha256"
        assert "target_room_id" not in report["anchors"]
        assert report["evaluation"]["target_room_id"] == "A"
    logs = [json.loads(line) for line in
            (output / "timing_runs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(logs) == 5 * (1 + 1 + 2)
    assert {item["phase"] for item in logs} == {"first_query", "warmup", "warm"}
    assert "per-query and timing exports" in result["report_write_scope"]
    assert result["report_write_ms"] >= 0
    assert len((output / "ablation_table.csv").read_text(encoding="utf-8").splitlines()) == 6


def test_prepare_only_does_not_load_reference_or_fabricate_results(tmp_path) -> None:
    manifest, config = _inputs(tmp_path)
    manifest["cases"][0]["target_room_id"] = None
    output = tmp_path / "prepare"

    def forbidden_loader():
        raise AssertionError("prepare-only must not fetch reference data")

    result = runner.run_experiment(manifest, config, workspace_root=tmp_path,
                                   output=output, prepare_only=True,
                                   snapshot_loader=forbidden_loader)
    assert result["status"] == result["ablation_table_status"] == "not_run"
    assert "q1.target_room_id" in result["missing_preparation_fields"]
    assert _json(output / "metrics_by_variant.json")["by_variant"] == {}
    assert not (output / "timing_runs.jsonl").exists()


def test_observation_preparation_failure_writes_incomplete_artifacts(tmp_path) -> None:
    manifest, config = _inputs(tmp_path)
    manifest["cases"][0]["ovg_path"] = "absent.json"
    output = tmp_path / "failed"
    with pytest.raises(FileNotFoundError):
        runner.run_experiment(manifest, config, workspace_root=tmp_path,
                              output=output, snapshot=_snapshot())
    result = _json(output / "run_manifest.json")
    assert result["status"] == "incomplete"
    assert result["failure"]["stage"] == "observation_preparation"
    assert result["failure"]["query_id"] == "q1"
    assert "incomplete" in (output / "ablation_table.md").read_text(encoding="utf-8")


def test_snapshot_loader_failure_keeps_setup_timing_and_redacts_message(tmp_path) -> None:
    manifest, config = _inputs(tmp_path)
    setup = {}

    def loader():
        setup["connection_preflight_ms"] = 1.5
        raise ValueError("bolt://secret-user:secret-password@host")

    output = tmp_path / "loader-failed"
    with pytest.raises(ValueError):
        runner.run_experiment(manifest, config, workspace_root=tmp_path, output=output,
                              snapshot_loader=loader, setup_timing=setup)
    result = _json(output / "run_manifest.json")
    assert result["status"] == "incomplete"
    assert result["setup_timing"]["connection_preflight_ms"] == 1.5
    assert result["failure"]["reason"] == "ValueError"
    for artifact in output.rglob("*"):
        if artifact.is_file():
            assert "secret-password" not in artifact.read_text(encoding="utf-8")


def test_successful_loader_updates_initially_empty_setup_timing(tmp_path) -> None:
    manifest, config = _inputs(tmp_path, variants=["full"])
    setup = {}

    def loader():
        setup["connection_preflight_ms"] = 4
        return _snapshot()

    output = tmp_path / "loaded"
    result = runner.run_experiment(manifest, config, workspace_root=tmp_path, output=output,
                                   snapshot_loader=loader, setup_timing=setup)
    assert result["status"] == "completed"
    assert result["ablation_table_status"] == "incomplete"  # Four variants were not requested.
    assert _json(output / "run_manifest.json")["setup_timing"]["connection_preflight_ms"] == 4


def test_elapsed_solver_fields_do_not_make_repeats_nondeterministic(tmp_path, monkeypatch) -> None:
    manifest, config = _inputs(tmp_path, variants=["full"])
    original = runner.rank_rooms
    calls = [0]

    def rank(*args, **kwargs):
        report = original(*args, **kwargs)
        calls[0] += 1
        report["ranking"][0]["solver"]["elapsed_ms"] = calls[0]
        return report

    monkeypatch.setattr(runner, "rank_rooms", rank)
    result = runner.run_experiment(manifest, config, workspace_root=tmp_path,
                                   output=tmp_path / "elapsed", snapshot=_snapshot())
    assert result["status"] == "completed"
    assert calls[0] == 4


def test_changed_repeat_evidence_is_a_recorded_incomplete_failure(tmp_path, monkeypatch) -> None:
    manifest, config = _inputs(tmp_path, variants=["full"])
    original = runner.rank_rooms
    calls = [0]

    def rank(*args, **kwargs):
        report = original(*args, **kwargs)
        calls[0] += 1
        report["ranking"][0]["score"] -= calls[0] * 0.01
        return report

    monkeypatch.setattr(runner, "rank_rooms", rank)
    output = tmp_path / "unstable"
    result = runner.run_experiment(manifest, config, workspace_root=tmp_path,
                                   output=output, snapshot=_snapshot())
    assert result["status"] == "incomplete"
    report = _json(output / "per_query" / "q1" / "full.json")
    assert report["status"] == "execution_failed"
    assert report["failure_stage"] == "warmup"
    assert report["timing_stats"]["mean_ms"] is None
    assert "last_successful_repeat" in report and "last_attempt_result" in report


def test_metric_contract_failure_keeps_invalid_report_and_null_metrics(tmp_path, monkeypatch):
    manifest, config = _inputs(tmp_path, variants=["full"])
    original = runner.rank_rooms

    def truncated(*args, **kwargs):
        report = original(*args, **kwargs)
        report["ranking"] = report["ranking"][:1]
        return report

    monkeypatch.setattr(runner, "rank_rooms", truncated)
    output = tmp_path / "truncated"
    result = runner.run_experiment(manifest, config, workspace_root=tmp_path,
                                   output=output, snapshot=_snapshot())
    assert result["status"] == "incomplete"
    report = _json(output / "per_query" / "q1" / "full.json")
    assert report["failure_stage"] == "metric_validation"
    assert report["invalid_report"]["candidate_count"] == 2
    assert report["evaluation"]["mrr"] is None
    assert report["timing_stats"]["status"] == "incomplete"


def test_unproven_solver_stops_repeats_without_a_zero_latency(tmp_path, monkeypatch) -> None:
    manifest, config = _inputs(tmp_path, variants=["full"])
    original = runner.rank_rooms

    def incomplete(*args, **kwargs):
        report = original(*args, **kwargs)
        report["status"] = "solver_incomplete"
        report["ranking"][0]["solver"]["optimality_proven"] = False
        return report

    monkeypatch.setattr(runner, "rank_rooms", incomplete)
    output = tmp_path / "solver-incomplete"
    result = runner.run_experiment(manifest, config, workspace_root=tmp_path,
                                   output=output, snapshot=_snapshot())
    assert result["status"] == "incomplete"
    report = _json(output / "per_query" / "q1" / "full.json")
    assert report["status"] == "solver_incomplete"
    assert report["timing_stats"]["repeat"] == 0
    assert report["timing_stats"]["mean_ms"] is None
    assert len((output / "timing_runs.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_absent_gt_invalidates_run_and_preserves_candidate_inventory(tmp_path) -> None:
    manifest, config = _inputs(tmp_path)
    manifest["cases"][0]["target_room_id"] = "C"
    output = tmp_path / "missing-gt"
    with pytest.raises(ValueError, match="absent from candidate pool"):
        runner.run_experiment(manifest, config, workspace_root=tmp_path,
                              output=output, snapshot=_snapshot())
    assert _json(output / "run_manifest.json")["status"] == "incomplete"
    assert _json(output / "candidate_pool.json")["candidate_ids"] == ["A", "B"]
    assert _json(output / "evaluated_queries.json")["records"] == []


def test_protocol_defaults_are_used_in_records_without_certifying_oracle(tmp_path) -> None:
    manifest, config = _inputs(tmp_path, variants=["full"])
    config["protocol"].update({"input_condition": "automatic_pipeline",
                               "oracle_validation_status": "not_applicable"})
    output = tmp_path / "automatic"
    runner.run_experiment(manifest, config, workspace_root=tmp_path,
                          output=output, snapshot=_snapshot())
    record = _json(output / "evaluated_queries.json")["records"][0]
    assert record["input_condition"] == "automatic_pipeline"
    assert record["oracle_validation_status"] == "not_applicable"


def test_timing_statistics_state_sample_variance_and_percentile_definition() -> None:
    assert runner._timing_stats([])["mean_ms"] is None
    assert runner._timing_stats([4])["std_ms"] is None
    stats = runner._timing_stats([1, 2, 3, 4])
    assert stats["p95_ms"] == 4
    assert stats["p95_method"] == "nearest_rank"
    assert stats["repeat"] == 4


@pytest.mark.parametrize("tie,target", [(False, "B"), (True, "A")])
def test_wrong_top1_and_ties_are_diagnostics_not_execution_failures(tmp_path, tie, target):
    manifest, config = _inputs(tmp_path, variants=["full"])
    manifest["cases"][0]["target_room_id"] = target
    snapshot = _snapshot()
    if tie:
        snapshot["nodes"][-1]["category"] = "chair"
    output = tmp_path / "diagnostic"
    result = runner.run_experiment(manifest, config, workspace_root=tmp_path,
                                   output=output, snapshot=snapshot)
    assert result["status"] == "completed"
    analysis = _json(output / "failure_analysis.json")
    assert analysis["execution_failures_and_abstentions"] == []
    diagnostic = analysis["retrieval_diagnostics"][0]
    assert diagnostic["top_score_tie"] is tie
    assert diagnostic["wrong_top1"] is not tie
    assert diagnostic["evidence_path"] == "per_query/q1/full.json"
    assert "object_numerator" in diagnostic["top_candidate"]
    assert "unmatched_edge_count" in diagnostic["target_candidate"]
    assert "target rank" in (output / "failure_analysis.md").read_text(encoding="utf-8")
