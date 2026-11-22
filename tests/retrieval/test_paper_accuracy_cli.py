"""The active paper path preserves the original solver/map and every outcome."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from citygml_sg.retrieval.graph_matching import joint_matcher
from citygml_sg.retrieval.hosted_reference import POLICY

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location(
        "paper_accuracy_test", ROOT / "scripts/retrieval/run_paper_accuracy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(tmp_path, *extra):
    return ["--manifest", "tests/fixtures/oracle/manifest.json", "--graph-json",
            "tests/fixtures/oracle/reference_graph.json", "--output-dir", str(tmp_path / "run"), *extra]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_paper_inference_never_dispatches_optimized_solver_and_does_not_report_latency(cli, tmp_path, monkeypatch):
    calls = []
    original = cli.rank_rooms

    def forbidden(*args, **kwargs):
        pytest.fail("The active paper path must not use an optimized solver")

    def checked(anchors, candidates, **kwargs):
        assert "target_room_id" not in anchors
        assert "query_id" not in anchors
        assert kwargs["config"]["solver_mode"] == "exact_reference"
        calls.append(kwargs["variant"])
        return original(anchors, candidates, **kwargs)

    # Also guards against accidentally reintroducing an optimized dispatch.
    monkeypatch.setattr(joint_matcher, "_joint_mapping_pruned_v2", forbidden, raising=False)
    monkeypatch.setattr(cli, "rank_rooms", checked)
    assert cli.main(arguments(tmp_path)) == 0
    assert len(calls) == 6  # Three synthetic inputs, Full and Objects, exactly once.
    output = tmp_path / "run"
    manifest = read(output / "run_manifest.json")
    assert manifest["case_variant_count"] == 6
    assert manifest["solver_mode"] == "exact_reference"
    assert manifest["latency_status"] == "not_collected"
    for path in (output / "per_query").glob("*/*.json"):
        report = read(path)
        assert "timings" not in report and "timing_stats" not in report
        assert report["evaluation"]["target_room_id"] in report["candidate_ids"]


def test_prepare_only_never_infers(cli, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "rank_rooms", lambda *a, **k: pytest.fail("prepare-only inferred"))
    assert cli.main(arguments(tmp_path, "--prepare-only")) == 0
    assert read(tmp_path / "run/run_manifest.json")["status"] == "not_run"
    assert not (tmp_path / "run/accuracy_metrics.json").exists()


@pytest.mark.parametrize("mode", ["exact_pruned_v2", "exact_pruned_v3"])
def test_optimized_config_is_rejected_before_output(cli, tmp_path, mode):
    config = yaml.safe_load((ROOT / "configs/paper_accuracy_reference.yaml").read_text(encoding="utf-8"))
    config["method"]["solver_mode"] = mode
    path = tmp_path / "fast.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cli.main(arguments(tmp_path, "--method-config", str(path)))
    assert error.value.code == 2
    assert not (tmp_path / "run").exists()


def test_failed_variant_stays_in_denominator_and_invalidates_primary(cli, tmp_path, monkeypatch):
    original = cli.rank_rooms

    def fail_full(*args, **kwargs):
        if kwargs["variant"] == "full":
            raise RuntimeError("synthetic failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "rank_rooms", fail_full)
    assert cli.main(arguments(tmp_path)) == 2
    records = read(tmp_path / "run/evaluated_queries.json")["records"]
    assert len(records) == 6
    failed = [row for row in records if row["variant_id"] == "full"]
    assert len(failed) == 3
    assert all(row["evaluation_status"] == "incomplete" and row["hit_at_1"] is None for row in failed)
    metrics = read(tmp_path / "run/accuracy_metrics.json")
    assert metrics["by_variant"]["full"]["status"] == "incomplete"
    assert metrics["by_variant"]["full"]["hit_at_1"] is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_explicit_groups_preserve_full_declared_pool(cli, tmp_path, wrapped):
    manifest = read(ROOT / "tests/fixtures/oracle/manifest.json")
    manifest["candidate_ids"] = ["synthetic_room_a", "synthetic_room_c"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    groups = [{"space_id": "synthetic_room_a", "space_name": "Synthetic union",
               "member_room_ids": ["synthetic_room_a", "synthetic_room_b"]}]
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(json.dumps({"groups": groups} if wrapped else groups), encoding="utf-8")
    assert cli.main(arguments(tmp_path, "--manifest", str(manifest_path),
                              "--space-groups", str(groups_path))) == 0
    run = read(tmp_path / "run/run_manifest.json")
    assert run["candidate_ids"] == manifest["candidate_ids"]
    assert run["space_groups_file_sha256"] is not None
    assert run["case_variant_count"] == 6


@pytest.mark.parametrize("payload", [[], {}, {"groups": []}, {"groups": None}, {"group": [{}]}])
def test_declared_group_file_must_supply_a_nonempty_group_list(cli, tmp_path, payload):
    path = tmp_path / "groups.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(arguments(tmp_path, "--space-groups", str(path), "--prepare-only"))
    assert not (tmp_path / "run").exists()


def test_existing_output_is_not_replaced(cli, tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    sentinel = output / "run_manifest.json"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(arguments(tmp_path))
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_hosted_ledger_opt_in_changes_only_existing_hosted_edges(cli, tmp_path):
    snapshot = read(ROOT / "tests/fixtures/oracle/reference_graph.json")
    baseline = cli.build_room_candidates(snapshot)
    relations = {}
    for candidate in baseline:
        for edge in candidate["edges"]:
            if edge["type"] == "HOSTED_BY":
                relations[edge["id"]] = {"edge_id": edge["id"], "source": edge["source"],
                                         "target": edge["target"], "supported_room_ids": []}
    assert relations  # The fixture exercises removal of real existing edges.
    ledger = {"policy": POLICY, "provenance": {"snapshot_content_sha256": cli.content_hash(snapshot)},
              "room_geometry_status": {c["room_id"]: "synthetic_fixture" for c in baseline},
              "relations": list(relations.values())}
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    assert cli.main(arguments(tmp_path, "--prepare-only", "--hosted-support-ledger", str(path))) == 0
    after = read(tmp_path / "run/candidate_pool.json")["candidates"]
    for old, new in zip(baseline, after):
        assert old["nodes"] == new["nodes"]
        assert new["edges"] == [e for e in old["edges"] if e["type"] != "HOSTED_BY"]
    identity = read(tmp_path / "run/run_manifest.json")
    assert identity["reference_policy"] == POLICY
    assert identity["hosted_support_ledger_file_sha256"] is not None
    assert identity["baseline_candidate_content_sha256"] == cli.content_hash(baseline)
    assert identity["candidate_content_sha256"] != identity["baseline_candidate_content_sha256"]


def test_table8_configuration_runs_the_selected_ranker_and_requires_map_evidence(cli, tmp_path):
    with pytest.raises(SystemExit):
        cli.main(arguments(tmp_path), table8=True)
    assert not (tmp_path / "run").exists()
    snapshot = read(ROOT / "tests/fixtures/oracle/reference_graph.json")
    pool = cli.build_room_candidates(snapshot)
    # The fixture's stored confidence is not a unit hosted edge. Mark it
    # unresolved so the pattern ledger contains no positive reference match.
    edges = {e["id"]: e for c in pool for e in c["edges"] if e["type"] == "HOSTED_BY"}
    ledger = {
        "policy": POLICY, "provenance": {"snapshot_content_sha256": cli.content_hash(snapshot)},
        "room_geometry_status": {c["room_id"]: "synthetic_fixture" for c in pool},
        "relations": [{"edge_id": e["id"], "source": e["source"], "target": e["target"],
                       "supported_room_ids": []} for e in edges.values()],
    }
    path = tmp_path / "host_ledger.json"
    path.write_text(json.dumps(ledger), encoding="utf-8")
    assert cli.main(arguments(tmp_path, "--hosted-support-ledger", str(path)), table8=True) == 0
    run = read(tmp_path / "run/run_manifest.json")
    assert run["method_version"] == "shared_wall_or_single_opening_v1"
    assert run["case_variant_count"] == 9
    assert run["variants"] == ["full", "objects_only", "relations_only"]
    assert run["reference_policy"] == POLICY and run["hosted_support_ledger_file_sha256"]
    for report in (tmp_path / "run/per_query").glob("*/*.json"):
        result = read(report)
        assert result["method_version"] == result["rank_key_policy"] == run["method_version"]
        assert "gate_reason" in result
