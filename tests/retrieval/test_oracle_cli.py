"""Offline benchmark CLI preparation/inventory and failure-artifact safeguards."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from citygml_sg.retrieval import oracle_runner
from citygml_sg.retrieval.evaluation.oracle_metrics import candidate_ids_hash


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cli(tmp_path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "run_oracle_evaluation_test", ROOT / "scripts/retrieval/run_oracle_evaluation.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(module, "git_provenance", lambda root: {"commit": "fixture", "dirty": True})
    monkeypatch.setattr(oracle_runner, "git_provenance", lambda root: {"commit": "fixture", "dirty": True})
    return module


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prepare_inputs(tmp_path: Path) -> None:
    _write_json(tmp_path / "view.json", {
        "objects": [{"alias": "seen", "type": "Furniture", "category": "chair"}], "relations": [],
    })
    _write_json(tmp_path / "manifest.json", {
        "cases": [{"query_id": "q1", "ovg_path": "view.json", "target_room_id": None}],
    })
    (tmp_path / "method.yaml").write_text("method: {}\ntiming:\n  warmup: 0\n  repeat: 1\n", encoding="utf-8")


def _snapshot() -> dict:
    return {
        "nodes": [{"id": "B", "type": "Room"}, {"id": "A", "type": "Room"},
                  {"id": "chair", "type": "BuildingFurniture", "category": "chair"}],
        "edges": [{"source_id": "chair", "target_id": "A", "relation": "INSIDE"}],
    }


def test_prepare_only_validates_without_graph_or_fabricated_results(cli, tmp_path: Path, monkeypatch) -> None:
    _prepare_inputs(tmp_path)
    def forbidden_loader(*args, **kwargs):
        raise AssertionError("prepare-only must not even construct a graph loader")
    monkeypatch.setattr(cli, "_snapshot_loader", forbidden_loader)
    code = cli.main([
        "--prepare-only", "--manifest", "manifest.json", "--method-config", "method.yaml",
        "--graph-json", "absent-and-not-needed.json", "--output-dir", "prepared",
    ])
    assert code == 0
    output = tmp_path / "prepared"
    report = _read_json(output / "run_manifest.json")
    assert report["status"] == "not_run"
    assert report["case_variant_count"] == 0
    assert "q1.target_room_id" in report["missing_preparation_fields"]
    assert _read_json(output / "preparation.json")["status"] == "not_run"
    assert _read_json(output / "metrics_by_variant.json")["status"] == "not_run"
    assert "not_run" in (output / "ablation_table.csv").read_text(encoding="utf-8")
    assert not (output / "timing_runs.jsonl").exists()
    assert not (output / "candidate_pool.json").exists()
    assert not (output / "per_query").exists()
    resolved = _read_json(output / "resolved_config.json")
    assert "variant" not in resolved["method"]
    assert "stored_metadata_policy" not in resolved["method"]


def test_inventory_only_exports_canonical_pool_without_manifest_or_observation(cli, tmp_path: Path) -> None:
    _write_json(tmp_path / "graph.json", _snapshot())
    code = cli.main([
        "--inventory-only", "--graph-json", "graph.json", "--output-dir", "inventory",
        "--manifest", "not-present.json", "--method-config", "not-present.yaml",
    ])
    assert code == 0
    output = tmp_path / "inventory"
    manifest = _read_json(output / "run_manifest.json")
    assert manifest["status"] == "not_run"
    assert manifest["inventory_status"] == "completed"
    pool = _read_json(output / "candidate_pool.json")
    assert pool["candidate_ids"] == ["A", "B"]
    assert pool["candidate_count"] == 2
    assert pool["candidate_ids_hash"] == candidate_ids_hash(["A", "B"])
    assert _read_json(output / "graph_inventory.json")["room_count"] == 2
    assert not (output / "metrics_by_variant.json").exists()
    assert not (output / "timing_runs.jsonl").exists()


@pytest.mark.parametrize("as_file", [False, True])
def test_existing_output_is_never_overwritten(cli, tmp_path: Path, as_file: bool) -> None:
    output = tmp_path / "existing"
    if as_file:
        output.write_text("keep", encoding="utf-8")
        sentinel = output
    else:
        output.mkdir()
        sentinel = output / "run_manifest.json"
        sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        cli.main(["--prepare-only", "--output-dir", str(output)])
    assert error.value.code == 2
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_invalid_snapshot_leaves_an_incomplete_inventory_artifact(cli, tmp_path: Path) -> None:
    (tmp_path / "bad.json").write_text("{broken", encoding="utf-8")
    assert cli.main(["--inventory-only", "--graph-json", "bad.json", "--output-dir", "failed"]) == 2
    output = tmp_path / "failed"
    manifest = _read_json(output / "run_manifest.json")
    assert manifest["status"] == "incomplete"
    assert manifest["error_type"] == "JSONDecodeError"
    assert "inventory_status" not in manifest
    assert not (output / "candidate_pool.json").exists()


def test_neo4j_loader_valueerror_cannot_copy_connection_details(cli, tmp_path: Path, monkeypatch, capsys) -> None:
    secret_marker = "synthetic-user:synthetic-secret@private-host"
    def loader_factory(*args, **kwargs):
        def fail():
            raise ValueError("bolt://" + secret_marker)
        return fail
    monkeypatch.setattr(cli, "_snapshot_loader", loader_factory)
    assert cli.main([
        "--inventory-only", "--config", "unused.local.yaml", "--output-dir", "db-failed",
    ]) == 2
    payload = (tmp_path / "db-failed/run_manifest.json").read_text(encoding="utf-8")
    assert secret_marker not in payload
    assert _read_json(tmp_path / "db-failed/run_manifest.json")["error_type"] == "ValueError"
    assert secret_marker not in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [["--inventory-only"], ["--prepare-only", "--repeat", "0"],
                                        ["--prepare-only", "--warmup", "-1"]])
def test_invalid_cli_options_fail_before_creating_results(cli, tmp_path: Path, arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main([*arguments, "--output-dir", "invalid"])
    assert error.value.code == 2
    assert not (tmp_path / "invalid").exists()
