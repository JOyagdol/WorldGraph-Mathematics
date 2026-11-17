"""Offline CLI integration and the explicit-OVG legacy-default regression."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from citygml_sg.retrieval.scoring.view_params import build_room_retrieval_params


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location(
        "room_localization_queries_test", ROOT / "scripts/retrieval/room_localization_queries.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _params(path: Path | None, **kwargs) -> dict:
    return build_room_retrieval_params(
        view_graph_path=None if path is None else str(path), limit=None,
        furniture_keywords=kwargs.get("furniture_keywords"), source_keywords=None, target_keywords=None,
    )


def test_legacy_defaults_remain_without_ovg_and_disappear_with_explicit_empty_ovg(tmp_path: Path) -> None:
    assert _params(None)["furniture_keywords"] == ["storage", "fridge", "sofa", "table"]
    assert _params(None)["installation_keywords"] == ["column", "pillar", "기둥"]
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"objects": [], "relations": []}), encoding="utf-8")
    params = _params(path)
    assert params["furniture_keywords"] == []
    assert params["installation_keywords"] == []
    assert params["installation_target_ids"] == []
    assert _params(path, furniture_keywords=["sofa"])["furniture_keywords"] == ["sofa"]


def test_structural_only_ovg_never_inherits_furniture_or_installation_keywords(tmp_path: Path) -> None:
    path = tmp_path / "structure.json"
    path.write_text(json.dumps({"objects": [{"alias": "d", "category": "Door", "type": "Opening"}]}), encoding="utf-8")
    params = _params(path)
    assert params["furniture_keywords"] == []
    assert params["installation_keywords"] == []
    path.write_text(json.dumps({"objects": [{"alias": "i", "category": "column", "type": "Installation"}]}), encoding="utf-8")
    assert _params(path)["installation_keywords"] == ["column"]


def test_cli_defaults_preserve_legacy_and_joint_requires_an_observation(cli) -> None:
    assert cli.build_parser().parse_args([]).method == "legacy_heuristic"
    args = cli.build_parser().parse_args(["--method", "joint_evidence_v1_proposed"])
    with pytest.raises(ValueError, match="requires --view-graph"):
        cli._run_joint(args)


def test_joint_offline_cli_keeps_full_pool_when_console_limit_is_one(cli, tmp_path: Path, monkeypatch, capsys) -> None:
    observation = tmp_path / "observation.json"
    observation.write_text(json.dumps({
        "objects": [{"alias": "seen", "type": "BuildingFurniture", "category": "chair", "confidence": 1, "visibility": 1}],
        "relations": [], "constraints": {"matching": {"min_object_matches": 999}, "room": {"candidate_limit": 1}},
    }), encoding="utf-8")
    snapshot = tmp_path / "reference.json"
    snapshot.write_text(json.dumps({
        "nodes": [{"id": "a", "type": "Room"}, {"id": "b", "type": "Room"}, {"id": "c", "type": "Room"},
                  {"id": "chair", "type": "BuildingFurniture", "category": "chair"}],
        "edges": [{"source_id": "chair", "target_id": "b", "relation": "INSIDE"}],
    }), encoding="utf-8")
    output = tmp_path / "report.json"
    args = cli.build_parser().parse_args([
        "--method", "joint_evidence_v1_proposed", "--variant", "objects_only",
        "--method-config", str(ROOT / "configs/joint_evidence_v1.yaml"),
        "--view-graph", str(observation), "--reference-graph", str(snapshot),
        "--limit", "1", "--output", str(output),
    ])
    def no_database(*args, **kwargs):
        raise AssertionError("Offline reference snapshots must not open Neo4j")
    monkeypatch.setattr(cli, "Neo4jClient", no_database)
    assert cli._run_joint(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["full_ranking"] is True
    assert report["candidate_count"] == 3
    assert len(report["ranking"]) == 3
    assert report["ranking"][0]["room_id"] == "b"
    assert report["evaluation_status"] == "not_evaluated_no_ground_truth"
    assert report["oracle_validation_status"] == "pending"
    assert report["timings"]["online_retrieval_ms"] >= 0
    console = capsys.readouterr().out
    assert "1. b" in console
    assert "2. " not in console
    assert "min_object_matches" in console


def test_joint_cli_rejects_legacy_override_flags(cli) -> None:
    args = cli.build_parser().parse_args([
        "--method", "joint_evidence_v1_proposed", "--view-graph", "unused.json",
        "--scenario", "combined_room_score",
    ])
    with pytest.raises(ValueError, match="legacy-only"):
        cli._run_joint(args)
