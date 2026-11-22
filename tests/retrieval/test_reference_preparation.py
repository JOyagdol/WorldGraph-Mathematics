"""Offline synthetic preparation checks, not retrieval experiments."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from citygml_sg.retrieval.reference_preparation import content_hash, prepare_reference_inputs


def snapshot():
    return {"nodes": [
        {"id": "parent", "type": "BuildingPart"},
        {"id": "B", "type": "Room"}, {"id": "A", "type": "Room"},
        {"id": "chair", "type": "Furniture", "category": "chair"},
        {"id": "elsewhere", "type": "Furniture", "category": "refrigerator"},
    ], "edges": [
        {"source": "parent", "target": "A", "type": "CONTAINS"},
        {"source": "parent", "target": "B", "type": "CONTAINS"},
        {"source": "chair", "target": "A", "type": "INSIDE"},
    ]}


def observation():
    return {"objects": [
        {"alias": "seen", "type": "Furniture", "category": "refrigerator"},
        {"alias": "unknown", "type": "Furniture", "category": "plant"},
    ], "relations": []}


def test_whole_snapshot_vocabulary_and_literal_scope_without_scores():
    graph, observed = snapshot(), {"q": observation()}
    before = deepcopy((graph, observed))
    prepared = prepare_reference_inputs(graph, observed, anchor_threshold=0.0)
    assert (graph, observed) == before
    assert prepared["status"] == "not_run"
    assert prepared["preparation_status"] == "completed"
    assert prepared["method_version"] is None
    assert prepared["oracle_certification"] == "not_performed"
    assert prepared["candidate_ids"] == ["A", "B"]
    scope = prepared["room_scopes"][0]
    assert scope["membership"]["other_room_ids"] == ["B"]
    assert "elsewhere" not in {n["id"] for n in scope["nodes"]}
    anchors = prepared["queries"]["q"]["anchors"]
    assert {n["id"] for n in anchors["nodes"]} == {"seen", "__context__Room"}
    assert "fridge" in prepared["reference_categories"]
    assert "buildingpart" not in prepared["reference_categories"]
    assert prepared["reference_vocabulary_sha256"] == anchors["reference_vocabulary_sha256"]
    assert not {"ranking", "metrics", "score", "timings"}.intersection(prepared)
    assert not {"ranking", "metrics", "score", "timings"}.intersection(prepared["queries"]["q"])


def test_content_identity_changes_with_parameters_and_sources_without_mutation():
    graph, raw = snapshot(), observation()
    first = prepare_reference_inputs(graph, {"q": raw}, anchor_threshold=0.0)
    assert first == prepare_reference_inputs(graph, {"q": raw}, anchor_threshold=0.0)
    second = prepare_reference_inputs(graph, {"q": raw}, anchor_threshold=0.5, hop_radius=0)
    assert first["config_sha256"] != second["config_sha256"]
    assert first["graph_snapshot_content_sha256"] == second["graph_snapshot_content_sha256"]
    assert len(second["room_scopes"][0]["nodes"]) == 1
    raw["objects"][0]["confidence"] = 0.4
    changed = prepare_reference_inputs(graph, {"q": raw}, anchor_threshold=0.5)
    assert changed["queries"]["q"]["observation_content_sha256"] != first["queries"]["q"]["observation_content_sha256"]
    assert changed["queries"]["q"]["anchors"]["excluded_nodes"][0]["reason"] == "below_anchor_threshold"
    assert first["graph_snapshot_content_sha256"] == content_hash(graph)


def test_target_metadata_never_changes_anchor_selection():
    raw = observation()
    raw["target_room_id"] = "A"
    a = prepare_reference_inputs(snapshot(), {"q": raw}, anchor_threshold=0.0)
    raw["target_room_id"] = "B"
    b = prepare_reference_inputs(snapshot(), {"q": raw}, anchor_threshold=0.0)
    assert a["queries"]["q"]["anchors"] == b["queries"]["q"]["anchors"]
    assert a["queries"]["q"]["observation_content_sha256"] != b["queries"]["q"]["observation_content_sha256"]
    assert "target_room_id" not in json.dumps(a["queries"]["q"]["anchors"])
    raw["objects"][0]["gml_id"] = "elsewhere"
    with pytest.raises(ValueError):
        prepare_reference_inputs(snapshot(), {"q": raw}, anchor_threshold=0.0)


@pytest.mark.parametrize("queries", [{}, {"../unsafe": {}}, {"": {}}, {1: {}}, []])
def test_invalid_query_mapping_fails(queries):
    with pytest.raises(ValueError):
        prepare_reference_inputs(snapshot(), queries, anchor_threshold=0.0)


def test_empty_room_pool_is_not_ready_input():
    with pytest.raises(ValueError, match="at least one Room"):
        prepare_reference_inputs({"nodes": [], "edges": []}, {"q": {}}, anchor_threshold=0.0)


def test_directed_inverse_and_symmetric_atoms_are_preserved_without_joint_dedup():
    raw = {"objects": [
        {"alias": "a", "type": "Furniture", "category": "chair"},
        {"alias": "b", "type": "Furniture", "category": "chair"}],
        "relations": [["a", "ABOVE", "b"], ["b", "BELOW", "a"],
                      ["a", "TOUCHES", "b"], ["b", "TOUCHES", "a"]]}
    prepared = prepare_reference_inputs(snapshot(), {"q": raw}, anchor_threshold=0.0)
    assert len(prepared["queries"]["q"]["anchors"]["edges"]) == 4
    assert prepared["config"]["observation_normalizer_config"] == {
        "symmetric_relations": [], "inverse_relations": {}}
    raw["relations"].append(["a", "ABOVE", "b"])
    with pytest.raises(ValueError, match="Duplicate observation relation"):
        prepare_reference_inputs(snapshot(), {"q": raw}, anchor_threshold=0.0)


@pytest.fixture
def cli(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "prepare_reference_inputs_test", root / "scripts/retrieval/prepare_reference_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "git_provenance", lambda root: {"commit": "synthetic", "dirty": True})
    return module


def files(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "view.json").write_text(json.dumps(observation()), encoding="utf-8")
    (inputs / "manifest.json").write_text(json.dumps({
        "input_condition": "ai_assisted", "oracle_validation_status": "pending",
        "cases": [{"query_id": "q", "ovg_path": "view.json", "target_room_id": "A"}],
    }), encoding="utf-8")
    (inputs / "graph.json").write_text(json.dumps(snapshot()), encoding="utf-8")
    return inputs


def arguments(inputs, output):
    return ["--manifest", str(inputs / "manifest.json"), "--graph-json", str(inputs / "graph.json"),
            "--output-dir", str(output), "--anchor-threshold", "0.5"]


def test_cli_preserves_pending_declaration_and_does_not_overwrite(cli, tmp_path):
    inputs = files(tmp_path)
    output = tmp_path / "output"
    assert cli.main(arguments(inputs, output)) == 0
    artifact = output / "preparation.json"
    first_bytes = artifact.read_bytes()
    prepared = json.loads(first_bytes)
    assert prepared["source_metadata"]["q"]["declared_annotation_metadata"] == {
        "input_condition": "ai_assisted", "oracle_validation_status": "pending"}
    assert prepared["source_metadata"]["q"]["annotation_verification_performed"] is False
    assert prepared["status"] == "not_run"
    assert len(prepared["graph_snapshot_file_sha256"]) == 64
    assert prepared["code_provenance"]["commit"] == "synthetic"
    assert {p.name for p in output.iterdir()} == {"preparation.json"}
    assert cli.main(arguments(inputs, output)) == 2
    assert artifact.read_bytes() == first_bytes


def test_cli_records_conflicting_annotation_declarations_without_certifying(cli, tmp_path):
    inputs = files(tmp_path)
    manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
    manifest["cases"][0]["oracle_validation_status"] = "oracle_verified"
    (inputs / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert cli.main(arguments(inputs, tmp_path / "output")) == 0
    artifact = json.loads((tmp_path / "output/preparation.json").read_text(encoding="utf-8"))
    source = artifact["source_metadata"]["q"]
    assert source["conflicting_annotation_fields"] == ["oracle_validation_status"]
    assert source["annotation_declarations_by_source"]["manifest"]["oracle_validation_status"] == "pending"
    assert source["annotation_declarations_by_source"]["case"]["oracle_validation_status"] == "oracle_verified"
    assert source["annotation_verification_performed"] is False
    assert artifact["oracle_certification"] == "not_performed"


@pytest.mark.parametrize("fault", ["dangling", "duplicate_case", "missing_observation", "threshold"])
def test_cli_validates_all_inputs_before_writing(cli, tmp_path, fault):
    inputs = files(tmp_path)
    output = tmp_path / "output"
    args = arguments(inputs, output)
    if fault == "dangling":
        graph = snapshot()
        graph["edges"][0]["target"] = "missing"
        (inputs / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    elif fault == "duplicate_case":
        manifest = json.loads((inputs / "manifest.json").read_text(encoding="utf-8"))
        manifest["cases"].append(manifest["cases"][0])
        (inputs / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif fault == "missing_observation":
        (inputs / "view.json").unlink()
    else:
        args[-1] = "nan"
    assert cli.main(args) == 2
    assert not output.exists()
