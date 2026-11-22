"""Synthetic source-freeze/bundle contracts; no empirical experiment claims."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from citygml_sg.retrieval import diagnostic_runner
from citygml_sg.retrieval.experiment_bundle import prepare_diagnostic_bundle
from citygml_sg.retrieval.oracle_runner import content_hash


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    shapes = {
        "objects_relations": {"objects": [
            {"alias": "chair", "type": "BuildingFurniture", "category": "chair"}],
            "relations": [["Room", "CONTAINS", "chair"]]},
        "objects_only": {"objects": [
            {"alias": "door", "type": "Door", "category": "door"}], "relations": []},
        "context_only": {"objects": [
            {"alias": "wall", "type": "WallSurface", "category": "wall"}], "relations": []},
        "empty": {"objects": [], "relations": []},
    }
    cases, expected, originals = [], {}, {}
    # Deliberately oppose file/manifest order and lexicographic selection order.
    for prefix in ("z", "a"):
        for stratum, shape in reversed(list(shapes.items())):
            query = f"{prefix}_{stratum}"
            raw = {**deepcopy(shape), "metadata": {
                "query_id": query, "image_id": f"image-{query}",
                "oracle_validation_status": "pending", "note": "관측 원문 보존",
            }}
            payload = ("\ufeff" + json.dumps(raw, ensure_ascii=False, indent=2) + "\r\n").encode("utf-8")
            (source / f"{query}.json").write_bytes(payload)
            expected[query] = sha256(payload).hexdigest()
            originals[query] = payload
            cases.append({"query_id": query, "ovg_path": f"{query}.json",
                          "target_room_id": "evaluator-only-room", "view_id": f"image-{query}"})
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps({
        "input_condition": "ai_assisted_image_annotation", "oracle_validation_status": "pending",
        "ground_truth": {"note": "evaluator-only declaration, never copied"}, "cases": cases,
    }), encoding="utf-8")
    snapshot = source / "snapshot.json"
    snapshot.write_text(json.dumps({
        "nodes": [{"id": "R", "type": "Room"},
                  {"id": "f", "type": "BuildingFurniture", "category": "chair"}],
        "edges": [{"source": "R", "target": "f", "type": "CONTAINS"}],
    }, indent=2) + "\n", encoding="utf-8")
    return manifest, snapshot, expected, originals


def _prepare(tmp_path, inputs):
    manifest, snapshot, expected, _ = inputs
    return prepare_diagnostic_bundle(manifest, snapshot, tmp_path / "bundle", expected_hashes=expected)


def test_exact_bytes_pending_and_complete_freeze_identity_are_preserved(tmp_path):
    inputs = _fixture(tmp_path)
    manifest, snapshot, expected, originals = inputs
    old_manifest, old_snapshot = manifest.read_bytes(), snapshot.read_bytes()
    summary = _prepare(tmp_path, inputs)
    output = tmp_path / "bundle"
    bundled = _read(output / "diagnostic_manifest.json")
    assert summary["status"] == "prepared_pending_input"
    assert summary["retrieval_status"] == "not_run"
    assert summary["annotation_verification_performed"] is False
    assert summary["query_count"] == len(expected)
    assert summary["source_manifest_file_sha256"] == sha256(old_manifest).hexdigest()
    assert summary["snapshot_file_sha256"] == sha256(old_snapshot).hexdigest()
    assert bundled["graph_snapshot_sha256"] == content_hash(_read(snapshot))
    assert (output / "snapshot.json").read_bytes() == old_snapshot
    assert snapshot.read_bytes() == old_snapshot and manifest.read_bytes() == old_manifest
    assert bundled["oracle_validation_status"] == "pending"
    for case in bundled["cases"]:
        query = case["query_id"]
        assert (output / case["ovg_path"]).read_bytes() == originals[query]
        assert (manifest.parent / f"{query}.json").read_bytes() == originals[query]
        assert case["observation_sha256"] == expected[query]
        assert case["oracle_validation_status"] == "pending"
        assert case["input_condition"] == "ai_assisted_image_annotation"
    for path in output.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "target_room_id" not in text and "evaluator-only-room" not in text
    assert not (output / "metrics_by_variant.json").exists()


def test_pilot_selects_first_query_of_each_normalized_stratum_without_gt(tmp_path):
    inputs = _fixture(tmp_path)
    summary = _prepare(tmp_path, inputs)
    assert summary["strata"] == {
        "objects_relations": 2, "objects_only": 2, "context_only": 2, "empty": 2,
    }
    assert summary["pilot_query_ids"] == [
        "a_objects_relations", "a_objects_only", "a_context_only", "a_empty",
    ]
    pilot = _read(tmp_path / "bundle/pilot_manifest.json")
    assert [case["query_id"] for case in pilot["cases"]] == summary["pilot_query_ids"]
    assert pilot["selection_policy"] == "lexicographic_first_per_stratum_before_outcomes"
    # Mutating evaluator GT and reversing original order must not select different inputs.
    manifest, snapshot, expected, _ = inputs
    source = _read(manifest)
    for case in source["cases"]:
        case["target_room_id"] = "different-evaluator-room"
    source["cases"].reverse()
    manifest.write_text(json.dumps(source), encoding="utf-8")
    second = prepare_diagnostic_bundle(manifest, snapshot, tmp_path / "second", expected_hashes=expected)
    assert second["pilot_query_ids"] == summary["pilot_query_ids"]
    assert second["strata"] == summary["strata"]


@pytest.mark.parametrize("change,pattern", [
    ("changed_bytes", "Frozen observation hash"), ("missing_expected", "Frozen observation hash"),
    ("extra_expected", "query sets differ"), ("duplicate_query", "unique and safe"),
    ("unsafe_query", "unique and safe"), ("empty_pool", "no Room candidates"),
])
def test_invalid_complete_inventory_is_rejected_before_publication(tmp_path, change, pattern):
    manifest, snapshot, expected, originals = inputs = _fixture(tmp_path)
    query = next(iter(expected))
    if change == "changed_bytes":
        with (manifest.parent / f"{query}.json").open("ab") as stream:
            stream.write(b" ")
    elif change == "missing_expected":
        expected.pop(query)
    elif change == "extra_expected":
        expected["unlisted"] = "0" * 64
    elif change in {"duplicate_query", "unsafe_query"}:
        source = _read(manifest)
        if change == "duplicate_query":
            source["cases"].append(deepcopy(source["cases"][0]))
        else:
            source["cases"][0]["query_id"] = "../unsafe"
        manifest.write_text(json.dumps(source), encoding="utf-8")
    else:
        snapshot.write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    with pytest.raises(ValueError, match=pattern):
        _prepare(tmp_path, inputs)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("metadata", [
    {"target_room_id": "R"}, {"review": [{"ground_truth": {"room": "R"}}]},
    {"room_name": "Room R"}, {"nested": {"gml_id": "stored-furniture"}},
])
def test_ignored_raw_metadata_cannot_smuggle_map_gt_ids_into_bundle(tmp_path, metadata):
    manifest, snapshot, expected, _ = inputs = _fixture(tmp_path)
    query = next(iter(expected))
    path = manifest.parent / f"{query}.json"
    raw = _read(path)
    raw["metadata"].update(metadata)
    path.write_text(json.dumps(raw), encoding="utf-8")
    expected[query] = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="map/GT identifiers"):
        _prepare(tmp_path, inputs)
    assert not (tmp_path / "bundle").exists()


def test_existing_output_preserves_all_artifacts(tmp_path):
    inputs = _fixture(tmp_path)
    _prepare(tmp_path, inputs)
    output = tmp_path / "bundle"
    before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="new or empty"):
        _prepare(tmp_path, inputs)
    after = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
    assert before == after


def test_absent_strata_are_not_fabricated(tmp_path):
    manifest, snapshot, expected, _ = _fixture(tmp_path)
    source = _read(manifest)
    source["cases"] = [c for c in source["cases"] if c["query_id"].endswith("empty")]
    expected = {c["query_id"]: expected[c["query_id"]] for c in source["cases"]}
    manifest.write_text(json.dumps(source), encoding="utf-8")
    result = prepare_diagnostic_bundle(manifest, snapshot, tmp_path / "bundle", expected_hashes=expected)
    assert result["pilot_query_ids"] == ["a_empty"]
    assert result["strata"] == {"empty": 2}


def test_direct_runner_rejects_ignored_raw_metadata_gt_before_worker(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path)
    _prepare(tmp_path, inputs)
    output = tmp_path / "bundle"
    manifest = _read(output / "pilot_manifest.json")
    case = manifest["cases"][0]
    observation = output / case["ovg_path"]
    raw = _read(observation)
    raw["metadata"]["review"] = [{"gt_room_id": "R"}]
    observation.write_text(json.dumps(raw), encoding="utf-8")
    case["observation_sha256"] = sha256(observation.read_bytes()).hexdigest()
    monkeypatch.setattr(diagnostic_runner, "_run_worker", lambda *args: pytest.fail("must not start"))
    result = diagnostic_runner.run_diagnostics(
        manifest, {"method": {}, "variants": ["full"]}, workspace_root=output,
        output=tmp_path / "run", snapshot=_read(output / "snapshot.json"),
        max_queries=1, budget_seconds=10, per_query_timeout_seconds=1,
    )
    assert result["status"] == "diagnostic_incomplete"
    assert result["completed_attempt_count"] == 0
    assert "map/GT identifiers" in result["failures"][0]["reason"]


def test_runner_keeps_cli_source_file_identity_distinct_from_resolved_manifest(tmp_path, monkeypatch):
    inputs = _fixture(tmp_path)
    _prepare(tmp_path, inputs)
    bundle = tmp_path / "bundle"
    payload = (bundle / "pilot_manifest.json").read_bytes()
    manifest = json.loads(payload)
    identity = {"file_sha256": sha256(payload).hexdigest(),
                "content_sha256_before_path_resolution": content_hash(manifest),
                "path_resolution": "relative_to_manifest_parent"}
    manifest["source_file_identity"] = deepcopy(identity)
    for case in manifest["cases"]:
        case["ovg_path"] = str((bundle / case["ovg_path"]).resolve())
    monkeypatch.setattr(diagnostic_runner, "git_provenance", lambda root: {"commit": "synthetic"})
    monkeypatch.setattr(diagnostic_runner, "_run_worker", lambda *args: {
        "status": "execution_failed", "failure_reason": "synthetic_failure"})
    result = diagnostic_runner.run_diagnostics(
        manifest, {"method": {}, "variants": ["full"]}, workspace_root=tmp_path,
        output=tmp_path / "run", snapshot=_read(bundle / "snapshot.json"),
        max_queries=1, budget_seconds=10, per_query_timeout_seconds=1,
    )
    assert result["source_file_identity"] == identity
    assert result["source_manifest_sha256"] == content_hash(manifest)
    assert result["source_manifest_sha256"] != identity["content_sha256_before_path_resolution"]
    assert _read(tmp_path / "run/selected_manifest.json")["source_file_identity"] == identity
    assert _read(tmp_path / "run/run_manifest.json")["source_file_identity"] == identity
    manifest["source_file_identity"]["file_sha256"] = "changed_later"
    assert result["source_file_identity"] == identity


def test_prepare_cli_rejects_duplicate_frozen_query_declarations(tmp_path, monkeypatch, capsys):
    manifest, snapshot, expected, _ = _fixture(tmp_path)
    frozen = tmp_path / "freeze.json"
    entry = {"query_id": next(iter(expected)), "sha256": next(iter(expected.values()))}
    frozen.write_text(json.dumps({"observations": [entry, entry]}), encoding="utf-8")
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "prepare_bundle_cli_test", root / "scripts/retrieval/prepare_diagnostic_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["prepare", "--manifest", str(manifest),
        "--graph-json", str(snapshot), "--observation-freeze", str(frozen),
        "--output-dir", str(tmp_path / "bundle")])
    assert module.main() == 2
    assert "Duplicate query IDs" in capsys.readouterr().err
    assert not (tmp_path / "bundle").exists()
