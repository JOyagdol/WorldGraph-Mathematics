from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from citygml_sg.retrieval import snapshot_freeze
from citygml_sg.retrieval.room_scope import build_room_hop_subgraphs
from citygml_sg.retrieval.snapshot_freeze import freeze_semantic_snapshot


def _source(tmp_path: Path, payload: dict) -> Path:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return source


def _graph() -> dict:
    return {
        "summary": {"node_count": 999, "edge_count": 999},
        "nodes": [
            {"id": "r", "type": "Room", "properties": {"name": "stored"}},
            {"id": "p", "type": "BuildingPart"},
            {"id": "f", "type": "BuildingFurniture", "properties": {"category": "chair"}},
            {"id": "geometry", "type": "Polygon", "properties": {"payload": [1, 2]}},
            {"id": "unsupported", "type": "NovelSourceType"},
        ],
        "edges": [
            {"source_id": "p", "target_id": "r", "relation": "INTERIOR_ROOM"},
            {"source_id": "f", "target_id": "r", "relation": "INSIDE",
             "properties": {"confidence": 0.7, "provenance": {"kind": "stored"}}},
            {"source_id": "f", "target_id": "r", "relation": "INSIDE"},
            {"source_id": "f", "target_id": "geometry", "relation": "HAS_GEOMETRY"},
            {"source_id": "unsupported", "target_id": "r", "relation": "CUSTOM"},
        ],
    }


@pytest.fixture(autouse=True)
def _require_streaming_dependency(request):
    if request.node.name != "test_missing_optional_dependency_has_actionable_error":
        pytest.importorskip("ijson")


def test_freeze_preserves_raw_records_counts_actual_payload_and_records_identity(tmp_path):
    payload = _graph()
    source = _source(tmp_path, payload)
    original = source.read_bytes()
    output = tmp_path / "frozen"
    report = freeze_semantic_snapshot(source, output)
    frozen_bytes = (output / "snapshot.json").read_bytes()
    frozen = json.loads(frozen_bytes)
    assert source.read_bytes() == original
    assert frozen["nodes"] == payload["nodes"][:3]
    assert frozen["edges"] == payload["edges"][:3]
    assert report["inventory"]["full_node_count"] == 5
    assert report["inventory"]["full_edge_count"] == 5
    assert report["inventory"]["geometry_node_count"] == 1
    assert report["inventory"]["dropped_node_type_counts"] == {"NovelSourceType": 1, "Polygon": 1}
    assert report["inventory"]["dropped_relation_counts"] == {"CUSTOM": 1, "HAS_GEOMETRY": 1}
    assert report["room_ids"] == ["r"]
    assert report["source"]["sha256"] == sha256(original).hexdigest()
    assert report["snapshot"]["file_sha256"] == sha256(frozen_bytes).hexdigest()
    canonical = json.dumps(frozen, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode("utf-8")
    assert report["snapshot"]["content_sha256"] == sha256(canonical).hexdigest()
    assert json.loads((output / "freeze_manifest.json").read_bytes()) == report
    assert str(source) not in json.dumps(report)
    candidates = build_room_hop_subgraphs(frozen, 2)
    assert candidates[0]["input_snapshot_sha256"] == report["snapshot"]["content_sha256"]
    assert len(candidates[0]["edges"]) == 3


def test_empty_existing_directory_is_supported_but_frozen_output_is_immutable(tmp_path):
    source = _source(tmp_path, _graph())
    output = tmp_path / "frozen"
    output.mkdir()
    freeze_semantic_snapshot(source, output)
    original_files = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError, match="new or empty"):
        freeze_semantic_snapshot(source, output)
    assert original_files == {path.name: path.read_bytes() for path in output.iterdir()}


@pytest.mark.parametrize("payload", [
    {"nodes": [], "edges": []},
    {"nodes": [{"id": "p", "type": "Polygon"}], "edges": []},
])
def test_zero_room_graph_is_frozen_without_an_evaluation_readiness_claim(tmp_path, payload):
    report = freeze_semantic_snapshot(_source(tmp_path, payload), tmp_path / "frozen")
    assert report["room_count"] == 0
    assert report["readiness"] == "reference_snapshot_only_no_gt_or_evaluation_claim"


@pytest.mark.parametrize("mutation,message", [
    (lambda graph: graph["nodes"].append(deepcopy(graph["nodes"][3])), "Duplicate.*node ID"),
    (lambda graph: graph["edges"].append(
        {"source": "geometry", "target": "missing", "type": "GEOMETRY"}), "unknown endpoint"),
    (lambda graph: graph["edges"][0].update({"id": "duplicate"}) or
     graph["edges"][1].update({"id": "duplicate"}), "Duplicate retained edge ID"),
    (lambda graph: graph["nodes"][0].update({"labels": None}), "labels must be a list"),
])
def test_invalid_identity_or_dangling_edges_fail_without_publication(tmp_path, mutation, message):
    payload = _graph()
    mutation(payload)
    output = tmp_path / "frozen"
    with pytest.raises(ValueError, match=message):
        freeze_semantic_snapshot(_source(tmp_path, payload), output)
    assert not output.exists() or not list(output.iterdir())


@pytest.mark.parametrize("text", [
    '[]', '{"nodes":[]}', '{"nodes":{},"edges":[]}',
    '{"nodes":[],"edges":null}', '{"nodes":[],"nodes":[],"edges":[]}',
    '{"nodes":[],"edges":[',
])
def test_invalid_top_level_graph_structure_is_rejected(tmp_path, text):
    source = tmp_path / "source.json"
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        freeze_semantic_snapshot(source, tmp_path / "frozen")


def test_source_change_during_scan_is_rejected_before_publication(tmp_path, monkeypatch):
    source = _source(tmp_path, _graph())
    original_stream = snapshot_freeze._stream_items

    def mutate_after_edges(ijson, path, key):
        yield from original_stream(ijson, path, key)
        if key == "edges":
            with path.open("ab") as handle:
                handle.write(b"\n")

    monkeypatch.setattr(snapshot_freeze, "_stream_items", mutate_after_edges)
    output = tmp_path / "frozen"
    with pytest.raises(ValueError, match="changed during"):
        freeze_semantic_snapshot(source, output)
    assert not output.exists()


def test_numeric_overflow_is_rejected_even_in_excluded_geometry(tmp_path):
    source = tmp_path / "source.json"
    source.write_text('{"nodes":[{"id":"g","type":"Polygon",'
                      '"properties":{"coordinate":1e400}}],"edges":[]}', encoding="utf-8")
    output = tmp_path / "frozen"
    with pytest.raises(ValueError, match="Invalid graph export"):
        freeze_semantic_snapshot(source, output)
    assert not output.exists()


def test_manifest_publication_failure_removes_only_own_snapshot(tmp_path, monkeypatch):
    source = _source(tmp_path, _graph())
    real_write = snapshot_freeze._write_new

    def fail_manifest(path, data):
        if path.name == "freeze_manifest.json":
            raise OSError("simulated write failure")
        real_write(path, data)

    monkeypatch.setattr(snapshot_freeze, "_write_new", fail_manifest)
    output = tmp_path / "frozen"
    with pytest.raises(OSError, match="simulated"):
        freeze_semantic_snapshot(source, output)
    assert not list(output.iterdir())


def test_missing_optional_dependency_has_actionable_error(tmp_path, monkeypatch):
    source = _source(tmp_path, _graph())

    def unavailable(name):
        raise ImportError(name)

    monkeypatch.setattr(snapshot_freeze.importlib, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="dedicated project environment"):
        freeze_semantic_snapshot(source, tmp_path / "frozen")
