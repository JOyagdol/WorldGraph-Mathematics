from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from citygml_sg.retrieval.reference_graph import build_room_candidates
from citygml_sg.retrieval.room_scope import (
    MEMBERSHIP_POLICY,
    SCOPE_VERSION,
    build_room_hop_subgraphs,
    canonical_node_id,
    canonical_node_type,
    edge_endpoints,
    extract_room_hop_subgraph,
)


def node(node_id: str, node_type: str) -> dict:
    return {"id": node_id, "type": node_type}


def edge(source: str, target: str, relation: str = "CONTAINS", **metadata: object) -> dict:
    return {"source": source, "target": target, "type": relation, **metadata}


def test_undirected_hops_retain_directed_induced_cross_edges_and_parallel_records() -> None:
    snapshot = {
        "nodes": [node("r", "Room"), node("f", "Furniture"), node("wall", "WallSurface"),
                  node("door", "Door"), node("remote", "Installation")],
        "edges": [edge("f", "r", "INSIDE"), edge("r", "wall", "BOUNDED_BY"),
                  edge("f", "wall", "ATTACHED_TO", id="cross-1"),
                  edge("f", "wall", "TOUCHES", id="cross-2"),
                  edge("wall", "door", "HAS_OPENING"), edge("door", "remote", "TOUCHES")],
    }
    result = extract_room_hop_subgraph(snapshot, "r", 1)
    assert result["membership"]["node_hops"] == {"f": 1, "r": 0, "wall": 1}
    assert result["edges"] == snapshot["edges"][:4]
    # The two edges joining distance-one nodes are not BFS-tree edges.
    assert [item["id"] for item in result["edges"] if "id" in item] == ["cross-1", "cross-2"]
    at_two = extract_room_hop_subgraph(snapshot, "r", 2)
    assert at_two["membership"]["node_hops"]["door"] == 2
    assert "remote" not in at_two["membership"]["node_hops"]
    assert at_two["edges"] == snapshot["edges"][:5]


def test_shared_parent_reaches_sibling_rooms_and_objects_without_ownership_filter() -> None:
    snapshot = {
        "nodes": [node("parent", "BuildingPart"), node("a", "Room"), node("b", "Room"),
                  node("unscoped", "Furniture"), node("fb", "Furniture")],
        "edges": [edge("parent", "a", "INTERIOR_ROOM"),
                  edge("parent", "b", "INTERIOR_ROOM"),
                  edge("parent", "unscoped"), edge("b", "fb", "INTERIOR_FURNITURE")],
    }
    result = extract_room_hop_subgraph(snapshot, "a", 2)
    assert result["membership"]["node_hops"] == {"a": 0, "b": 2, "parent": 1, "unscoped": 2}
    assert result["membership"]["other_room_ids"] == ["b"]
    assert "fb" in extract_room_hop_subgraph(snapshot, "a", 3)["membership"]["node_hops"]
    # This explicit counterexample must not be silently made equivalent to the
    # existing typed ownership adapter, which remains a separate policy.
    typed_a = next(item for item in build_room_candidates(snapshot) if item["room_id"] == "a")
    assert {item["id"] for item in typed_a["nodes"]} == {"a"}


def test_zero_hops_self_loops_isolated_rooms_and_cycles() -> None:
    snapshot = {
        "nodes": [node("z", "Room"), node("r", "Room"), node("a", "Furniture"),
                  node("b", "Opening")],
        "edges": [edge("r", "r", "SELF", id="self"), edge("r", "a"),
                  edge("a", "b"), edge("b", "r")],
    }
    zero = extract_room_hop_subgraph(snapshot, "r", 0)
    assert zero["nodes"] == [snapshot["nodes"][1]]
    assert zero["edges"] == [snapshot["edges"][0]]
    many = build_room_hop_subgraphs(snapshot, 20)
    assert [item["room_id"] for item in many] == ["r", "z"]
    assert many[0]["membership"]["node_hops"] == {"a": 1, "b": 1, "r": 0}
    assert many[1]["membership"]["node_hops"] == {"z": 0}
    assert many[1]["edges"] == []
    assert many[0] == extract_room_hop_subgraph(snapshot, "r", 20)


def test_records_and_metadata_are_preserved_without_aliasing_or_fabrication() -> None:
    snapshot = {
        "snapshot_name": "synthetic-only",
        "nodes": [{"properties": {"gml_id": "r", "object_type": "Room", "custom": [1]}},
                  {"node_id": "f", "labels": ["CityObject", "BuildingFurniture"]}],
        "edges": [{"source_id": "f", "target_id": "r", "relation": "INSIDE",
                   "properties": {"gamma": 0.2, "custom": {"value": "stored"}}}],
    }
    original = deepcopy(snapshot)
    result = extract_room_hop_subgraph(snapshot, "r", 1)
    assert snapshot == original
    assert result["nodes"] == original["nodes"]
    assert result["edges"] == original["edges"]
    assert result["membership"]["policy"] == MEMBERSHIP_POLICY
    assert result["membership"]["version"] == SCOPE_VERSION
    result["nodes"][0]["properties"]["custom"].append(2)
    result["edges"][0]["properties"]["custom"]["value"] = "changed"
    assert snapshot == original


def test_anonymous_identical_parallel_edges_are_retained_and_hash_is_content_exact() -> None:
    snapshot = {"nodes": [node("r", "Room"), node("f", "Furniture")],
                "edges": [edge("r", "f"), edge("r", "f")]}
    result = extract_room_hop_subgraph(snapshot, "r")
    digest = sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    assert len(result["edges"]) == 2
    assert result["input_snapshot_sha256"] == digest == result["subgraph_sha256"]
    assert "id" not in result["edges"][0]
    snapshot["edges"][0]["properties"] = {"timestamp": "new"}
    assert extract_room_hop_subgraph(snapshot, "r")["input_snapshot_sha256"] != digest


@pytest.mark.parametrize(("raw_type", "canonical"), [
    ("Building", "Building"), ("building_part", "BuildingPart"), ("room", "Room"),
    ("BuildingFurniture", "Furniture"), ("IntBuildingInstallation", "Installation"),
    ("Window", "Opening"), ("InteriorWallSurface", "BoundarySurface"),
])
def test_supported_semantic_aliases(raw_type: str, canonical: str) -> None:
    raw = {"properties": {"gml_id": "stored", "node_type": raw_type}}
    assert canonical_node_id(raw) == "stored"
    assert canonical_node_type(raw) == canonical
    assert edge_endpoints({"source_id": "a", "target_id": "b"}) == ("a", "b")


@pytest.mark.parametrize("h", [True, False, -1, 1.5, "2", None])
def test_invalid_hop_radius_is_rejected_even_without_rooms(h: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        build_room_hop_subgraphs({"nodes": [], "edges": []}, h)


@pytest.mark.parametrize("unsupported", ["Geometry", "Position", "Unrecognized"])
def test_out_of_domain_nodes_are_rejected_even_when_disconnected(unsupported: str) -> None:
    snapshot = {"nodes": [node("r", "Room"), node("unsupported", unsupported)], "edges": []}
    with pytest.raises(ValueError, match="Unsupported semantic graph node type"):
        extract_room_hop_subgraph(snapshot, "r", 0)


@pytest.mark.parametrize(("snapshot", "message"), [
    ({"nodes": [node("r", "Room"), node("r", "Room")], "edges": []}, "Duplicate.*node ID"),
    ({"nodes": [node("r", "Room")], "edges": [edge("r", "unknown")]}, "unknown endpoint"),
    ({"nodes": [node("r", "Room")],
      "edges": [edge("r", "r", id="dup"), edge("r", "r", id="dup")]}, "Duplicate.*edge ID"),
    ({"nodes": [node("r", "Room")], "edges": [edge("r", "r", id=12)]}, "edge ID"),
    ({"nodes": [node("r", "Room")], "edges": [edge("r", "r", id="")]}, "edge ID"),
    ({"nodes": [node("r", "Room")], "edges": [edge("r", "r", id=False)]}, "edge ID"),
    ({"nodes": [node("r", "Room")],
      "edges": [{"source": "r", "target": "r", "type": ""}]}, "relation type"),
    ({"nodes": [node("r", "Room")], "edges": [{"source": "r", "target": None}]}, "target IDs"),
])
def test_invalid_graph_identity_and_edges_are_rejected(snapshot: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        extract_room_hop_subgraph(snapshot, "r")


def test_root_must_be_an_existing_room_and_empty_batch_is_valid() -> None:
    snapshot = {"nodes": [node("f", "Furniture")], "edges": []}
    with pytest.raises(ValueError, match="not a Room"):
        extract_room_hop_subgraph(snapshot, "f")
    with pytest.raises(ValueError, match="identify a node"):
        extract_room_hop_subgraph(snapshot, "missing")
    assert build_room_hop_subgraphs(snapshot) == []
    assert build_room_hop_subgraphs({"nodes": [], "edges": []}) == []


def test_nonfinite_metadata_cannot_produce_a_snapshot_identity() -> None:
    snapshot = {"nodes": [node("r", "Room")], "edges": [], "metadata": float("nan")}
    with pytest.raises(ValueError, match="finite JSON-compatible data"):
        extract_room_hop_subgraph(snapshot, "r")


@pytest.mark.parametrize("labels", [None, "Room", [{}]])
def test_malformed_labels_raise_a_validation_error(labels: object) -> None:
    snapshot = {"nodes": [{"id": "r", "labels": labels}], "edges": []}
    with pytest.raises(ValueError, match="labels must be a list of strings"):
        extract_room_hop_subgraph(snapshot, "r")
