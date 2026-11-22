from __future__ import annotations

from copy import deepcopy

import pytest

from citygml_sg.retrieval.reference_graph import (
    BASE_EDGE_POLICY,
    build_room_candidates,
    fetch_reference_snapshot,
    normalize_semantic_node,
    reference_snapshot_stats,
)


def node(node_id: str, node_type: str, **properties: object) -> dict:
    return {"id": node_id, "type": node_type, "properties": properties}


def edge(source_id: str, relation: str, target_id: str, **properties: object) -> dict:
    return {"source_id": source_id, "target_id": target_id, "relation": relation,
            "properties": properties}


def candidates_by_id(snapshot: dict) -> dict:
    return {candidate["room_id"]: candidate for candidate in build_room_candidates(snapshot)}


def node_ids(candidate: dict) -> set[str]:
    return {item["id"] for item in candidate["nodes"]}


def test_sibling_room_furniture_is_not_reached_through_building_part() -> None:
    snapshot = {
        "nodes": [node("part", "BuildingPart"), node("a", "Room"), node("b", "Room"),
                  node("empty", "Room"), node("fa", "BuildingFurniture", gml_name="Sofa"),
                  node("fb", "BuildingFurniture", gml_name="Dining table"),
                  node("unscoped", "BuildingFurniture", gml_name="Chair")],
        "edges": [edge("part", "INTERIOR_ROOM", "a"), edge("part", "INTERIOR_ROOM", "b"),
                  edge("fa", "INSIDE", "a"), edge("fb", "INSIDE", "b"),
                  edge("part", "CONTAINS", "unscoped"),
                  edge("fa", "ADJACENT_TO", "fb", confidence=0.9, evidence_score=0.8)],
    }
    result = candidates_by_id(snapshot)
    assert set(result) == {"a", "b", "empty"}
    assert node_ids(result["a"]) == {"a", "fa"}
    assert node_ids(result["b"]) == {"b", "fb"}
    assert node_ids(result["empty"]) == {"empty"}
    assert result["empty"]["edges"] == []
    assert all(item["type"] != "ADJACENT_TO" for item in result["a"]["edges"])


def test_shared_boundary_and_opening_retain_membership_paths() -> None:
    snapshot = {
        "nodes": [node("a", "Room"), node("b", "Room"),
                  node("wall", "BoundarySurface", surface_type="InteriorWallSurface"),
                  node("door", "Opening", opening_type="Door")],
        "edges": [edge("a", "BOUNDED_BY", "wall"), edge("b", "BOUNDED_BY", "wall"),
                  edge("wall", "HAS_OPENING", "door")],
    }
    result = candidates_by_id(snapshot)
    for room_id in ("a", "b"):
        candidate = result[room_id]
        assert node_ids(candidate) == {room_id, "wall", "door"}
        assert set(candidate["membership"]["shared_node_ids"]) == {"wall", "door"}
        path = candidate["membership"]["node_sources"]["door"][0]
        assert path["source"] == "room_boundary_opening"
        assert len(path["edge_ids"]) == 2
        assert all(item["source"] != ("b" if room_id == "a" else "a")
                   for item in candidate["edges"])


def test_container_boundary_fallback_is_conditional_and_does_not_invent_edges() -> None:
    snapshot = {
        "nodes": [node("part", "BuildingPart"), node("direct", "Room"), node("fallback", "Room"),
                  node("room_wall", "BoundarySurface", surface_type="WallSurface"),
                  node("part_wall", "BoundarySurface", surface_type="WallSurface"),
                  node("window", "Opening", opening_type="Window")],
        "edges": [edge("part", "INTERIOR_ROOM", "direct"),
                  edge("part", "INTERIOR_ROOM", "fallback"),
                  edge("direct", "BOUNDED_BY", "room_wall"),
                  edge("part", "BOUNDED_BY", "part_wall"),
                  edge("part_wall", "HAS_OPENING", "window")],
    }
    result = candidates_by_id(snapshot)
    assert "part_wall" not in node_ids(result["direct"])
    assert node_ids(result["fallback"]) == {"fallback", "part_wall", "window"}
    sources = result["fallback"]["membership"]["node_sources"]["part_wall"]
    assert sources[0]["source"] == "container_boundary_fallback"
    assert len(sources[0]["edge_ids"]) == 2
    assert not any(item["type"] == "BOUNDED_BY" for item in result["fallback"]["edges"])


def test_connected_opening_host_wall_and_one_floor_adjacency_are_scoped() -> None:
    snapshot = {
        "nodes": [node("room", "Room"), node("door", "Opening", opening_type="Door"),
                  node("window", "Opening", opening_type="Window"),
                  node("wall", "BoundarySurface", surface_type="WallSurface"),
                  node("floor", "BoundarySurface", surface_type="FloorSurface"),
                  node("far_wall", "BoundarySurface", surface_type="WallSurface")],
        "edges": [edge("door", "CONNECTS", "room", source="connects_fallback",
                       confidence=0.85, evidence_score=0.85),
                  edge("door", "HOSTED_BY", "wall", source="semantic_boundary_opening",
                       confidence=1.0, evidence_score=1.0),
                  edge("window", "HOSTED_BY", "wall", confidence=1.0, evidence_score=1.0),
                  edge("wall", "ADJACENT_SURFACE", "floor", confidence=0.9, evidence_score=0.8),
                  edge("floor", "ADJACENT_SURFACE", "far_wall",
                       confidence=0.9, evidence_score=0.8)],
    }
    candidate = build_room_candidates(snapshot)[0]
    assert node_ids(candidate) == {"room", "door", "window", "wall", "floor"}
    assert candidate["membership"]["node_sources"]["wall"][0]["source"] == (
        "connected_opening_host_wall"
    )
    assert "surface_adjacency" in candidate["membership"]["node_sources"]["floor"][0]["source"]


def test_direct_installation_membership_and_unrestricted_pair_edges_are_not_confused() -> None:
    snapshot = {
        "nodes": [node("room", "Room"), node("other", "Room"),
                  node("column", "IntBuildingInstallation", gml_name="pillar"),
                  node("other_column", "BuildingInstallation", gml_name="column")],
        "edges": [edge("room", "ROOM_INSTALLATION", "column"),
                  edge("other", "ROOM_INSTALLATION", "other_column"),
                  edge("column", "TOUCHES", "other_column", confidence=0.9, evidence_score=0.7)],
    }
    result = candidates_by_id(snapshot)
    assert node_ids(result["room"]) == {"room", "column"}
    normalized = next(item for item in result["room"]["nodes"] if item["id"] == "column")
    assert normalized["label"] == "Installation"
    assert normalized["category"] == "column"


def test_metadata_mapping_and_missing_inferred_values_are_preserved() -> None:
    snapshot = {
        "nodes": [node("room", "Room"), node("a", "BuildingFurniture", gml_name="sofa"),
                  node("b", "BuildingFurniture", gml_name="table")],
        "edges": [edge("a", "INSIDE", "room"), edge("b", "INSIDE", "room"),
                  edge("a", "TOUCHES", "b", confidence=0.0, evidence_score=0.75,
                       method="bbox_aabb_v1", source="geometry", computed_at="2026-01-01"),
                  edge("a", "ABOVE", "b", confidence=0.8, method="bbox_vertical_v1")],
    }
    candidate = build_room_candidates(snapshot)[0]
    by_type = {item["type"]: item for item in candidate["edges"]}
    assert by_type["TOUCHES"]["gamma"] == 0.0
    assert by_type["TOUCHES"]["epsilon"] == 0.75
    assert by_type["TOUCHES"]["provenance"] == "geometry:bbox_aabb_v1"
    assert by_type["TOUCHES"]["metadata"]["computed_at"] == "2026-01-01"
    assert by_type["ABOVE"]["gamma"] == 0.8
    assert by_type["ABOVE"]["epsilon"] is None
    assert by_type["INSIDE"]["gamma"] == by_type["INSIDE"]["epsilon"] == 1.0
    assert by_type["INSIDE"]["provenance"] == BASE_EDGE_POLICY


def test_explicit_inferred_provenance_does_not_receive_base_unit_defaults() -> None:
    snapshot = {"nodes": [node("r", "Room"), node("f", "BuildingFurniture")],
                "edges": [edge("f", "INSIDE", "r", method="geometric_inside_v1")]}
    stored_edge = build_room_candidates(snapshot)[0]["edges"][0]
    assert stored_edge["gamma"] is None
    assert stored_edge["epsilon"] is None
    assert stored_edge["provenance"] == "geometric_inside_v1"


@pytest.mark.parametrize("invalid", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_edge_metadata_is_rejected(invalid: object) -> None:
    snapshot = {"nodes": [node("r", "Room"), node("f", "BuildingFurniture")],
                "edges": [edge("f", "INSIDE", "r", confidence=invalid)]}
    with pytest.raises(ValueError, match="gamma"):
        build_room_candidates(snapshot)


@pytest.mark.parametrize(
    ("raw", "label", "subtype", "category"),
    [
        (node("f", "BuildingFurniture", gml_name="M_Refrigerator 01"), "Furniture", None, "fridge"),
        (node("f", "BuildingFurniture", gml_name="TV stand"), "Furniture", None, "tv"),
        (node("i", "IntBuildingInstallation", gml_name="기둥"), "Installation", None, "column"),
        (node("o", "Opening", opening_type="Door"), "Opening", "Door", "door"),
        (node("b", "BoundarySurface", surface_type="GroundSurface"),
         "BoundarySurface", "FloorSurface", "floor"),
        (node("b", "BoundarySurface", surface_type="InteriorWallSurface"),
         "BoundarySurface", "WallSurface", "wall"),
        ({"id": "b", "label": "BoundarySurface", "category": "floor_surface"},
         "BoundarySurface", "FloorSurface", "floor"),
    ],
)
def test_semantic_normalization(raw: dict, label: str, subtype: str | None, category: str) -> None:
    normalized = normalize_semantic_node(raw)
    assert (normalized["label"], normalized["subtype"], normalized["category"]) == (
        label, subtype, category
    )


def test_distinct_kitchen_categories_and_original_surface_type_are_preserved() -> None:
    for category in ("sink", "counter", "cabinet", "storage"):
        assert normalize_semantic_node(node("f", "BuildingFurniture", gml_name=category))[
            "category"
        ] == category
    normalized = normalize_semantic_node(node("b", "BoundarySurface", surface_type="GroundSurface"))
    assert normalized["original_subtype"] == "GroundSurface"


@pytest.mark.parametrize("product_name", [
    "Brand-Model42-IntegratedFrenchDoorFridgeFreezer-NZ:Default:1234",
    "BuiltInRefrigerator:Model42",
    "IntegratedFRIDGEFreezer",
    "UnderCounterFreezer",
])
@pytest.mark.parametrize("field", ["gml_name", "attr_ifc_object_type", "attr_other_family"])
def test_furniture_compound_fridge_names_normalize_without_mutation(
    product_name: str, field: str,
) -> None:
    raw = node("appliance", "BuildingFurniture", **{field: product_name})
    original = deepcopy(raw)
    normalized = normalize_semantic_node(raw)
    assert normalized["category"] == "fridge"
    assert normalize_semantic_node(normalized) == normalized
    assert raw == original


@pytest.mark.parametrize("product_name", ["Refrigeratorish", "RefrigeratedPanel", "RefrigerationUnit"])
def test_fridge_name_detection_does_not_match_arbitrary_substrings(product_name: str) -> None:
    normalized = normalize_semantic_node(node("f", "Furniture", gml_name=product_name))
    assert normalized["category"] == product_name.lower()


def test_compound_fridge_rule_preserves_category_precedence_and_other_labels() -> None:
    explicit = node("f", "Furniture", category="storage", gml_name="IntegratedFridgeFreezer")
    assert normalize_semantic_node(explicit)["category"] == "storage"
    ordered = node("f", "Furniture", gml_name="storage IntegratedFridgeFreezer")
    assert normalize_semantic_node(ordered)["category"] == "storage"
    installation = node("i", "Installation", gml_name="IntegratedFridgeFreezer")
    assert normalize_semantic_node(installation)["category"] == "integratedfridgefreezer"
    assert normalize_semantic_node(node("r", "Room", gml_name="IntegratedFridgeFreezer"))[
        "category"
    ] == "room"
    # Splitting is limited to fridge synonyms, not a new general product taxonomy.
    assert normalize_semantic_node(node("f", "Furniture", gml_name="DiningTable"))[
        "category"
    ] == "diningtable"


def test_category_does_not_leak_from_identifier_or_room_name() -> None:
    assert normalize_semantic_node(node("sofa_E102", "BuildingFurniture"))["category"] == "unknown"
    assert normalize_semantic_node(node("r", "Room", gml_name="kitchen_E102"))["category"] == "room"
    canonical = {"id": "f", "label": "Furniture", "category": "cabinet",
                 "properties": {"gml_name": "sink"}}
    assert normalize_semantic_node(canonical)["category"] == "cabinet"


def test_canonical_records_stable_ids_and_input_are_preserved() -> None:
    snapshot = {
        "nodes": [{"id": "r", "label": "Room"},
                  {"id": "a", "label": "Furniture", "category": "sofa"},
                  {"id": "b", "label": "Furniture", "category": "table"}],
        "edges": [{"id": "given", "source": "r", "target": "a", "type": "CONTAINS"},
                  {"source": "r", "target": "b", "type": "CONTAINS"},
                  {"source": "a", "target": "b", "type": "TOUCHES", "gamma": 0.9,
                   "epsilon": 0.8, "provenance": {"producer": "fixture"}, "computed_at": "first"}],
    }
    original = deepcopy(snapshot)
    first = build_room_candidates(snapshot)
    assert snapshot == original
    assert "given" in {item["id"] for item in first[0]["edges"]}
    snapshot["nodes"].reverse()
    snapshot["edges"].reverse()
    for item in snapshot["edges"]:
        if item["type"] == "TOUCHES":
            item["computed_at"] = "second"
    second = build_room_candidates(snapshot)
    assert {item["id"] for item in first[0]["edges"]} == {item["id"] for item in second[0]["edges"]}
    touch = next(item for item in first[0]["edges"] if item["type"] == "TOUCHES")
    assert touch["metadata"]["provenance"] == {"producer": "fixture"}
    assert isinstance(touch["provenance"], str)


def test_snapshot_inventory_separates_geometry_from_search_semantics() -> None:
    snapshot = {
        "nodes": [node("part", "BuildingPart"), node("r", "Room"),
                  node("p", "Position"), node("unscoped", "BuildingFurniture")],
        "edges": [edge("part", "INTERIOR_ROOM", "r")],
    }
    stats = reference_snapshot_stats(snapshot)
    assert stats["room_count"] == 1
    assert stats["total_node_count"] == 4
    assert stats["geometry_node_count"] == 1
    assert stats["search_semantic_node_count"] == 1
    assert stats["search_semantic_edge_count"] == 0


def test_duplicate_nodes_and_dangling_edges_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate reference node"):
        build_room_candidates({"nodes": [node("r", "Room"), node("r", "Room")], "edges": []})
    with pytest.raises(ValueError, match="unknown endpoint"):
        build_room_candidates({"nodes": [node("r", "Room")],
                               "edges": [edge("r", "CONTAINS", "missing")]})


def test_fetch_prefetches_semantics_but_preserves_full_inventory() -> None:
    class Result:
        def __init__(self, rows: list[dict]) -> None:
            self.rows = rows

        def data(self) -> list[dict]:
            return self.rows

    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        def run(self, query: str, **parameters: object) -> Result:
            self.calls.append((query, parameters))
            assert not any(word in query.upper().split()
                           for word in ("CREATE", "MERGE", "SET", "DELETE"))
            assert "elementId" not in query
            if "count(n)" in query:
                return Result([{"node_type": "Room", "count": 2},
                               {"node_type": "Position", "count": 1000}])
            assert "Position" not in parameters["types"]
            if "properties(n)" in query:
                return Result([node("a", "Room"), node("b", "Room")])
            return Result([])

    session = Session()
    snapshot = fetch_reference_snapshot(session)
    assert len(session.calls) == 3
    stats = reference_snapshot_stats(snapshot)
    assert stats["total_node_count"] == 1002
    assert stats["geometry_node_count"] == 1000
    assert stats["prefetched_node_count"] == stats["room_count"] == 2
