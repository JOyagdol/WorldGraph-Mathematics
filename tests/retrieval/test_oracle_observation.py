from copy import deepcopy

import pytest

from citygml_sg.ovg.validation.oracle_observation import normalize_observation, select_anchors


def test_typed_floor_reuses_explicit_observation_without_phantom_instance():
    result = normalize_observation({"objects": [
        {"alias": "chair", "category": "chair"},
        {"alias": "floor_1", "type": "FloorSurface", "category": "floor_surface"},
    ], "relations": [["chair", "ATTACHED_TO", "FloorSurface"]]})
    assert len(result["nodes"]) == 2
    assert result["edges"][0]["target"] == "floor_1"
    assert result["typed_endpoint_resolution"] == {"FloorSurface": "floor_1"}


def test_ambiguous_typed_context_requires_explicit_alias():
    raw = {"objects": [
        {"alias": "chair", "category": "chair"},
        {"alias": "wall_1", "type": "WallSurface", "category": "wall"},
        {"alias": "wall_2", "type": "WallSurface", "category": "wall"},
    ], "relations": [["chair", "ADJACENT_TO", "WallSurface"]]}
    with pytest.raises(ValueError, match="Ambiguous typed endpoint"):
        normalize_observation(raw)


@pytest.mark.parametrize("item", [
    {"alias": "d", "type": "Door", "category": "window"},
    {"alias": "d", "type": "Opening", "subtype": "Door", "category": "window"},
    {"alias": "f", "type": "FloorSurface", "category": "wall"},
    {"alias": "f", "type": "FloorSurface", "subtype": "WallSurface", "category": "wall"},
])
def test_conflicting_semantic_type_is_rejected(item):
    with pytest.raises(ValueError, match="conflict"):
        normalize_observation({"objects": [item]})


@pytest.mark.parametrize("relations", [
    [["a", "TOUCHES", "b"], ["b", "TOUCHES", "a"]],
    [["a", "ABOVE", "b"], ["b", "BELOW", "a"]],
])
def test_same_semantic_relation_cannot_be_counted_twice(relations):
    with pytest.raises(ValueError, match="Duplicate observation relation"):
        normalize_observation({"objects": [{"alias": "a", "category": "chair"},
                                           {"alias": "b", "category": "chair"}],
                               "relations": relations})


def observation():
    return {"objects": [{"alias": "chair1", "type": "BuildingFurniture", "category": "chair",
                         "confidence": 0.5, "visibility": 0.8},
                        {"alias": "chair2", "type": "BuildingFurniture", "category": "chair"}],
            "relations": [["chair1", "ATTACHED_TO", "FloorSurface"]]}


def test_instances_and_typed_context_are_preserved():
    graph = normalize_observation(observation())
    assert len(graph["nodes"]) == 3
    assert graph["nodes"][0]["weight"] == pytest.approx(0.4)
    context = graph["nodes"][2]
    assert context["provenance"] == "implicit_typed_context"
    assert context["scoring"] is False
    assert graph["edges"][0]["weight_source"] == "default_unit"


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_observation_confidence_fails(value):
    raw = observation()
    raw["objects"][0]["confidence"] = value
    with pytest.raises(ValueError):
        normalize_observation(raw)


def test_alias_and_endpoint_validation():
    raw = observation()
    raw["objects"][1]["alias"] = "chair1"
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_observation(raw)
    raw = observation()
    raw["relations"][0][2] = "misspelled_floor"
    with pytest.raises(ValueError, match="Unknown"):
        normalize_observation(raw)


def test_anchor_deletion_removes_incident_relations():
    normalized = normalize_observation(observation())
    before = deepcopy(normalized)
    anchors = select_anchors(normalized, 0.5)
    assert "chair1" not in {n["id"] for n in anchors["nodes"]}
    assert anchors["edges"] == []
    assert anchors["excluded_edges"][0]["reason"] == "excluded_endpoint"
    assert normalized == before


@pytest.mark.parametrize("identifier", ["gml_id", "target_id", "target_room_id", "id"])
def test_map_ids_cannot_enter_inference(identifier):
    raw = observation()
    raw["objects"][0][identifier] = "secret-ground-truth"
    with pytest.raises(ValueError):
        normalize_observation(raw)


def test_weights_are_observation_weights_not_legacy_bonuses():
    raw = observation()
    raw["objects"][0]["weight"] = 4.0
    graph = normalize_observation(raw)
    assert graph["nodes"][0]["weight"] == pytest.approx(0.4)
    assert graph["warnings"]
