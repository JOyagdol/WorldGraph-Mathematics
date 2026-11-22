"""Reference-vocabulary selection checks; no dataset performance claims."""

from copy import deepcopy

import pytest

from citygml_sg.ovg.validation.oracle_observation import normalize_observation, select_anchors
from citygml_sg.ovg.validation.reference_anchors import select_reference_anchors


def _observation():
    return normalize_observation({
        "objects": [
            {"alias": "chair", "type": "Furniture", "category": "chair", "confidence": 0.5},
            {"alias": "plant", "type": "Furniture", "category": "plant"},
            {"alias": "floor", "type": "FloorSurface"},
        ],
        "relations": [
            {"source": "chair", "type": "ATTACHED_TO", "target": "floor", "weight": 0.5},
            {"source": "plant", "type": "ATTACHED_TO", "target": "floor", "weight": 1},
            {"source": "chair", "type": "INSIDE", "target": "Room", "weight": 0.4},
        ],
    })


def test_inclusive_node_and_edge_threshold_and_unknown_category_exclusion():
    observation = _observation()
    original = deepcopy(observation)
    anchors = select_reference_anchors(observation, {"chair", "floor"}, 0.5)
    assert {n["id"] for n in anchors["nodes"]} == {"chair", "floor", "__context__Room"}
    assert [e["id"] for e in anchors["edges"]] == ["obs_edge_0"]
    assert anchors["excluded_nodes"][0]["reason"] == "category_absent_from_reference"
    assert {e["id"]: e["reason"] for e in anchors["excluded_edges"]} == {
        "obs_edge_1": "excluded_endpoint", "obs_edge_2": "below_anchor_threshold",
    }
    assert observation == original
    # The earlier proposed-method contract remains unchanged.
    old = select_anchors(observation, 0.5)
    assert "plant" in {n["id"] for n in old["nodes"]}
    assert "obs_edge_2" in {e["id"] for e in old["edges"]}


def test_zero_threshold_retains_zero_weight_atoms_and_adds_unknown_room():
    observation = normalize_observation({
        "objects": [{"alias": "x", "type": "Furniture", "category": "chair", "confidence": 0},
                    {"alias": "y", "type": "Furniture", "category": "chair", "confidence": 0}],
        "relations": [{"source": "x", "target": "y", "type": "ABOVE", "weight": 0}],
    })
    anchors = select_reference_anchors(observation, ["chair"], 0)
    assert len(anchors["nodes"]) == 3 and len(anchors["edges"]) == 1
    assert anchors["nodes"][0]["label"] == "Room"
    empty = select_reference_anchors(observation, [], 1)
    assert [n["label"] for n in empty["nodes"]] == ["Room"]
    assert empty["edges"] == []


def test_reference_vocabulary_is_global_and_not_fuzzy_opening_compatibility():
    observation = normalize_observation({"objects": [
        {"alias": "o", "type": "Opening", "category": "opening"},
        {"alias": "f", "type": "Furniture", "category": "fridge"},
    ]})
    # A fridge elsewhere in the snapshot remains an anchor, even if absent in
    # a particular Room. Generic opening is not silently expanded to window.
    anchors = select_reference_anchors(observation, ["window", "fridge"])
    assert {n["id"] for n in anchors["nodes"]} == {"f", "__context__Room"}
    same = select_reference_anchors(observation, ["fridge", "window", "fridge"])
    assert same == anchors


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        select_reference_anchors(_observation(), ["chair"], threshold)


def test_invalid_graphs_fail_before_selection():
    observation = _observation()
    observation["edges"][0]["target"] = "missing"
    with pytest.raises(ValueError, match="unknown endpoint"):
        select_reference_anchors(observation, ["chair"])
    observation = _observation()
    observation["nodes"][-1]["weight"] = 0
    with pytest.raises(ValueError, match="unit weight"):
        select_reference_anchors(observation, ["chair"])
    with pytest.raises(ValueError, match="normalized categories"):
        select_reference_anchors(_observation(), ["Chair"])


@pytest.mark.parametrize("vocabulary", [None, 123, [None], [["chair"]], [{"chair": True}]])
def test_invalid_vocabulary_raises_value_error(vocabulary):
    with pytest.raises(ValueError, match="reference_categories"):
        select_reference_anchors(_observation(), vocabulary)


def test_generator_vocabulary_is_supported_without_changing_selection():
    generated = select_reference_anchors(_observation(), (item for item in ["chair", "floor"]))
    assert generated == select_reference_anchors(_observation(), ["chair", "floor"])


@pytest.mark.parametrize("room_fields", [
    {"category": "kitchen"}, {"subtype": "office"}, {"scoring": True},
])
def test_existing_room_must_be_an_unknown_generic_placeholder(room_fields):
    observation = _observation()
    room = next(item for item in observation["nodes"] if item["label"] == "Room")
    room.update(room_fields)
    with pytest.raises(ValueError, match="unknown-Room placeholder requires"):
        select_reference_anchors(observation, ["chair", "floor"])


def test_normalized_explicit_room_with_category_context_is_rejected():
    observation = normalize_observation({"objects": [
        {"alias": "here", "type": "Room", "category": "kitchen", "subtype": "office"},
    ]})
    with pytest.raises(ValueError, match="unknown-Room placeholder requires"):
        select_reference_anchors(observation, ["room", "kitchen"])


@pytest.mark.parametrize("opening_type", ["Door", "Window"])
def test_specific_opening_with_generic_category_requires_canonicalization(opening_type):
    observation = normalize_observation({"objects": [
        {"alias": "opening", "type": opening_type, "category": "opening"},
    ]})
    original = deepcopy(observation)
    with pytest.raises(ValueError, match="canonical category mismatch"):
        select_reference_anchors(observation, [opening_type.casefold()])
    # Neither the shared normalizer nor the legacy anchor policy is rewritten.
    assert observation == original
    assert select_anchors(observation)["nodes"][0]["category"] == "opening"
    canonical = normalize_observation({"objects": [
        {"alias": "opening", "type": opening_type, "category": opening_type.casefold()},
    ]})
    result = select_reference_anchors(canonical, [opening_type.casefold()])
    assert "opening" in {item["id"] for item in result["nodes"]}


@pytest.mark.parametrize("field", ["ground_truth", "gt_room_id", "room_id", "gml_id"])
def test_direct_normalized_input_cannot_carry_map_or_gt_identifiers(field):
    observation = _observation()
    observation["metadata"] = {"nested": [{field: "synthetic-target"}]}
    with pytest.raises(ValueError, match="map/GT identifiers"):
        select_reference_anchors(observation, ["chair", "floor"])


def test_unhashable_labels_and_endpoints_raise_value_error():
    observation = _observation()
    observation["nodes"][0]["label"] = ["Furniture"]
    with pytest.raises(ValueError, match="unsupported normalized observation label"):
        select_reference_anchors(observation, ["chair"])
    observation = _observation()
    observation["edges"][0]["source"] = {"id": "chair"}
    with pytest.raises(ValueError, match="edge.source"):
        select_reference_anchors(observation, ["chair"])


def test_excessively_large_numeric_threshold_raises_value_error():
    with pytest.raises(ValueError, match="finite number"):
        select_reference_anchors(_observation(), ["chair"], 10 ** 400)
