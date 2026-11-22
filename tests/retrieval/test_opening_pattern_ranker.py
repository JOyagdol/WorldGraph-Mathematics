from copy import deepcopy
from fractions import Fraction

import pytest

from citygml_sg.retrieval.evaluation.oracle_metrics import evaluate_retrieval
from citygml_sg.retrieval.graph_matching.joint_matcher import rank_rooms
from citygml_sg.retrieval.opening_pattern_ranker import (
    gate_reason, match_probability, opening_pattern, prepare_opening_patterns, rank_opening_evidence,
)


CONFIG = {"version": "joint_evidence_v1_proposed"}


def graph(categories, walls, chair=False):
    nodes = [{"id": wall, "label": "BoundarySurface", "category": "wall", "scoring": False,
              "weight": 1} for wall in sorted(set(walls))]
    edges = []
    for i, (category, wall) in enumerate(zip(categories, walls, strict=True)):
        opening_id = f"o{i}"
        nodes.append({"id": opening_id, "label": "Opening", "category": category,
                      "scoring": True, "weight": 1})
        edges.append({"id": f"e{i}", "type": "HOSTED_BY", "source": opening_id,
                      "target": wall, "weight": 1, "gamma": 1, "epsilon": 1,
                      "provenance": "synthetic"})
    if chair:
        nodes.append({"id": "chair", "label": "Furniture", "category": "chair",
                      "scoring": True, "weight": 1})
    return {"nodes": nodes, "edges": edges}


def candidates():
    def candidate(room, contents):
        return {"room_id": room, "room_name": room, **contents,
                "nodes": [{"id": room, "label": "Room", "category": "room"}, *contents["nodes"]]}
    return [
        candidate("a", graph(["door", "door", "window"], ["w"] * 3, chair=True)),
        candidate("b", graph(["door", "door"], ["w", "w"])),
    ]


@pytest.mark.parametrize("variant,expected,calls", [
    ("full", "a", ["objects_only", "full"]),
    ("objects_only", "a", ["objects_only"]),
    ("relations_only", "b", ["relations_only"]),
])
def test_full_keeps_object_priority_and_relations_only_uses_no_object_score(variant, expected, calls):
    seen = []
    def checked(*args, **kwargs):
        seen.append(kwargs["variant"])
        return rank_rooms(*args, **kwargs)

    query = graph(["door", "door"], ["wall", "wall"], chair=True)
    pool = candidates()
    result = rank_opening_evidence(query, pool, config=CONFIG, variant=variant,
                                   reference_patterns=prepare_opening_patterns(pool), core_rank=checked)
    assert seen == calls
    assert result["ranking"][0]["room_id"] == expected
    assert evaluate_retrieval(result, expected)["hit_at_1"] == 1
    if variant == "relations_only":
        assert all(row["rank_key"][0] == row["object_numerator"] == 0 for row in result["ranking"])
    if variant != "objects_only":
        by_room = {r["room_id"]: r["rank_key"][1] for r in result["ranking"]}
        assert by_room == {"a": [1, 3], "b": [1, 1]}


def test_single_opening_can_break_an_object_tie_but_other_scored_objects_close_the_gate():
    bare = graph(["door"], ["w"])
    rich = graph(["door"], ["w"], chair=True)
    assert gate_reason(bare, opening_pattern(bare)) == "complete_single_opening"
    assert gate_reason(rich, opening_pattern(rich)) == "closed"
    assert rank_opening_evidence(bare, candidates(), config=CONFIG, variant="full")["ranking"][0]["room_id"] == "b"
    fallback = rank_opening_evidence(rich, candidates(), config=CONFIG, variant="full")
    assert fallback["component_statuses"] == {"objects_only": "completed"}
    assert fallback["ranking"][0]["room_id"] == "a"
    absent = rank_opening_evidence(rich, candidates(), config=CONFIG, variant="relations_only",
                                  core_rank=lambda *a, **k: pytest.fail("The closed gate has no active score"))
    assert absent["ranking"] == [] and absent["status"] == "abstained_no_evidence"
    assert evaluate_retrieval(absent, "a")["mtr"] == 3


def test_gates_use_distinct_openings_and_injective_host_identity():
    one = graph(["door"], ["w"])
    one["edges"] += [dict(one["edges"][0], id="duplicate_edge")]
    assert gate_reason(one, opening_pattern(one)) == "closed"
    split = graph(["door", "opening"], ["a", "b"])
    assert gate_reason(split, opening_pattern(split)) == "closed"
    assert match_probability(opening_pattern(split), [("door", "x"), ("window", "y")]) == Fraction(1, 2)
    assert match_probability(opening_pattern(split), [("door", "x"), ("window", "x")]) == 0
    assert match_probability(opening_pattern(split), []) == 0


def test_failed_auxiliary_objective_is_not_a_valid_full_answer():
    def partial(anchors, pool, *, config, variant):
        if variant == "full":
            return {"status": "solver_incomplete", "ranking": []}
        return rank_rooms(anchors, pool, config=config, variant=variant)

    result = rank_opening_evidence(graph(["door"], ["w"]), candidates(),
                                   config=CONFIG, variant="full", core_rank=partial)
    assert result["status"] == "solver_incomplete" and not result["ranking"]
    assert result["failure_stage"] == "full"
    assert evaluate_retrieval(result, "a")["hit_at_1"] is None


def test_invalid_rank_key_and_reference_are_rejected():
    query = graph(["door"], ["w"])
    result = rank_opening_evidence(query, candidates(), config=CONFIG, variant="relations_only")
    forged = deepcopy(result)
    forged["ranking"][0]["rank_key"][1] = [1, 0]
    with pytest.raises(ValueError, match="rational"):
        evaluate_retrieval(forged, "a")
    with pytest.raises(ValueError, match="pool differs"):
        rank_opening_evidence(query, candidates(), config=CONFIG,
                              variant="full", reference_patterns={"a": []})
