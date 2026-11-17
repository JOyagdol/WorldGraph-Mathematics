"""Counterexamples for the proposed matcher; no database or private data needed."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import math
import random

import pytest

from citygml_sg.retrieval.graph_matching.joint_matcher import rank_rooms


def _node(alias: str, category: str = "chair", *, label: str = "Furniture",
          subtype: str | None = None, weight: float = 1.0) -> dict:
    return {"id": alias, "label": label, "category": category, "subtype": subtype,
            "weight": weight, "scoring": label in {"Furniture", "Installation", "Opening"}}


def _edge(source: str, target: str, relation: str = "ADJACENT_TO", *,
          edge_id: str = "e", weight: float = 1.0, quality: float | None = 1.0) -> dict:
    return {"id": edge_id, "source": source, "target": target, "type": relation,
            "weight": weight, "weight_source": "explicit", "gamma": quality,
            "epsilon": 1.0, "provenance": "synthetic_fixture"}


def _observation(nodes: list[dict], edges: list[dict] | None = None) -> dict:
    return {"nodes": nodes, "edges": edges or [], "excluded_nodes": [], "excluded_edges": []}


def _room(room_id: str, nodes: list[dict], edges: list[dict] | None = None) -> dict:
    return {"room_id": room_id, "room_name": room_id,
            "nodes": [_node(room_id, "room", label="Room"), *nodes], "edges": edges or [], "membership": {}}


def _brute_objective(observation: dict, room: dict, pool: list[dict], variant: str) -> float:
    """Independent enumeration of small Furniture fixtures, including None."""
    alpha = 0 if variant == "relations_only" else 1
    beta = 0 if variant == "objects_only" else 1
    labels = {"Furniture", "Installation", "Opening"}
    categories = {node["category"] for node in observation["nodes"]}
    rho = {
        category: 1.0 if variant == "uniform_category" else math.log1p(len(pool) / (1 + sum(
            any(node["label"] in labels and node["category"] == category for node in candidate["nodes"])
            for candidate in pool
        ))) for category in categories
    }
    alternatives = [[None, *[stored["id"] for stored in room["nodes"]
                            if stored["label"] == observed["label"] and stored["category"] == observed["category"]]]
                    for observed in observation["nodes"]]
    compatible_types = {"ADJACENT_TO": {"ADJACENT_TO", "TOUCHES", "INTERSECTS"},
                        "TOUCHES": {"TOUCHES", "INTERSECTS"}}
    symmetric = {"ADJACENT_TO", "TOUCHES", "INTERSECTS", "ADJACENT_SURFACE"}
    inverse = {"ABOVE": "BELOW", "BELOW": "ABOVE"}
    best = 0.0
    for choices in product(*alternatives):
        selected = [choice for choice in choices if choice is not None]
        if len(selected) != len(set(selected)):
            continue
        mapping = dict(zip((node["id"] for node in observation["nodes"]), choices))
        obj = sum(alpha * node["weight"] * rho[node["category"]]
                  for node in observation["nodes"] if node["scoring"] and mapping[node["id"]] is not None)
        rel = 0.0
        for observed in observation["edges"]:
            a, b = mapping[observed["source"]], mapping[observed["target"]]
            qualities = []
            for stored in room["edges"]:
                forward = stored["source"] == a and stored["target"] == b
                reverse = stored["source"] == b and stored["target"] == a
                allowed = stored["type"] in compatible_types.get(observed["type"], {observed["type"]})
                if (allowed and (forward or (reverse and observed["type"] in symmetric and stored["type"] in symmetric))) or (
                    reverse and inverse.get(observed["type"]) == stored["type"]
                ):
                    qualities.append(1.0 if variant == "boolean_metadata" else stored["gamma"] * stored["epsilon"])
            rel += beta * observed["weight"] * max(qualities, default=0)
        best = max(best, obj + rel)
    return best


@pytest.mark.parametrize("variant", ["full", "objects_only", "relations_only", "uniform_category", "boolean_metadata"])
def test_exact_solver_agrees_with_independent_small_brute_force(variant: str) -> None:
    rng = random.Random(847)
    for _ in range(5):
        observation = _observation(
            [_node("q0", weight=0.2), _node("q1", weight=0.9), _node("q2", "table", weight=0.6)],
            [_edge("q0", "q1", "ABOVE", edge_id="q01", weight=0.7),
             _edge("q1", "q2", edge_id="q12", weight=0.4)],
        )
        pool = []
        for room_id in ("r1", "r2"):
            nodes = [_node("a"), _node("b"), _node("c"), _node("t", "table")]
            edges = [_edge(a, b, rng.choice(["ABOVE", "BELOW", "ADJACENT_TO", "INTERSECTS"]),
                           edge_id=f"{a}_{b}", quality=rng.choice([0.0, 0.25, 0.8, 1.0]))
                     for a, b in [("a", "b"), ("a", "t"), ("c", "b"), ("b", "t")]]
            pool.append(_room(room_id, nodes, edges))
        report = rank_rooms(observation, pool, variant=variant)
        for row in report["ranking"]:
            candidate = next(item for item in pool if item["room_id"] == row["room_id"])
            assert row["numerator"] == pytest.approx(_brute_objective(observation, candidate, pool, variant))
            assert row["solver"]["optimality_proven"]
            assert row["solver"]["upper_bound"] == pytest.approx(row["score"])


def test_duplicate_categories_remain_injective_and_candidate_surplus_cannot_add_score() -> None:
    observation = _observation([_node("low", weight=0.25), _node("high", weight=1.0)])
    pool = [_room("one", [_node("a")]), _room("many", [_node(str(i)) for i in range(8)])]
    rows = {row["room_id"]: row for row in rank_rooms(observation, pool, variant="objects_only")["ranking"]}
    assert rows["one"]["score"] == pytest.approx(0.8)
    assert rows["one"]["node_matches"][0]["observation_id"] == "high"
    assert rows["many"]["score"] == pytest.approx(1)
    assert len(rows["many"]["node_matches"]) == 2


def test_joint_correspondence_finds_edge_missed_by_first_category_pairing() -> None:
    observation = _observation([_node("x"), _node("y")], [_edge("x", "y", "ABOVE")])
    room = _room("r", [_node("a"), _node("b"), _node("c")], [_edge("c", "b", "ABOVE")])
    row = rank_rooms(observation, [room])["ranking"][0]
    assert row["score"] == pytest.approx(1)
    assert row["edge_matches"][0]["mapped_source"] == "c"
    assert row["edge_matches"][0]["mapped_target"] == "b"


def test_equal_node_and_edge_counts_do_not_hide_incidence_difference() -> None:
    observation = _observation([_node(f"q{i}") for i in range(4)],
                               [_edge("q0", f"q{i}", edge_id=f"q{i}") for i in range(1, 4)])
    star = _room("star", [_node(f"s{i}") for i in range(4)],
                 [_edge("s0", f"s{i}", edge_id=f"s{i}") for i in range(1, 4)])
    path = _room("path", [_node(f"p{i}") for i in range(4)],
                 [_edge(f"p{i}", f"p{i+1}", edge_id=f"p{i}") for i in range(3)])
    report = rank_rooms(observation, [path, star])
    assert report["ranking"][0]["room_id"] == "star"
    assert report["ranking"][0]["score"] == pytest.approx(1)
    assert report["ranking"][1]["relation_numerator"] == pytest.approx(2)


@pytest.mark.parametrize(("observed_type", "stored_type", "reverse", "expected"), [
    ("ADJACENT_TO", "TOUCHES", False, 1),
    ("ADJACENT_TO", "INTERSECTS", True, 1),
    ("TOUCHES", "ADJACENT_TO", False, 0),
    ("ABOVE", "ABOVE", False, 1),
    ("ABOVE", "ABOVE", True, 0),
    ("ABOVE", "BELOW", True, 1),
    ("ABOVE", "BELOW", False, 0),
    ("ATTACHED_TO", "ATTACHED_TO", True, 0),
])
def test_comp_is_one_way_and_direction_is_explicit(observed_type: str, stored_type: str,
                                                  reverse: bool, expected: int) -> None:
    observation = _observation([_node("qchair"), _node("qtable", "table")],
                               [_edge("qchair", "qtable", observed_type)])
    a, b = ("table", "chair") if reverse else ("chair", "table")
    room = _room("r", [_node("chair"), _node("table", "table")], [_edge(a, b, stored_type)])
    report = rank_rooms(observation, [room], variant="relations_only")
    assert report["ranking"][0]["score"] == expected


def test_contexts_are_injective_and_room_placeholder_is_fixed() -> None:
    observation = _observation(
        [_node("q"), _node("wall1", "wall", label="BoundarySurface"),
         _node("wall2", "wall", label="BoundarySurface"), _node("unknown", "room", label="Room")],
        [_edge("q", "wall1", "ATTACHED_TO", edge_id="a"),
         _edge("q", "wall2", "ATTACHED_TO", edge_id="b"), _edge("q", "unknown", "INSIDE", edge_id="c")],
    )
    room = _room("r", [_node("f"), _node("w", "wall", label="BoundarySurface"),
                       _node("other_room", "room", label="Room")],
                 [_edge("f", "w", "ATTACHED_TO", edge_id="a"), _edge("f", "other_room", "INSIDE", edge_id="b")])
    row = rank_rooms(observation, [room], variant="relations_only")["ranking"][0]
    assert row["relation_numerator"] == 1
    matches = {item["observation_id"]: item["candidate_id"] for item in row["node_matches"]}
    assert matches["unknown"] == "r"
    assert len(matches.values()) == len(set(matches.values()))


def test_opening_wildcard_and_specific_subtype_use_exact_weighted_matching() -> None:
    observation = _observation([_node("generic", "opening", label="Opening", weight=1),
                               _node("specific", "door", label="Opening", subtype="Door", weight=0.8)])
    room = _room("r", [_node("a_door", "door", label="Opening", subtype="Door"),
                       _node("z_window", "window", label="Opening", subtype="Window")])
    row = rank_rooms(observation, [room], variant="objects_only")["ranking"][0]
    matches = {item["observation_id"]: item["candidate_id"] for item in row["node_matches"]}
    assert matches == {"generic": "z_window", "specific": "a_door"}
    assert row["score"] == pytest.approx(1)


def test_best_parallel_edge_is_counted_once_and_metadata_below_one_limits_score() -> None:
    observation = _observation([_node("x"), _node("y", "table")], [_edge("x", "y", weight=0.5)])
    room = _room("r", [_node("a"), _node("b", "table")],
                 [_edge("a", "b", edge_id="weak", quality=0.2), _edge("a", "b", edge_id="best", quality=0.7)])
    row = rank_rooms(observation, [room], variant="relations_only")["ranking"][0]
    assert row["score"] == pytest.approx(0.7)
    assert row["relation_numerator"] == pytest.approx(0.35)
    assert len(row["edge_matches"]) == 1
    assert row["edge_matches"][0]["stored_edge_id"] == "best"


@pytest.mark.parametrize("variant", ["full", "uniform_category", "relations_only"])
def test_active_nonboolean_scoring_rejects_missing_metadata(variant: str) -> None:
    observation = _observation([_node("x"), _node("y", "table")], [_edge("x", "y")])
    room = _room("r", [_node("a"), _node("b", "table")], [_edge("a", "b", quality=None)])
    with pytest.raises(ValueError, match="missing metadata"):
        rank_rooms(observation, [room], variant=variant)


@pytest.mark.parametrize("variant", ["objects_only", "boolean_metadata"])
def test_inactive_or_boolean_metadata_can_be_missing_but_is_reported(variant: str) -> None:
    observation = _observation([_node("x"), _node("y", "table")], [_edge("x", "y")])
    room = _room("r", [_node("a"), _node("b", "table")], [_edge("a", "b", quality=None)])
    report = rank_rooms(observation, [room], variant=variant)
    assert report["status"] == "completed"
    assert report["missing_metadata"]["r"]["gamma_count"] == 1
    room["edges"][0]["gamma"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        rank_rooms(observation, [room], variant=variant)


def test_boolean_metadata_does_not_invent_edges() -> None:
    observation = _observation([_node("x"), _node("y", "table")], [_edge("x", "y")])
    room = _room("r", [_node("a"), _node("b", "table")])
    report = rank_rooms(observation, [room], config={"alpha": 0, "beta": 1}, variant="boolean_metadata")
    assert report["status"] == "abstained_no_evidence"
    assert report["ranking"][0]["edge_matches"] == []


def test_objects_only_ties_never_use_relation_evidence() -> None:
    observation = _observation([_node("x"), _node("y", "table")], [_edge("x", "y")])
    a = _room("a", [_node("a1"), _node("a2", "table")])
    z = _room("z", [_node("z1"), _node("z2", "table")], [_edge("z1", "z2")])
    report = rank_rooms(observation, [z, a], variant="objects_only")
    assert [row["room_id"] for row in report["ranking"]] == ["a", "z"]
    assert all(row["relation_numerator"] == 0 and row["edge_matches"] == [] for row in report["ranking"])
    assert [(row["best_tie_rank"], row["worst_tie_rank"], row["tie_size"]) for row in report["ranking"]] == [(1, 2, 2)] * 2
    assert report["top_score_tie"]
    assert rank_rooms(observation, [a, z])["ranking"][0]["room_id"] == "z"


def test_relations_only_does_not_use_object_bonus_for_ties() -> None:
    observation = _observation([_node("x"), _node("y", "table"), _node("tv", "tv")], [_edge("x", "y")])
    a = _room("a", [_node("a1"), _node("a2", "table")], [_edge("a1", "a2")])
    z = _room("z", [_node("z1"), _node("z2", "table"), _node("z3", "tv")], [_edge("z1", "z2")])
    report = rank_rooms(observation, [z, a], variant="relations_only")
    assert [row["room_id"] for row in report["ranking"]] == ["a", "z"]
    assert all(row["object_numerator"] == 0 for row in report["ranking"])


def test_idf_denominator_and_inputs_are_fixed_across_candidate_order_and_variants() -> None:
    observation = _observation([_node("x", "rare"), _node("y", "common")], [_edge("x", "y")])
    pool = [_room("a", [_node("x1", "rare"), _node("y1", "common")], [_edge("x1", "y1", quality=0.5)]),
            _room("b", [_node("y2", "common")])]
    original = deepcopy((observation, pool))
    reports = {variant: rank_rooms(observation, pool, variant=variant) for variant in
               ["full", "objects_only", "relations_only", "uniform_category", "boolean_metadata"]}
    for report in reports.values():
        assert report["candidate_ids"] == ["a", "b"]
        assert all(row["denominator"] == report["denominator"] for row in report["ranking"])
        assert report["candidate_ids_hash"] == reports["full"]["candidate_ids_hash"]
    assert reports["full"]["category_weights"]["rare"] == pytest.approx(math.log(2))
    assert reports["full"]["category_weights"]["common"] == pytest.approx(math.log1p(2 / 3))
    assert reports["boolean_metadata"]["denominator"] == reports["full"]["denominator"]
    assert reports["relations_only"]["denominator"] == 1
    assert reports["uniform_category"]["denominator"] == 3
    assert reports["objects_only"]["denominator"] == pytest.approx(reports["full"]["denominator"] - 1)
    assert (observation, pool) == original
    reverse = rank_rooms(observation, list(reversed(pool)))
    assert reverse["ranking"] == reports["full"]["ranking"]


def test_abstentions_invalid_values_and_zero_component_config() -> None:
    room = _room("r", [_node("a", "table")])
    assert rank_rooms(_observation([]), [room])["status"] == "abstained_no_anchor"
    assert rank_rooms(_observation([_node("x")]), [room])["status"] == "abstained_no_evidence"
    observation = _observation([_node("x", "table")])
    assert rank_rooms(observation, [room], config={"alpha": 0, "beta": 1})["status"] == "abstained_no_anchor"
    assert rank_rooms(observation, [room], config={"alpha": 1, "beta": 0})["status"] == "completed"
    assert rank_rooms(observation, [room], config={"alpha": 0, "beta": 1}, variant="objects_only")["status"] == "abstained_no_anchor"
    with pytest.raises(ValueError, match="both be zero"):
        rank_rooms(observation, [room], config={"alpha": 0, "beta": 0})
    observation["nodes"][0]["weight"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        rank_rooms(observation, [room])
    with pytest.raises(ValueError, match="Duplicate candidate"):
        rank_rooms(_observation([]), [room, room])


def test_budget_exhaustion_reports_feasible_best_and_honest_upper_bound() -> None:
    observation = _observation([_node("x"), _node("y")], [_edge("x", "y", "ABOVE")])
    room = _room("r", [_node("a"), _node("b"), _node("c")], [_edge("c", "b", "ABOVE")])
    incomplete = rank_rooms(observation, [room], config={"max_states": 1})
    exact = rank_rooms(observation, [room])
    assert incomplete["status"] == "solver_incomplete"
    row = incomplete["ranking"][0]
    assert row["solver"]["states_explored"] == 1
    assert not row["solver"]["optimality_proven"]
    assert row["score"] <= exact["ranking"][0]["score"] <= row["solver"]["upper_bound"]
    assert incomplete["full_ranking"]


def test_boundary_subtype_must_match_when_specified() -> None:
    observation = _observation([_node("f"), _node("floor", "floor", label="BoundarySurface", subtype="FloorSurface")],
                               [_edge("f", "floor", "ATTACHED_TO")])
    wrong = _room("r", [_node("f1"), _node("ground", "floor", label="BoundarySurface", subtype="GroundSurface")],
                  [_edge("f1", "ground", "ATTACHED_TO")])
    assert rank_rooms(observation, [wrong], variant="relations_only")["status"] == "abstained_no_evidence"


def test_generic_boundary_context_is_a_label_wildcard() -> None:
    observation = _observation([_node("f"), _node("surface", "", label="BoundarySurface")],
                               [_edge("f", "surface", "ATTACHED_TO")])
    room = _room("r", [_node("stored_f"), _node("wall", "wall", label="BoundarySurface", subtype="WallSurface")],
                 [_edge("stored_f", "wall", "ATTACHED_TO")])
    assert rank_rooms(observation, [room], variant="relations_only")["ranking"][0]["score"] == 1


@pytest.mark.parametrize("config", [{"version": "different_method"}, {"password": "must_not_be_saved"},
                                     {"alpah": 1}, {"category_aliases": {"chair": "table"}}])
def test_effective_config_rejects_unknown_fields_or_unimplemented_versions(config: dict) -> None:
    with pytest.raises(ValueError):
        rank_rooms(_observation([]), [_room("r", [])], config=config)


def test_missing_map_metadata_is_inactive_without_positive_observed_relations() -> None:
    room = _room("r", [_node("a"), _node("b", "table")], [_edge("a", "b", quality=None)])
    empty = rank_rooms(_observation([]), [room])
    assert empty["status"] == "abstained_no_anchor"
    assert empty["missing_metadata"]["r"]["gamma_count"] == 1
    assert empty["effective_config"]["stored_metadata_policy"] == "validate_present_allow_missing_no_active_observed_relation"
    objects = rank_rooms(_observation([_node("seen")]), [room])
    assert objects["status"] == "completed"
    assert objects["ranking"][0]["score"] == pytest.approx(1)
    assert objects["ranking"][0]["relation_numerator"] == 0
    assert objects["effective_config"]["beta"] == 1  # The method itself was not changed.
    room["edges"][0]["gamma"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        rank_rooms(_observation([]), [room])
