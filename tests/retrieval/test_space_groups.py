"""Space grouping must preserve source evidence and the fixed-pool contract."""
from copy import deepcopy

import pytest

from citygml_sg.retrieval.evaluation.oracle_metrics import candidate_ids_hash, evaluate_retrieval
from citygml_sg.retrieval.graph_matching.joint_matcher import rank_rooms
from citygml_sg.retrieval.reference_graph import build_room_candidates
from citygml_sg.retrieval.space_groups import (
    group_room_candidates, project_room_ranking_to_spaces, room_to_space_map,
)


GROUPS = [{"space_id": "space_AB", "space_name": "Open space", "member_room_ids": ["a", "b"],
           "provenance": "declared connected source space"}]


def _pool():
    nodes = [{"id": i, "type": "Room"} for i in ("a", "b", "c")]
    nodes += [{"id": "sofa", "type": "Furniture", "category": "sofa"},
              {"id": "fridge", "type": "Furniture", "category": "fridge"},
              {"id": "bed", "type": "Furniture", "category": "bed"},
              {"id": "wall", "type": "WallSurface"}, {"id": "window", "type": "Window"}]
    edges = [{"id": i, "source": a, "target": b, "type": t} for i, a, t, b in [
        ("inside_a", "sofa", "INSIDE", "a"), ("inside_b", "fridge", "INSIDE", "b"),
        ("inside_c", "bed", "INSIDE", "c"), ("bound_a", "a", "BOUNDED_BY", "wall"),
        ("bound_b", "b", "BOUNDED_BY", "wall"), ("opening", "wall", "HAS_OPENING", "window")]]
    return build_room_candidates({"nodes": nodes, "edges": edges})


def _observed(alias, label, category):
    return {"id": alias, "label": label, "category": category, "subtype": None,
            "weight": 1.0, "scoring": label in {"Furniture", "Opening", "Installation"}}


def test_union_deduplicates_shared_instances_preserves_original_edges_and_singleton():
    pool = _pool()
    original = deepcopy(pool)
    result = group_room_candidates(pool, GROUPS)
    merged = next(c for c in result if c["room_id"] == "space_AB")
    assert pool == original
    assert next(c for c in result if c["room_id"] == "c") == original[2]
    assert {n["id"] for n in merged["nodes"]} == {"space_AB", "sofa", "fridge", "wall", "window"}
    source_edges = {e["id"]: e for c in pool[:2] for e in c["edges"]}
    assert len(merged["edges"]) == len(source_edges)
    for edge in merged["edges"]:
        old = source_edges[edge["id"]]
        assert edge["type"] == old["type"]
        assert edge["gamma"] == old["gamma"] and edge["epsilon"] == old["epsilon"]
        if "space_group_projection" in edge:
            p = edge["space_group_projection"]
            assert p["original_source"] == old["source"] and p["original_target"] == old["target"]
        else:
            assert edge == old
    assert not any(e["source"] == "sofa" and e["target"] == "fridge" for e in merged["edges"])
    assert merged["membership"]["source_room_memberships"]["a"] == pool[0]["membership"]
    assert room_to_space_map(pool, GROUPS) == {"a": "space_AB", "b": "space_AB", "c": "c"}
    result[0]["nodes"][0]["category"] = "changed"
    assert pool == original


def test_group_spec_and_input_order_do_not_change_union_or_definition_hash():
    pool = _pool()
    expected = group_room_candidates(pool, GROUPS)
    shuffled = deepcopy(pool[::-1])
    for c in shuffled:
        c["nodes"].reverse()
        c["edges"].reverse()
    groups = deepcopy(GROUPS)
    groups[0]["member_room_ids"].reverse()
    actual = group_room_candidates(shuffled, groups)
    # Singleton graphs remain byte-structure identical to their input; compare union.
    assert actual[1] == expected[1]
    assert group_room_candidates(pool, []) == pool


def test_union_room_context_supports_existing_membership_relations_in_matcher():
    pool = group_room_candidates(_pool(), GROUPS)
    observed = {"nodes": [_observed("unknown", "Room", "room"),
                          _observed("s", "Furniture", "sofa"),
                          _observed("f", "Furniture", "fridge")],
                "edges": [{"id": i, "source": alias, "target": "unknown", "type": "INSIDE", "weight": 1.0}
                          for i, alias in [("qs", "s"), ("qf", "f")]]}
    report = rank_rooms(observed, pool)
    assert report["status"] == "completed"
    top = report["ranking"][0]
    assert top["room_id"] == "space_AB" and top["score"] == pytest.approx(1.0)
    assert top["relation_numerator"] == 2.0
    assert len(top["edge_matches"]) == 2
    assert any(n["observation_id"] == "unknown" and n["candidate_id"] == "space_AB" for n in top["node_matches"])


def test_shared_window_is_not_duplicated_into_extra_matching_capacity():
    pool = group_room_candidates(_pool(), GROUPS)
    observed = {"nodes": [_observed("w1", "Opening", "window"), _observed("w2", "Opening", "window")], "edges": []}
    report = rank_rooms(observed, pool, variant="objects_only")
    merged = next(row for row in report["ranking"] if row["room_id"] == "space_AB")
    assert merged["score"] == pytest.approx(0.5)
    assert len(merged["node_matches"]) == 1


@pytest.mark.parametrize("groups,match", [
    ([{"space_id": "s", "member_room_ids": ["a"]}], "at least two"),
    ([{"space_id": "s", "member_room_ids": ["a", "missing"]}], "outside"),
    ([{"space_id": "s", "member_room_ids": ["a", "a"]}], "Duplicate member"),
    ([{"space_id": "c", "member_room_ids": ["a", "b"]}], "own member"),
    ([{"space_id": "wall", "member_room_ids": ["a", "b"]}], "collides"),
    ([{"space_id": "s", "member_room_ids": ["a", "b"]}, {"space_id": "t", "member_room_ids": ["b", "c"]}], "overlapping"),
    ([{"space_id": "s", "member_room_ids": ["a", "b"], "target_room_id": "a"}], "accepts only"),
])
def test_invalid_group_definitions_are_rejected(groups, match):
    with pytest.raises(ValueError, match=match):
        group_room_candidates(_pool(), groups)


@pytest.mark.parametrize("kind", ["node", "edge"])
def test_conflicting_shared_records_are_not_silently_overwritten(kind):
    pool = [deepcopy(c) for c in _pool()]
    if kind == "node":
        next(n for n in pool[1]["nodes"] if n["id"] == "wall")["category"] = "floor"
    else:
        next(e for e in pool[1]["edges"] if e["id"] == "opening")["gamma"] = 0.25
    with pytest.raises(ValueError, match="Conflicting shared"):
        group_room_candidates(pool, GROUPS)


def _report(status="completed", full=True):
    ids = ["a", "b", "c"]
    return {"status": status, "candidate_ids": ids, "candidate_count": 3,
            "candidate_ids_hash": candidate_ids_hash(ids), "full_ranking": full,
            "score_tie_decimals": 12, "variant": "full", "method_version": "joint_evidence_v1_proposed",
            "ranking": [{"room_id": i, "room_name": i, "rank": j, "score": score,
                         "relation_numerator": rel, "solver": {"optimality_proven": True}}
                        for j, (i, score, rel) in enumerate([("a", 0.8, 1.0), ("b", 0.8, 0.5), ("c", 0.8, 0.5)], 1)],
            "category_room_counts": {"window": 3}}


def test_projection_selects_best_member_and_recomputes_space_ranks_ties_pool():
    original = _report()
    saved = deepcopy(original)
    result = project_room_ranking_to_spaces(original, GROUPS)
    assert original == saved
    assert result["candidate_ids"] == ["c", "space_AB"]
    assert result["candidate_ids_hash"] == candidate_ids_hash(["space_AB", "c"])
    assert result["source_room_ranking"] == original["ranking"]
    assert result["source_room_candidate_count"] == 3
    assert result["ranking"][0]["source_room_id"] == "a"
    assert result["ranking"][0]["room_id"] == "space_AB"
    assert result["ranking"][0]["tie_size"] == 1
    assert result["top_score_tie"] is True  # score-only ties differ from relation tie-break ties
    assert result["category_room_counts"] == original["category_room_counts"]
    assert result["space_projection"]["scoring_pool"] == "original_room_pool"
    metrics = evaluate_retrieval(result, "space_AB", candidate_ids=result["candidate_ids"])
    assert metrics["recall_at_1"] == 1


def test_projection_reorders_equal_scores_by_space_id_and_retains_member_evidence():
    report = _report()
    for row in report["ranking"]:
        row["relation_numerator"] = 0.0
    result = project_room_ranking_to_spaces(report, GROUPS)
    assert [r["room_id"] for r in result["ranking"]] == ["c", "space_AB"]
    assert result["ranking"][1]["source_room_id"] == "a"
    assert result["ranking"][1]["source_member_room_ids"] == ["a", "b"]
    assert all(r["tie_size"] == 2 and r["best_tie_rank"] == 1 and r["worst_tie_rank"] == 2 for r in result["ranking"])


@pytest.mark.parametrize("status", ["execution_failed", "solver_incomplete", "abstained_no_anchor", "abstained_no_evidence"])
def test_projection_never_upgrades_failure_or_abstention(status):
    original = _report(status, False)
    original["ranking"] = []
    original["failure_reason"] = "example"
    result = project_room_ranking_to_spaces(original, GROUPS)
    assert result["status"] == status and result["ranking"] == []
    assert result["full_ranking"] is False and result["failure_reason"] == "example"
    assert result["candidate_count"] == 2


def test_projection_rejects_inconsistent_source_pool_and_partial_completed_ranking():
    report = _report()
    report["candidate_ids_hash"] = "bad"
    with pytest.raises(ValueError, match="identity"):
        project_room_ranking_to_spaces(report, GROUPS)
    report = _report()
    report["ranking"].pop()
    with pytest.raises(ValueError, match="missing a candidate"):
        project_room_ranking_to_spaces(report, GROUPS)


def test_group_can_reuse_a_member_id_to_preserve_canonical_tie_position():
    groups = [{"space_id": "a", "space_name": "A+B", "member_room_ids": ["a", "b"]}]
    pool = group_room_candidates(_pool(), groups)
    assert [c["room_id"] for c in pool] == ["a", "c"]
    assert sum(n["label"] == "Room" for n in pool[0]["nodes"]) == 1
    assert pool[0]["space_group"]["space_id_policy"] == "representative_member_id"
    report = _report()
    for row in report["ranking"]:
        row["relation_numerator"] = 0.0
    projected = project_room_ranking_to_spaces(report, groups)
    assert [r["room_id"] for r in projected["ranking"]] == ["a", "c"]
    assert projected["ranking"][0]["source_member_room_ids"] == ["a", "b"]
