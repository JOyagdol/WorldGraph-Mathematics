"""Reject withdrawn solver identities rather than silently substituting modes."""
from copy import deepcopy

import pytest

from citygml_sg.retrieval.graph_matching.joint_matcher import rank_rooms
from citygml_sg.retrieval.scoring.joint_score import effective_config


def inputs():
    observation = {
        "nodes": [{"id": "seen", "label": "Opening", "category": "door", "weight": 1.0, "scoring": True},
                  {"id": "wall", "label": "BoundarySurface", "category": "wall", "weight": 1.0, "scoring": False}],
        "edges": [{"id": "relation", "source": "seen", "target": "wall", "type": "HOSTED_BY", "weight": 1.0}],
    }
    candidates = [{"room_id": "A", "nodes": [
        {"id": "A", "label": "Room", "category": "room"},
        {"id": "stored_door", "label": "Opening", "category": "door"},
        {"id": "stored_wall", "label": "BoundarySurface", "category": "wall"}],
        "edges": [{"id": "host", "source": "stored_door", "target": "stored_wall", "type": "HOSTED_BY", "gamma": 1.0, "epsilon": 1.0}]}]
    return observation, candidates


@pytest.mark.parametrize("mode", ["exact_pruned_v2", "exact_pruned_v3"])
@pytest.mark.parametrize("variant", ["full", "objects_only"])
def test_withdrawn_fast_modes_raise_instead_of_falling_back(mode, variant):
    with pytest.raises(ValueError, match="Only solver_mode=exact_reference is implemented"):
        effective_config({"solver_mode": mode}, variant)
    with pytest.raises(ValueError, match="Only solver_mode=exact_reference is implemented"):
        rank_rooms(*inputs(), config={"solver_mode": mode}, variant=variant)


def test_default_restores_reference_search_and_evidence():
    observation, candidates = inputs()
    before = deepcopy((observation, candidates))
    default = rank_rooms(observation, candidates)
    reference = rank_rooms(observation, candidates, config={"solver_mode": "exact_reference"})
    for report in (default, reference):
        report.pop("timings")
        assert report["effective_config"]["solver_mode"] == "exact_reference"
        assert report["ranking"][0]["solver"]["algorithm"] == "joint_branch_and_bound"
        assert report["ranking"][0]["solver"]["optimality_proven"] is True
        assert report["mapping_tie_policy"] == "first_optimum_in_deterministic_domain_search_order"
    assert default == reference
    assert (observation, candidates) == before
