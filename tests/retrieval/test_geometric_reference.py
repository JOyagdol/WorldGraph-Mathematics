"""Membership integration contracts, independent of geometric inference."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from citygml_sg.retrieval.geometric_reference import build_geometric_room_candidates
from citygml_sg.retrieval.reference_graph import build_room_candidates


def fixture():
    nodes = [{"id": key, "type": kind, "properties": props} for key, kind, props in [
        ("building", "Building", {}), ("a", "Room", {"gml_name": "A"}),
        ("b", "Room", {"gml_name": "B"}), ("wall", "WallSurface", {}),
        ("floor", "FloorSurface", {}), ("door", "Door", {}),
        ("sink", "IntBuildingInstallation", {"gml_name": "sink"}),
        ("geometry", "Polygon", {}),
    ]]
    edges = [{"id": key, "source_id": a, "target_id": b, "relation": relation,
              "properties": {"confidence": 1.0, "evidence_score": 0.8}}
             for key, a, b, relation in [
                 ("ba", "building", "a", "INTERIOR_ROOM"),
                 ("bb", "building", "b", "INTERIOR_ROOM"),
                 ("bw", "building", "wall", "BOUNDED_BY"),
                 ("bf", "building", "floor", "BOUNDED_BY"),
                 ("wd", "wall", "door", "HAS_OPENING"),
                 ("dw", "door", "wall", "HOSTED_BY"),
                 ("wf", "wall", "floor", "ADJACENT_SURFACE"),
                 ("is", "building", "sink", "INTERIOR_BUILDING_INSTALLATION"),
             ]]
    snapshot = {"nodes": nodes, "edges": edges}
    digest = sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False,
                              separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    ledger = {"policy": "validated_prism_membership_v1",
              "provenance": {"snapshot_content_sha256": digest},
              "allowed_node_ids_by_room": {"a": ["a", "wall", "door", "sink"], "b": ["b", "floor"]}}
    return snapshot, ledger


def test_membership_is_exact_and_does_not_expand_through_parent_or_adjacency():
    snapshot, ledger = fixture()
    before = deepcopy((snapshot, ledger))
    old = build_room_candidates(snapshot)
    assert all("door" in {n["id"] for n in c["nodes"]} for c in old)
    new = build_geometric_room_candidates(snapshot, ledger)
    assert [set(n["id"] for n in c["nodes"]) for c in new] == [
        {"a", "wall", "door", "sink"}, {"b", "floor"}]
    assert {e["id"] for e in new[0]["edges"]} == {"wd", "dw"}
    assert new[1]["edges"] == []
    assert all(e["gamma"] == 1.0 and e["epsilon"] == 0.8 for e in new[0]["edges"])
    assert (snapshot, ledger) == before
    assert build_room_candidates(snapshot) == old


@pytest.mark.parametrize("change", ["snapshot", "policy", "missing_room", "foreign_room", "unknown", "geometry", "duplicate", "no_self"])
def test_rejects_inconsistent_or_unsafe_ledger(change):
    snapshot, ledger = fixture()
    if change == "snapshot":
        snapshot["nodes"][0]["properties"]["changed"] = True
    elif change == "policy":
        ledger["policy"] = "unknown"
    elif change == "missing_room":
        del ledger["allowed_node_ids_by_room"]["b"]
    elif change == "no_self":
        ledger["allowed_node_ids_by_room"]["a"].remove("a")
    else:
        ledger["allowed_node_ids_by_room"]["a"].append({
            "foreign_room": "b", "unknown": "absent", "geometry": "geometry", "duplicate": "door"}[change])
    with pytest.raises(ValueError):
        build_geometric_room_candidates(snapshot, ledger)


def test_shared_boundary_requires_explicit_membership_and_creates_no_edge():
    snapshot, ledger = fixture()
    ledger["allowed_node_ids_by_room"]["b"].append("wall")
    candidates = build_geometric_room_candidates(snapshot, ledger)
    assert all(c["membership"]["shared_node_ids"] == ["wall"] for c in candidates)
    assert {e["id"] for e in candidates[1]["edges"]} == {"wf"}
    assert all(e["source"] != c["room_id"] for c in candidates for e in c["edges"])
