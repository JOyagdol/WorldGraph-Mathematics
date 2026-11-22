"""Source-only host scope and unchanged object-evidence contracts."""
from copy import deepcopy
import math

import pytest

from citygml_sg.retrieval.hosted_reference import build_hosted_support_ledger, scope_hosted_by
from citygml_sg.retrieval.reference_graph import _snapshot_records, normalize_semantic_node


def box(key, x0, x1, y0=0, y1=2, z0=0, z1=3):
    bottom = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)]
    top = [(x, y, z1) for x, y, _ in bottom]
    faces = [bottom, top[::-1]]
    for i in range(4):
        j = (i + 1) % 4
        faces.append([bottom[i], bottom[j], top[j], top[i]])
    return {"id": key, "geometry_errors": [], "polygons": [
        {"exterior": [list(v) for v in face + face[:1]], "interiors": []} for face in faces]}


def fixture():
    specs = [("a", "Room", {}), ("b", "Room", {}),
             ("opening", "Window", {"attr_other_host_id": "42"}),
             ("host", "WallSurface", {"attr_ifc_tag": "42", "attr_construction_width": "150"}),
             ("core", "WallSurface", {"attr_construction_width": "150"}),
             ("finish", "WallSurface", {"attr_construction_width": "50"})]
    snapshot = {"nodes": [{"id": key, "type": kind, "properties": props} for key, kind, props in specs],
                "edges": [{"id": key, "source_id": a, "target_id": b, "relation": relation,
                           "properties": {"confidence": 0.9, "evidence_score": 0.8}}
                          for key, a, b, relation in [("hosted", "opening", "host", "HOSTED_BY"),
                                                     ("opening_semantic", "host", "opening", "HAS_OPENING")]]}
    geometry = {"source": {"sha256": "synthetic"}, "nodes": {
        "a": box("a", 0, 2), "b": box("b", 3, 5),
        "opening": box("opening", -0.25, -0.2, .5, 1.5, 1, 2),
        "host": box("host", -.35, -.2), "core": box("core", -.2, -.05),
        "finish": box("finish", -.05, 0)}}
    return snapshot, geometry


def support(snapshot, geometry):
    return build_hosted_support_ledger(snapshot, geometry)["relations"][0]


def candidates(snapshot):
    raw, edges = _snapshot_records(snapshot)
    return [{"room_id": key, "room_name": key,
             "nodes": [normalize_semantic_node(v) for k, v in raw.items() if k not in {"a", "b"} or k == key],
             "edges": deepcopy(edges), "membership": {"policy": "synthetic_broad_baseline"}}
            for key in ("a", "b")]


def test_layers_explain_offset_without_expanding_object_membership():
    snapshot, geometry = fixture()
    row = support(snapshot, geometry)
    assert row["supported_room_ids"] == ["a"]
    proof = row["room_evidence"]["a"]
    assert all(p["basis"] != "opening_source_face_contact" for p in proof)
    assert any(p["layer_ids"] == ["core", "finish", "host"] for p in proof)


@pytest.mark.parametrize("broken", ["gap", "width", "missing_width", "host_tag", "missing_semantic", "patch", "bad_room", "unresolved_geometry"])
def test_unsupported_source_does_not_certify_a_room(broken):
    snapshot, geometry = fixture()
    if broken == "gap":
        geometry["nodes"].pop("core")
    elif broken == "width":
        snapshot["nodes"][4]["properties"]["attr_construction_width"] = "500"
    elif broken == "missing_width":
        snapshot["nodes"][4]["properties"].clear()
    elif broken == "host_tag":
        snapshot["nodes"][2]["properties"]["attr_other_host_id"] = "wrong"
    elif broken == "missing_semantic":
        snapshot["edges"].pop()
    elif broken == "patch":
        geometry["nodes"]["core"] = box("core", -.2, -.05, 0, .75)
    elif broken == "bad_room":
        geometry["nodes"]["a"]["polygons"].pop()
    elif broken == "unresolved_geometry":
        geometry["nodes"]["opening"]["geometry_errors"].append("unresolved_geometry_xlink")
    assert support(snapshot, geometry)["supported_room_ids"] == []


def test_direct_contact_requires_the_same_semantic_host_checks():
    snapshot, geometry = fixture()
    geometry["nodes"]["opening"] = box("opening", -.02, .02, .5, 1.5, 1, 2)
    row = support(snapshot, geometry)
    assert row["supported_room_ids"] == ["a"]
    assert any(p["basis"] == "opening_source_face_contact" for p in row["room_evidence"]["a"])
    snapshot["nodes"][2]["properties"]["attr_other_host_id"] = "wrong"
    assert support(snapshot, geometry)["supported_room_ids"] == []


def test_transform_preserves_nodes_other_edges_and_metadata_without_mutation():
    snapshot, geometry = fixture()
    ledger = build_hosted_support_ledger(snapshot, geometry)
    baseline = candidates(snapshot)
    before = deepcopy((snapshot, geometry, ledger, baseline))
    revised = scope_hosted_by(baseline, ledger, snapshot)
    for old, new in zip(baseline, revised):
        assert old["nodes"] == new["nodes"]
        assert [e for e in old["edges"] if e["type"] != "HOSTED_BY"] == [e for e in new["edges"] if e["type"] != "HOSTED_BY"]
        assert old["membership"] == new["membership"]
    assert revised[0]["edges"] == baseline[0]["edges"]
    assert [e["type"] for e in revised[1]["edges"]] == ["HAS_OPENING"]
    edge = next(e for e in revised[0]["edges"] if e["type"] == "HOSTED_BY")
    assert edge["gamma"] == .9 and edge["epsilon"] == .8
    assert (snapshot, geometry, ledger, baseline) == before


def test_space_union_uses_source_member_support():
    snapshot, geometry = fixture()
    ledger = build_hosted_support_ledger(snapshot, geometry)
    baseline = candidates(snapshot)[1:]
    baseline[0]["space_group"] = {"member_room_ids": ["a", "b"]}
    assert scope_hosted_by(baseline, ledger, snapshot)[0]["edges"] == baseline[0]["edges"]


def test_rotation_and_large_projected_coordinate_origin_preserve_support():
    snapshot, geometry = fixture()
    angle = .53
    for shape in geometry["nodes"].values():
        for polygon in shape["polygons"]:
            for p in polygon["exterior"]:
                x, y = p[:2]
                p[:2] = [374000 + x * math.cos(angle) - y * math.sin(angle),
                         283000 + x * math.sin(angle) + y * math.cos(angle)]
    assert support(snapshot, geometry)["supported_room_ids"] == ["a"]


@pytest.mark.parametrize("broken", ["snapshot", "policy", "unknown_edge", "duplicate_edge", "unknown_room"])
def test_adapter_rejects_inconsistent_inputs(broken):
    snapshot, geometry = fixture()
    ledger = build_hosted_support_ledger(snapshot, geometry)
    baseline = candidates(snapshot)
    if broken == "snapshot":
        snapshot["nodes"][0]["properties"]["extra"] = True
    elif broken == "policy":
        ledger["policy"] = "unsupported"
    elif broken == "unknown_edge":
        next(e for e in baseline[0]["edges"] if e["type"] == "HOSTED_BY")["id"] = "unknown"
    elif broken == "duplicate_edge":
        ledger["relations"].append(deepcopy(ledger["relations"][0]))
    elif broken == "unknown_room":
        baseline[0]["space_group"] = {"member_room_ids": ["missing"]}
    with pytest.raises(ValueError):
        scope_hosted_by(baseline, ledger, snapshot)
