"""Geometric counterexamples for the opt-in membership proposal contract."""
from copy import deepcopy
from hashlib import sha256
import json
import math

import pytest

from citygml_sg.retrieval.geometric_membership import (
    load_citygml_geometry, point_in_prism, propose_geometric_memberships,
    validate_room_prism,
)


def polygon(points):
    return {"exterior": [list(p) for p in points + points[:1]], "interiors": []}


def prism(node_id, footprint=None, low=0, high=3):
    footprint = footprint or [(0,0),(2,0),(2,2),(0,2)]
    bottom = [(x,y,low) for x,y in footprint]
    top = [(x,y,high) for x,y in footprint]
    faces = [polygon(bottom), polygon(top[::-1])]
    for i in range(len(bottom)):
        j = (i+1)%len(bottom)
        faces.append(polygon([bottom[i],bottom[j],top[j],top[i]]))
    return {"id":node_id,"polygons":faces,"geometry_errors":[]}


def scene(rooms, objects, edges=None):
    nodes = [{"id":r["id"],"type":"Room","properties":{"gml_name":r["id"]}} for r in rooms]
    geometries = {r["id"]:r for r in rooms}
    for name,label,shape in objects:
        nodes.append({"id":name,"type":label,"properties":{"gml_name":name}})
        geometries[name]=shape
    return {"nodes":nodes,"edges":edges or []},{"source":{"sha256":"synthetic"},"nodes":geometries}


def assignment(ledger,node_id):
    return next(x for x in ledger["assignments"] if x["node_id"]==node_id)


def test_rotated_prism_point_in_aabb_but_outside_footprint():
    room=validate_room_prism(prism("diamond",[(0,1),(1,0),(2,1),(1,2)]))
    assert room["status"]=="valid_prism"
    assert room["volume"]==pytest.approx(6)
    assert point_in_prism((0.1,0.1,1),room)=="outside"
    assert point_in_prism((1,1,1),room)=="inside"
    assert point_in_prism((0.5,0.5,1),room)=="boundary"
    assert point_in_prism((1,1,3),room)=="boundary"
    assert point_in_prism((1,1,4),room)=="outside"


@pytest.mark.parametrize("mutation,reason",[
    (lambda g:g["polygons"].pop(1),"expected_single_floor_and_ceiling"),
    (lambda g:g["polygons"].append(deepcopy(g["polygons"][-1])),"incomplete_or_duplicate_walls"),
    (lambda g:g["polygons"][0]["exterior"].pop(),"unclosed_ring"),
    (lambda g:g["polygons"][0]["interiors"].append([[0,0,0]]),"room_holes_unsupported"),
    (lambda g:g["geometry_errors"].append("unresolved_geometry_xlink"),"source_geometry_errors"),
])
def test_rejects_uncertified_room_shell(mutation,reason):
    geometry=prism("r")
    mutation(geometry)
    result=validate_room_prism(geometry)
    assert result["status"]=="unresolved"
    assert result["reason"]==reason


def test_bow_tie_wall_rejected_even_with_correct_vertex_set():
    geometry=prism("r")
    vertices=geometry["polygons"][2]["exterior"]
    vertices[1],vertices[2]=vertices[2],vertices[1]
    result=validate_room_prism(geometry)
    assert result["status"]=="unresolved"
    assert result["reason"]=="invalid_wall_ring"


def test_mismatched_ceiling_rejected():
    geometry=prism("r")
    geometry["polygons"][1]["exterior"][1][0]+=0.1
    assert validate_room_prism(geometry)["reason"]=="non_matching_floor_ceiling"


def test_nonfinite_room_geometry_rejected():
    geometry=prism("r")
    geometry["polygons"][0]["exterior"][1][0]=float("nan")
    assert validate_room_prism(geometry)["status"]=="unresolved"


def test_concave_prism_contains_arm_but_not_notch():
    room=validate_room_prism(prism("L",[(0,0),(3,0),(3,1),(1,1),(1,3),(0,3)]))
    assert room["status"]=="valid_prism"
    assert point_in_prism((0.5,2,1),room)=="inside"
    assert point_in_prism((2,2,1),room)=="outside"


def test_full_envelope_check_rejects_triangle_crossing_concave_notch():
    room=prism("L",[(0,0),(3,0),(3,1),(1,1),(1,3),(0,3)])
    obj=prism("obj",[(0.5,0.5),(2.5,0.5),(0.5,2.5)],low=1,high=2)
    snapshot,geometry=scene([room],[("obj","IntBuildingInstallation",obj)])
    item=assignment(propose_geometric_memberships(snapshot,geometry),"obj")
    # All three XY vertices are in L, but the diagonal crosses its missing part.
    assert item["full_shape_containment_room_ids"]==[]
    assert item["status"]=="ambiguous"
    assert item["proposed_room_ids"]==[]


def test_unique_installation_full_containment():
    obj=prism("sink",[(0.2,0.2),(0.8,0.2),(0.8,0.8),(0.2,0.8)],low=1,high=2)
    snapshot,geometry=scene([prism("r")],[("sink","IntBuildingInstallation",obj)])
    before=deepcopy((snapshot,geometry))
    ledger=propose_geometric_memberships(snapshot,geometry)
    item=assignment(ledger,"sink")
    assert item["status"]=="assigned_containment"
    assert item["proposed_room_ids"]==["r"]
    assert ledger["allowed_node_ids_by_room"]=={"r":["r","sink"]}
    assert (snapshot,geometry)==before
    encoded=json.dumps(snapshot,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
    assert ledger["provenance"]["snapshot_content_sha256"]==sha256(encoded).hexdigest()


def test_rotated_room_cannot_receive_installation_only_in_aabb():
    obj=prism("obj",[(0.05,0.05),(0.1,0.05),(0.1,0.1),(0.05,0.1)],low=1,high=2)
    snapshot,geometry=scene([prism("diamond",[(0,1),(1,0),(2,1),(1,2)])],[("obj","BuildingInstallation",obj)])
    ledger=propose_geometric_memberships(snapshot,geometry)
    assert assignment(ledger,"obj")["proposed_room_ids"]==[]
    assert ledger["allowed_node_ids_by_room"]=={"diamond":["diamond"]}


def test_overlapping_rooms_keep_installation_ambiguous():
    obj=prism("obj",[(0.2,0.2),(0.8,0.2),(0.8,0.8),(0.2,0.8)],low=1,high=2)
    snapshot,geometry=scene([prism("a"),prism("b")],[("obj","BuildingInstallation",obj)])
    ledger=propose_geometric_memberships(snapshot,geometry)
    assert assignment(ledger,"obj")["status"]=="ambiguous"
    assert assignment(ledger,"obj")["full_shape_containment_room_ids"]==["a","b"]
    assert ledger["allowed_node_ids_by_room"]=={"a":["a"],"b":["b"]}


def test_furniture_semantic_owner_preserved_despite_geometry_conflict():
    obj=prism("chair",[(3.2,0.2),(3.8,0.2),(3.8,0.8),(3.2,0.8)],low=1,high=2)
    edges=[{"source_id":"chair","target_id":"a","relation":"INSIDE"}]
    snapshot,geometry=scene([prism("a"),prism("b",[(3,0),(5,0),(5,2),(3,2)])],[("chair","BuildingFurniture",obj)],edges)
    ledger=propose_geometric_memberships(snapshot,geometry)
    item=assignment(ledger,"chair")
    assert item["geometry_membership_conflict"] is True
    assert item["full_shape_containment_room_ids"]==["b"]
    assert item["proposed_room_ids"]==["a"]
    assert "chair" not in ledger["allowed_node_ids_by_room"]["b"]


def test_shared_surface_contact_preserved_for_adjacent_rooms():
    wall={"id":"wall","polygons":[polygon([(2,0,0),(2,2,0),(2,2,3),(2,0,3)])],"geometry_errors":[]}
    snapshot,geometry=scene([prism("a"),prism("b",[(2,0),(4,0),(4,2),(2,2)])],[("wall","WallSurface",wall)])
    ledger=propose_geometric_memberships(snapshot,geometry)
    item=assignment(ledger,"wall")
    assert item["status"]=="shared_contact"
    assert item["proposed_room_ids"]==["a","b"]
    assert all("wall" in x for x in ledger["allowed_node_ids_by_room"].values())
    assert item["surface_contact_evidence"]["a"][0]["max_projected_overlap_area"]==pytest.approx(6)


@pytest.mark.parametrize("offset,expected",[(0.049,True),(0.051,False)])
def test_fixed_contact_tolerance(offset,expected):
    opening={"id":"opening","polygons":[polygon([(0.4,-offset,0.4),(1.6,-offset,0.4),(1.6,-offset,2),(0.4,-offset,2)])],"geometry_errors":[]}
    snapshot,geometry=scene([prism("r")],[("opening","Door",opening)])
    item=assignment(propose_geometric_memberships(snapshot,geometry),"opening")
    assert bool(item["proposed_room_ids"]) is expected


def test_edge_only_contact_is_not_surface_membership():
    wall={"id":"wall","polygons":[polygon([(0,-2,0),(2,-2,0),(2,0,0),(0,0,0)])],"geometry_errors":[]}
    snapshot,geometry=scene([prism("r")],[("wall","FloorSurface",wall)])
    item=assignment(propose_geometric_memberships(snapshot,geometry),"wall")
    assert item["proposed_room_ids"]==[]


def test_missing_room_geometry_preserves_empty_candidate_and_ledger():
    snapshot,geometry=scene([prism("r")],[])
    geometry["nodes"].clear()
    ledger=propose_geometric_memberships(snapshot,geometry)
    assert ledger["allowed_node_ids_by_room"]=={"r":["r"]}
    assert ledger["room_geometry"]["r"]["status"]=="unresolved"


def test_warped_source_polygon_cannot_certify_contact():
    opening={"id":"opening","polygons":[polygon([(0.4,0,0.4),(1.6,0,0.4),(1.6,0,2),(0.4,0.02,2)])],"geometry_errors":[]}
    snapshot,geometry=scene([prism("r")],[("opening","Door",opening)])
    assert assignment(propose_geometric_memberships(snapshot,geometry),"opening")["proposed_room_ids"]==[]


def test_concave_floor_overlap_uses_actual_polygon_area():
    floor={"id":"floor","polygons":[polygon([(0,0,0),(3,0,0),(3,3,0),(0,3,0)])],"geometry_errors":[]}
    snapshot,geometry=scene([prism("L",[(0,0),(3,0),(3,1),(1,1),(1,3),(0,3)])],[("floor","FloorSurface",floor)])
    item=assignment(propose_geometric_memberships(snapshot,geometry),"floor")
    assert item["proposed_room_ids"]==["L"]
    assert item["surface_contact_evidence"]["L"][0]["max_projected_overlap_area"]==pytest.approx(5)


def test_large_translation_does_not_change_membership():
    obj=prism("sink",[(0.2,0.2),(0.8,0.2),(0.8,0.8),(0.2,0.8)],low=1,high=2)
    snapshot,geometry=scene([prism("r")],[("sink","IntBuildingInstallation",obj)])
    for shape in geometry["nodes"].values():
        for poly in shape["polygons"]:
            for p in poly["exterior"]:
                p[0]+=374460
                p[1]+=283440
    assert assignment(propose_geometric_memberships(snapshot,geometry),"sink")["proposed_room_ids"]==["r"]


def test_missing_object_geometry_is_not_assigned():
    snapshot,geometry=scene([prism("r")],[("object","BuildingInstallation",{"id":"object","geometry_errors":["missing_geometry"]})])
    ledger=propose_geometric_memberships(snapshot,geometry)
    assert assignment(ledger,"object")["status"]=="unresolved"
    assert ledger["allowed_node_ids_by_room"]=={"r":["r"]}


@pytest.mark.parametrize("epsilon,contact",[(0,0.05),(float("nan"),0.05),(1e-6,-1),(1e-6,float("inf"))])
def test_invalid_tolerances_rejected(epsilon,contact):
    snapshot,geometry=scene([prism("r")],[])
    with pytest.raises(ValueError,match="tolerances"):
        propose_geometric_memberships(snapshot,geometry,epsilon=epsilon,contact_tolerance=contact)


def test_loader_excludes_nested_semantic_geometry_and_keeps_xlink_unresolved(tmp_path):
    path=tmp_path/"source.gml"
    path.write_text('''<root xmlns:gml="http://www.opengis.net/gml" xmlns:b="http://www.opengis.net/citygml/building/2.0" xmlns:xlink="http://www.w3.org/1999/xlink"><b:Room gml:id="room"><b:lod4MultiSurface><gml:Polygon><gml:exterior><gml:LinearRing><gml:posList>0 0 0 2 0 0 2 2 0 0 2 0 0 0 0</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon></b:lod4MultiSurface><b:interiorFurniture><b:BuildingFurniture gml:id="chair"><b:lod4Geometry xlink:href="#missing"/></b:BuildingFurniture></b:interiorFurniture></b:Room></root>''',encoding="utf-8")
    result=load_citygml_geometry(path,{"room","chair"})
    assert len(result["nodes"]["room"]["polygons"])==1
    assert "unresolved_geometry_xlink" in result["nodes"]["chair"]["geometry_errors"]
    assert result["source"]["sha256"]==sha256(path.read_bytes()).hexdigest()
