"""Opt-in room support for stored HOSTED_BY edges, preserving object evidence.

Only map geometry and existing semantic host evidence enter this preparation.
Wall layers are explicit source envelopes, not a claim about physical visibility
or solid material across an aperture. The baseline candidate nodes and every
other relation remain unchanged. This is not enabled by the default builder.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math

from citygml_sg.retrieval.geometric_membership import (
    GEOMETRY_EPSILON, CONTACT_TOLERANCE, _area, _covers, _cross, _frame,
    _hull, _project, _ring, _surface_contacts, validate_room_prism,
)
from citygml_sg.retrieval.reference_graph import _snapshot_records, normalize_semantic_node


POLICY = "semantic_host_wall_layer_room_support_v1"


def content_hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _clip(subject, boundary):
    """Intersect convex projected envelopes with a convex room wall face."""
    if _area(boundary) < 0:
        boundary = list(reversed(boundary))
    result = list(subject)
    for a, b in zip(boundary, boundary[1:] + boundary[:1]):
        previous_polygon, result = result, []
        if not previous_polygon:
            break
        previous = previous_polygon[-1]
        before = _cross(a, b, previous)
        for current in previous_polygon:
            after = _cross(a, b, current)
            if (after >= 0) != (before >= 0):
                fraction = before / (before - after)
                result.append(tuple(previous[i] + fraction * (current[i] - previous[i]) for i in (0, 1)))
            if after >= 0:
                result.append(current)
            previous, before = current, after
    return result


def _vertices(shape, epsilon):
    # These points define an explicit *envelope*, not a certified solid. A
    # degenerate or warped source face does not invalidate its finite vertex
    # envelope; direct face contact performs its own planar-polygon checks.
    if not shape or shape.get("geometry_errors"):
        raise ValueError("missing_or_unsupported_geometry")
    vertices = []
    for polygon in shape.get("polygons", []):
        for raw in [polygon["exterior"]] + polygon.get("interiors", []):
            ring = _ring(raw, epsilon)
            vertices.extend(ring)
    if not vertices:
        raise ValueError("missing_explicit_polygons")
    return vertices


def _projection(vertices, frame):
    return _hull([(*_project(p, frame), 0.0) for p in vertices])


def _interval(vertices, frame):
    origin, normal, _, _ = frame
    values = [sum((p[i] - origin[i]) * normal[i] for i in range(3)) for p in vertices]
    return min(values), max(values)


def _layer_certificate(opening, host_id, face, layers, epsilon, tolerance):
    frame = _frame(face, epsilon)
    if abs(frame[1][2]) > epsilon:
        return None  # Only vertical architectural wall support is modeled.
    projected = _projection(opening, frame)
    if len(projected) < 3:
        return None
    extent = [max(p[i] for p in projected) - min(p[i] for p in projected) for i in (0, 1)]
    opening_interval = _interval(opening, frame)
    if opening_interval[1] - opening_interval[0] >= min(extent) - epsilon:
        return None  # Exclude the side view of a thick opening envelope.
    portal = _clip(projected, [_project(p, frame) for p in face])
    area = abs(_area(portal))
    if len(portal) < 3 or area <= epsilon * epsilon:
        return None
    eligible = {}
    for wall_id, (vertices, declared_width) in layers.items():
        interval = _interval(vertices, frame)
        if abs(interval[1] - interval[0] - declared_width) > epsilon:
            continue  # A wall on another axis or a thick/non-panel envelope.
        if not _covers(_projection(vertices, frame), portal, epsilon):
            continue
        eligible[wall_id] = interval
    if host_id not in eligible:
        return None
    low, high = eligible[host_id]
    if opening_interval[0] > high + tolerance or opening_interval[1] < low - tolerance:
        return None
    selected = {host_id}
    changed = True
    while changed:
        changed = False
        for wall_id, (left, right) in sorted(eligible.items()):
            if wall_id not in selected and left <= high + tolerance and right >= low - tolerance:
                selected.add(wall_id)
                low, high = min(low, left), max(high, right)
                changed = True
    if not low - tolerance <= 0 <= high + tolerance:
        return None
    return {
        "basis": "semantic_host_connected_wall_layer_envelopes",
        "projected_common_area": area,
        "opening_normal_interval": list(opening_interval),
        "connected_normal_interval": [low, high],
        "room_face_normal_coordinate": 0.0,
        "layer_ids": sorted(selected),
        "layer_intervals": {key: list(eligible[key]) for key in sorted(selected)},
        "layer_widths_m": {key: layers[key][1] for key in sorted(selected)},
        "portal_projection": [list(p) for p in portal],
    }


def build_hosted_support_ledger(snapshot, geometry, *, epsilon=GEOMETRY_EPSILON,
                                contact_tolerance=CONTACT_TOLERANCE):
    """Certify source-envelope support, without observations, targets or scores.

    Direct face contact or a connected sequence of measured wall-layer envelopes
    supports a room. All layers cover the *same* opening/room projected patch.
    Declared widths are in millimetres and must match measured metre geometry;
    a missing or inconsistent width cannot justify a layer. Unresolved host
    scope is reported, and no existing object membership is changed.
    """
    if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(contact_tolerance) or contact_tolerance < epsilon:
        raise ValueError("Invalid geometry tolerances")
    raw, edges = _snapshot_records(snapshot)
    nodes = {}
    for key, record in raw.items():
        try:
            nodes[key] = normalize_semantic_node(record)
        except ValueError:
            pass
    shapes = geometry["nodes"]
    rooms = {key: validate_room_prism(shapes.get(key, {"id": key}), epsilon)
             for key, node in nodes.items() if node["label"] == "Room"}
    vertices, errors, layers = {}, {}, {}
    for key, node in nodes.items():
        if node["label"] not in {"Opening", "BoundarySurface"}:
            continue
        try:
            vertices[key] = _vertices(shapes.get(key), epsilon)
        except (ValueError, TypeError, KeyError) as exc:
            errors[key] = str(exc)
            continue
        if node["label"] == "BoundarySurface" and node["category"] == "wall":
            value = raw[key].get("properties", {}).get("attr_construction_width")
            try:
                width = float(value) * 0.001
            except (ValueError, TypeError):
                continue
            if math.isfinite(width) and width > epsilon:
                layers[key] = vertices[key], width
    semantic_pairs = {(edge["source"], edge["target"]) for edge in edges if edge["type"] == "HAS_OPENING"}
    relations = []
    for edge in edges:
        if edge["type"] != "HOSTED_BY":
            continue
        opening_id, host_id = edge["source"], edge["target"]
        op = raw[opening_id].get("properties", {})
        hp = raw[host_id].get("properties", {})
        host_tag, declared_host = hp.get("attr_ifc_tag"), op.get("attr_other_host_id")
        semantic_valid = ((host_id, opening_id) in semantic_pairs and host_tag is not None
                          and declared_host is not None and str(host_tag) == str(declared_host)
                          and nodes[opening_id]["label"] == "Opening"
                          and nodes[host_id]["label"] == "BoundarySurface"
                          and nodes[host_id]["category"] == "wall")
        evidence = {}
        if semantic_valid and opening_id in vertices and host_id in vertices:
            for room_id, room in rooms.items():
                if room["status"] != "valid_prism":
                    continue
                direct = _surface_contacts(shapes[opening_id]["polygons"], room, epsilon, contact_tolerance)
                rows = [{"basis": "opening_source_face_contact", **item} for item in direct]
                for face_index, face in enumerate(room["faces"]):
                    certificate = _layer_certificate(vertices[opening_id], host_id, face, layers, epsilon, contact_tolerance)
                    if certificate:
                        rows.append({"room_face_index": face_index, **certificate})
                if rows:
                    evidence[room_id] = rows
        relations.append({
            "edge_id": edge["id"], "source": opening_id, "target": host_id,
            "semantic_host_validated": semantic_valid,
            "semantic_evidence": {"reverse_HAS_OPENING": (host_id, opening_id) in semantic_pairs,
                                  "opening_declared_host_tag": declared_host, "host_ifc_tag": host_tag},
            "supported_room_ids": sorted(evidence), "room_evidence": evidence,
            "status": "supported" if evidence else "unresolved",
        })
    return {
        "policy": POLICY, "epsilon": epsilon, "contact_tolerance": contact_tolerance,
        "provenance": {"snapshot_content_sha256": content_hash(snapshot),
                       "geometry_content_sha256": content_hash(geometry),
                       "geometry_source": geometry.get("source", {})},
        "room_geometry_status": {key: value["status"] for key, value in rooms.items()},
        "relations": relations, "geometry_errors": errors,
        "threshold_basis": "Existing 1e-6 geometry epsilon and 0.05m contact tolerance; no retrieval outcome tuning.",
        "limitations": ["Wall envelopes may bridge source apertures; this is envelope support, not a material or visibility certificate.",
                        "Unresolved scope does not establish physical absence; unsupported HOSTED_BY edges are omitted only in this opt-in reference.",
                        "All baseline candidate nodes and non-HOSTED_BY relations are preserved.",
                        "No observations, GT labels, perception outputs, or retrieval outcomes enter preparation."],
    }


def scope_hosted_by(candidates, ledger, snapshot):
    """Copy a baseline pool and gate HOSTED_BY by source-room support only."""
    if ledger.get("policy") != POLICY or ledger.get("provenance", {}).get("snapshot_content_sha256") != content_hash(snapshot):
        raise ValueError("Support ledger policy or snapshot mismatch")
    rows = ledger.get("relations", [])
    lookup = {row["edge_id"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate support edge ID")
    result = deepcopy(candidates)
    if len({c["room_id"] for c in result}) != len(result):
        raise ValueError("Duplicate candidate ID")
    for before, candidate in zip(candidates, result):
        members = candidate.get("space_group", {}).get("member_room_ids", [candidate["room_id"]])
        if not members or any(key not in ledger["room_geometry_status"] for key in members):
            raise ValueError("Unknown source Room in candidate")
        kept, removed = [], []
        for edge in candidate["edges"]:
            if edge["type"] != "HOSTED_BY":
                kept.append(edge)
                continue
            row = lookup.get(edge["id"])
            if row is None or (row["source"], row["target"]) != (edge["source"], edge["target"]):
                raise ValueError("Unknown or mismatched HOSTED_BY edge")
            if set(members).intersection(row["supported_room_ids"]):
                kept.append(edge)
            else:
                removed.append(edge["id"])
        candidate["edges"] = kept
        candidate["reference_relation_scope"] = {
            "policy": POLICY, "ledger_content_sha256": content_hash(ledger),
            "baseline_candidate_content_sha256": content_hash(before),
            "removed_hosted_by_edge_ids": removed, "node_records_unchanged": True,
            "other_relation_records_unchanged": True,
        }
        assert candidate["nodes"] == before["nodes"]
    return result
