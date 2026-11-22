"""Opt-in, conservative Room membership proposals from explicit CityGML geometry.

No ground truth, observations, score, or database is used. Rooms must validate
as closed vertical prisms. Object inclusion uses the convex envelope of every
source vertex, a sufficient (not necessary) full-shape containment certificate.
Boundaries/openings use positive polygon-area contact with a Room face, allowing
shared boundaries. AABBs only reject impossible pairs. Unsupported geometry and
ambiguous assignments remain in the ledger. Existing furniture ownership is
preserved and independently diagnosed. This module never creates scored edges.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from citygml_sg.retrieval.reference_graph import normalize_semantic_node

POLICY = "validated_prism_membership_v1"
GEOMETRY_EPSILON = 1e-6  # Existing spatial.intersection_epsilon, metres.
CONTACT_TOLERANCE = 0.05  # Existing spatial.touch_epsilon, metres.
GML = "{http://www.opengis.net/gml}"
XLINK = "{http://www.w3.org/1999/xlink}"


def _digest(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _near(a, b, epsilon):
    return max(abs(x - y) for x, y in zip(a, b)) <= epsilon


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _area(points):
    if len(points) < 3:
        return 0.0
    origin = points[0]
    return sum(_cross(origin, points[i], points[i + 1])
               for i in range(1, len(points) - 1)) / 2


def _segment_parameters(a, b, c, d, epsilon):
    """Parameters of intersections on AB, including collinear CD endpoints."""
    vx, vy = b[0] - a[0], b[1] - a[1]
    wx, wy = d[0] - c[0], d[1] - c[1]
    denom = vx * wy - vy * wx
    length = math.hypot(vx, vy)
    other_length = math.hypot(wx, wy)
    if length <= epsilon:
        return []
    cx, cy = c[0] - a[0], c[1] - a[1]
    if abs(denom) > epsilon * max(length, other_length):
        t = (cx * wy - cy * wx) / denom
        u = (cx * vy - cy * vx) / denom
        if -epsilon / length <= t <= 1 + epsilon / length and -epsilon / max(other_length, epsilon) <= u <= 1 + epsilon / max(other_length, epsilon):
            return [min(1.0, max(0.0, t))]
        return []
    if abs(cx * vy - cy * vx) > epsilon * length:
        return []
    length2 = length * length
    candidates = [((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / length2 for p in (c, d)]
    if max(candidates) < 0 or min(candidates) > 1:
        return []
    return [max(0.0, min(1.0, t)) for t in candidates]


def _point_location(point, polygon, epsilon):
    """Return outside/boundary/inside for a simple polygon without holes."""
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2)) if length2 else 0.0
        if math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy) <= epsilon:
            return "boundary"
        if (a[1] > point[1]) != (b[1] > point[1]):
            crossing = a[0] + (point[1] - a[1]) * dx / dy
            if point[0] < crossing:
                inside = not inside
    return "inside" if inside else "outside"


def _simple(polygon, epsilon):
    if len(polygon) < 3 or abs(_area(polygon)) <= epsilon * epsilon:
        return False
    segments = list(zip(polygon, polygon[1:] + polygon[:1]))
    for i, (a, b) in enumerate(segments):
        if _near(a, b, epsilon):
            return False
        for j in range(i + 1, len(segments)):
            if j == i + 1 or (i == 0 and j == len(segments) - 1):
                continue
            if _segment_parameters(a, b, *segments[j], epsilon):
                return False
    return True


def _hull(points):
    points = sorted(set(tuple(p[:2]) for p in points))
    if len(points) < 3:
        return points
    lower, upper = [], []
    for sequence, target in ((points, lower), (reversed(points), upper)):
        for p in sequence:
            while len(target) >= 2 and _cross(target[-2], target[-1], p) <= 0:
                target.pop()
            target.append(p)
    return lower[:-1] + upper[:-1]


def _covers(outer, inner, epsilon):
    if any(_point_location(p, outer, epsilon) == "outside" for p in inner):
        return False
    if len(inner) < 2:
        return bool(inner)
    for a, b in zip(inner, inner[1:] + inner[:1]):
        splits = [0.0, 1.0]
        for c, d in zip(outer, outer[1:] + outer[:1]):
            splits.extend(_segment_parameters(a, b, c, d, epsilon))
        splits = sorted(set(splits))
        for left, right in zip(splits, splits[1:]):
            t = (left + right) / 2
            p = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if _point_location(p, outer, epsilon) == "outside":
                return False
    return True


def _triangles(polygon, epsilon):
    points = list(polygon)
    if _area(points) < 0:
        points.reverse()
    triangles = []
    while len(points) > 3:
        for i in range(len(points)):
            a, b, c = points[i - 1], points[i], points[(i + 1) % len(points)]
            if _cross(a, b, c) <= epsilon * epsilon:
                continue
            others = [p for j, p in enumerate(points) if j not in {(i - 1) % len(points), i, (i + 1) % len(points)}]
            if any(min(_cross(a, b, p), _cross(b, c, p), _cross(c, a, p)) >= -epsilon * epsilon for p in others):
                continue
            triangles.append([a, b, c])
            points.pop(i)
            break
        else:
            # Collinear vertices can prevent finding an ear; remove only an
            # intermediate collinear vertex, preserving the polygon boundary.
            for i in range(len(points)):
                a, b, c = points[i - 1], points[i], points[(i + 1) % len(points)]
                if abs(_cross(a, b, c)) <= epsilon * epsilon and (b[0]-a[0])*(b[0]-c[0]) + (b[1]-a[1])*(b[1]-c[1]) <= 0:
                    points.pop(i)
                    break
            else:
                raise ValueError("polygon_triangulation_failed")
    if len(points) == 3 and abs(_area(points)) > epsilon * epsilon:
        triangles.append(points)
    return triangles


def _triangle_overlap(first, second):
    output = list(first)
    for a, b in zip(second, second[1:] + second[:1]):
        source, output = output, []
        if not source:
            break
        previous = source[-1]
        pv = _cross(a, b, previous)
        for current in source:
            cv = _cross(a, b, current)
            if (cv >= 0) != (pv >= 0):
                t = pv / (pv - cv)
                output.append((previous[0] + t*(current[0]-previous[0]), previous[1] + t*(current[1]-previous[1])))
            if cv >= 0:
                output.append(current)
            previous, pv = current, cv
    return abs(_area(output))


def load_citygml_geometry(path, node_ids=None):
    """Read explicit direct LoD geometry; unresolved xlinks stay explicit."""
    path = Path(path)
    before = _digest(path)
    root = ET.parse(path).getroot()
    selected = set(node_ids) if node_ids is not None else None
    nodes = {}
    for element in root.iter():
        node_id = element.get(GML + "id")
        if not node_id or (selected is not None and node_id not in selected):
            continue
        properties = [c for c in element if c.tag.rsplit("}", 1)[-1].startswith("lod")]
        if selected is None and not properties:
            continue
        errors, polygons = [], []
        for prop in properties:
            if any(XLINK + "href" in child.attrib for child in prop.iter()):
                errors.append("unresolved_geometry_xlink")
            for polygon in prop.iter(GML + "Polygon"):
                rings = []
                for boundary in list(polygon):
                    if boundary.tag not in {GML + "exterior", GML + "interior"}:
                        continue
                    linear = boundary.find(GML + "LinearRing")
                    if linear is None:
                        errors.append("unsupported_ring")
                        continue
                    values = []
                    for coordinate in list(linear):
                        if coordinate.tag not in {GML + "posList", GML + "pos"}:
                            continue
                        if coordinate.get("srsDimension", "3") != "3":
                            errors.append("unsupported_coordinate_dimension")
                            continue
                        try:
                            numbers = [float(x) for x in (coordinate.text or "").split()]
                        except ValueError:
                            errors.append("invalid_coordinates")
                            continue
                        if len(numbers) % 3 or any(not math.isfinite(x) for x in numbers):
                            errors.append("invalid_coordinates")
                            continue
                        values.extend([numbers[i:i+3] for i in range(0, len(numbers), 3)])
                    rings.append((boundary.tag, values))
                exteriors = [values for tag, values in rings if tag == GML + "exterior"]
                if len(exteriors) != 1:
                    errors.append("missing_or_duplicate_exterior")
                    continue
                polygons.append({"exterior": exteriors[0], "interiors": [values for tag, values in rings if tag == GML + "interior"]})
        if not polygons:
            errors.append("missing_explicit_polygons")
        if node_id in nodes:
            raise ValueError("Duplicate geometry source node ID")
        nodes[node_id] = {"id": node_id, "source_type": element.tag.rsplit("}", 1)[-1], "polygons": polygons, "geometry_errors": sorted(set(errors))}
    if _digest(path) != before:
        raise ValueError("Source changed while reading geometry")
    return {"source": {"path": str(path.resolve()), "sha256": before, "crs_names": sorted({e.get("srsName") for e in root.iter() if e.get("srsName")})}, "nodes": nodes}


def _ring(raw, epsilon):
    if len(raw) < 4 or any(len(p) != 3 or any(not math.isfinite(v) for v in p) for p in raw):
        raise ValueError("invalid_ring_coordinates")
    if not _near(raw[0], raw[-1], epsilon):
        raise ValueError("unclosed_ring")
    return [tuple(p) for p in raw[:-1]]


def validate_room_prism(node_geometry, epsilon=GEOMETRY_EPSILON):
    """Certify a simple closed vertical prism; unsupported forms are unresolved."""
    try:
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("invalid_epsilon")
        if node_geometry.get("geometry_errors"):
            raise ValueError("source_geometry_errors")
        faces = []
        for polygon in node_geometry.get("polygons", []):
            if polygon.get("interiors"):
                raise ValueError("room_holes_unsupported")
            faces.append(_ring(polygon["exterior"], epsilon))
        if not faces:
            raise ValueError("missing_faces")
        zs = [p[2] for face in faces for p in face]
        zmin, zmax = min(zs), max(zs)
        if zmax - zmin <= epsilon:
            raise ValueError("zero_height")
        floors = [f for f in faces if all(abs(p[2]-zmin) <= epsilon for p in f)]
        tops = [f for f in faces if all(abs(p[2]-zmax) <= epsilon for p in f)]
        if len(floors) != 1 or len(tops) != 1:
            raise ValueError("expected_single_floor_and_ceiling")
        footprint = [p[:2] for p in floors[0]]
        if not _simple(footprint, epsilon):
            raise ValueError("invalid_footprint")
        ceiling = [p[:2] for p in tops[0]]
        if len(ceiling) != len(footprint) or not any(all(_near(p, ceiling[(offset+direction*i) % len(ceiling)], epsilon) for i, p in enumerate(footprint)) for offset in range(len(ceiling)) for direction in (1, -1)):
            raise ValueError("non_matching_floor_ceiling")
        walls = [f for f in faces if f is not floors[0] and f is not tops[0]]
        if len(walls) != len(footprint):
            raise ValueError("incomplete_or_duplicate_walls")
        expected = [[(a[0], a[1], zmin), (b[0], b[1], zmin), (b[0], b[1], zmax), (a[0], a[1], zmax)] for a, b in zip(footprint, footprint[1:] + footprint[:1])]
        used = set()
        for wall in walls:
            matches = [i for i, vertices in enumerate(expected) if len(wall) == 4 and all(any(_near(p, v, epsilon) for v in vertices) for p in wall) and all(any(_near(v, p, epsilon) for p in wall) for v in vertices)]
            if len(matches) != 1 or matches[0] in used:
                raise ValueError("non_vertical_or_duplicate_wall")
            # Prevent bow-tie wall ordering even if the vertex set is right.
            a, b = footprint[matches[0]], footprint[(matches[0]+1) % len(footprint)]
            length = math.dist(a, b)
            projected = [((p[0]-a[0])*(b[0]-a[0])/length + (p[1]-a[1])*(b[1]-a[1])/length, p[2]) for p in wall]
            if not _simple(projected, epsilon):
                raise ValueError("invalid_wall_ring")
            used.add(matches[0])
        return {"status": "valid_prism", "room_id": node_geometry.get("id"), "footprint": footprint, "z_min": zmin, "z_max": zmax, "face_count": len(faces), "closed_prism_verified": True, "faces": faces, "volume": abs(_area(footprint))*(zmax-zmin)}
    except (ValueError, TypeError, KeyError) as error:
        return {"status": "unresolved", "room_id": node_geometry.get("id"), "reason": str(error)}


def point_in_prism(point, prism, epsilon=GEOMETRY_EPSILON):
    if prism.get("status") != "valid_prism":
        return "unresolved"
    if len(point) != 3 or any(not math.isfinite(v) for v in point):
        return "unresolved"
    if point[2] < prism["z_min"]-epsilon or point[2] > prism["z_max"]+epsilon:
        return "outside"
    location = _point_location(point[:2], prism["footprint"], epsilon)
    if location == "outside":
        return location
    if location == "boundary" or abs(point[2]-prism["z_min"]) <= epsilon or abs(point[2]-prism["z_max"]) <= epsilon:
        return "boundary"
    return "inside"


def _frame(face, epsilon):
    origin = face[0]
    for b, c in zip(face[1:], face[2:]):
        u = [b[i]-origin[i] for i in range(3)]
        v = [c[i]-origin[i] for i in range(3)]
        normal = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
        length = math.sqrt(sum(x*x for x in normal))
        if length > epsilon*epsilon:
            normal = [x/length for x in normal]
            ulength = math.sqrt(sum(x*x for x in u))
            u = [x/ulength for x in u]
            v = [normal[1]*u[2]-normal[2]*u[1], normal[2]*u[0]-normal[0]*u[2], normal[0]*u[1]-normal[1]*u[0]]
            return origin, normal, u, v
    raise ValueError("degenerate_polygon")


def _project(point, frame):
    origin, normal, u, v = frame
    delta = [point[i]-origin[i] for i in range(3)]
    return (sum(delta[i]*u[i] for i in range(3)), sum(delta[i]*v[i] for i in range(3)))


def _surface_contacts(polygons, prism, epsilon, tolerance):
    contacts = []
    for face_index, room_face in enumerate(prism["faces"]):
        frame = _frame(room_face, epsilon)
        origin, normal, _, _ = frame
        target = [_project(p, frame) for p in room_face]
        target_triangles = _triangles(target, epsilon)
        matches = []
        for polygon_index, polygon in enumerate(polygons):
            if polygon.get("interiors"):
                continue  # No exterior-only contact over a hole.
            vertices = _ring(polygon["exterior"], epsilon)
            distances = [sum((p[i]-origin[i])*normal[i] for i in range(3)) for p in vertices]
            if max(abs(d) for d in distances) > tolerance:
                continue
            try:
                object_frame = _frame(vertices, epsilon)
            except ValueError:
                continue
            if any(abs(sum((p[i]-object_frame[0][i])*object_frame[1][i]
                           for i in range(3))) > epsilon for p in vertices):
                continue  # A warped polygon cannot certify plane contact.
            if 1-abs(sum(a*b for a,b in zip(normal, object_frame[1]))) > epsilon*epsilon:
                continue
            projected = [_project(p, frame) for p in vertices]
            if not _simple(projected, epsilon):
                continue
            try:
                triangles = _triangles(projected, epsilon)
            except ValueError:
                continue
            area = sum(_triangle_overlap(a,b) for a in triangles for b in target_triangles)
            if area > epsilon*epsilon:
                matches.append({"source_polygon_index": polygon_index, "projected_overlap_area": area, "max_plane_distance": max(abs(d) for d in distances)})
        if matches:
            contacts.append({"room_face_index": face_index, "matching_polygon_count": len(matches), "max_projected_overlap_area": max(m["projected_overlap_area"] for m in matches), "minimum_max_plane_distance": min(m["max_plane_distance"] for m in matches), "source_polygon_indices": [m["source_polygon_index"] for m in matches]})
    return contacts


def propose_geometric_memberships(snapshot, geometry, *, epsilon=GEOMETRY_EPSILON, contact_tolerance=CONTACT_TOLERANCE):
    """Return a strict allowed-node ledger; callers induce stored edges only."""
    if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(contact_tolerance) or contact_tolerance < epsilon:
        raise ValueError("Invalid geometry tolerances")
    raw = {node["id"]: node for node in snapshot["nodes"]}
    normalized = {}
    for node_id, node in raw.items():
        try:
            normalized[node_id] = normalize_semantic_node(node)
        except ValueError:
            pass
    room_ids = sorted(k for k,n in normalized.items() if n["label"] == "Room")
    shapes = geometry["nodes"]
    rooms = {k: validate_room_prism(shapes.get(k, {"id":k,"geometry_errors":["missing_geometry"]}), epsilon) for k in room_ids}
    allowed = {k:{k} for k in room_ids}
    semantic = {}
    for edge in snapshot["edges"]:
        source = edge.get("source_id", edge.get("source"))
        target = edge.get("target_id", edge.get("target"))
        relation = edge.get("relation",edge.get("type"))
        if relation == "INSIDE" and target in allowed and normalized.get(source,{}).get("label") == "Furniture":
            semantic.setdefault(source,set()).add(target)
        if relation in {"INTERIOR_FURNITURE", "CONTAINS"} and source in allowed and normalized.get(target,{}).get("label") == "Furniture":
            semantic.setdefault(target,set()).add(source)
    assignments = []
    for node_id,node in sorted(normalized.items()):
        if node["label"] == "Room":
            continue
        item = {"node_id":node_id,"label":node["label"],"category":node["category"],"status":"unresolved","proposed_room_ids":[],"semantic_room_ids":sorted(semantic.get(node_id,set()))}
        for room_id in semantic.get(node_id,set()):
            allowed[room_id].add(node_id)
        shape = shapes.get(node_id)
        try:
            if not shape or shape.get("geometry_errors"):
                raise ValueError("missing_or_unsupported_geometry")
            vertices = [p for polygon in shape["polygons"] for ring in [polygon["exterior"]]+polygon.get("interiors",[]) for p in _ring(ring,epsilon)]
            if not vertices:
                raise ValueError("empty_geometry")
            hull = _hull(vertices)
            zmin,zmax = min(p[2] for p in vertices),max(p[2] for p in vertices)
            bounds = [[min(p[i] for p in vertices) for i in range(3)],[max(p[i] for p in vertices) for i in range(3)]]
            item["source_geometry"]={"polygon_count":len(shape["polygons"]),"vertex_count":len(vertices),"convex_envelope_xy":hull,"z_range":[zmin,zmax]}
            containing,contacts,possible = [],{},[]
            for room_id,prism in rooms.items():
                if prism["status"] != "valid_prism":
                    continue
                fp = prism["footprint"]
                if zmin > prism["z_max"]+contact_tolerance or zmax < prism["z_min"]-contact_tolerance or any(bounds[0][i] > max(p[i] for p in fp)+contact_tolerance or bounds[1][i] < min(p[i] for p in fp)-contact_tolerance for i in (0,1)):
                    continue
                possible.append(room_id)
                if zmin >= prism["z_min"]-epsilon and zmax <= prism["z_max"]+epsilon and _covers(fp,hull,epsilon):
                    containing.append(room_id)
                if node["label"] in {"BoundarySurface","Opening"}:
                    evidence = _surface_contacts(shape["polygons"],prism,epsilon,contact_tolerance)
                    if evidence:
                        contacts[room_id]=evidence
            item.update({"full_shape_containment_room_ids":containing,"broad_phase_room_ids":possible,"surface_contact_evidence":contacts})
            if node["label"] == "Furniture":
                item["status"]="semantic_preserved"
                item["proposed_room_ids"]=item["semantic_room_ids"]
                item["geometry_membership_conflict"]=set(containing)!=set(item["semantic_room_ids"])
            elif node["label"] == "Installation":
                if len(containing)==1:
                    item["status"]="assigned_containment"
                    item["proposed_room_ids"]=containing
                else:
                    item["status"]="ambiguous" if len(containing)>1 or possible else "unresolved"
                    item["reason"]="multiple_containing_rooms" if len(containing)>1 else "no_unique_full_shape_certificate"
            elif contacts:
                item["status"]="shared_contact" if len(contacts)>1 else "assigned_contact"
                item["proposed_room_ids"]=sorted(contacts)
            else:
                item["reason"]="no_certified_surface_contact"
            for room_id in item["proposed_room_ids"]:
                allowed[room_id].add(node_id)
        except (ValueError,TypeError,KeyError) as error:
            item["reason"]=str(error)
            if item["semantic_room_ids"]:
                item["status"]="semantic_preserved_geometry_unresolved"
                item["proposed_room_ids"]=item["semantic_room_ids"]
        assignments.append(item)
    return {"policy":POLICY,"geometry_epsilon":epsilon,"contact_tolerance":contact_tolerance,
            "threshold_basis":"Existing spatial.intersection_epsilon=1e-6 and spatial.touch_epsilon=0.05; frozen engineering policy, not accuracy tuning.",
            "provenance":{"snapshot_content_sha256":_canonical_hash(snapshot),"geometry_source":geometry.get("source",{}),"geometry_nodes_content_sha256":_canonical_hash(shapes)},
            "room_geometry":rooms,"assignments":assignments,
            "allowed_node_ids_by_room":{k:sorted(v) for k,v in allowed.items()},
            "summary":{"room_count":len(room_ids),"room_status_counts":dict(Counter(r["status"] for r in rooms.values())),"assignment_status_counts":dict(Counter(r["status"] for r in assignments))},
            "limitations":["Convex-envelope containment is sufficient but conservative; rejected envelopes may hide valid nonconvex object containment.","Surface contact certifies nearby overlapping source faces, not physical visibility or observation identity.","No new relation edges, GT labels, or annotation certification are created."]}
