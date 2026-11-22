"""Read-only, room-scoped reference graphs for the experimental joint matcher.

Membership follows typed paths from ``spatial_scope`` and the existing room
queries. Container fallback supplies boundaries only, never sibling furniture.
It does not fabricate Room->boundary edges. Layered boundary collapse is not
repeated: semantic snapshots do not contain the geometry needed to justify it.

Stored ``confidence`` and ``evidence_score`` become gamma and epsilon. The
importer's explicit hierarchy edges have no such properties, so this adapter
records ``citygml_semantic_hierarchy_v1`` and supplies unit values only for those
typed base edges (including the importer's Furniture->Room INSIDE). Inferred
edges with missing metadata retain None. Existing metadata is never replaced.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
import re
from typing import Any

from citygml_sg.relations.spatial_scope import (
    FLOOR_LIKE_SURFACE_TYPES,
    WALL_LIKE_SURFACE_TYPES,
)


NORMALIZATION_VERSION = "semantic_reference_v2_fridge_compounds"
MEMBERSHIP_POLICY = "typed_room_paths_v1"
BASE_EDGE_POLICY = "citygml_semantic_hierarchy_v1"
GEOMETRY_TYPES = frozenset({
    "Geometry", "ImplicitGeometry", "Solid", "MultiSurface", "MultiCurve",
    "Polygon", "LinearRing", "Position",
})
LABEL_ALIASES = {
    "Room": "Room", "Furniture": "Furniture", "BuildingFurniture": "Furniture",
    "Installation": "Installation", "BuildingInstallation": "Installation",
    "IntBuildingInstallation": "Installation", "Opening": "Opening",
    "Door": "Opening", "Window": "Opening", "BoundarySurface": "BoundarySurface",
}
SURFACE_CATEGORIES = {
    **{name: ("FloorSurface", "floor") for name in FLOOR_LIKE_SURFACE_TYPES},
    **{name: ("WallSurface", "wall") for name in WALL_LIKE_SURFACE_TYPES},
    "CeilingSurface": ("CeilingSurface", "ceiling"),
    "OuterCeilingSurface": ("CeilingSurface", "ceiling"),
    "RoofSurface": ("RoofSurface", "roof"),
    "ClosureSurface": ("ClosureSurface", "closure"),
}
SURFACE_ALIASES = {
    **{name.lower(): name for name in SURFACE_CATEGORIES},
    "wall": "WallSurface", "wall_surface": "WallSurface",
    "floor": "FloorSurface", "floor_surface": "FloorSurface",
    "ceiling": "CeilingSurface", "ceiling_surface": "CeilingSurface",
    "roof": "RoofSurface", "ground": "GroundSurface",
}
CATEGORY_ALIASES = {
    "refrigerator": "fridge", "freezer": "fridge", "television": "tv",
    "display": "tv", "couch": "sofa", "pillar": "column", "기둥": "column",
    "wallsurface": "wall", "wall_surface": "wall", "floorsurface": "floor",
    "floor_surface": "floor", "groundsurface": "floor", "ground_surface": "floor",
    "ceilingsurface": "ceiling", "ceiling_surface": "ceiling",
    "roofsurface": "roof", "roof_surface": "roof", "closuresurface": "closure",
    "closure_surface": "closure", "ground": "floor",
}
CATEGORY_KEYWORDS = (
    ("sink", ("sink",)), ("counter", ("counter",)),
    ("cabinet", ("cabinet",)), ("storage", ("storage",)),
    ("fridge", ("fridge", "freezer", "refrigerator")),
    ("sofa", ("sofa", "couch")), ("tv", ("tv", "television", "display")),
    ("table", ("table",)), ("chair", ("chair",)), ("bed", ("bed",)),
    ("column", ("column", "pillar", "기둥")),
)
FRIDGE_COMPOUND_TOKENS = frozenset({"fridge", "freezer", "refrigerator"})
SEMANTIC_TYPES = frozenset(LABEL_ALIASES) | frozenset(SURFACE_CATEGORIES)
PREFETCH_TYPES = SEMANTIC_TYPES | {"Building", "BuildingPart"}
NODE_TYPE_ALIASES = {name.casefold(): name for name in PREFETCH_TYPES | GEOMETRY_TYPES}
BASE_RELATIONS = frozenset({
    "CONTAINS", "BOUNDED_BY", "HAS_OPENING", "INTERIOR_FURNITURE",
    "ROOM_INSTALLATION", "INTERIOR_BUILDING_INSTALLATION",
    "OUTER_BUILDING_INSTALLATION", "INTERIOR_ROOM", "CONSISTS_OF_BUILDING_PART",
})


def _properties(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("properties", {})
    if not isinstance(value, dict):
        raise ValueError("Reference properties must be an object")
    return value


def _node_type(raw: dict[str, Any]) -> str:
    props = _properties(raw)
    value = (raw.get("node_type") or raw.get("type") or raw.get("label")
             or props.get("node_type") or props.get("object_type"))
    if not value:
        value = next((label for label in raw.get("labels", [])
                      if label in PREFETCH_TYPES or label in GEOMETRY_TYPES), "Unknown")
    text = str(value)
    return NODE_TYPE_ALIASES.get(text.replace("_", "").casefold(), text)


def _node_id(raw: dict[str, Any]) -> str:
    props = _properties(raw)
    value = raw.get("id") or raw.get("node_id") or props.get("id") or props.get("gml_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Reference node must have a nonempty canonical string ID")
    return value


def normalize_semantic_node(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize semantic labels without deriving categories from node/room IDs.

    Explicit categories take precedence. Otherwise named semantic fields use an
    ordered, token-based vocabulary; sink/counter/cabinet/storage stay distinct.
    Furniture product names also expose fridge synonyms at CamelCase boundaries.
    Unknown names remain normalized text (or ``unknown``), not discarded nodes.
    Surface subtype canonicalization retains ``original_subtype`` for auditing.
    """
    props = _properties(raw)
    node_type = _node_type(raw)
    label = LABEL_ALIASES.get(node_type)
    if node_type in SURFACE_CATEGORIES:
        label = "BoundarySurface"
    if label is None:
        raise ValueError(f"Unsupported reference semantic node type: {node_type}")
    subtype = raw.get("subtype", props.get("subtype"))
    if subtype is not None and not isinstance(subtype, str):
        raise ValueError("Reference semantic subtype must be a string or null")
    category = str(raw.get("category") or props.get("category") or "").strip().lower()
    category = CATEGORY_ALIASES.get(category, category)
    original_subtype = subtype
    if label == "Room":
        subtype, category = None, "room"
    elif label == "Opening":
        original_subtype = (subtype or props.get("opening_type")
                            or (node_type if node_type != "Opening" else None))
        name = str(original_subtype or category).strip().lower()
        subtype = {"door": "Door", "window": "Window"}.get(name, original_subtype)
        category = str(subtype or category or "opening").lower()
    elif label == "BoundarySurface":
        original_subtype = (subtype or props.get("surface_type")
                            or (node_type if node_type != "BoundarySurface" else None))
        surface = str(original_subtype or category or "BoundarySurface")
        surface = SURFACE_ALIASES.get(surface.lower(), surface)
        subtype, category = SURFACE_CATEGORIES.get(surface, (surface, category or "boundary"))
    elif not category:
        fields = ("gml_name", "attr_ifc_object_type", "attr_identity_data_type_name",
                  "attr_other_category_0", "attr_other_family", "attr_other_family_and_type")
        values = [str(props.get(key) or raw.get(key) or "") for key in fields]
        text = re.sub(r"[_\-\s]+", " ", " ".join(values)).strip().lower()
        tokens = set(re.findall(r"[^\W_]+", text, flags=re.UNICODE))
        if label == "Furniture":
            # Imported product families may concatenate words, e.g.
            # IntegratedFrenchDoorFridgeFreezer. Split before lowercasing;
            # retain only fridge synonyms so other category rules stay fixed.
            compound_text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", " ".join(values))
            compound_text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", compound_text)
            compound_tokens = set(re.findall(r"[^\W_]+", compound_text.lower(), flags=re.UNICODE))
            tokens.update(compound_tokens.intersection(FRIDGE_COMPOUND_TOKENS))
        category = next((name for name, words in CATEGORY_KEYWORDS if tokens.intersection(words)),
                        re.sub(r"[_\-\s]+", " ", next((v for v in values if v), "unknown"))
                        .strip().lower())
    return {
        "id": _node_id(raw), "label": label, "subtype": subtype,
        "category": category, "original_subtype": original_subtype,
    }


def _unit_interval(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Reference edge {field} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Reference edge {field} must be numeric") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"Reference edge {field} must be finite and within [0, 1]")
    return result


def _normalize_edge(raw: dict[str, Any], raw_nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = raw.get("source_id", raw.get("source"))
    target = raw.get("target_id", raw.get("target"))
    relation = str(raw.get("relation") or raw.get("type") or "").strip().upper()
    if not isinstance(source, str) or not isinstance(target, str) or not relation:
        raise ValueError("Reference edge requires source, target and relation type")
    metadata = dict(_properties(raw))
    for field in ("gamma", "epsilon", "confidence", "evidence_score", "method",
                  "computed_at", "provenance"):
        if field in raw:
            metadata[field] = raw[field]
    gamma = _unit_interval(metadata.get("gamma", metadata.get("confidence")), "gamma")
    epsilon = _unit_interval(metadata.get("epsilon", metadata.get("evidence_score")), "epsilon")
    provenance_value = metadata.get("provenance")
    if isinstance(provenance_value, dict):
        provenance = json.dumps(provenance_value, sort_keys=True, ensure_ascii=False)
    else:
        provenance = str(provenance_value or "")
    if not provenance:
        provenance = ":".join(str(metadata[key]) for key in ("source", "method")
                              if metadata.get(key))
    source_type = _node_type(raw_nodes[source]) if source in raw_nodes else "Unknown"
    target_type = _node_type(raw_nodes[target]) if target in raw_nodes else "Unknown"
    semantic_inside = (relation == "INSIDE" and source_type in {"BuildingFurniture", "Furniture"}
                       and target_type == "Room")
    base_edge = relation in BASE_RELATIONS or semantic_inside
    # A geometric/inferred producer must not acquire semantic unit metadata.
    semantic_producer = not provenance or provenance.startswith(("semantic_", BASE_EDGE_POLICY))
    if base_edge and semantic_producer and (gamma is None or epsilon is None):
        gamma = 1.0 if gamma is None else gamma
        epsilon = 1.0 if epsilon is None else epsilon
        metadata["adapter_metadata_policy"] = BASE_EDGE_POLICY
        provenance = provenance or BASE_EDGE_POLICY
    provenance = provenance or "stored_edge_metadata_unavailable"
    edge_id = raw.get("id") or raw.get("edge_id") or metadata.get("id")
    if edge_id is None:
        stable_metadata = {key: value for key, value in metadata.items()
                           if key not in {"computed_at", "created_at", "updated_at", "element_id"}}
        identity = json.dumps([source, relation, target, stable_metadata], sort_keys=True,
                              ensure_ascii=False, separators=(",", ":"))
        edge_id = "edge:" + sha256(identity.encode("utf-8")).hexdigest()[:24]
    return {
        "id": str(edge_id), "source": source, "target": target, "type": relation,
        "gamma": gamma, "epsilon": epsilon, "provenance": provenance, "metadata": metadata,
    }


def _snapshot_records(
    snapshot: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    nodes = snapshot.get("nodes")
    edges = snapshot.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Reference snapshot requires nodes and edges lists")
    raw_nodes: dict[str, dict[str, Any]] = {}
    for raw in nodes:
        if not isinstance(raw, dict):
            raise ValueError("Reference node must be an object")
        node_id = _node_id(raw)
        if node_id in raw_nodes:
            raise ValueError(f"Duplicate reference node ID: {node_id}")
        raw_nodes[node_id] = raw
    normalized_edges: dict[str, dict[str, Any]] = {}
    for raw in edges:
        if not isinstance(raw, dict):
            raise ValueError("Reference edge must be an object")
        source = raw.get("source_id", raw.get("source"))
        target = raw.get("target_id", raw.get("target"))
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError("Reference edge requires string source and target IDs")
        if source not in raw_nodes or target not in raw_nodes:
            raise ValueError("Reference edge has an unknown endpoint")
        # A full graph export may contain millions of geometry edges. Keep its
        # inventory count, but do not hash/copy geometry into the search graph.
        if (_node_type(raw_nodes[source]) not in PREFETCH_TYPES
                or _node_type(raw_nodes[target]) not in PREFETCH_TYPES):
            continue
        edge = _normalize_edge(raw, raw_nodes)
        previous = normalized_edges.get(edge["id"])
        if previous is not None and previous != edge:
            raise ValueError(f"Conflicting reference edge ID: {edge['id']}")
        normalized_edges[edge["id"]] = edge
    return raw_nodes, sorted(normalized_edges.values(), key=lambda edge: edge["id"])


def build_room_candidates(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Build every Room candidate, preserving typed path evidence and empty Rooms.

    Supports graph export ``type/properties`` and canonical ``label/category``
    records. Membership provenance is separate from relation evidence: a
    fallback never creates a scored edge that is absent from the stored graph.
    """
    raw_nodes, edges = _snapshot_records(snapshot)
    nodes = {node_id: normalize_semantic_node(raw) for node_id, raw in raw_nodes.items()
             if _node_type(raw) in SEMANTIC_TYPES}
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        outgoing[edge["source"]].append(edge)
        incoming[edge["target"]].append(edge)
    rooms = sorted(node_id for node_id, node in nodes.items() if node["label"] == "Room")
    candidates: list[dict[str, Any]] = []
    for room_id in rooms:
        membership: dict[str, list[dict[str, Any]]] = defaultdict(list)

        def include(node_id: str, source: str, path: list[str]) -> None:
            if node_id not in nodes or (nodes[node_id]["label"] == "Room" and node_id != room_id):
                return
            evidence = {"source": source, "edge_ids": path}
            if evidence not in membership[node_id]:
                membership[node_id].append(evidence)

        include(room_id, "candidate_room", [])
        direct_boundaries: list[str] = []
        for edge in outgoing[room_id]:
            target = nodes.get(edge["target"], {})
            if (target.get("label") in {"Furniture", "Installation"}
                    and edge["type"] in {"CONTAINS", "INTERIOR_FURNITURE", "ROOM_INSTALLATION",
                                         "INTERIOR_BUILDING_INSTALLATION",
                                         "OUTER_BUILDING_INSTALLATION"}):
                include(edge["target"], "room_" + edge["type"].lower(), [edge["id"]])
            if edge["type"] == "BOUNDED_BY" and target.get("label") == "BoundarySurface":
                include(edge["target"], "room_boundary", [edge["id"]])
                direct_boundaries.append(edge["target"])
        for edge in incoming[room_id]:
            source_label = nodes.get(edge["source"], {}).get("label")
            if edge["type"] == "INSIDE" and source_label in {"Furniture", "Installation"}:
                include(edge["source"], "room_inside", [edge["id"]])
            if edge["type"] == "CONNECTS" and source_label == "Opening":
                include(edge["source"], "connected_opening", [edge["id"]])
                for hosted in outgoing[edge["source"]]:
                    host = nodes.get(hosted["target"], {})
                    if (hosted["type"] == "HOSTED_BY" and host.get("category") == "wall"
                            and host.get("label") == "BoundarySurface"):
                        include(hosted["target"], "connected_opening_host_wall",
                                [edge["id"], hosted["id"]])
        if not direct_boundaries:
            for parent in incoming[room_id]:
                parent_type = _node_type(raw_nodes[parent["source"]])
                if (parent["type"] not in {"INTERIOR_ROOM", "CONTAINS"}
                        or parent_type not in {"Building", "BuildingPart"}):
                    continue
                for edge in outgoing[parent["source"]]:
                    if (edge["type"] == "BOUNDED_BY"
                            and nodes.get(edge["target"], {}).get("label") == "BoundarySurface"):
                        include(edge["target"], "container_boundary_fallback",
                                [parent["id"], edge["id"]])
        # The existing query permits one wall/floor adjacency step, not a flood fill.
        boundary_seeds = [node_id for node_id in membership
                          if nodes[node_id]["label"] == "BoundarySurface"]
        # Freeze the pre-expansion evidence as well as the seed IDs. Otherwise
        # paths appended to another seed are reused later in this loop, making
        # a one-step membership expansion claim multi-step/cyclic provenance.
        boundary_seed_paths = {node_id: list(membership[node_id]) for node_id in boundary_seeds}
        for boundary_id in boundary_seeds:
            for edge in outgoing[boundary_id] + incoming[boundary_id]:
                other = edge["target"] if edge["source"] == boundary_id else edge["source"]
                pair_categories = {nodes[boundary_id]["category"],
                                   nodes.get(other, {}).get("category")}
                if (edge["type"] == "ADJACENT_SURFACE" and other in nodes
                        and nodes[other]["label"] == "BoundarySurface"
                        and pair_categories == {"wall", "floor"}):
                    for path in boundary_seed_paths[boundary_id]:
                        include(other, path["source"] + "_surface_adjacency",
                                path["edge_ids"] + [edge["id"]])
        for boundary_id in list(membership):
            if nodes[boundary_id]["label"] != "BoundarySurface":
                continue
            for edge in outgoing[boundary_id] + incoming[boundary_id]:
                if edge["source"] == boundary_id and edge["type"] == "HAS_OPENING":
                    opening_id = edge["target"]
                elif edge["target"] == boundary_id and edge["type"] == "HOSTED_BY":
                    opening_id = edge["source"]
                else:
                    continue
                if nodes.get(opening_id, {}).get("label") == "Opening":
                    for path in list(membership[boundary_id]):
                        include(opening_id, path["source"] + "_opening",
                                path["edge_ids"] + [edge["id"]])
        selected_ids = set(membership)
        props = _properties(raw_nodes[room_id])
        candidates.append({
            "room_id": room_id,
            "room_name": str(props.get("gml_name")
                             or raw_nodes[room_id].get("room_name") or room_id),
            "nodes": [nodes[node_id] for node_id in sorted(selected_ids)],
            "edges": [edge for edge in edges
                      if edge["source"] in selected_ids and edge["target"] in selected_ids],
            "membership": {
                "policy": MEMBERSHIP_POLICY, "normalization_version": NORMALIZATION_VERSION,
                "node_sources": dict(sorted(membership.items())),
                "layered_boundary_collapse": "not_applied_semantic_snapshot",
            },
        })
    occurrence = Counter(node["id"] for candidate in candidates for node in candidate["nodes"])
    for candidate in candidates:
        candidate["membership"]["shared_node_ids"] = [node["id"] for node in candidate["nodes"]
                                                       if occurrence[node["id"]] > 1]
    return candidates


def reference_snapshot_stats(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Separate full stored inventory, semantic prefetch, and candidate union size."""
    candidates = build_room_candidates(snapshot)
    type_counts = Counter(_node_type(node) for node in snapshot["nodes"])
    inventory = snapshot.get("inventory", {})
    full_counts = inventory.get("node_type_counts", dict(type_counts))
    return {
        "room_count": len(candidates),
        "total_node_count": sum(full_counts.values()),
        "geometry_node_count": sum(count for kind, count in full_counts.items()
                                   if kind in GEOMETRY_TYPES),
        "prefetched_node_count": len(snapshot["nodes"]),
        "prefetched_edge_count": len(snapshot["edges"]),
        "search_semantic_node_count": len({node["id"] for candidate in candidates
                                           for node in candidate["nodes"]}),
        "search_semantic_edge_count": len({edge["id"] for candidate in candidates
                                           for edge in candidate["edges"]}),
        "node_type_counts": dict(sorted(full_counts.items())),
    }


def fetch_reference_snapshot(session: Any) -> dict[str, Any]:
    """Prefetch semantic nodes/edges with an existing read-only Neo4j session.

    The caller owns credentials, transaction/session lifetime and read access.
    Three read queries retrieve complete node-type counts and semantic data;
    geometry payloads are never downloaded. Database element IDs are not used.
    """
    inventory = session.run("""
        MATCH (n:CityObject)
        RETURN coalesce(n.node_type, head([label IN labels(n) WHERE label <> 'CityObject']),
                        'Unknown') AS node_type, count(n) AS count
        ORDER BY node_type
    """).data()
    nodes = session.run("""
        MATCH (n:CityObject)
        WHERE n.node_type IN $types OR any(label IN labels(n) WHERE label IN $types)
        RETURN n.id AS id, n.node_type AS type, labels(n) AS labels, properties(n) AS properties
        ORDER BY id
    """, types=sorted(PREFETCH_TYPES)).data()
    edges = session.run("""
        MATCH (s:CityObject)-[r]->(t:CityObject)
        WHERE (s.node_type IN $types OR any(label IN labels(s) WHERE label IN $types))
          AND (t.node_type IN $types OR any(label IN labels(t) WHERE label IN $types))
        RETURN r.id AS id, s.id AS source_id, t.id AS target_id,
               type(r) AS relation, properties(r) AS properties
        ORDER BY source_id, relation, target_id
    """, types=sorted(PREFETCH_TYPES)).data()
    return {
        "nodes": nodes, "edges": edges,
        "snapshot_scope": "semantic_with_container_context",
        "inventory": {"node_type_counts": {row["node_type"]: int(row["count"])
                                           for row in inventory}},
    }
