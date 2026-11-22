"""Induced semantic subgraphs within an undirected hop radius of each Room.

The input domain is an explicit semantic World Graph with Building,
BuildingPart, Room, BoundarySurface, Opening, Furniture and Installation nodes.
The aliases already recognized by ``reference_graph`` are supported, including
CityGML surface subtypes. Geometry and other node types are rejected throughout
the input, even when disconnected from a requested Room; no implicit filtering
can remove a path or change its length.

Traversal uses every supplied edge without direction or relation filtering.
The induced result retains all original directed edges, parallel records and
self-loops whose endpoints are selected. This is not an ownership filter:
shared parents or relation edges can bring other Rooms and their objects into
the result. A fixed radius alone does not bound size by one Room's inventory.

Returned records are deep copies, not normalized scoring records. No edge IDs,
weights, evidence, provenance or hierarchy edges are generated. Hashes identify
canonical JSON content, including record order and stored metadata, rather than
the bytes of an input file or an order-independent graph isomorphism class.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

from citygml_sg.retrieval.reference_graph import (
    LABEL_ALIASES,
    SURFACE_CATEGORIES,
    _node_id,
    _node_type,
    _properties,
)


SCOPE_VERSION = "room_hop_scope_v1"
MEMBERSHIP_POLICY = "undirected_semantic_h_hop_induced_v1"
SEMANTIC_NODE_TYPES = frozenset({
    "Building", "BuildingPart", "Room", "BoundarySurface", "Opening",
    "Furniture", "Installation",
})


def canonical_node_id(raw: dict[str, Any]) -> str:
    """Read a stored canonical ID with the reference adapter's field precedence."""
    if not isinstance(raw, dict):
        raise ValueError("Semantic graph node must be an object")
    return _node_id(raw)


def canonical_node_type(raw: dict[str, Any]) -> str:
    """Resolve a supported semantic alias without changing the original record."""
    if not isinstance(raw, dict):
        raise ValueError("Semantic graph node must be an object")
    labels = raw.get("labels", [])
    if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
        raise ValueError("Semantic graph node labels must be a list of strings when supplied")
    node_type = _node_type(raw)
    canonical = ("BoundarySurface" if node_type in SURFACE_CATEGORIES
                 else LABEL_ALIASES.get(node_type, node_type))
    if canonical not in SEMANTIC_NODE_TYPES:
        raise ValueError(f"Unsupported semantic graph node type: {node_type}")
    return canonical


def edge_endpoints(raw: dict[str, Any]) -> tuple[str, str]:
    """Read stored source and target IDs; preserve their original direction."""
    if not isinstance(raw, dict):
        raise ValueError("Semantic graph edge must be an object")
    source = raw.get("source_id", raw.get("source"))
    target = raw.get("target_id", raw.get("target"))
    if (not isinstance(source, str) or not source.strip()
            or not isinstance(target, str) or not target.strip()):
        raise ValueError("Semantic graph edge requires nonempty source and target IDs")
    return source, target


def _content_hash(value: Any) -> str:
    try:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Semantic graph must contain finite JSON-compatible data") from exc
    return sha256(serialized.encode("utf-8")).hexdigest()


def _validate_h(h: int) -> None:
    if isinstance(h, bool) or not isinstance(h, int) or h < 0:
        raise ValueError("Hop radius h must be a nonnegative integer, not boolean")


@dataclass(frozen=True)
class _ValidatedGraph:
    nodes: dict[str, dict[str, Any]]
    node_types: dict[str, str]
    edges: list[dict[str, Any]]
    endpoints: list[tuple[str, str]]
    adjacency: dict[str, set[str]]
    input_snapshot_sha256: str


def _validate_snapshot(snapshot: dict[str, Any]) -> _ValidatedGraph:
    if not isinstance(snapshot, dict):
        raise ValueError("Semantic graph snapshot must be an object")
    nodes, edges = snapshot.get("nodes"), snapshot.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Semantic graph snapshot requires nodes and edges lists")
    by_id: dict[str, dict[str, Any]] = {}
    node_types: dict[str, str] = {}
    for node in nodes:
        node_id = canonical_node_id(node)
        if node_id in by_id:
            raise ValueError(f"Duplicate semantic graph node ID: {node_id}")
        by_id[node_id] = node
        node_types[node_id] = canonical_node_type(node)
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in by_id}
    endpoints: list[tuple[str, str]] = []
    edge_ids: set[str] = set()
    for edge in edges:
        source, target = edge_endpoints(edge)
        if source not in by_id or target not in by_id:
            raise ValueError("Semantic graph edge has an unknown endpoint")
        relation = edge.get("relation") or edge.get("type")
        if not isinstance(relation, str) or not relation.strip():
            raise ValueError("Semantic graph edge requires a nonempty relation type")
        properties = _properties(edge)
        edge_id = next((value for value in (edge.get("id"), edge.get("edge_id"),
                                           properties.get("id")) if value is not None), None)
        if edge_id is not None:
            if not isinstance(edge_id, str) or not edge_id.strip():
                raise ValueError("Semantic graph edge ID must be a nonempty string when supplied")
            if edge_id in edge_ids:
                raise ValueError(f"Duplicate semantic graph edge ID: {edge_id}")
            edge_ids.add(edge_id)
        endpoints.append((source, target))
        adjacency[source].add(target)
        adjacency[target].add(source)
    return _ValidatedGraph(by_id, node_types, edges, endpoints, adjacency,
                           _content_hash(snapshot))


def _extract(graph: _ValidatedGraph, room_id: str, h: int) -> dict[str, Any]:
    if not isinstance(room_id, str) or room_id not in graph.nodes:
        raise ValueError("Room ID must identify a node in the semantic graph")
    if graph.node_types[room_id] != "Room":
        raise ValueError(f"Requested root is not a Room: {room_id}")
    node_hops = {room_id: 0}
    queue = deque([room_id])
    while queue:
        node_id = queue.popleft()
        if node_hops[node_id] == h:
            continue
        for neighbor in graph.adjacency[node_id]:
            if neighbor not in node_hops:
                node_hops[neighbor] = node_hops[node_id] + 1
                queue.append(neighbor)
    # Iterating original records preserves list order and every parallel edge.
    subgraph = {
        "nodes": [deepcopy(node) for node_id, node in graph.nodes.items()
                  if node_id in node_hops],
        "edges": [deepcopy(edge) for edge, (source, target)
                  in zip(graph.edges, graph.endpoints)
                  if source in node_hops and target in node_hops],
    }
    return {
        "room_id": room_id,
        **subgraph,
        "input_snapshot_sha256": graph.input_snapshot_sha256,
        "subgraph_sha256": _content_hash(subgraph),
        "membership": {
            "policy": MEMBERSHIP_POLICY,
            "version": SCOPE_VERSION,
            "h": h,
            "traversal": "undirected_all_supplied_edges",
            "domain": "explicit_semantic_graph_reject_unsupported_nodes",
            "node_hops": dict(sorted(node_hops.items())),
            "other_room_ids": sorted(node_id for node_id in node_hops
                                     if node_id != room_id
                                     and graph.node_types[node_id] == "Room"),
        },
    }


def extract_room_hop_subgraph(
    snapshot: dict[str, Any], room_id: str, h: int = 2,
) -> dict[str, Any]:
    """Return the induced graph on nodes at undirected distance at most ``h``.

    All node IDs and supplied edge IDs must be unique. Anonymous parallel edges
    are permitted and retained individually. Dangling endpoints, unsupported
    node types and malformed records anywhere in the input cause ``ValueError``.
    ``h=0`` retains the Room and its existing self-loops. The input is unchanged,
    and mutating output records cannot change it.
    """
    _validate_h(h)
    return _extract(_validate_snapshot(snapshot), room_id, h)


def build_room_hop_subgraphs(snapshot: dict[str, Any], h: int = 2) -> list[dict[str, Any]]:
    """Extract every Room in ID order, validating and hashing the input once."""
    _validate_h(h)
    graph = _validate_snapshot(snapshot)
    return [_extract(graph, room_id, h) for room_id in sorted(graph.nodes)
            if graph.node_types[room_id] == "Room"]
