"""Opt-in candidate construction from a separately audited geometry ledger.

The legacy typed-path builder is unchanged. Geometry restricts membership only:
this adapter retains stored edges between admitted nodes and creates no scored
relationship. The ledger records unresolved memberships outside the candidate
graphs, which remain tied to its immutable content identity.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from typing import Any

from citygml_sg.retrieval.reference_graph import (
    NORMALIZATION_VERSION, SEMANTIC_TYPES, _node_type, _properties,
    _snapshot_records, normalize_semantic_node,
)


GEOMETRIC_MEMBERSHIP_POLICY = "validated_prism_membership_v1"
GEOMETRIC_REFERENCE_VERSION = "geometry_induced_reference_v1"


def _identity(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def build_geometric_room_candidates(snapshot: dict[str, Any], ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply exact allowed node sets; never run container/adjacency expansion.

    The caller must supply the complete ledger for this snapshot and every Room.
    Geometric membership is not a substitute for human annotation verification.
    Equal underlying node/edge records preserve the legacy normalization rules.
    """
    if not isinstance(ledger, dict) or ledger.get("policy") != GEOMETRIC_MEMBERSHIP_POLICY:
        raise ValueError("Unsupported geometric membership policy")
    provenance = ledger.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("snapshot_content_sha256") != _identity(snapshot):
        raise ValueError("Membership ledger belongs to a different snapshot")
    raw_nodes, edges = _snapshot_records(snapshot)
    nodes = {key: normalize_semantic_node(raw) for key, raw in raw_nodes.items()
             if _node_type(raw) in SEMANTIC_TYPES}
    room_ids = {key for key, node in nodes.items() if node["label"] == "Room"}
    allowed = ledger.get("allowed_node_ids_by_room")
    if not isinstance(allowed, dict) or set(allowed) != room_ids:
        raise ValueError("Ledger must explicitly cover every source Room")
    ledger_hash = _identity(ledger)
    candidates = []
    for room_id in sorted(room_ids):
        ids = allowed[room_id]
        if (not isinstance(ids, list) or any(not isinstance(key, str) for key in ids)
                or len(ids) != len(set(ids))):
            raise ValueError("Allowed membership IDs must be unique strings")
        selected = set(ids)
        if room_id not in selected:
            raise ValueError("Each candidate must contain its own Room")
        if not selected <= nodes.keys():
            raise ValueError("Allowed membership contains an unknown or nonsemantic node")
        if selected.intersection(room_ids) != {room_id}:
            raise ValueError("A candidate cannot contain another Room")
        props = _properties(raw_nodes[room_id])
        candidates.append({
            "room_id": room_id,
            "room_name": str(props.get("gml_name") or raw_nodes[room_id].get("room_name") or room_id),
            "nodes": [dict(nodes[key]) for key in sorted(selected)],
            "edges": [edge for edge in edges if edge["source"] in selected and edge["target"] in selected],
            "membership": {
                "policy": GEOMETRIC_MEMBERSHIP_POLICY,
                "reference_version": GEOMETRIC_REFERENCE_VERSION,
                "normalization_version": NORMALIZATION_VERSION,
                "ledger_content_sha256": ledger_hash,
                "node_sources": {key: [{"source": "geometry_ledger", "edge_ids": []}]
                                 for key in sorted(selected)},
                "layered_boundary_collapse": "not_applied",
                "container_boundary_fallback": "disabled",
                "adjacency_membership_expansion": "disabled",
                "relation_policy": "stored_induced_edges_only",
            },
        })
    occurrence = Counter(node["id"] for c in candidates for node in c["nodes"])
    for candidate in candidates:
        candidate["membership"]["shared_node_ids"] = [n["id"] for n in candidate["nodes"]
                                                        if occurrence[n["id"]] > 1]
    return candidates
