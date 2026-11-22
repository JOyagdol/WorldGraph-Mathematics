"""Inclusive, reference-vocabulary anchor selection, independent of scoring.

This opt-in policy retains the unknown-Room placeholder, applies the same
threshold to nodes and edges, and excludes categories absent from the supplied
whole-reference vocabulary. It does not change the existing joint policy.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite
from typing import Any, Iterable

from citygml_sg.ovg.validation.oracle_observation import _reject_map_identifiers


POLICY_VERSION = "reference_vocabulary_threshold_v1"
LABELS = {"Room", "Furniture", "Installation", "Opening", "BoundarySurface"}


def _unit(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number in [0,1]")
    try:
        numeric = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite number in [0,1]") from exc
    if not isfinite(numeric) or not 0 <= numeric <= 1:
        raise ValueError(f"{field} must be a finite number in [0,1]")
    return numeric


def _id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be a nonempty local identifier")
    return value


def select_reference_anchors(
    normalized: dict[str, Any], reference_categories: Iterable[str], threshold: float = 0.0,
) -> dict[str, Any]:
    """Select an anchor graph from normalized observations without using GT.

    Categories must already use the same normalization on both sides. Specific
    Door/Window observations with a generic ``opening`` category are rejected;
    this selector does not repair or broaden the shared category vocabulary. The
    caller supplies the vocabulary of the entire declared reference snapshot,
    never a target Room's vocabulary. Zero-weight atoms are retained at zero
    threshold. This is selection only: no scoring domain or matcher is chosen.
    """
    threshold = _unit(threshold, "anchor_threshold")
    if isinstance(reference_categories, (str, bytes)):
        raise ValueError("reference_categories must be a collection of canonical categories")
    try:
        categories = list(reference_categories)
    except TypeError as exc:
        raise ValueError("reference_categories must be an iterable of canonical categories") from exc
    if any(not isinstance(k, str) or not k or k != k.strip().casefold() for k in categories):
        raise ValueError("reference_categories must contain nonempty normalized categories")
    vocabulary = set(categories)
    if (not isinstance(normalized, dict) or not isinstance(normalized.get("nodes"), list)
            or not isinstance(normalized.get("edges"), list)):
        raise ValueError("normalized observation requires nodes and edges lists")
    _reject_map_identifiers(normalized, "normalized")
    result = deepcopy(normalized)
    nodes, edges = result["nodes"], result["edges"]
    node_ids: set[str] = set()
    rooms = []
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("observation nodes must be objects")
        node_id = _id(node.get("id"), "node.id")
        if node_id in node_ids:
            raise ValueError("duplicate observation node ID")
        node_ids.add(node_id)
        if not isinstance(node.get("label"), str) or node["label"] not in LABELS:
            raise ValueError("unsupported normalized observation label")
        category = node.get("category")
        if not isinstance(category, str) or category != category.strip().casefold():
            raise ValueError("node.category must already be normalized")
        _unit(node.get("weight"), "node.weight")
        if node["label"] == "Room":
            rooms.append(node)
            if node["weight"] != 1:
                raise ValueError("unknown-Room placeholder must have unit weight")
            if (category != "room" or node.get("subtype") is not None
                    or node.get("scoring") is not False):
                raise ValueError("unknown-Room placeholder requires category='room', "
                                 "subtype=None and scoring=False")
        if node["label"] == "Opening" and node.get("subtype") in ("Door", "Window"):
            if category != node["subtype"].casefold():
                raise ValueError("Opening subtype and canonical category mismatch; "
                                 "explicit Door/Window requires category='door'/'window'")
    if len(rooms) > 1:
        raise ValueError("only one unknown-Room placeholder is allowed")
    edge_ids: set[str] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("observation edges must be objects")
        edge_id = _id(edge.get("id"), "edge.id")
        if edge_id in edge_ids:
            raise ValueError("duplicate observation edge ID")
        edge_ids.add(edge_id)
        source = _id(edge.get("source"), "edge.source")
        target = _id(edge.get("target"), "edge.target")
        if source not in node_ids or target not in node_ids:
            raise ValueError("observation edge has an unknown endpoint")
        _id(edge.get("type"), "edge.type")
        _unit(edge.get("weight"), "edge.weight")
    if not rooms:
        if "__context__Room" in node_ids:
            raise ValueError("reserved Room placeholder ID is occupied")
        star = {"id": "__context__Room", "label": "Room", "subtype": None,
                "category": "room", "weight": 1.0, "confidence": 1.0,
                "visibility": 1.0, "scoring": False,
                "provenance": "required_unknown_room_placeholder"}
        nodes.append(star)
        rooms = [star]
    selected, excluded = [], []
    for node in nodes:
        reason = None
        if node["label"] != "Room":
            if node["category"] not in vocabulary:
                reason = "category_absent_from_reference"
            elif node["weight"] < threshold:
                reason = "below_anchor_threshold"
        (excluded if reason else selected).append({**node, "reason": reason} if reason else node)
    included = {node["id"] for node in selected}
    kept_edges, excluded_edges = [], []
    for edge in edges:
        reason = ("excluded_endpoint" if edge["source"] not in included or edge["target"] not in included
                  else "below_anchor_threshold" if edge["weight"] < threshold else None)
        (excluded_edges if reason else kept_edges).append({**edge, "reason": reason} if reason else edge)
    categories = sorted(vocabulary)
    result.update({
        "nodes": sorted(selected, key=lambda n: n["id"]),
        "edges": sorted(kept_edges, key=lambda e: e["id"]),
        "excluded_nodes": sorted(excluded, key=lambda n: n["id"]),
        "excluded_edges": sorted(excluded_edges, key=lambda e: e["id"]),
        "anchor_threshold": threshold, "anchor_policy": POLICY_VERSION,
        "threshold_comparison": "inclusive_greater_than_or_equal",
        "unknown_category_policy": "exclude_if_absent_from_whole_reference_vocabulary",
        "unknown_room_placeholder_id": rooms[0]["id"],
        "reference_categories": categories,
        "reference_vocabulary_sha256": sha256(json.dumps(
            categories, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest(),
        "component_scope": "anchor_selection_only_no_scoring_domain_or_matching_decision",
    })
    return result
