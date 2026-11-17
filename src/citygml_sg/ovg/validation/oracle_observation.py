"""Strict observation-only contract for the proposed joint retrieval method.

Legacy category bonuses, room constraints and map instance identifiers are never
copied into this inference representation.
"""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

NORMALIZATION_VERSION = "semantic_observation_v1"
LABELS = {
    "room": "Room", "furniture": "Furniture", "buildingfurniture": "Furniture",
    "installation": "Installation", "buildinginstallation": "Installation",
    "intbuildinginstallation": "Installation", "opening": "Opening",
    "door": "Opening", "window": "Opening", "boundarysurface": "BoundarySurface",
    "wallsurface": "BoundarySurface", "floorsurface": "BoundarySurface",
    "groundsurface": "BoundarySurface", "ceilingsurface": "BoundarySurface",
    "roofsurface": "BoundarySurface", "closuresurface": "BoundarySurface",
    "interiorwallsurface": "BoundarySurface", "outerfloorsurface": "BoundarySurface",
    "outerceilingsurface": "BoundarySurface",
}
CATEGORY_ALIASES = {
    "wallsurface": "wall", "wall_surface": "wall", "floorsurface": "floor",
    "floor_surface": "floor", "groundsurface": "floor", "ground_surface": "floor",
    "ceilingsurface": "ceiling", "ceiling_surface": "ceiling",
    "roofsurface": "roof", "roof_surface": "roof", "closuresurface": "closure",
    "closure_surface": "closure", "pillar": "column", "기둥": "column",
    "refrigerator": "fridge", "freezer": "fridge", "television": "tv",
    "display": "tv", "couch": "sofa", "ground": "floor",
}
TYPED_CONTEXT = {
    "Room": ("Room", None, "room"),
    "BoundarySurface": ("BoundarySurface", None, ""),
    "WallSurface": ("BoundarySurface", "WallSurface", "wall"),
    "FloorSurface": ("BoundarySurface", "FloorSurface", "floor"),
    "GroundSurface": ("BoundarySurface", "FloorSurface", "floor"),
    "OuterFloorSurface": ("BoundarySurface", "FloorSurface", "floor"),
    "InteriorWallSurface": ("BoundarySurface", "WallSurface", "wall"),
    "OuterCeilingSurface": ("BoundarySurface", "CeilingSurface", "ceiling"),
    "CeilingSurface": ("BoundarySurface", "CeilingSurface", "ceiling"),
    "RoofSurface": ("BoundarySurface", "RoofSurface", "roof"),
    "ClosureSurface": ("BoundarySurface", "ClosureSurface", "closure"),
}
FORBIDDEN_IDENTIFIERS = {
    "gml_id", "target_id", "target_ids", "target_room", "target_room_id",
    "target_room_name", "room_id", "room_name", "ground_truth", "gt_room_id",
}
SCORING_LABELS = {"Furniture", "Installation", "Opening"}


def _unit(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    result = float(value)
    if not isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    return result


def _reject_map_identifiers(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_IDENTIFIERS:
                raise ValueError(f"{path}.{key}: map/GT identifiers are not observation inputs")
            _reject_map_identifiers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_map_identifiers(child, f"{path}[{index}]")


def normalize_observation(raw: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize local aliases and observation weights, without selecting anchors.

    A typed endpoint reuses a unique compatible explicit context. Otherwise it
    creates one existential context per canonical type; ambiguous aliases fail.
    """
    if not isinstance(raw, dict):
        raise ValueError("Observation must be an object")
    config = config or {}
    aliases = dict(CATEGORY_ALIASES)
    if config.get("category_aliases"):
        raise ValueError("Custom category aliases require a shared map/observation normalizer version")
    objects = raw.get("objects", [])
    relations = raw.get("relations", [])
    if not isinstance(objects, list) or not isinstance(relations, list):
        raise ValueError("Observation objects and relations must be lists")
    nodes: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    endpoint_resolution: dict[str, str] = {}
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ValueError(f"objects[{index}] must be an object")
        _reject_map_identifiers(item, f"objects[{index}]")
        if "id" in item:
            raise ValueError("Use a local alias, not an object/map id, in observations")
        alias = item.get("alias")
        if (not isinstance(alias, str) or not alias.strip() or alias.strip() != alias
                or alias.startswith("__context__") or alias in TYPED_CONTEXT):
            raise ValueError(f"objects[{index}] requires a non-reserved local alias")
        if alias in nodes:
            raise ValueError(f"Duplicate observation alias: {alias}")
        category = item.get("category", "")
        if not isinstance(category, str):
            raise ValueError(f"{alias}.category must be a string")
        category = category.strip().casefold()
        category = aliases.get(category, category)
        node_type = str(item.get("label", item.get("type", "BuildingFurniture")))
        label = LABELS.get(node_type.replace("_", "").casefold())
        if label is None:
            raise ValueError(f"Unsupported observation label: {node_type}")
        if not category and label in SCORING_LABELS:
            raise ValueError(f"Scoring object {alias} requires a category")
        context_type = next((key for key in TYPED_CONTEXT
                             if key.casefold() == node_type.replace("_", "").casefold()), None)
        subtype = item.get("subtype")
        if subtype is not None and not isinstance(subtype, str):
            raise ValueError(f"{alias}.subtype must be a string or null")
        if not subtype and context_type:
            subtype = TYPED_CONTEXT[context_type][1]
        subtype_context = next((key for key in TYPED_CONTEXT
                                if key.casefold() == str(subtype).replace("_", "").casefold()), None)
        if subtype_context:
            subtype = TYPED_CONTEXT[subtype_context][1]
        if not category and context_type:
            category = TYPED_CONTEXT[context_type][2]
        if label == "BoundarySurface" and not subtype:
            subtype = {"floor": "FloorSurface", "wall": "WallSurface",
                       "ceiling": "CeilingSurface", "roof": "RoofSurface",
                       "closure": "ClosureSurface"}.get(category)
        if label == "BoundarySurface":
            declared = TYPED_CONTEXT[context_type][1] if context_type else None
            if declared and subtype != declared:
                raise ValueError(f"{alias}: surface type and subtype conflict")
            expected_category = TYPED_CONTEXT.get(subtype, (None, None, None))[2]
            if category and expected_category and category != expected_category:
                raise ValueError(f"{alias}: surface subtype and category conflict")
            category = category or expected_category or ""
        if label == "Opening":
            kinds = {value for value in (node_type.casefold(), str(subtype).casefold(), category)
                     if value in {"door", "window"}}
            if len(kinds) > 1:
                raise ValueError(f"{alias}: opening type, subtype and category conflict")
            if kinds:
                specific = next(iter(kinds))
                if category not in {specific, "opening"}:
                    raise ValueError(f"{alias}: opening category conflicts with subtype")
                subtype = specific.title()
        confidence = _unit(item.get("confidence", 1.0), f"{alias}.confidence")
        visibility = _unit(item.get("visibility", 1.0), f"{alias}.visibility")
        if "weight" in item:
            warnings.append(f"{alias}: legacy weight ignored; weight=confidence*visibility")
        nodes[alias] = {
            "id": alias, "label": label, "subtype": subtype, "category": category,
            "confidence": confidence, "visibility": visibility,
            "weight": confidence * visibility, "scoring": label in SCORING_LABELS,
            "provenance": "explicit_observation",
        }

    def endpoint(value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("Relation endpoints must be nonempty alias strings")
        if value in nodes:
            return value
        if value not in TYPED_CONTEXT:
            raise ValueError(f"Unknown observation relation endpoint: {value}")
        label, subtype, category = TYPED_CONTEXT[value]
        compatible = [node for node in nodes.values()
                      if node["provenance"] == "explicit_observation"
                      and node["label"] == label
                      and (subtype is None or node["subtype"] == subtype)
                      and (not category or node["category"] == category)]
        if len(compatible) > 1:
            raise ValueError(f"Ambiguous typed endpoint {value}; use an explicit observation alias")
        if compatible:
            endpoint_resolution[value] = compatible[0]["id"]
            return compatible[0]["id"]
        context_id = f"__context__{subtype or label}"
        endpoint_resolution[value] = context_id
        nodes.setdefault(context_id, {
            "id": context_id, "label": label, "subtype": subtype, "category": category,
            "confidence": 1.0, "visibility": 1.0, "weight": 1.0, "scoring": False,
            "provenance": "implicit_typed_context",
        })
        return context_id

    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for index, item in enumerate(relations):
        if isinstance(item, list) and len(item) == 3:
            source, relation, target = item
            weight, weight_source = 1.0, "default_unit"
        elif isinstance(item, dict):
            _reject_map_identifiers(item, f"relations[{index}]")
            source = item.get("source_alias", item.get("source"))
            target = item.get("target_alias", item.get("target"))
            relation = item.get("type")
            weight = _unit(item.get("weight", 1.0), f"relations[{index}].weight")
            weight_source = "explicit_observation" if "weight" in item else "default_unit"
        else:
            raise ValueError(f"relations[{index}] must be a triple or relation object")
        if not isinstance(relation, str) or not relation.strip():
            raise ValueError("Relation type must be a nonempty string")
        source_id, target_id = endpoint(source), endpoint(target)
        relation = relation.strip().upper()
        key = (source_id, relation, target_id)
        equivalents = {key}
        if relation in config.get("symmetric_relations", [
            "ADJACENT_TO", "TOUCHES", "INTERSECTS", "ADJACENT_SURFACE",
        ]):
            equivalents.add((target_id, relation, source_id))
        inverse = config.get("inverse_relations", {"ABOVE": "BELOW", "BELOW": "ABOVE"}).get(relation)
        if inverse:
            equivalents.add((target_id, inverse, source_id))
        if seen_edges.intersection(equivalents):
            raise ValueError(f"Duplicate observation relation: {key}")
        seen_edges.add(key)
        edges.append({"id": f"obs_edge_{index}", "source": source_id, "target": target_id,
                      "type": relation, "weight": weight, "weight_source": weight_source})
    if raw.get("constraints") or raw.get("query"):
        warnings.append("Legacy constraints/query settings are not inference parameters")
    return {"normalization_version": NORMALIZATION_VERSION, "nodes": list(nodes.values()),
            "edges": edges, "normalization_map": aliases,
            "typed_endpoint_resolution": endpoint_resolution, "warnings": warnings}


def select_anchors(normalized: dict[str, Any], threshold: float = 0.0) -> dict[str, Any]:
    """Select positive-weight anchors and remove every excluded incident edge."""
    threshold = _unit(threshold, "anchor_threshold")
    result = deepcopy(normalized)
    included = {n["id"] for n in normalized["nodes"]
                if n["weight"] > 0 and n["weight"] >= threshold}
    result["nodes"] = [n for n in normalized["nodes"] if n["id"] in included]
    result["excluded_nodes"] = [{**n, "reason": "zero_weight_or_below_threshold"}
                                for n in normalized["nodes"] if n["id"] not in included]
    result["edges"], result["excluded_edges"] = [], []
    for edge in normalized["edges"]:
        reason = None
        if edge["source"] not in included or edge["target"] not in included:
            reason = "excluded_endpoint"
        elif edge["weight"] <= 0:
            reason = "zero_observation_weight"
        if reason:
            result["excluded_edges"].append({**edge, "reason": reason})
        else:
            result["edges"].append(edge)
    result["anchor_threshold"] = threshold
    result["unknown_category_policy"] = "retain_as_unmatched_no_gt_filter"
    return result
