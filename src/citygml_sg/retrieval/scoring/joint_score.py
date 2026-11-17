"""Fixed-pool evidence weights for the proposed joint matching method.

This module is independent of the legacy Cypher heuristics. Scores are evidence
ratios, not calibrated probabilities. The matcher supplies a partial injective
mapping and one best stored edge per observed edge.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


METHOD_VERSION = "joint_evidence_v1_proposed"
VARIANTS = ("full", "objects_only", "relations_only", "uniform_category", "boolean_metadata")
DEFAULT_CONFIG: dict[str, Any] = {
    "version": METHOD_VERSION,
    "alpha": 1.0,
    "beta": 1.0,
    "max_states": 100000,
    "score_tie_decimals": 12,
    "solver_mode": "exact_reference",
    "node_scoring_labels": ["Furniture", "Installation", "Opening"],
    "category_weight": "room_idf",
    "missing_stored_metadata": "report_error",
    "relation_compatibility": {
        "ADJACENT_TO": ["TOUCHES", "INTERSECTS"],
        "TOUCHES": ["INTERSECTS"],
    },
    "symmetric_relations": ["ADJACENT_TO", "TOUCHES", "INTERSECTS", "ADJACENT_SURFACE"],
    "inverse_relations": {"ABOVE": "BELOW", "BELOW": "ABOVE"},
}


def finite_number(value: Any, *, field: str, unit_interval: bool = False) -> float:
    """Validate numeric evidence without accepting bool, NaN, or infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (unit_interval and number > 1):
        interval = "[0, 1]" if unit_interval else "[0, infinity)"
        raise ValueError(f"{field} must be finite and in {interval}")
    return number


def effective_config(config: dict[str, Any] | None, variant: str = "full") -> dict[str, Any]:
    """Resolve the five interventions without changing the input graph/pool."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    result = deepcopy(DEFAULT_CONFIG)
    if config is not None:
        if not isinstance(config, dict):
            raise ValueError("method config must be a dictionary")
        unknown = set(config) - set(DEFAULT_CONFIG) - {"anchor_threshold"}
        if unknown:
            raise ValueError("Unsupported method config keys: " + ", ".join(sorted(str(key) for key in unknown)))
        result.update(deepcopy(config))
    if result["version"] != METHOD_VERSION:
        raise ValueError(f"This implementation requires version={METHOD_VERSION}")
    if "anchor_threshold" in result:
        finite_number(result["anchor_threshold"], field="anchor_threshold", unit_interval=True)
    alpha = finite_number(result["alpha"], field="alpha")
    beta = finite_number(result["beta"], field="beta")
    if alpha == 0 and beta == 0:
        raise ValueError("alpha and beta cannot both be zero before ablation")
    for field in ("max_states", "score_tie_decimals"):
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an integer")
    if result["max_states"] < 1:
        raise ValueError("max_states must be >= 1")
    if not 0 <= result["score_tie_decimals"] <= 15:
        raise ValueError("score_tie_decimals must be in [0, 15]")
    if result["solver_mode"] != "exact_reference":
        raise ValueError("Only solver_mode=exact_reference is implemented")
    if result["category_weight"] != "room_idf":
        raise ValueError("Set variant=uniform_category to intervene on room IDF")
    if result["missing_stored_metadata"] != "report_error":
        raise ValueError("Only explicit report_error metadata policy is implemented")
    labels = result["node_scoring_labels"]
    if not isinstance(labels, list) or set(labels) != {"Furniture", "Installation", "Opening"}:
        raise ValueError("node_scoring_labels must be Furniture, Installation, Opening")
    compatibility = result["relation_compatibility"]
    if not isinstance(compatibility, dict) or any(
        not isinstance(key, str) or not isinstance(values, list)
        or any(not isinstance(value, str) for value in values)
        for key, values in compatibility.items()
    ):
        raise ValueError("relation_compatibility must map strings to lists of strings")
    if not isinstance(result["symmetric_relations"], list) or any(
        not isinstance(value, str) for value in result["symmetric_relations"]
    ):
        raise ValueError("symmetric_relations must be a string list")
    inverses = result["inverse_relations"]
    if not isinstance(inverses, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or inverses.get(value) != key
        for key, value in inverses.items()
    ):
        raise ValueError("inverse_relations must be an involutive string map")
    result["base_alpha"], result["base_beta"] = alpha, beta
    result["alpha"] = 0.0 if variant == "relations_only" else alpha
    result["beta"] = 0.0 if variant == "objects_only" else beta
    result["variant"] = variant
    if variant == "uniform_category":
        result["category_weight"] = "uniform"
    result["stored_metadata_policy"] = (
        "validate_present_allow_missing_inactive" if result["beta"] == 0 else
        "validate_present_allow_missing_boolean_intervention" if variant == "boolean_metadata" else
        "require_gamma_epsilon"
    )
    result["solver_budget_scope"] = "branch_and_bound_states_per_candidate"
    return result


def stored_edge_quality(edge: dict[str, Any], config: dict[str, Any]) -> float:
    """Return the active metadata product; never synthesize an absent edge."""
    values: list[float] = []
    missing: list[str] = []
    for field in ("gamma", "epsilon"):
        value = edge.get(field)
        if value is None:
            missing.append(field)
        else:
            values.append(finite_number(value, field=f"edge {edge['id']} {field}", unit_interval=True))
    if config["beta"] == 0:
        return 0.0
    if config["variant"] == "boolean_metadata":
        return 1.0
    if missing:
        raise ValueError(f"Stored edge {edge['id']} missing metadata: {', '.join(missing)}")
    return values[0] * values[1]


def build_score_context(
    observation: dict[str, Any], candidates: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    """Freeze Room IDF and the query/variant denominator over the entire pool."""
    labels = set(config["node_scoring_labels"])
    room_counts: dict[str, int] = {}
    for candidate in candidates:
        categories = {
            str(node["category"]).casefold() for node in candidate["nodes"] if node["label"] in labels
        }
        for category in categories:
            room_counts[category] = room_counts.get(category, 0) + 1
    observed_categories = {
        str(node["category"]).casefold() for node in observation["nodes"] if node["scoring"]
    }
    rho = {
        category: 1.0 if config["category_weight"] == "uniform" else
        math.log1p(len(candidates) / (1 + room_counts.get(category, 0)))
        for category in sorted(observed_categories)
    }
    object_weights = {
        node["id"]: config["alpha"] * node["weight"] * rho[str(node["category"]).casefold()]
        if node["scoring"] and node["label"] in labels else 0.0
        for node in observation["nodes"]
    }
    relation_weights = {edge["id"]: config["beta"] * edge["weight"] for edge in observation["edges"]}
    try:
        denominator = math.fsum((math.fsum(object_weights.values()), math.fsum(relation_weights.values())))
    except OverflowError as exc:
        raise ValueError("Active evidence weights overflow the denominator") from exc
    if not math.isfinite(denominator):
        raise ValueError("Active evidence denominator must be finite")
    return {
        "object_weights": object_weights,
        "relation_weights": relation_weights,
        "denominator": denominator,
        "rho": rho,
        "category_room_counts": room_counts,
        "candidate_count": len(candidates),
    }


def score_mapping(
    observation: dict[str, Any], mapping: dict[str, str | None],
    edge_options: dict[str, dict[tuple[str, str], dict[str, Any]]],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Build active evidence for one mapping, counting each observation once."""
    node_matches: list[dict[str, Any]] = []
    unmatched_nodes: list[dict[str, Any]] = []
    for node in observation["nodes"]:
        target = mapping.get(node["id"])
        if target is None:
            unmatched_nodes.append({"observation_id": node["id"], "reason": "not_mapped"})
        else:
            node_matches.append({
                "observation_id": node["id"], "candidate_id": target,
                "category": node["category"], "scoring": node["scoring"],
                "observation_weight": node["weight"],
                "contribution": context["object_weights"][node["id"]],
            })
    edge_matches: list[dict[str, Any]] = []
    unmatched_edges: list[dict[str, Any]] = []
    for edge in observation["edges"]:
        active_weight = context["relation_weights"][edge["id"]]
        source, target = mapping.get(edge["source"]), mapping.get(edge["target"])
        selected = edge_options.get(edge["id"], {}).get((source, target))
        if active_weight == 0:
            reason = "inactive_or_zero_weight"
        elif source is None or target is None:
            reason = "endpoint_unmatched"
        elif selected is None:
            reason = "no_compatible_stored_edge"
        else:
            stored = selected["edge"]
            edge_matches.append({
                "observation_id": edge["id"], "stored_edge_id": stored["id"],
                "observed_type": edge["type"], "stored_type": stored["type"],
                "source_alias": edge["source"], "target_alias": edge["target"],
                "mapped_source": source, "mapped_target": target,
                "stored_source": stored["source"], "stored_target": stored["target"],
                "orientation": selected["orientation"],
                "observation_weight": edge["weight"],
                "weight_source": edge.get("weight_source", "explicit"),
                "gamma": stored.get("gamma"), "epsilon": stored.get("epsilon"),
                "active_metadata_product": selected["quality"],
                "provenance": stored.get("provenance", ""),
                "contribution": active_weight * selected["quality"],
            })
            continue
        unmatched_edges.append({"observation_id": edge["id"], "reason": reason})
    objects = math.fsum(item["contribution"] for item in node_matches)
    relations = math.fsum(item["contribution"] for item in edge_matches)
    numerator = math.fsum((objects, relations))
    denominator = context["denominator"]
    return {
        "score": numerator / denominator if denominator else 0.0,
        "numerator": numerator, "object_numerator": objects,
        "relation_numerator": relations, "denominator": denominator,
        "node_matches": node_matches, "edge_matches": edge_matches,
        "unmatched_nodes": unmatched_nodes, "unmatched_edges": unmatched_edges,
    }
