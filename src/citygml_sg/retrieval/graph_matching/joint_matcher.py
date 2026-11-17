"""Deterministic partial injective matching for the proposed evidence score.

The general solver uses an admissible branch-and-bound relaxation, with an
explicit state budget per Room. Unfinished searches return feasible evidence
and an upper bound, never an assertion of exact optimality. Objects-only uses
exact weighted bipartite matching and does not inspect relation evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from time import perf_counter
from typing import Any

from citygml_sg.retrieval.scoring.joint_score import (
    build_score_context, effective_config, finite_number, score_mapping, stored_edge_quality,
)


LABELS = {"Room", "Furniture", "Installation", "Opening", "BoundarySurface"}
SCORING_LABELS = {"Furniture", "Installation", "Opening"}


def _text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{field} must be a {'possibly empty ' if allow_empty else 'nonempty '}string")
    return value


def _validate_graph(graph: dict[str, Any], *, observed: bool) -> dict[str, Any]:
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        raise ValueError("Each normalized graph requires nodes and edges lists")
    nodes: list[dict[str, Any]] = []
    for raw in graph["nodes"]:
        if not isinstance(raw, dict):
            raise ValueError("Nodes must be dictionaries")
        node = dict(raw)
        _text(node.get("id"), "node.id")
        if node.get("label") not in LABELS:
            raise ValueError(f"Unsupported normalized label: {node.get('label')}")
        _text(node.get("category"), "node.category", allow_empty=node["label"] in {"Room", "BoundarySurface"})
        if node.get("subtype") is not None:
            _text(node["subtype"], "node.subtype", allow_empty=True)
        if observed:
            node["weight"] = finite_number(node.get("weight"), field="node.weight", unit_interval=True)
            if not isinstance(node.get("scoring"), bool):
                raise ValueError("Observation node.scoring must be boolean")
            if node["scoring"] and node["label"] not in SCORING_LABELS:
                raise ValueError("Room and BoundarySurface are relation contexts, not scoring objects")
        nodes.append(node)
    node_ids = {node["id"] for node in nodes}
    if len(node_ids) != len(nodes):
        raise ValueError("Duplicate node IDs")
    if observed and sum(node["label"] == "Room" for node in nodes) > 1:
        raise ValueError("An observation may contain only one unknown-Room placeholder")
    edges: list[dict[str, Any]] = []
    for raw in graph["edges"]:
        if not isinstance(raw, dict):
            raise ValueError("Edges must be dictionaries")
        edge = dict(raw)
        for field in ("id", "source", "target", "type"):
            _text(edge.get(field), f"edge.{field}")
        if edge["source"] not in node_ids or edge["target"] not in node_ids:
            raise ValueError(f"Edge {edge['id']} references an absent endpoint")
        if observed:
            edge["weight"] = finite_number(edge.get("weight"), field="edge.weight", unit_interval=True)
        edges.append(edge)
    if len({edge["id"] for edge in edges}) != len(edges):
        raise ValueError("Duplicate edge IDs")
    return {**graph, "nodes": sorted(nodes, key=lambda item: item["id"]), "edges": sorted(edges, key=lambda item: item["id"])}


def _compatible(observed: dict[str, Any], stored: dict[str, Any], room_id: str) -> bool:
    if observed["label"] != stored["label"]:
        return False
    if observed["label"] == "Room":
        return stored["id"] == room_id
    observed_subtype = str(observed.get("subtype") or "").casefold()
    stored_subtype = str(stored.get("subtype") or "").casefold()
    category = observed["category"].casefold()
    stored_category = stored["category"].casefold()
    if observed["label"] == "Opening":
        observed_subtype = observed_subtype or (category if category in {"door", "window"} else "")
        stored_subtype = stored_subtype or (stored_category if stored_category in {"door", "window"} else "")
        if observed_subtype and observed_subtype != stored_subtype:
            return False
        return category == "opening" or category == stored_category or category == stored_subtype
    if observed_subtype and observed_subtype != stored_subtype:
        return False
    if observed["label"] == "BoundarySurface" and category in {"", "boundary", "boundarysurface", "boundary_surface"}:
        return True
    return category == stored_category


def _build_edge_options(
    observation: dict[str, Any], candidate: dict[str, Any], domains: dict[str, list[str]],
    config: dict[str, Any], qualities: dict[str, float],
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    options: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    if config["beta"] == 0:
        return options
    symmetric = set(config["symmetric_relations"])
    for observed in observation["edges"]:
        accepted = {observed["type"], *config["relation_compatibility"].get(observed["type"], [])}
        observed_options: dict[tuple[str, str], dict[str, Any]] = {}
        source_domain, target_domain = set(domains[observed["source"]]), set(domains[observed["target"]])
        for stored in candidate["edges"]:
            orientations: list[tuple[str, str, str]] = []
            if stored["type"] in accepted:
                orientations.append((stored["source"], stored["target"], "forward"))
                if observed["type"] in symmetric and stored["type"] in symmetric:
                    orientations.append((stored["target"], stored["source"], "symmetric_reverse"))
            if config["inverse_relations"].get(observed["type"]) == stored["type"]:
                orientations.append((stored["target"], stored["source"], "inverse_reverse"))
            for source, target, orientation in orientations:
                if source not in source_domain or target not in target_domain:
                    continue
                if (observed["source"] == observed["target"]) != (source == target):
                    continue
                selected = {"edge": stored, "quality": qualities[stored["id"]], "orientation": orientation}
                previous = observed_options.get((source, target))
                if previous is None or selected["quality"] > previous["quality"]:
                    observed_options[(source, target)] = selected
                # Candidate edges are sorted by ID: equal metadata keeps the first.
        options[observed["id"]] = observed_options
    return options


def _objects_only_mapping(
    observation: dict[str, Any], domains: dict[str, list[str]], context: dict[str, Any]
) -> tuple[dict[str, str | None], dict[str, Any]]:
    """Greedy weighted independent-set selection in a transversal matroid.

    Each feasible set of matched left vertices is an independent set. Processing
    nonnegative left weights in descending order and testing independence with
    augmenting paths therefore gives an exact maximum-weight matching.
    """
    owners: dict[str, str] = {}
    mapping: dict[str, str | None] = {node["id"]: None for node in observation["nodes"]}
    attempts = 0

    def augment(alias: str, visited: set[str]) -> bool:
        nonlocal attempts
        attempts += 1
        for target in domains[alias]:
            if target in visited:
                continue
            visited.add(target)
            owner = owners.get(target)
            if owner is None or augment(owner, visited):
                owners[target] = alias
                mapping[alias] = target
                return True
        return False

    for node in sorted(observation["nodes"], key=lambda item: (-context["object_weights"][item["id"]], item["id"])):
        augment(node["id"], set())
    numerator = math.fsum(context["object_weights"][alias] for alias, target in mapping.items() if target is not None)
    return mapping, {
        "algorithm": "weighted_bipartite_augmenting_paths",
        "optimality_proven": True, "termination_reason": "polynomial_optimum",
        "states_explored": 0, "augmenting_path_visits": attempts,
        "upper_bound_numerator": numerator,
        "upper_bound": numerator / context["denominator"] if context["denominator"] else 0.0,
    }


def _joint_mapping(
    observation: dict[str, Any], domains: dict[str, list[str]],
    options: dict[str, dict[tuple[str, str], dict[str, Any]]],
    context: dict[str, Any], config: dict[str, Any],
) -> tuple[dict[str, str | None], dict[str, Any]]:
    weights, relation_weights = context["object_weights"], context["relation_weights"]
    edges = observation["edges"]
    degrees = {node["id"]: sum(
        edge["source"] == node["id"] or edge["target"] == node["id"]
        for edge in edges if relation_weights[edge["id"]] > 0
    ) for node in observation["nodes"]}
    mapping: dict[str, str | None] = {
        node["id"]: domains[node["id"]][0] for node in observation["nodes"] if node["label"] == "Room"
    }
    order = sorted((alias for alias in domains if alias not in mapping),
                   key=lambda alias: (len(domains[alias]), -degrees[alias], -weights[alias], alias))
    used = {target for target in mapping.values() if target is not None}
    best_mapping: dict[str, str | None] = dict(mapping)
    best_numerator = best_relation = 0.0
    states = 0
    exhausted = False
    unexplored_upper = 0.0

    def objective() -> tuple[float, float]:
        objects = math.fsum(weights[alias] for alias, target in mapping.items() if target is not None)
        relations = math.fsum(
            relation_weights[edge["id"]] * options.get(edge["id"], {}).get(
                (mapping.get(edge["source"]), mapping.get(edge["target"])), {}
            ).get("quality", 0.0)
            for edge in edges
        )
        return math.fsum((objects, relations)), relations

    def bound() -> tuple[float, float]:
        available = {
            alias: [mapping[alias]] if mapping.get(alias) is not None else []
            if alias in mapping else [target for target in domains[alias] if target not in used]
            for alias in domains
        }
        objects = math.fsum(weights[alias] for alias in domains if available[alias])
        relation_bounds = []
        for edge in edges:
            best = max((
                selected["quality"] for (source, target), selected in options.get(edge["id"], {}).items()
                if source in available[edge["source"]] and target in available[edge["target"]]
            ), default=0.0)
            relation_bounds.append(relation_weights[edge["id"]] * best)
        relations = math.fsum(relation_bounds)
        return math.fsum((objects, relations)), relations

    def visit(depth: int) -> None:
        nonlocal best_mapping, best_numerator, best_relation, states, exhausted, unexplored_upper
        upper, relation_upper = bound()
        if upper < best_numerator or (upper == best_numerator and relation_upper <= best_relation):
            return
        if states >= config["max_states"]:
            exhausted = True
            unexplored_upper = max(unexplored_upper, upper)
            return
        states += 1
        numerator, relation = objective()
        if numerator > best_numerator or (numerator == best_numerator and relation > best_relation):
            best_numerator, best_relation = numerator, relation
            best_mapping = dict(mapping)
        if depth == len(order):
            return
        alias = order[depth]
        for target in [*domains[alias], None]:
            if target is not None and target in used:
                continue
            mapping[alias] = target
            if target is not None:
                used.add(target)
            visit(depth + 1)
            if target is not None:
                used.remove(target)
            del mapping[alias]
            if exhausted:
                # This ancestor bounds all still-unvisited siblings as well.
                unexplored_upper = max(unexplored_upper, upper)
                return

    visit(0)
    upper = max(best_numerator, unexplored_upper) if exhausted else best_numerator
    return best_mapping, {
        "algorithm": "joint_branch_and_bound", "optimality_proven": not exhausted,
        "termination_reason": "max_states_exceeded" if exhausted else "optimality_proven",
        "states_explored": states, "upper_bound_numerator": upper,
        "upper_bound": upper / context["denominator"] if context["denominator"] else 0.0,
    }


def rank_rooms(
    observation: dict[str, Any], candidates: list[dict[str, Any]], *,
    config: dict[str, Any] | None = None, variant: str = "full",
) -> dict[str, Any]:
    """Rank the entire fixed Room pool with explicit solver/evidence status.

    Inputs must already be normalized and anchor-filtered. No target Room,
    filename, or evaluator metadata is consulted. ``max_states`` limits the
    general solver per Room; the polynomial objects-only path has no search
    states. Strict active metadata errors raise ValueError before retrieval.
    """
    start = perf_counter()
    settings = effective_config(config, variant)
    observed = _validate_graph(observation, observed=True)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("A nonempty fixed candidate Room pool is required")
    pool = [_validate_graph(candidate, observed=False) for candidate in candidates]
    for candidate in pool:
        room_id = _text(candidate.get("room_id"), "candidate.room_id")
        if not any(node["id"] == room_id and node["label"] == "Room" for node in candidate["nodes"]):
            raise ValueError(f"Candidate {room_id} requires its canonical Room node")
    ids = sorted(candidate["room_id"] for candidate in pool)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate candidate Room IDs")
    pool.sort(key=lambda item: item["room_id"])
    context = build_score_context(observed, pool, settings)
    metadata_settings = settings
    if not any(context["relation_weights"].values()):
        # No observed relation atom can use stored metadata in this query.
        # Still reject invalid values that are present; absent values remain
        # absent and are never synthesized as unit evidence.
        metadata_settings = {**settings, "beta": 0.0}
        if settings["beta"] > 0:
            settings["stored_metadata_policy"] = "validate_present_allow_missing_no_active_observed_relation"
    qualities: dict[str, dict[str, float]] = {}
    missing_metadata: dict[str, dict[str, int]] = {}
    for candidate in pool:
        qualities[candidate["room_id"]] = {
            edge["id"]: stored_edge_quality(edge, metadata_settings) for edge in candidate["edges"]
        }
        missing_metadata[candidate["room_id"]] = {
            "edge_count": sum(edge.get("gamma") is None or edge.get("epsilon") is None for edge in candidate["edges"]),
            "gamma_count": sum(edge.get("gamma") is None for edge in candidate["edges"]),
            "epsilon_count": sum(edge.get("epsilon") is None for edge in candidate["edges"]),
        }
    solved = []
    for candidate in pool:
        domains = {
            node["id"]: [stored["id"] for stored in candidate["nodes"] if _compatible(node, stored, candidate["room_id"])]
            for node in observed["nodes"]
        }
        options = _build_edge_options(observed, candidate, domains, settings, qualities[candidate["room_id"]])
        if context["denominator"] == 0:
            mapping: dict[str, str | None] = {
                node["id"]: candidate["room_id"] for node in observed["nodes"] if node["label"] == "Room"
            }
            solver = {"algorithm": "none", "optimality_proven": True, "termination_reason": "no_active_anchor",
                      "states_explored": 0, "upper_bound": 0.0, "upper_bound_numerator": 0.0}
        elif settings["beta"] == 0 or not any(context["relation_weights"].values()):
            mapping, solver = _objects_only_mapping(observed, domains, context)
        else:
            mapping, solver = _joint_mapping(observed, domains, options, context, settings)
        solved.append((candidate, mapping, options, solver))
    matching_end = perf_counter()
    ranking = [{
        "room_id": candidate["room_id"], "room_name": candidate.get("room_name", candidate["room_id"]),
        **score_mapping(observed, mapping, options, context), "solver": solver,
        "missing_metadata": missing_metadata[candidate["room_id"]],
    } for candidate, mapping, options, solver in solved]
    decimals = settings["score_tie_decimals"]
    def tie_key(row: dict[str, Any]) -> tuple[float, float]:
        return round(row["score"], decimals), round(row["relation_numerator"], decimals)
    ranking.sort(key=lambda row: (-tie_key(row)[0], -tie_key(row)[1], row["room_id"]))
    groups: dict[tuple[float, float], list[int]] = {}
    for index, row in enumerate(ranking, start=1):
        groups.setdefault(tie_key(row), []).append(index)
    for index, row in enumerate(ranking, start=1):
        positions = groups[tie_key(row)]
        row.update(rank=index, best_tie_rank=positions[0], worst_tie_rank=positions[-1], tie_size=len(positions))
    if context["denominator"] == 0:
        status = "abstained_no_anchor"
    elif any(not row["solver"]["optimality_proven"] for row in ranking):
        status = "solver_incomplete"
    elif all(row["numerator"] == 0 for row in ranking):
        status = "abstained_no_evidence"
    else:
        status = "completed"
    return {
        "status": status, "ranking": ranking, "full_ranking": True,
        "candidate_count": len(ids), "candidate_ids": ids,
        "candidate_ids_hash": hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "variant": variant, "method_version": settings["version"], "effective_config": settings,
        "denominator": context["denominator"], "category_weights": context["rho"],
        "category_room_counts": context["category_room_counts"],
        "excluded_nodes": observation.get("excluded_nodes", []), "excluded_edges": observation.get("excluded_edges", []),
        "missing_metadata": missing_metadata,
        "score_tie_decimals": decimals,
        "top_score_tie": len(ranking) > 1 and round(ranking[0]["score"], decimals) == round(ranking[1]["score"], decimals),
        "tie_policy": "rounded_active_score_desc,rounded_active_relation_numerator_desc,canonical_room_id_asc",
        "mapping_tie_policy": "first_optimum_in_deterministic_domain_search_order",
        "timings": {"matching_ms": (matching_end - start) * 1000.0,
                    "scoring_sorting_ms": (perf_counter() - matching_end) * 1000.0},
    }
