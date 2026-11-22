"""Explicit space unions and a separate best-member projection baseline.

Grouping is declared independently of query evidence and evaluation targets.
No source snapshot, legacy Room adapter, observation or input report is mutated.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
import math
from typing import Any

from citygml_sg.retrieval.evaluation.oracle_metrics import candidate_ids_hash


SPACE_GROUP_VERSION = "explicit_space_union_v1"
PROJECTION_VERSION = "best_member_room_projection_v1"


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _definitions(room_ids: list[str], groups: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str], str]:
    if not isinstance(room_ids, list) or not room_ids:
        raise ValueError("A nonempty source Room pool is required")
    for room_id in room_ids:
        _text(room_id, "room_id")
    if len(set(room_ids)) != len(room_ids):
        raise ValueError("Duplicate source Room IDs")
    if not isinstance(groups, list):
        raise ValueError("groups must be a list of explicit space definitions")
    source_ids = set(room_ids)
    mapping = {room_id: room_id for room_id in room_ids}
    used: set[str] = set()
    group_ids: set[str] = set()
    definitions = []
    for raw in groups:
        if not isinstance(raw, dict) or set(raw) - {"space_id", "space_name", "member_room_ids", "provenance"}:
            raise ValueError("Each group accepts only space_id, space_name, member_room_ids and provenance")
        space_id = _text(raw.get("space_id"), "space_id")
        if space_id in group_ids:
            raise ValueError("Space ID must be unique within the grouped pool")
        members = raw.get("member_room_ids")
        if not isinstance(members, list) or len(members) < 2:
            raise ValueError("Each explicit space group requires at least two members")
        for member in members:
            _text(member, "member_room_id")
        if len(set(members)) != len(members):
            raise ValueError("Duplicate member Room in a space definition")
        if not set(members) <= source_ids:
            raise ValueError("A space group references a Room outside the source pool")
        if space_id in source_ids and space_id not in members:
            raise ValueError("An existing space ID must be one of its own member Rooms")
        if used.intersection(members):
            raise ValueError("Source Rooms cannot belong to overlapping space groups")
        definition = {
            "space_id": space_id,
            "space_name": _text(raw.get("space_name", space_id), "space_name"),
            "member_room_ids": sorted(members),
        }
        if "provenance" in raw:
            definition["provenance"] = _text(raw["provenance"], "group.provenance")
        definitions.append(definition)
        used.update(members)
        group_ids.add(space_id)
        mapping.update({member: space_id for member in members})
    definitions.sort(key=lambda d: d["space_id"])
    digest = sha256(_canonical({"version": SPACE_GROUP_VERSION, "groups": definitions}).encode("utf-8")).hexdigest()
    return definitions, dict(sorted(mapping.items())), digest


def room_to_space_map(candidates: list[dict[str, Any]], groups: list[dict[str, Any]]) -> dict[str, str]:
    """Return an explicit all-Room evaluation mapping, including singleton IDs."""
    if not isinstance(candidates, list) or any(not isinstance(c, dict) for c in candidates):
        raise ValueError("candidates must be a list of Room dictionaries")
    return _definitions([c.get("room_id") for c in candidates], groups)[1]


def _records(candidates: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Check original global identities before a union can hide conflicts."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        room_id = candidate["room_id"]
        if not isinstance(candidate.get("nodes"), list) or not isinstance(candidate.get("edges"), list):
            raise ValueError("Each candidate requires nodes and edges lists")
        local_nodes: set[str] = set()
        local_rooms = []
        for node in candidate["nodes"]:
            if not isinstance(node, dict):
                raise ValueError("Candidate nodes must be dictionaries")
            node_id = _text(node.get("id"), "node.id")
            if node_id in local_nodes:
                raise ValueError("Duplicate node ID within a candidate")
            local_nodes.add(node_id)
            if node.get("label") == "Room":
                local_rooms.append(node_id)
            if node_id in nodes and nodes[node_id] != node:
                raise ValueError(f"Conflicting shared node: {node_id}")
            nodes[node_id] = node
        if local_rooms != [room_id]:
            raise ValueError("Each source candidate requires exactly its own Room context node")
        local_edges: set[str] = set()
        for edge in candidate["edges"]:
            if not isinstance(edge, dict):
                raise ValueError("Candidate edges must be dictionaries")
            edge_id = _text(edge.get("id"), "edge.id")
            if edge_id in local_edges:
                raise ValueError("Duplicate edge ID within a candidate")
            local_edges.add(edge_id)
            if edge.get("source") not in local_nodes or edge.get("target") not in local_nodes:
                raise ValueError("Candidate edge references an absent node")
            if edge_id in edges and edges[edge_id] != edge:
                raise ValueError(f"Conflicting shared edge: {edge_id}")
            edges[edge_id] = edge
    return nodes, edges


def group_room_candidates(candidates: list[dict[str, Any]], groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union member graphs and project their Room contexts to one declared space.

    Shared object and edge IDs occur once. Only existing edge endpoints that
    refer to a member Room change; their original endpoints remain in provenance.
    Missing relations, objects or ownership links are never synthesized.
    Ungrouped candidates remain deep-copy identical, including their historical
    membership metadata. Source membership is retained separately on unions.
    """
    if not isinstance(candidates, list) or any(not isinstance(c, dict) for c in candidates):
        raise ValueError("candidates must be a list of Room dictionaries")
    definitions, mapping, digest = _definitions([c.get("room_id") for c in candidates], groups)
    source_nodes, _ = _records(candidates)
    if any(d["space_id"] in source_nodes and d["space_id"] not in d["member_room_ids"] for d in definitions):
        raise ValueError("Space ID collides with a source node ID")
    by_id = {candidate["room_id"]: candidate for candidate in candidates}
    grouped_members = {m for d in definitions for m in d["member_room_ids"]}
    result = [deepcopy(c) for c in candidates if c["room_id"] not in grouped_members]
    for definition in definitions:
        space_id, members = definition["space_id"], definition["member_room_ids"]
        member_set = set(members)
        union_nodes: dict[str, dict[str, Any]] = {}
        union_edges: dict[str, dict[str, Any]] = {}
        node_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
        original_memberships = {}
        for member in members:
            candidate = by_id[member]
            original_memberships[member] = deepcopy(candidate.get("membership", {}))
            for node in candidate["nodes"]:
                if node["id"] not in member_set:
                    union_nodes[node["id"]] = deepcopy(node)
                    node_sources[node["id"]].append({"source_member_room_id": member,
                        "paths": deepcopy(candidate.get("membership", {}).get("node_sources", {}).get(node["id"], []))})
            for original in candidate["edges"]:
                edge = deepcopy(original)
                for endpoint in ("source", "target"):
                    if edge[endpoint] in member_set:
                        edge[endpoint] = space_id
                if edge["source"] != original["source"] or edge["target"] != original["target"]:
                    if "space_group_projection" in edge:
                        raise ValueError("Source edge already has reserved space projection provenance")
                    edge["space_group_projection"] = {
                        "version": SPACE_GROUP_VERSION, "space_id": space_id,
                        "original_source": original["source"], "original_target": original["target"],
                        "group_definition_sha256": digest,
                    }
                union_edges[edge["id"]] = edge
        union_nodes[space_id] = {
            "id": space_id, "label": "Room", "category": "room", "subtype": None,
            "original_subtype": None,
            "space_group_context": {"version": SPACE_GROUP_VERSION, "member_room_ids": members},
        }
        node_sources[space_id] = [{"source": "explicit_space_group_context", "member_room_ids": members}]
        source_versions = sorted({str(by_id[m].get("membership", {}).get("normalization_version", "unavailable")) for m in members})
        result.append({
            "room_id": space_id, "room_name": definition["space_name"],
            "nodes": [union_nodes[k] for k in sorted(union_nodes)],
            "edges": [union_edges[k] for k in sorted(union_edges)],
            "membership": {
                "policy": SPACE_GROUP_VERSION, "normalization_versions": source_versions,
                "node_sources": dict(sorted(node_sources.items())),
                "source_room_memberships": original_memberships,
                "room_context_policy": "member_room_quotient_with_original_edge_endpoints",
                "missing_relations": "not_inferred",
            },
            "space_group": {**deepcopy(definition), "version": SPACE_GROUP_VERSION,
                "group_definition_sha256": digest,
                "space_id_policy": "representative_member_id" if space_id in member_set else "new_declared_id",
                "member_room_names": {m: by_id[m].get("room_name", m) for m in members}},
        })
    result.sort(key=lambda candidate: candidate["room_id"])
    # Group metadata describes the new pool; singleton source metadata is unchanged.
    occurrences: dict[str, int] = defaultdict(int)
    for candidate in result:
        for node in candidate["nodes"]:
            occurrences[node["id"]] += 1
    for candidate in result:
        if "space_group" in candidate:
            candidate["membership"]["shared_node_ids"] = [n["id"] for n in candidate["nodes"] if occurrences[n["id"]] > 1]
    return result


def project_room_ranking_to_spaces(report: dict[str, Any], groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse each declared space to its best original Room ranking evidence.

    This is a separate baseline, not graph-union retrieval: original pool IDF,
    scores and solver status remain unchanged. Timing this call belongs to the
    layout's online scope. Failure/incomplete statuses are never upgraded.
    """
    if not isinstance(report, dict):
        raise ValueError("report must be a dictionary")
    definitions, mapping, digest = _definitions(report.get("candidate_ids"), groups)
    room_ids = report["candidate_ids"]
    if report.get("candidate_count") != len(room_ids) or report.get("candidate_ids_hash") != candidate_ids_hash(room_ids):
        raise ValueError("Source candidate pool identity is inconsistent")
    decimals = report.get("score_tie_decimals", 12)
    if type(decimals) is not int or not 0 <= decimals <= 15:
        raise ValueError("score_tie_decimals must be an integer in [0, 15]")
    ranking = report.get("ranking", [])
    if not isinstance(ranking, list) or any(not isinstance(row, dict) for row in ranking):
        raise ValueError("Source ranking must be a list of dictionaries")
    seen = set()
    for row in ranking:
        room_id = row.get("room_id")
        if room_id not in mapping or room_id in seen:
            raise ValueError("Source ranking contains duplicate or unknown Room IDs")
        seen.add(room_id)
        for field in ("score", "relation_numerator"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Source ranking {field} must be finite and nonnegative")
    if report.get("full_ranking") is True and seen != set(room_ids):
        raise ValueError("Source full ranking is missing a candidate")
    if report.get("status") == "completed" and (report.get("full_ranking") is not True or seen != set(room_ids)):
        raise ValueError("Completed projection requires an entire source ranking")
    projected = deepcopy(report)
    projected["source_room_ranking"] = deepcopy(ranking)
    projected["source_room_candidate_ids"] = deepcopy(room_ids)
    projected["source_room_candidate_ids_hash"] = report["candidate_ids_hash"]
    projected["source_room_candidate_count"] = report["candidate_count"]
    names = {d["space_id"]: d["space_name"] for d in definitions}
    members_by_space: dict[str, list[str]] = defaultdict(list)
    for room_id, space_id in mapping.items():
        members_by_space[space_id].append(room_id)
    def key(row: dict[str, Any]) -> tuple[float, float, str]:
        return -round(row["score"], decimals), -round(row["relation_numerator"], decimals), row["room_id"]
    best: dict[str, dict[str, Any]] = {}
    for row in sorted(ranking, key=key):
        space_id = mapping[row["room_id"]]
        if space_id not in best:
            selected = deepcopy(row)
            selected.update(room_id=space_id, room_name=names.get(space_id, row.get("room_name", space_id)),
                            source_room_id=row["room_id"], source_room_rank=row.get("rank"),
                            source_member_room_ids=members_by_space[space_id])
            best[space_id] = selected
    new_ranking = sorted(best.values(), key=key)
    ties: dict[tuple[float, float], list[int]] = defaultdict(list)
    for rank, row in enumerate(new_ranking, 1):
        ties[key(row)[:2]].append(rank)
    for rank, row in enumerate(new_ranking, 1):
        positions = ties[key(row)[:2]]
        row.update(rank=rank, best_tie_rank=positions[0], worst_tie_rank=positions[-1], tie_size=len(positions))
    ids = sorted(members_by_space)
    projected.update(ranking=new_ranking, candidate_ids=ids, candidate_count=len(ids),
        candidate_ids_hash=candidate_ids_hash(ids),
        top_score_tie=len(new_ranking) > 1 and round(new_ranking[0]["score"], decimals) == round(new_ranking[1]["score"], decimals),
        tie_policy="rounded_active_score_desc,rounded_active_relation_numerator_desc,canonical_room_id_asc",
        space_projection={"version": PROJECTION_VERSION, "group_definition_sha256": digest,
            "groups": definitions, "room_to_space_id": mapping,
            "scoring_pool": "original_room_pool", "selection": "best_member_by_original_score_relation_room_id",
            "original_status": report.get("status"), "missing_metadata_scope": "original_room_pool"})
    return projected
