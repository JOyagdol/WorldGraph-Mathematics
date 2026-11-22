"""Offline input preparation; no matching, ranking, metrics or Oracle certification."""

from __future__ import annotations

from hashlib import sha256
import json
import re
from typing import Any

from citygml_sg.ovg.validation.oracle_observation import (
    NORMALIZATION_VERSION as OBSERVATION_NORMALIZATION_VERSION,
    normalize_observation,
)
from citygml_sg.ovg.validation.reference_anchors import POLICY_VERSION, select_reference_anchors
from citygml_sg.retrieval.reference_graph import (
    CATEGORY_ALIASES, NORMALIZATION_VERSION as REFERENCE_NORMALIZATION_VERSION,
    normalize_semantic_node,
)
from citygml_sg.retrieval.room_scope import (
    SCOPE_VERSION, build_room_hop_subgraphs, canonical_node_type,
)

COMPONENT_VERSION = "reference_input_preparation_v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def content_hash(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def prepare_reference_inputs(
    snapshot: dict[str, Any], observations: dict[str, dict[str, Any]], *,
    anchor_threshold: float, hop_radius: int = 2,
) -> dict[str, Any]:
    """Prepare literal hop scopes and exact vocabulary-filtered anchor graphs.

    The vocabulary is built from the entire supplied semantic snapshot, not
    from a target or a candidate's contents. No GT field is accepted as an
    argument. Annotation status remains outside the inference representation.
    Existing normalizers are explicitly versioned, not assumed to implement a
    final publication category taxonomy. In particular, cabinet and storage
    remain distinct and generic opening does not expand to window or door.
    """
    if not isinstance(observations, dict) or not observations:
        raise ValueError("observations must be a nonempty query-ID mapping")
    if any(not isinstance(key, str) or not SAFE_ID.fullmatch(key) for key in observations):
        raise ValueError("query IDs must be safe, nonempty identifiers")
    scopes = build_room_hop_subgraphs(snapshot, h=hop_radius)
    if not scopes:
        raise ValueError("reference snapshot must contain at least one Room")
    vocabulary: set[str] = set()
    for node in snapshot["nodes"]:
        label = canonical_node_type(node)
        if label in {"Building", "BuildingPart"}:
            # Containers can carry explicit categories but never borrow their ID.
            props = node.get("properties", {})
            category = node.get("category", props.get("category", ""))
            if not isinstance(category, str):
                raise ValueError("container category must be a string")
            category = category.strip().casefold()
            category = CATEGORY_ALIASES.get(category, category)
        else:
            category = normalize_semantic_node(node)["category"].casefold()
        if category:
            vocabulary.add(category)
    config = {
        "component_version": COMPONENT_VERSION, "hop_radius": hop_radius,
        "room_scope_version": SCOPE_VERSION, "anchor_policy": POLICY_VERSION,
        "anchor_threshold": anchor_threshold,
        "reference_normalization_version": REFERENCE_NORMALIZATION_VERSION,
        "observation_normalization_version": OBSERVATION_NORMALIZATION_VERSION,
        "vocabulary_scope": "entire_supplied_semantic_snapshot",
        "category_match": "exact_after_declared_normalization",
        "container_category_policy": "explicit_category_only_no_label_or_id_fallback",
        "observation_relation_policy": "preserve_directed_atoms_reject_exact_duplicates",
        "observation_normalizer_config": {"symmetric_relations": [], "inverse_relations": {}},
    }
    queries = {}
    for query_id, raw in sorted(observations.items()):
        normalized = normalize_observation(raw, config["observation_normalizer_config"])
        anchors = select_reference_anchors(normalized, vocabulary, anchor_threshold)
        queries[query_id] = {
            "observation_content_sha256": content_hash(raw),
            "normalized_observation_sha256": content_hash(normalized),
            "anchor_graph_sha256": content_hash(anchors), "anchors": anchors,
        }
    candidate_ids = [scope["room_id"] for scope in scopes]
    return {
        "status": "not_run", "preparation_status": "completed",
        "component_version": COMPONENT_VERSION, "method_version": None,
        "config": config, "config_sha256": content_hash(config),
        "graph_snapshot_content_sha256": content_hash(snapshot),
        "reference_categories": sorted(vocabulary),
        "reference_vocabulary_sha256": content_hash(sorted(vocabulary)),
        "candidate_ids": candidate_ids, "candidate_count": len(candidate_ids),
        "candidate_ids_sha256": content_hash(candidate_ids),
        "room_scopes": scopes, "queries": queries,
        "oracle_certification": "not_performed",
        "unresolved_components": [
            "final_category_taxonomy", "scoring_domain_and_idf_population",
            "instance_matching_and_tie_policy", "score_and_proof_contract",
            "reviewed_observations_and_canonical_gt", "frozen_evaluation_protocol",
        ],
        "notes": [
            "No matching, scores, rankings, retrieval timings or metrics were computed.",
            "Hop scopes may contain other Rooms through shared parents or relations.",
            "Inherited normalization scoring flags do not define a scoring domain here.",
            "A snapshot hash identifies supplied content; it does not certify data quality.",
        ],
    }
