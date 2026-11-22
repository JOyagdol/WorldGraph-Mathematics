"""Prepare immutable pending-input diagnostics, with GT kept out of inference."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from citygml_sg.ovg.validation.oracle_observation import (
    normalize_observation, SCORING_LABELS, _reject_map_identifiers,
)
from citygml_sg.retrieval.evaluation.oracle_metrics import candidate_ids_hash
from citygml_sg.retrieval.oracle_runner import content_hash, SAFE_ID
from citygml_sg.retrieval.reference_graph import build_room_candidates


def prepare_diagnostic_bundle(manifest_path: Path, snapshot_path: Path, output: Path,
                              *, expected_hashes: dict[str, str]) -> dict[str, Any]:
    """Copy exact observation bytes only after validating the complete frozen inventory.

    Original manifests may contain evaluator GT, but none is used for selection or
    copied. Input hashes must come from a prior observation freeze. A missing or
    changed observation is an error. Annotation verification is never inferred.
    """
    manifest_path, snapshot_path, output = map(Path, (manifest_path, snapshot_path, output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty")
    manifest_bytes, snapshot_bytes = manifest_path.read_bytes(), snapshot_path.read_bytes()
    source = json.loads(manifest_bytes.decode("utf-8-sig"))
    snapshot = json.loads(snapshot_bytes.decode("utf-8-sig"))
    cases = source.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Manifest requires nonempty cases")
    pool = sorted(candidate["room_id"] for candidate in build_room_candidates(snapshot))
    if not pool:
        raise ValueError("Snapshot has no Room candidates")
    pool_hash, snapshot_hash = candidate_ids_hash(pool), content_hash(snapshot)
    seen, prepared = set(), []
    for case in cases:
        query = case.get("query_id")
        if not isinstance(query, str) or not SAFE_ID.fullmatch(query) or query in seen:
            raise ValueError("query_id must be unique and safe")
        seen.add(query)
        path = manifest_path.parent / case["ovg_path"]
        payload = path.read_bytes()
        digest = sha256(payload).hexdigest()
        if expected_hashes.get(query) != digest:
            raise ValueError(f"Frozen observation hash mismatch or absent: {query}")
        raw = json.loads(payload.decode("utf-8-sig"))
        # The inference normalizer intentionally ignores metadata. A blind
        # diagnostic bundle must also reject map/GT IDs in those ignored fields.
        _reject_map_identifiers(raw, "observation")
        normalized = normalize_observation(raw)
        n_objects = sum(node["label"] in SCORING_LABELS for node in normalized["nodes"])
        stratum = ("objects_relations" if normalized["edges"] else "objects_only") if n_objects else (
            "context_only" if normalized["nodes"] else "empty")
        prepared.append(({"query_id": query, "ovg_path": f"observations/{query}.json",
                          "observation_sha256": digest, "stratum": stratum,
                          "input_condition": source.get("input_condition", "researcher_curated"),
                          "oracle_validation_status": "pending"}, payload))
    if seen != set(expected_hashes):
        raise ValueError("Frozen inventory and manifest query sets differ")
    # Predeclare one lexicographically first query in each observed stratum.
    ordered = sorted((case for case, _ in prepared), key=lambda case: case["query_id"])
    chosen = [next((case for case in ordered if case["stratum"] == name), None)
              for name in ("objects_relations", "objects_only", "context_only", "empty")]
    base = {"manifest_version": "no_gt_diagnostic_inputs_v1",
            "graph_snapshot_id": f"semantic_{snapshot_hash[:16]}",
            "graph_snapshot_sha256": snapshot_hash, "candidate_pool_id": f"pool_{pool_hash[:16]}",
            "candidate_ids_hash": pool_hash, "experiment_stage": "diagnostic_pending_input",
            "oracle_validation_status": "pending", "annotation_verification_performed": False}
    summary = {"status": "prepared_pending_input", "query_count": len(prepared),
               "candidate_count": len(pool), "strata": dict(Counter(c["stratum"] for c in ordered)),
               "pilot_selection": "lexicographic_first_per_stratum_before_outcomes",
               "pilot_query_ids": [c["query_id"] for c in chosen if c],
               "accuracy_status": "not_ready_requires_separate_canonical_gt_and_review",
               "source_manifest_file_sha256": sha256(manifest_bytes).hexdigest(),
               "snapshot_file_sha256": sha256(snapshot_bytes).hexdigest(),
               "annotation_verification_performed": False, "retrieval_status": "not_run"}
    output.mkdir(parents=True, exist_ok=True)
    (output / "observations").mkdir()
    for case, payload in prepared:
        with (output / case["ovg_path"]).open("xb") as stream:
            stream.write(payload)
    with (output / "snapshot.json").open("xb") as stream:
        stream.write(snapshot_bytes)
    for name, value in {
        "diagnostic_manifest.json": {**base, "cases": ordered},
        "pilot_manifest.json": {**base, "selection_policy": summary["pilot_selection"],
                                "cases": [c for c in chosen if c]},
        "candidate_pool.json": {"candidate_ids": pool, "candidate_ids_hash": pool_hash},
        "bundle_manifest.json": summary,
    }.items():
        with (output / name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return summary
