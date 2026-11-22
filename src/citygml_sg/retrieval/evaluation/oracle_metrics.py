"""Strict, full-pool metrics for the proposed Oracle retrieval component.

Retrieval abstention is a valid zero-hit outcome. Incomplete solvers and execution
failures invalidate primary aggregates; a separately labelled failure-count
diagnostic preserves their denominator without pretending they were exact runs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from copy import deepcopy
from hashlib import sha256
import json
from math import isfinite
from statistics import mean
from typing import Any


VARIANT_IDS = (
    "full", "objects_only", "relations_only", "uniform_category", "boolean_metadata",
)
PROVENANCE_FIELDS = ("experiment_stage", "input_condition", "oracle_validation_status")
METRIC_FIELDS = ("recall_at_1", "recall_at_3", "hit_at_1", "hit_at_3", "mrr", "mtr")
ABSTENTION_STATUSES = {"abstained_no_anchor", "abstained_no_evidence"}
INCOMPLETE_STATUSES = {"solver_incomplete", "execution_failed"}
VALID_STATUSES = {"completed", *ABSTENTION_STATUSES}
MTR_POLICY = "N_plus_1_on_retrieval_abstention"
CONTENT_IDENTITY_FIELDS = (
    "method_version", "observation_sha256", "config_sha256", "graph_snapshot_sha256",
    "code_provenance",
)
RUN_IDENTITY_FIELDS = (
    "candidate_count", "candidate_ids_hash", "graph_snapshot_id", "method_version",
    "config_sha256", "graph_snapshot_sha256", "code_provenance",
)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty canonical ID string")
    return value


def _ids(values: Any, name: str) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of canonical IDs")
    result = [_identifier(value, name) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate Room ID in {name}")
    return result


def candidate_ids_hash(candidate_ids: Sequence[str]) -> str:
    """Hash the unique candidate set, independent of retrieval/ranking order."""
    ids = _ids(candidate_ids, "candidate_ids")
    payload = json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def hit_recall_at_k(
    target_ids: Sequence[str], ranked_ids: Sequence[str], k: int,
) -> dict[str, float]:
    """General set-based Hit/Recall; the primary evaluator separately requires one GT."""
    targets = _ids(target_ids, "target_ids")
    ranked = _ids(ranked_ids, "ranked_ids")
    if not targets:
        raise ValueError("target_ids must not be empty")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    retrieved = len(set(targets).intersection(ranked[:k]))
    return {"hit": float(retrieved > 0), "recall": retrieved / len(targets)}


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _validate_ranking(
    report: dict[str, Any], candidates: list[str], *, exact: bool,
) -> tuple[list[dict[str, Any]], bool]:
    ranking = report.get("ranking")
    if not isinstance(ranking, list) or any(not isinstance(row, dict) for row in ranking):
        raise ValueError("ranking must be a list of candidate objects")
    ranked_ids = _ids([row.get("room_id") for row in ranking], "ranking")
    if not set(ranked_ids).issubset(candidates):
        raise ValueError("ranking contains a Room outside the candidate pool")
    if exact and (report.get("full_ranking") is not True or set(ranked_ids) != set(candidates)):
        raise ValueError("full-ranking required: ranking must contain the complete candidate pool")
    decimals = report.get("score_tie_decimals", 12)
    if type(decimals) is not int or not 0 <= decimals <= 15:
        raise ValueError("score_tie_decimals must be an integer from 0 to 15")
    opening_policy = report.get("rank_key_policy") == "shared_wall_or_single_opening_v1"
    keys = []
    for row in ranking:
        score = _number(row.get("score"), "score")
        relation = _number(row.get("relation_numerator"), "relation_numerator")
        if not 0 <= score <= 1 or relation < 0:
            raise ValueError("score must be in [0,1] and active relation numerator nonnegative")
        variant = report.get("variant_id", report.get("variant"))
        if variant == "objects_only" and relation != 0:
            raise ValueError("objects_only must not use relation evidence for tie-breaking")
        if opening_policy:
            from citygml_sg.retrieval.opening_pattern_ranker import decode_rank_key
            obj, probability, auxiliary, active_relation = decode_rank_key(row)
            obj = _number(obj, "object score key")
            auxiliary = _number(auxiliary, "auxiliary score key")
            active_relation = _number(active_relation, "relation numerator key")
            if (not 0 <= obj <= 1 or not 0 <= auxiliary <= 1
                    or active_relation != round(relation, decimals)):
                raise ValueError("Invalid opening-pattern ranking terms")
            if obj != round(_number(row["object_component_score"], "object component score"), decimals):
                raise ValueError("Object key differs from component score")
            if variant == "relations_only" and (obj != 0 or row["object_numerator"] != 0):
                raise ValueError("relations_only must not rank with object scores")
            if report["gate_reason"] in {"closed", "disabled"}:
                if probability or auxiliary or active_relation or obj != round(score, decimals):
                    raise ValueError("Inactive pattern must use only the object score")
            elif auxiliary != round(score, decimals):
                raise ValueError("Auxiliary score key differs from component score")
            keys.append((-obj, -probability, -auxiliary, -active_relation, row["room_id"]))
        else:
            keys.append((-round(score, decimals), -round(relation, decimals), row["room_id"]))
        solver = row.get("solver")
        if exact and (not isinstance(solver, dict) or solver.get("optimality_proven") is not True):
            raise ValueError(
                "completed exact ranking requires optimality_proven for every candidate"
            )
    if keys != sorted(keys):
        raise ValueError("ranking has unstable order under the declared evidence and Room ID keys")
    top_score_tie = len(keys) > 1 and (
        keys[0][:-1] == keys[1][:-1] if opening_policy else keys[0][0] == keys[1][0])
    if exact:
        groups = defaultdict(list)
        for rank, key in enumerate(keys, start=1):
            groups[key[:-1]].append(rank)
        for rank, (row, key) in enumerate(zip(ranking, keys), start=1):
            tied = groups[key[:-1]]
            expected = {
                "rank": rank, "best_tie_rank": tied[0],
                "worst_tie_rank": tied[-1], "tie_size": len(tied),
            }
            for name, value in expected.items():
                if type(row.get(name)) is not int or row[name] != value:
                    raise ValueError(f"invalid {name} for Room {row['room_id']}")
        if "top_score_tie" in report and report["top_score_tie"] is not top_score_tie:
            raise ValueError("top_score_tie disagrees with the quantized ranking")
    return ranking, top_score_tie


def evaluate_retrieval(
    report: dict[str, Any], target_room_id: str, *, candidate_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate one exact closed-set single-GT query, including honest failure status.

    A truncated ranking or absent GT is a data-contract error, never an estimated
    rank. The target is supplied only here, after inference has returned.
    """
    target = _identifier(target_room_id, "target_room_id (single GT)")
    pool = _ids(candidate_ids if candidate_ids is not None else report.get("candidate_ids"),
                "candidate_ids")
    if not pool:
        raise ValueError("candidate pool must not be empty")
    if candidate_ids is not None and "candidate_ids" in report:
        if set(pool) != set(_ids(report["candidate_ids"], "report candidate_ids")):
            raise ValueError("explicit candidate pool differs from the report")
    if type(report.get("candidate_count")) is not int or report["candidate_count"] != len(pool):
        raise ValueError("candidate_count differs from the complete candidate pool")
    fingerprint = candidate_ids_hash(pool)
    if report.get("candidate_ids_hash") != fingerprint:
        raise ValueError("candidate_ids_hash differs from the complete candidate pool")
    if target not in pool:
        raise ValueError("GT Room is absent from the closed-set candidate pool")
    status = report.get("status")
    if status not in VALID_STATUSES | INCOMPLETE_STATUSES:
        raise ValueError(f"unsupported retrieval status: {status}")
    # An abstention may have a complete zero-evidence ranking, or no ranking.
    has_ranking = bool(report.get("ranking"))
    ranking, ambiguous = _validate_ranking(
        report, pool,
        exact=status == "completed" or (status in ABSTENTION_STATUSES and has_ranking),
    )
    metadata_fields = (
        "query_id", "variant_id", *PROVENANCE_FIELDS, "annotation_protocol_version",
        "view_id", "parent_query_id", "graph_snapshot_id", "run_id",
        *CONTENT_IDENTITY_FIELDS,
    )
    result = {key: deepcopy(report[key]) for key in metadata_fields if key in report}
    if "variant_id" not in result and "variant" in report:
        result["variant_id"] = report["variant"]
    result.update({
        "protocol": "closed_set_single_gt_room", "target_room_id": target,
        "candidate_count": len(pool), "candidate_ids_hash": fingerprint,
        "status": status, "evaluation_status": "completed",
        "mtr_policy": MTR_POLICY,
        "metric_units": {"recall_hit_mrr": "fraction", "mtr": "rank"},
        "target_rank": None, "evaluation_rank": None,
        "retrieval_failed": status != "completed", "ambiguous": ambiguous,
        "top_score_tie": ambiguous,
        "compatibility_aliases": {
            "top1_room_accuracy": "recall_at_1", "top3_room_inclusion": "recall_at_3",
        },
    })
    if status in INCOMPLETE_STATUSES:
        result.update({key: None for key in METRIC_FIELDS})
        result.update({
            "evaluation_status": "incomplete", "reciprocal_rank": None,
            "top1_room_accuracy": None, "top3_room_inclusion": None,
            "failure_stage": report.get("failure_stage"),
            "failure_reason": report.get("failure_reason", status),
            "diagnostic_failure_metrics": {
                "recall_at_1": 0.0, "recall_at_3": 0.0,
                "hit_at_1": 0.0, "hit_at_3": 0.0, "mrr": 0.0,
                "mtr": len(pool) + 1,
                "policy": "count_incomplete_as_failure_N_plus_1_diagnostic_only",
            },
        })
        return result
    if status in ABSTENTION_STATUSES:
        rank, rr = len(pool) + 1, 0.0
        hit1, hit3 = 0.0, 0.0
    else:
        row = next(row for row in ranking if row["room_id"] == target)
        rank, rr = row["rank"], 1.0 / row["rank"]
        hit1, hit3 = float(rank <= 1), float(rank <= min(3, len(pool)))
        result.update({key: row[key] for key in ("best_tie_rank", "worst_tie_rank", "tie_size")})
        result["target_rank"] = rank
    result.update({
        "evaluation_rank": rank, "reciprocal_rank": rr,
        "recall_at_1": hit1, "hit_at_1": hit1, "recall_at_3": hit3, "hit_at_3": hit3,
        "mrr": rr, "mtr": rank, "top1_room_accuracy": hit1, "top3_room_inclusion": hit3,
    })
    assert result["hit_at_1"] == result["recall_at_1"]
    assert result["hit_at_3"] == result["recall_at_3"]
    return result


def _means(records: list[dict[str, Any]]) -> dict[str, float | None]:
    return {key: mean(record[key] for record in records) if records else None
            for key in METRIC_FIELDS}


def _timing_mean(record: dict[str, Any]) -> float | None:
    stats = record.get("timing_stats", {})
    value = stats.get("mean_ms") if isinstance(stats, dict) else None
    if value is None:
        value = record.get("online_retrieval_ms_mean", record.get("online_retrieval_ms"))
    if value is None:
        return None
    result = _number(value, "online_retrieval_ms mean")
    if result < 0:
        raise ValueError("online retrieval time must be nonnegative")
    return result


def _summarize(records: list[dict[str, Any]], expected_queries: set[str]) -> dict[str, Any]:
    valid = [record for record in records if record["status"] in VALID_STATUSES]
    missing = sorted(expected_queries - {record["query_id"] for record in records})
    complete = len(valid) == len(records) and not missing
    failures = Counter(record["status"] for record in records if record["status"] != "completed")
    by_room: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in valid:
        by_room[record["target_room_id"]].append(record)
    room_means = [_means(room_records) for room_records in by_room.values()]
    timing = [_timing_mean(record) for record in records]
    measured_timing = [value for value in timing if value is not None]
    all_times_present = len(measured_timing) == len(records)
    summary = {
        "status": "completed" if complete else "incomplete",
        "query_count": len(records), "expected_query_count": len(expected_queries),
        "valid_query_count": len(valid), "missing_query_ids": missing,
        "unique_room_count": len({record["target_room_id"] for record in records}),
        "aggregation": "query_mean", "mtr_policy": MTR_POLICY,
        "metric_units": {"recall_hit_mrr": "fraction", "mtr": "rank", "mean_time": "ms"},
        "failure_counts": dict(sorted(failures.items())),
        "retrieval_failure_rate": sum(failures.values()) / len(records) if not missing else None,
        "ambiguous_query_rate": (
            mean(float(record.get("ambiguous", False)) for record in records) if complete else None
        ),
        "room_macro": _means(room_means) if complete else {key: None for key in METRIC_FIELDS},
        "room_query_counts": dict(Counter(record["target_room_id"] for record in records)),
        "mean_time_ms": mean(measured_timing) if complete and all_times_present else None,
        "timing_query_count": len(measured_timing),
        "timing_aggregation": "mean_of_query_means; timing repeats are not queries",
    }
    summary.update(_means(valid) if complete else {key: None for key in METRIC_FIELDS})
    summary["top1_room_accuracy"] = summary["recall_at_1"]
    summary["top3_room_inclusion"] = summary["recall_at_3"]
    summary["compatibility_aliases"] = {
        "top1_room_accuracy": "recall_at_1", "top3_room_inclusion": "recall_at_3",
    }
    if not complete:
        diagnostics = [record if record["status"] in VALID_STATUSES
                       else record["diagnostic_failure_metrics"] for record in records]
        summary["diagnostic_failure_count_policy"] = {
            **_means(diagnostics), "query_count": len(records),
            "policy": "count_incomplete_as_failure_N_plus_1_diagnostic_only",
            "excludes_missing_queries": missing,
        }
    return summary


def _paired(by_variant: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    full = {record["query_id"]: record for record in by_variant.get("full", [])}
    objects = {record["query_id"]: record for record in by_variant.get("objects_only", [])}
    common = sorted(full.keys() & objects.keys())
    counts = {"improved": 0, "tied": 0, "worsened": 0}
    pairs = []
    incomplete = []
    for query_id in common:
        left, right = full[query_id], objects[query_id]
        if left["status"] not in VALID_STATUSES or right["status"] not in VALID_STATUSES:
            incomplete.append(query_id)
            continue
        delta = left["evaluation_rank"] - right["evaluation_rank"]
        outcome = "improved" if delta < 0 else "worsened" if delta > 0 else "tied"
        counts[outcome] += 1
        pairs.append({
            "query_id": query_id, "target_room_id": left["target_room_id"],
            "full_rank": left["evaluation_rank"], "objects_only_rank": right["evaluation_rank"],
            "rank_delta_full_minus_objects_only": delta, "outcome": outcome,
        })
    missing = sorted(full.keys() ^ objects.keys())
    status = "not_run" if not full and not objects else "completed"
    if missing or incomplete:
        status = "incomplete"
    return {
        "status": status, "query_count": len(pairs), **counts,
        "mtr_policy": MTR_POLICY, "missing_query_ids": missing,
        "incomplete_query_ids": incomplete, "per_query": pairs,
    }


def _validate_run_identity(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Check a fixed full-pool run without certifying missing legacy provenance."""
    first = records[0]
    for field in RUN_IDENTITY_FIELDS:
        if any(record.get(field) != first.get(field) for record in records[1:]):
            raise ValueError(f"one full-pool run must share {field}; provenance differs or is missing")
    # Different queries may use different observations, but cannot silently mix
    # hashed observations with records whose observation identity was discarded.
    for field in CONTENT_IDENTITY_FIELDS:
        present = [record.get(field) is not None for record in records]
        if any(present) and not all(present):
            raise ValueError(f"{field} provenance is missing from some evaluation records")
    supplied = [field for field in CONTENT_IDENTITY_FIELDS if first.get(field) is not None]
    missing = [field for field in CONTENT_IDENTITY_FIELDS if field not in supplied]
    return {
        "status": ("legacy_unverified" if not supplied else
                   "partial_identity" if missing else "declared_identity_consistent"),
        "supplied_fields": supplied, "missing_fields": missing,
        "scope": "supplied run identity equality and per-query observation equality across variants",
        "limitation": "does not certify annotations or independently rehash source artifacts",
    }


def aggregate_evaluations(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Group exact results by input provenance; never merge conditions as one primary.

    Repeated timing trials must be inside each record, not additional query rows.
    Different conditions are returned under ``by_condition`` with an empty primary
    ``by_variant``. Missing or failed queries invalidate their variant's primary.
    The candidate pool, snapshot, method/config and code identity are fixed for
    this full-pool protocol. Uniformly missing legacy provenance remains usable
    for metric arithmetic, with an explicit unverified/partial identity status.
    """
    items = list(records)
    if not items:
        return {"status": "not_run", "by_variant": {}, "by_condition": {},
                "paired": {}, "provenance": None}
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, ...]] = set()
    contracts: dict[tuple[str, ...], tuple[Any, ...]] = {}
    run_ids = {record.get("run_id") for record in items}
    if len(run_ids) > 1:
        raise ValueError("aggregate one run at a time; run_id provenance differs")
    for record in items:
        condition = tuple(_identifier(record.get(key), key) for key in PROVENANCE_FIELDS)
        query = _identifier(record.get("query_id"), "query_id")
        variant = _identifier(record.get("variant_id"), "variant_id")
        identity = (*condition, query, variant)
        if identity in seen:
            raise ValueError("duplicate query_id within variant/condition; repeats are not queries")
        seen.add(identity)
        _identifier(record.get("target_room_id"), "target_room_id")
        if record.get("status") not in VALID_STATUSES | INCOMPLETE_STATUSES:
            raise ValueError("aggregate records must come from evaluate_retrieval")
        if record["status"] in VALID_STATUSES:
            for key in METRIC_FIELDS:
                _number(record.get(key), key)
            if any(record[f"hit_at_{k}"] != record[f"recall_at_{k}"] for k in (1, 3)):
                raise ValueError("single-GT Hit/Recall aliases must be equal")
        elif not isinstance(record.get("diagnostic_failure_metrics"), dict):
            raise ValueError("incomplete records require diagnostic failure metrics")
        contract = tuple(record.get(key) for key in (
            "target_room_id", "candidate_count", "candidate_ids_hash", "graph_snapshot_id",
            "annotation_protocol_version", "view_id", "parent_query_id", "run_id",
            *CONTENT_IDENTITY_FIELDS,
        ))
        contract_key = (*condition, query)
        if contract_key in contracts and contracts[contract_key] != contract:
            raise ValueError(
                "variants must share query GT, observation, candidate pool, snapshot, "
                "method/config/code identity and provenance"
            )
        contracts[contract_key] = contract
        grouped[condition].append(record)
    identity_validation = _validate_run_identity(items)
    conditions = {}
    for condition, condition_records in sorted(grouped.items()):
        by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        expected = {record["query_id"] for record in condition_records}
        for record in condition_records:
            by_variant[record["variant_id"]].append(record)
        provenance = dict(zip(PROVENANCE_FIELDS, condition))
        key = json.dumps(provenance, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        conditions[key] = {
            "provenance": provenance,
            "identity_validation": deepcopy(identity_validation),
            "by_variant": {variant: _summarize(rows, expected)
                           for variant, rows in sorted(by_variant.items())},
            "paired": _paired(by_variant),
        }
    if len(conditions) > 1:
        return {"status": "separated_conditions", "by_variant": {}, "by_condition": conditions,
                "paired": {}, "provenance": None, "identity_validation": identity_validation}
    condition = next(iter(conditions.values()))
    status = "completed" if all(
        value["status"] == "completed" for value in condition["by_variant"].values()
    ) else "incomplete"
    return {"status": status, "by_condition": conditions, **condition}
