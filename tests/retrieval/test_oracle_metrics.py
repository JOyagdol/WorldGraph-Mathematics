"""Hand-calculated fixtures; no experiment results or database access required."""

from __future__ import annotations

from copy import deepcopy
import csv
from io import StringIO

import pytest

from citygml_sg.retrieval.evaluation.oracle_metrics import (
    VARIANT_IDS,
    aggregate_evaluations,
    candidate_ids_hash,
    evaluate_retrieval,
    hit_recall_at_k,
)
from citygml_sg.retrieval.reporting.ablation_table import build_ablation_table


def _report(
    order: tuple[str, ...] = ("A", "B", "C"), *, status: str = "completed",
) -> dict:
    ranking = [
        {
            "room_id": room_id, "score": 1.0 / rank, "relation_numerator": 0.0,
            "rank": rank, "best_tie_rank": rank, "worst_tie_rank": rank, "tie_size": 1,
            "solver": {"optimality_proven": True},
        }
        for rank, room_id in enumerate(order, start=1)
    ]
    return {
        "status": status, "ranking": ranking if status == "completed" else [],
        "candidate_count": len(order), "candidate_ids": sorted(order),
        "candidate_ids_hash": candidate_ids_hash(order), "full_ranking": status == "completed",
        "score_tie_decimals": 12, "top_score_tie": False,
    }


def _record(
    query_id: str, target: str = "A", *, order: tuple[str, ...] = ("A", "B", "C"),
    variant_id: str = "full", status: str = "completed",
) -> dict:
    return {
        **evaluate_retrieval(_report(order, status=status), target),
        "query_id": query_id, "variant_id": variant_id,
        "experiment_stage": "oracle_preparation", "input_condition": "researcher_curated",
        "oracle_validation_status": "pending",
    }


def test_hand_calculated_ranks_1_1_3_and_single_gt_aliases() -> None:
    records = [_record("q1"), _record("q2"), _record("q3", "C")]
    summary = aggregate_evaluations(records)["by_variant"]["full"]
    assert summary["query_count"] == 3
    assert summary["status"] == "completed"
    assert summary["recall_at_1"] == pytest.approx(2 / 3)
    assert summary["recall_at_3"] == 1
    assert summary["mrr"] == pytest.approx(7 / 9)
    assert summary["mtr"] == pytest.approx(5 / 3)
    for record in [*records, summary]:
        assert record["hit_at_1"] == record["recall_at_1"]
        assert record["hit_at_3"] == record["recall_at_3"]
        assert record["top1_room_accuracy"] == record["recall_at_1"]
        assert record["top3_room_inclusion"] == record["recall_at_3"]


def test_general_hit_and_recall_are_different_with_multiple_gt() -> None:
    metrics = hit_recall_at_k(["A", "B", "C"], ["A", "D", "E"], 3)
    assert metrics == {"hit": 1.0, "recall": 1 / 3}
    with pytest.raises(ValueError, match="single GT"):
        evaluate_retrieval(_report(), ["A", "B"])


@pytest.mark.parametrize("status", ["abstained_no_anchor", "abstained_no_evidence"])
def test_abstention_is_included_with_n_plus_one_rank(status: str) -> None:
    failed = _record("empty", status=status)
    assert failed["evaluation_status"] == "completed"
    assert failed["target_rank"] is None
    assert failed["evaluation_rank"] == failed["mtr"] == 4
    assert failed["recall_at_1"] == failed["recall_at_3"] == failed["mrr"] == 0
    summary = aggregate_evaluations([_record("ok"), failed])["by_variant"]["full"]
    assert summary["query_count"] == 2
    assert summary["mtr"] == 2.5
    assert summary["mrr"] == summary["retrieval_failure_rate"] == 0.5


@pytest.mark.parametrize("status", ["solver_incomplete", "execution_failed"])
def test_incomplete_execution_invalidates_primary_and_preserves_denominator(status: str) -> None:
    failure = _record("failed", status=status)
    assert failure["mrr"] is None
    summary = aggregate_evaluations([_record("ok"), failure])["by_variant"]["full"]
    assert summary["status"] == "incomplete"
    assert summary["query_count"] == 2
    assert summary["valid_query_count"] == 1
    assert summary["mrr"] is summary["recall_at_1"] is summary["mtr"] is None
    assert summary["diagnostic_failure_count_policy"]["mrr"] == 0.5
    table = build_ablation_table({"full": summary})
    assert table["status"] == "incomplete"
    assert table["rows"][0]["cells"][1] == "incomplete"


def test_duplicate_absent_gt_truncated_count_and_hash_are_rejected() -> None:
    duplicate = _report()
    duplicate["ranking"][1]["room_id"] = "A"
    with pytest.raises(ValueError, match="duplicate Room ID"):
        evaluate_retrieval(duplicate, "A")
    with pytest.raises(ValueError, match="absent"):
        evaluate_retrieval(_report(), "Z")
    truncated = _report(("A", "B", "C", "D"))
    truncated["ranking"] = truncated["ranking"][:3]
    with pytest.raises(ValueError, match="full-ranking required"):
        evaluate_retrieval(truncated, "D")
    truncated = _report()
    truncated["full_ranking"] = False
    with pytest.raises(ValueError, match="full-ranking required"):
        evaluate_retrieval(truncated, "A")
    for field, value in [("candidate_count", 4), ("candidate_ids_hash", "wrong")]:
        invalid = _report()
        invalid[field] = value
        with pytest.raises(ValueError, match=field):
            evaluate_retrieval(invalid, "A")


def test_candidate_hash_is_order_independent_and_duplicate_sensitive() -> None:
    assert candidate_ids_hash(["A", "B", "C"]) == candidate_ids_hash(["C", "A", "B"])
    with pytest.raises(ValueError, match="duplicate"):
        candidate_ids_hash(["A", "A"])


def test_ties_use_active_score_relation_and_canonical_id() -> None:
    report = _report()
    report["top_score_tie"] = True
    for row in report["ranking"]:
        row.update({"score": 0.5, "best_tie_rank": 1, "worst_tie_rank": 3, "tie_size": 3})
    metrics = evaluate_retrieval(report, "C")
    assert metrics["target_rank"] == 3
    assert metrics["best_tie_rank"] == 1
    assert metrics["worst_tie_rank"] == 3
    assert metrics["ambiguous"] is True
    swapped = deepcopy(report)
    swapped["ranking"][0], swapped["ranking"][1] = swapped["ranking"][1], swapped["ranking"][0]
    with pytest.raises(ValueError, match="unstable order"):
        evaluate_retrieval(swapped, "A")
    bad_tie = deepcopy(report)
    bad_tie["ranking"][0]["tie_size"] = 1
    with pytest.raises(ValueError, match="tie_size"):
        evaluate_retrieval(bad_tie, "A")
    objects = deepcopy(report)
    objects["variant_id"] = "objects_only"
    objects["ranking"][0]["relation_numerator"] = 1
    with pytest.raises(ValueError, match="objects_only"):
        evaluate_retrieval(objects, "A")


def test_unproven_or_nonfinite_exact_results_are_rejected() -> None:
    report = _report()
    report["ranking"][0]["solver"]["optimality_proven"] = False
    with pytest.raises(ValueError, match="optimality_proven"):
        evaluate_retrieval(report, "A")
    for number in [float("nan"), float("inf")]:
        report = _report()
        report["ranking"][0]["score"] = number
        with pytest.raises(ValueError, match="finite"):
            evaluate_retrieval(report, "A")


def test_query_mean_and_room_macro_have_separate_denominators() -> None:
    summary = aggregate_evaluations([
        _record("room_a_view1"), _record("room_a_view2"), _record("room_c_view", "C"),
    ])["by_variant"]["full"]
    assert summary["recall_at_1"] == pytest.approx(2 / 3)
    assert summary["room_macro"]["recall_at_1"] == 0.5
    assert summary["room_query_counts"] == {"A": 2, "C": 1}


def test_duplicate_query_and_variant_candidate_mismatch_are_rejected() -> None:
    record = _record("q")
    with pytest.raises(ValueError, match="duplicate query_id"):
        aggregate_evaluations([record, deepcopy(record)])
    other = _record("q", variant_id="objects_only", order=("A", "B", "C", "D"))
    with pytest.raises(ValueError, match="variants must share"):
        aggregate_evaluations([record, other])


def test_input_conditions_are_separated_and_cannot_render_a_mixed_primary() -> None:
    curated = _record("q")
    automatic = deepcopy(curated)
    automatic["input_condition"] = "automatic_pipeline"
    result = aggregate_evaluations([curated, automatic])
    assert result["status"] == "separated_conditions"
    assert result["by_variant"] == {}
    assert len(result["by_condition"]) == 2
    with pytest.raises(ValueError, match="one provenance condition"):
        build_ablation_table(result)


def test_separate_runs_cannot_be_silently_merged() -> None:
    first, second = _record("q1"), _record("q2")
    first["run_id"], second["run_id"] = "first", "second"
    with pytest.raises(ValueError, match="one run at a time"):
        aggregate_evaluations([first, second])


def test_paired_full_objects_improved_tied_worsened_and_missing_queries() -> None:
    records = []
    for query, full_order, objects_order in [
        ("improved", ("A", "B", "C"), ("B", "A", "C")),
        ("tied", ("A", "B", "C"), ("A", "B", "C")),
        ("worsened", ("B", "A", "C"), ("A", "B", "C")),
    ]:
        records.extend([
            _record(query, order=full_order),
            _record(query, order=objects_order, variant_id="objects_only"),
        ])
    paired = aggregate_evaluations(records)["paired"]
    assert paired["status"] == "completed"
    assert paired["query_count"] == 3
    assert paired["improved"] == paired["tied"] == paired["worsened"] == 1
    records.pop()
    result = aggregate_evaluations(records)
    assert result["status"] == "incomplete"
    assert result["by_variant"]["objects_only"]["recall_at_1"] is None
    assert result["paired"]["missing_query_ids"] == ["worsened"]


def test_timing_repeats_do_not_increase_query_count_and_means_are_query_weighted() -> None:
    records = [_record("q1"), _record("q2")]
    records[0]["timing_stats"] = {"mean_ms": 10, "repeat_count": 30}
    records[1]["timing_stats"] = {"mean_ms": 30, "repeat_count": 5}
    summary = aggregate_evaluations(records)["by_variant"]["full"]
    assert summary["query_count"] == 2
    assert summary["mean_time_ms"] == 20
    assert summary["timing_query_count"] == 2


def test_table_five_rows_percent_units_and_empty_not_run() -> None:
    empty = build_ablation_table(aggregate_evaluations([]))
    assert empty["status"] == "not_run"
    assert len(empty["rows"]) == 5
    assert all(row["cells"][1] == "not_run" for row in empty["rows"])
    records = []
    for variant in VARIANT_IDS:
        for query, target in [("q1", "A"), ("q2", "A"), ("q3", "C")]:
            record = _record(query, target, variant_id=variant)
            record["timing_stats"] = {"mean_ms": 12}
            records.append(record)
    table = build_ablation_table(aggregate_evaluations(records))
    assert table["status"] == "completed"
    parsed = list(csv.reader(StringIO(table["csv"])))
    assert len(parsed) == 6
    assert parsed[0][1] == "Recall@1 (%) ↑"
    assert parsed[1][1:6] == ["66.67", "100.00", "0.778", "1.667", "12.000"]
    assert "N_plus_1_on_retrieval_abstention" in table["markdown"]
    hit_table = build_ablation_table(aggregate_evaluations(records), metric_label="Hit")
    assert "Hit@1 (%)" in hit_table["csv"]
    assert hit_table["rows"] == table["rows"]
    with pytest.raises(ValueError, match="Recall or Hit"):
        build_ablation_table({}, metric_label="Accuracy")


def test_table_without_measured_timing_is_not_complete() -> None:
    records = [_record("q", variant_id=variant) for variant in VARIANT_IDS]
    table = build_ablation_table(aggregate_evaluations(records))
    assert table["status"] == "incomplete"
    assert table["rows"][0]["cells"][1] == "100.00"
    assert table["rows"][0]["cells"][5] == "not_measured"
