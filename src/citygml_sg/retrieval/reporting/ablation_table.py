"""Render the five proposed ablations only from measured metric records."""

from __future__ import annotations

import csv
from io import StringIO
from math import isfinite
from typing import Any


VARIANT_LABELS = (
    ("full", "Full score"),
    ("objects_only", "Objects only"),
    ("relations_only", "Relations only"),
    ("uniform_category", "Uniform category weights"),
    ("boolean_metadata", "Boolean metadata"),
)


def _display(value: Any, status: str, *, percent: bool = False) -> str:
    if status != "completed":
        return status
    if value is None:
        return "not_measured"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError("table metrics must be finite numbers or null")
    if percent and not 0 <= value <= 1:
        raise ValueError("Recall/Hit JSON values must be fractions in [0,1]")
    return f"{value * 100:.2f}" if percent else f"{value:.3f}"


def build_ablation_table(
    metrics_by_variant: dict[str, Any], metric_label: str = "Recall",
) -> dict[str, Any]:
    """Return CSV, Markdown and five structured rows with explicit status and units.

    Accept an ``aggregate_evaluations`` result or its ``by_variant`` mapping.
    Select one ``by_condition`` result explicitly when conditions differ.
    """
    if metric_label not in {"Recall", "Hit"}:
        raise ValueError("metric_label must be Recall or Hit")
    if metrics_by_variant.get("status") == "separated_conditions":
        raise ValueError("select one provenance condition before exporting a primary table")
    variants = metrics_by_variant.get("by_variant", metrics_by_variant)
    headers = [
        "Variant", f"{metric_label}@1 (%) ↑", f"{metric_label}@3 (%) ↑", "MRR ↑", "MTR ↓",
        "Mean Time (ms) ↓", "Status",
    ]
    prefix = "recall" if metric_label == "Recall" else "hit"
    fields = (f"{prefix}_at_1", f"{prefix}_at_3", "mrr", "mtr", "mean_time_ms")
    rows = []
    for variant_id, label in VARIANT_LABELS:
        metrics = variants.get(variant_id, {})
        status = metrics.get("status", "not_run")
        if status not in {"completed", "not_run", "incomplete"}:
            raise ValueError(f"unsupported aggregate status for {variant_id}: {status}")
        if status == "completed" and any(metrics.get(key) is None for key in fields[:-1]):
            status = "incomplete"
        values = [_display(metrics.get(key), status, percent=index < 2)
                  for index, key in enumerate(fields)]
        rows.append({"variant_id": variant_id, "status": status, "cells": [label, *values, status]})
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(row["cells"] for row in rows)
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "|---|---:|---:|---:|---:|---:|---|\n"
    markdown += "".join("| " + " | ".join(row["cells"]) + " |\n" for row in rows)
    markdown += (
        "\nRecall/Hit are percentages; MRR is a fraction and MTR is a rank. "
        "Primary aggregation: equal query weights. "
        "MTR policy: N_plus_1_on_retrieval_abstention. "
        "Mean Time is warm online retrieval, including matching, scoring and evidence; "
        "timing repeats are not extra queries. not_run/incomplete/not_measured are not zeros.\n"
    )
    statuses = {row["status"] for row in rows}
    all_measured = all("not_measured" not in row["cells"] for row in rows)
    if statuses == {"completed"} and all_measured:
        status = "completed"
    else:
        status = "not_run" if statuses == {"not_run"} else "incomplete"
    return {"csv": stream.getvalue(), "markdown": markdown, "rows": rows, "status": status}
