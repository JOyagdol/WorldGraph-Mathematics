"""Freeze an explicit semantic snapshot from a read-only full JSON graph export.

The exporter is streamed twice. Only node IDs and retained semantic records
remain in memory, not geometry payloads. Original semantic records and parallel
edges are preserved; unsupported node types are counted before exclusion.
An exact-byte source hash and file identity are checked before and after the
scan. Publication uses new files only, with the manifest written last.

Numbers use ijson's ``use_float=True`` policy: JSON non-integral numbers become
finite binary64 values, with ordinary rounding/underflow semantics. Numeric
overflow or values unsupported by the selected parser backend are rejected,
including in dropped geometry records. Original numeric spellings are covered
by the source byte hash, not reproduced in the canonical snapshot encoding.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from citygml_sg.retrieval.reference_graph import (
    GEOMETRY_TYPES,
    PREFETCH_TYPES,
    _node_type,
    _properties,
)
from citygml_sg.retrieval.room_scope import (
    canonical_node_id,
    canonical_node_type,
    edge_endpoints,
)


FREEZE_VERSION = "semantic_snapshot_freeze_v1"
FILTER_POLICY = "semantic_alphabet_induced_from_full_export_v1"


def _stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino


def _source_identity(path: Path) -> tuple[tuple[int, int, int, int, int], str]:
    before = _stat_identity(path)
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if _stat_identity(path) != before:
        raise ValueError("Source graph changed while its exact-byte hash was read")
    return before, digest.hexdigest()


def _stream_items(ijson: Any, path: Path, key: str) -> Iterator[dict[str, Any]]:
    """Chain checked parse events into ijson's per-record object builder."""
    with path.open("rb") as handle:
        def checked_events() -> Iterator[tuple[str, str, Any]]:
            first = True
            seen_keys: set[str] = set()
            arrays: set[str] = set()
            for prefix, event, value in ijson.parse(handle, use_float=True):
                if first:
                    first = False
                    if prefix != "" or event != "start_map":
                        raise ValueError("Graph export must have a top-level JSON object")
                if prefix == "" and event == "map_key":
                    if value in seen_keys:
                        raise ValueError(f"Duplicate top-level graph field: {value}")
                    seen_keys.add(value)
                if prefix in {"nodes", "edges"}:
                    if event == "start_array":
                        arrays.add(prefix)
                    elif event != "end_array":
                        raise ValueError(f"Graph export {prefix} must be an array")
                yield prefix, event, value
            if arrays != {"nodes", "edges"}:
                raise ValueError("Graph export requires nodes and edges arrays")
        try:
            yield from ijson.items(checked_events(), key + ".item")
        except (ValueError, TypeError, OverflowError, ijson.JSONError) as exc:
            raise ValueError(f"Invalid graph export while reading {key}: {exc}") from exc


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_new(path: Path, data: bytes) -> None:
    """Publish a fully written file using a hard link that cannot overwrite."""
    temporary = path.with_name("." + path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_semantic_snapshot(source: Path, output: Path) -> dict[str, Any]:
    """Write ``snapshot.json`` and ``freeze_manifest.json`` to a new/empty folder.

    The return value is the manifest. The source is never written. Existing
    output content is rejected. All original node IDs must be unique, every
    edge must have existing endpoints, and supplied retained edge IDs must be
    unique. Anonymous parallel edges remain distinct. Unsupported node types
    are excluded with counters; their incident edges are excluded as well.

    ``ijson`` is optional and imported here so ordinary retrieval does not
    require the streaming dependency. Successful freezes require both output
    files; a source that changes during scanning is rejected before publication.
    """
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise ValueError("Source graph export must be an existing file")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Freeze output must be a new or empty directory")
    try:
        ijson = importlib.import_module("ijson")
    except ImportError as exc:
        raise RuntimeError("Snapshot freezing requires optional dependency 'ijson'; "
                           "install it in the dedicated project environment") from exc

    source_before, source_sha256 = _source_identity(source)
    all_ids: set[str] = set()
    retained_ids: set[str] = set()
    room_ids: list[str] = []
    retained_nodes: list[dict[str, Any]] = []
    node_types: Counter[str] = Counter()
    retained_types: Counter[str] = Counter()
    canonical_types: Counter[str] = Counter()
    dropped_types: Counter[str] = Counter()
    for raw in _stream_items(ijson, source, "nodes"):
        node_id = canonical_node_id(raw)
        if node_id in all_ids:
            raise ValueError(f"Duplicate source graph node ID: {node_id}")
        all_ids.add(node_id)
        labels = raw.get("labels", [])
        if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
            raise ValueError("Source graph node labels must be a list of strings")
        node_type = _node_type(raw)
        node_types[node_type] += 1
        if node_type not in PREFETCH_TYPES:
            dropped_types[node_type] += 1
            continue
        canonical_type = canonical_node_type(raw)
        retained_ids.add(node_id)
        retained_nodes.append(raw)
        retained_types[node_type] += 1
        canonical_types[canonical_type] += 1
        if canonical_type == "Room":
            room_ids.append(node_id)

    retained_edges: list[dict[str, Any]] = []
    edge_ids: set[str] = set()
    relation_counts: Counter[str] = Counter()
    retained_relations: Counter[str] = Counter()
    dropped_relations: Counter[str] = Counter()
    for raw in _stream_items(ijson, source, "edges"):
        src, dst = edge_endpoints(raw)
        if src not in all_ids or dst not in all_ids:
            raise ValueError("Source graph edge has an unknown endpoint")
        relation = raw.get("relation") or raw.get("type")
        if not isinstance(relation, str) or not relation.strip():
            raise ValueError("Source graph edge requires a nonempty relation type")
        properties = _properties(raw)
        relation_counts[relation] += 1
        if src not in retained_ids or dst not in retained_ids:
            dropped_relations[relation] += 1
            continue
        edge_id = next((value for value in (raw.get("id"), raw.get("edge_id"),
                                           properties.get("id")) if value is not None), None)
        if edge_id is not None:
            if not isinstance(edge_id, str) or not edge_id.strip():
                raise ValueError("Retained edge ID must be a nonempty string when supplied")
            if edge_id in edge_ids:
                raise ValueError(f"Duplicate retained edge ID: {edge_id}")
            edge_ids.add(edge_id)
        retained_edges.append(raw)
        retained_relations[relation] += 1

    source_after, final_source_sha256 = _source_identity(source)
    if source_after != source_before or final_source_sha256 != source_sha256:
        raise ValueError("Source graph changed during semantic snapshot extraction")
    counts = {
        "full_node_count": len(all_ids),
        "full_edge_count": sum(relation_counts.values()),
        "retained_node_count": len(retained_nodes),
        "retained_edge_count": len(retained_edges),
        "dropped_node_count": sum(dropped_types.values()),
        "dropped_edge_count": sum(dropped_relations.values()),
        "geometry_node_count": sum(count for kind, count in node_types.items()
                                   if kind in GEOMETRY_TYPES),
        "node_type_counts": dict(sorted(node_types.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "retained_node_type_counts": dict(sorted(retained_types.items())),
        "retained_canonical_type_counts": dict(sorted(canonical_types.items())),
        "retained_relation_counts": dict(sorted(retained_relations.items())),
        "dropped_node_type_counts": dict(sorted(dropped_types.items())),
        "dropped_relation_counts": dict(sorted(dropped_relations.items())),
    }
    snapshot = {
        "snapshot_scope": "semantic_with_container_context",
        "filter_policy": FILTER_POLICY,
        "nodes": retained_nodes,
        "edges": retained_edges,
        "inventory": counts,
    }
    content_bytes = _json_bytes(snapshot)
    file_bytes = content_bytes + b"\n"
    report = {
        "schema_version": 1,
        "freeze_version": FREEZE_VERSION,
        "filter_policy": FILTER_POLICY,
        "parser": {"library": "ijson", "version": getattr(ijson, "__version__", "unknown"),
                   "backend": getattr(ijson, "backend", "unknown"),
                   "numeric_policy": "use_float_true_binary64_overflow_rejected"},
        "source": {"basename": source.name, "size_bytes": source_before[0],
                   "mtime_ns": source_before[1], "sha256": source_sha256,
                   "unchanged_during_extraction": True},
        "snapshot": {"filename": "snapshot.json", "size_bytes": len(file_bytes),
                     "content_sha256": sha256(content_bytes).hexdigest(),
                     "file_sha256": sha256(file_bytes).hexdigest()},
        "inventory": counts,
        "room_ids": sorted(room_ids),
        "room_count": len(room_ids),
        "validation": {"all_node_ids_unique": True, "all_edge_endpoints_present": True,
                       "retained_supplied_edge_ids_unique": True,
                       "original_semantic_records_preserved": True},
        "readiness": "reference_snapshot_only_no_gt_or_evaluation_claim",
    }
    # Check again after the scan in case another process populated the folder.
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Freeze output became nonempty during extraction")
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "snapshot.json"
    manifest_path = output / "freeze_manifest.json"
    published_snapshot = False
    try:
        _write_new(snapshot_path, file_bytes)
        published_snapshot = True
        _write_new(manifest_path, _json_bytes(report) + b"\n")
    except Exception:
        if published_snapshot:
            snapshot_path.unlink(missing_ok=True)
        raise
    return report
