"""Read-only label-contract audit for completed Unity schema-4 capture sessions.

This evaluation-only report checks identities, byte hashes and recorded review
stamps. A catalog is an allowlist, not evidence of spatial correctness. Neither
region geometry nor image annotations are certified, and poses are never copied
to the report. Images are hashed as opaque bytes without decoding them. Capture
must be stopped before auditing; concurrent input changes invalidate the audit.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from citygml_sg.retrieval.evaluation.oracle_metrics import candidate_ids_hash


ROOM_ID_SOURCE = (
    "Exactly one explicitly reviewed RoomRegion; exact canonical catalog membership; "
    "capture fails otherwise"
)
_SHA = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"[0-9a-f]{32}")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,7})?(?:Z|\+00:00)")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _identifier(value: Any) -> bool:
    return (_text(value) and value == value.strip()
            and value.casefold() not in {"unknown", "overlap", "ambiguous"})


def _matches(pattern: re.Pattern, value: Any) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _utc(value: Any) -> datetime | None:
    if not _matches(_UTC, value):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _json(payload: bytes) -> dict[str, Any]:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("Non-finite JSON number")

    value = json.loads(payload.decode("utf-8-sig"), object_pairs_hook=unique_pairs,
                       parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _inventory(folder: Path) -> dict[str, tuple[int, int, bool]]:
    return {path.name: (path.stat().st_size, path.stat().st_mtime_ns,
                        path.is_file() and not path.is_symlink())
            for path in folder.iterdir()}


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _catalog(path: Path) -> tuple[dict[str, Any], bytes, str]:
    payload = path.read_bytes()
    catalog = _json(payload)
    if type(catalog.get("schemaVersion")) is not int or catalog["schemaVersion"] != 1:
        raise ValueError("Canonical catalog requires schemaVersion=1")
    for key in ("referenceSnapshotSha256", "candidatePoolSha256"):
        if not _matches(_SHA, catalog.get(key)):
            raise ValueError(f"Catalog {key} must be lowercase SHA-256")
    ids = catalog.get("canonicalRoomIds")
    if (not isinstance(ids, list) or not ids or not all(_identifier(v) for v in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Catalog requires unique explicit canonical Room IDs")
    if candidate_ids_hash(ids) != catalog["candidatePoolSha256"]:
        raise ValueError("Catalog candidatePoolSha256 does not hash its canonical Room ID set")
    # TextAsset.text is decoded UTF-8, without BOM. Preserve CRLF; read_text()
    # would silently normalize line endings and disagree with the Unity stamp.
    text_hash = sha256(payload.decode("utf-8-sig").encode("utf-8")).hexdigest()
    return catalog, payload, text_hash


def _record_errors(record: dict, label_name: str, catalog: dict, text_hash: str) -> list[str]:
    errors = []
    if type(record.get("schemaVersion")) is not int or record["schemaVersion"] != 4:
        errors.append("schemaVersion must be 4; older captures require recapture, not label inference")
    image_id = record.get("imageId")
    if not _matches(_IMAGE_ID, image_id):
        errors.append("imageId must be a lowercase 32-hex GUID")
    else:
        if label_name != f"{image_id}.json":
            errors.append("ground_truth filename must exactly match imageId")
        if record.get("fileName") != f"images/{image_id}.png":
            errors.append("fileName must exactly equal images/<imageId>.png")
    if record.get("roomId") not in catalog["canonicalRoomIds"]:
        errors.append("roomId is not an exact canonical catalog member; no label is inferred")
    if not _identifier(record.get("roomRegionId")):
        errors.append("roomRegionId must be an explicit nonempty identifier")
    if record.get("roomIdSource") != ROOM_ID_SOURCE:
        errors.append("roomIdSource must state the schema-4 single reviewed region contract")
    for key in ("referenceSnapshotSha256", "candidatePoolSha256"):
        if record.get(key) != catalog[key]:
            errors.append(f"{key} does not match the expected catalog")
    if not _matches(_SHA, record.get("roomRegionGeometrySha256")):
        errors.append("roomRegionGeometrySha256 must be lowercase SHA-256")
    if record.get("roomRegionCatalogTextSha256") != text_hash:
        errors.append("roomRegionCatalogTextSha256 does not match the supplied catalog text")
    for key in ("roomRegionReviewedBy", "roomRegionReviewEvidence"):
        if not _text(record.get(key)):
            errors.append(f"{key} must be nonempty")
    times = {}
    for key in ("capturedAtUtc", "roomRegionReviewedAtUtc"):
        times[key] = _utc(record.get(key))
        if times[key] is None:
            errors.append(f"{key} must be a valid ISO-8601 UTC timestamp")
    if all(times.values()) and times["roomRegionReviewedAtUtc"] > times["capturedAtUtc"]:
        errors.append("Region review timestamp is later than capture timestamp")
    return errors


def audit_capture_session(session_dir: Path, catalog_path: Path) -> dict[str, Any]:
    """Audit all raw image/label pairs; invalid pairs remain visible in the report.

    Invalid catalog or session layout raises ValueError/OSError. Per-capture
    failures are returned as label_contract_invalid. Only contract-valid sessions
    receive label_contract_valid, always with independent reviews still required.
    Unknown labels, overlap claims and older schemas are never repaired or guessed.
    """
    session_dir, catalog_path = Path(session_dir), Path(catalog_path)
    catalog, catalog_bytes, text_hash = _catalog(catalog_path)
    folders = {name: session_dir / name for name in ("images", "ground_truth")}
    if any(not p.is_dir() or p.is_symlink() for p in folders.values()):
        raise ValueError("Session requires real images/ and ground_truth/ directories")
    before = {name: _inventory(path) for name, path in folders.items()}
    errors, records, ids = [], [], defaultdict(list)
    expected_images = set()
    for label_name, (_, _, regular) in sorted(before["ground_truth"].items()):
        if not regular or not label_name.endswith(".json"):
            errors.append(f"Unexpected ground_truth entry: {label_name}")
            continue
        row = {"label_file": f"ground_truth/{label_name}", "errors": []}
        records.append(row)
        try:
            payload = (folders["ground_truth"] / label_name).read_bytes()
            row["label_sha256"] = sha256(payload).hexdigest()
            record = _json(payload)
        except (ValueError, UnicodeError):
            row["errors"].append("Label is not a valid UTF-8 JSON object with unique keys")
            continue
        row["errors"].extend(_record_errors(record, label_name, catalog, text_hash))
        image_id = record.get("imageId")
        if _matches(_IMAGE_ID, image_id):
            row["image_id"] = image_id
            ids[image_id].append(row)
            image_name = f"{image_id}.png"
            row["image_file"] = f"images/{image_name}"
            expected_images.add(image_name)
            entry = before["images"].get(image_name)
            if entry is None or not entry[2]:
                row["errors"].append("Exact image file is absent or not a regular file")
            elif entry[0] == 0:
                row["errors"].append("Image file is empty")
            else:
                row["image_sha256"] = _file_hash(folders["images"] / image_name)
        if record.get("roomId") in catalog["canonicalRoomIds"]:
            row["room_id"] = record["roomId"]
    for duplicates in ids.values():
        if len(duplicates) > 1:
            for row in duplicates:
                row["errors"].append("Duplicate imageId")
    for image_name, (_, _, regular) in sorted(before["images"].items()):
        if not regular or image_name not in expected_images:
            errors.append(f"Unpaired or unexpected images entry: {image_name}")
    if not records:
        errors.append("Session has no capture label records")
    if (before != {name: _inventory(path) for name, path in folders.items()}
            or catalog_path.read_bytes() != catalog_bytes):
        errors.append("Inputs changed during audit; stop capture and rerun")
    for row in records:
        row["label_contract_valid"] = not row["errors"]
    valid = not errors and all(row["label_contract_valid"] for row in records)
    return {
        "audit_schema_version": 1, "capture_schema_version": 4,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_scope": "evaluation_only; never provide this report to image inference",
        "status": "label_contract_valid" if valid else "label_contract_invalid",
        "independent_spatial_review": "required; stored stamps do not certify geometry or absence of overlap",
        "independent_annotation_review": "required; no image content was interpreted",
        "hash_scope": "image and label exact bytes; catalog UTF-8 text without BOM, preserving line endings",
        "catalog_file_sha256": sha256(catalog_bytes).hexdigest(),
        "catalog_text_sha256": text_hash,
        "reference_snapshot_sha256": catalog["referenceSnapshotSha256"],
        "candidate_pool_sha256": catalog["candidatePoolSha256"],
        "canonical_room_ids": catalog["canonicalRoomIds"],
        "capture_count": len(records),
        "valid_contract_count": sum(row["label_contract_valid"] for row in records),
        "room_counts": dict(sorted(Counter(row["room_id"] for row in records
                                          if row["label_contract_valid"]).items())),
        "errors": errors, "records": records,
    }


def write_capture_audit(report: dict[str, Any], output_path: Path) -> None:
    """Write one new evaluation-only report without overwriting an existing file."""
    output_path = Path(output_path)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
