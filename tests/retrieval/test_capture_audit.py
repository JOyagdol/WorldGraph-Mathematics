from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from citygml_sg.retrieval import capture_audit
from citygml_sg.retrieval.capture_audit import (
    ROOM_ID_SOURCE, audit_capture_session, write_capture_audit,
)
from citygml_sg.retrieval.evaluation.oracle_metrics import candidate_ids_hash


ID = "a" * 32


@pytest.fixture
def capture(tmp_path):
    catalog = {"schemaVersion": 1, "canonicalRoomIds": ["room-2", "room-1"],
               "referenceSnapshotSha256": "1" * 64,
               "candidatePoolSha256": candidate_ids_hash(["room-1", "room-2"])}
    catalog_path = tmp_path / "catalog.json"
    # CRLF is intentional: Unity hashes the imported text, preserving newlines.
    catalog_path.write_bytes(json.dumps(catalog, indent=2).replace("\n", "\r\n").encode())
    session = tmp_path / "session"
    (session / "images").mkdir(parents=True)
    (session / "ground_truth").mkdir()
    image_path = session / "images" / f"{ID}.png"
    image_path.write_bytes(b"opaque synthetic image bytes: never decoded")
    record = {
        "schemaVersion": 4, "imageId": ID, "fileName": f"images/{ID}.png",
        "roomId": "room-1", "roomRegionId": "region-1", "roomIdSource": ROOM_ID_SOURCE,
        "referenceSnapshotSha256": catalog["referenceSnapshotSha256"],
        "candidatePoolSha256": catalog["candidatePoolSha256"],
        "roomRegionCatalogTextSha256": sha256(catalog_path.read_bytes()).hexdigest(),
        "roomRegionGeometrySha256": "2" * 64, "roomRegionReviewedBy": "Synthetic reviewer",
        "roomRegionReviewEvidence": "Synthetic review fixture, not actual annotation",
        "roomRegionReviewedAtUtc": "2026-11-17T00:00:00.1234567Z",
        "capturedAtUtc": "2026-11-17T01:00:00.1234567Z",
        "cameraPositionUnity": {"x": 1234, "y": 5678, "z": 9876},
        "ecefMetres": [1, 2, 3], "buildingId": "unknown", "floorId": "unknown",
    }
    label_path = session / "ground_truth" / f"{ID}.json"
    label_path.write_text(json.dumps(record), encoding="utf-8")
    return session, catalog_path, record, label_path


def _save(record, path):
    path.write_text(json.dumps(record), encoding="utf-8")


def _errors(report):
    return "\n".join(report["errors"] + [e for r in report["records"] for e in r["errors"]])


def test_valid_contract_preserves_inputs_and_never_exports_poses_or_certifies_review(capture):
    session, catalog, _, _ = capture
    before = {p: p.read_bytes() for p in session.rglob("*") if p.is_file()}
    report = audit_capture_session(session, catalog)
    assert report["status"] == "label_contract_valid"
    assert report["capture_count"] == report["valid_contract_count"] == 1
    assert report["room_counts"] == {"room-1": 1}
    assert report["independent_spatial_review"].startswith("required")
    assert report["independent_annotation_review"].startswith("required")
    assert report["records"][0]["image_sha256"] == sha256(before[session / "images" / f"{ID}.png"]).hexdigest()
    assert report["records"][0]["label_sha256"] == sha256(before[session / "ground_truth" / f"{ID}.json"]).hexdigest()
    text = json.dumps(report)
    for forbidden in ("cameraPositionUnity", "ecefMetres", "oracle_verified", "Synthetic reviewer"):
        assert forbidden not in text
    assert before == {p: p.read_bytes() for p in session.rglob("*") if p.is_file()}


@pytest.mark.parametrize(("field", "value", "error"), [
    ("schemaVersion", 3, "schemaVersion must be 4"),
    ("schemaVersion", "4", "schemaVersion must be 4"),
    ("imageId", "../unsafe", "imageId must"),
    ("imageId", "A" * 32, "imageId must"),
    ("fileName", "images/other.png", "fileName must exactly"),
    ("fileName", "../outside.png", "fileName must exactly"),
    ("roomId", "unknown", "not an exact canonical"),
    ("roomId", "overlap", "not an exact canonical"),
    ("roomId", ["room-1", "room-2"], "not an exact canonical"),
    ("roomIdSource", "overlapping regions", "single reviewed region contract"),
    ("roomRegionId", " ", "roomRegionId must"),
    ("referenceSnapshotSha256", "3" * 64, "referenceSnapshotSha256 does not match"),
    ("candidatePoolSha256", "3" * 64, "candidatePoolSha256 does not match"),
    ("roomRegionCatalogTextSha256", "3" * 64, "catalog text"),
    ("roomRegionGeometrySha256", "A" * 64, "lowercase SHA-256"),
    ("roomRegionReviewedBy", " ", "must be nonempty"),
    ("roomRegionReviewEvidence", None, "must be nonempty"),
    ("roomRegionReviewedAtUtc", "2026-99-17T00:00:00Z", "valid ISO-8601 UTC"),
    ("roomRegionReviewedAtUtc", "2026-11-18T00:00:00Z", "later than capture"),
    ("capturedAtUtc", "2026-11-17T01:00:00", "valid ISO-8601 UTC"),
])
def test_malformed_contract_is_explicit_failure(capture, field, value, error):
    session, catalog, record, label = capture
    record[field] = value
    _save(record, label)
    report = audit_capture_session(session, catalog)
    assert report["status"] == "label_contract_invalid"
    assert error in _errors(report)


def test_missing_and_orphan_images_and_label_filename_mismatch_are_not_repaired(capture):
    session, catalog, record, label = capture
    (session / "images" / f"{ID}.png").unlink()
    (session / "images" / "orphan.png").write_bytes(b"synthetic")
    label.rename(label.with_name("wrong.json"))
    report = audit_capture_session(session, catalog)
    errors = _errors(report)
    assert report["status"] == "label_contract_invalid"
    assert "Exact image file is absent" in errors
    assert "Unpaired or unexpected images entry: orphan.png" in errors
    assert "ground_truth filename must exactly match" in errors
    assert record["fileName"] == f"images/{ID}.png"


def test_duplicate_image_ids_invalidate_every_affected_record(capture):
    session, catalog, record, label = capture
    _save(record, label.with_name("duplicate.json"))
    report = audit_capture_session(session, catalog)
    assert report["valid_contract_count"] == 0
    assert all("Duplicate imageId" in row["errors"] for row in report["records"])


@pytest.mark.parametrize("payload", [b"not-json", b"[]", b'{"schemaVersion":4,"schemaVersion":4}'])
def test_malformed_json_is_recorded_without_stopping_inventory(capture, payload):
    session, catalog, _, label = capture
    label.write_bytes(payload)
    report = audit_capture_session(session, catalog)
    assert report["status"] == "label_contract_invalid"
    assert "valid UTF-8 JSON object with unique keys" in _errors(report)


def test_inconsistent_catalog_pool_fails_before_auditing(capture):
    session, catalog, _, _ = capture
    raw = json.loads(catalog.read_bytes())
    raw["canonicalRoomIds"].append("room-3")
    _save(raw, catalog)
    with pytest.raises(ValueError, match="does not hash"):
        audit_capture_session(session, catalog)


def test_changed_input_during_hashing_invalidates_report(capture, monkeypatch):
    session, catalog, _, _ = capture
    original = capture_audit._file_hash

    def change(path):
        digest = original(path)
        path.write_bytes(path.read_bytes() + b"changed after hash")
        return digest

    monkeypatch.setattr(capture_audit, "_file_hash", change)
    report = audit_capture_session(session, catalog)
    assert "Inputs changed during audit" in _errors(report)
    assert report["status"] == "label_contract_invalid"


def test_report_writer_refuses_overwrite(capture, tmp_path):
    session, catalog, _, _ = capture
    output = tmp_path / "audit.local.json"
    report = audit_capture_session(session, catalog)
    write_capture_audit(report, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_capture_audit(report, output)
    assert output.read_bytes() == original


def test_cli_outputs_new_ignored_report_and_refuses_raw_session(capture, tmp_path, monkeypatch):
    session, catalog, _, _ = capture
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("audit_unity_cli", root / "scripts/retrieval/audit_unity_capture.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    # CLI output belongs to the checkout; raw fixtures may use any pytest temp.
    with TemporaryDirectory(dir=root / "data/output", prefix="capture_audit_test_") as output_dir:
        output = Path(output_dir) / "audit.local.json"
        argv = ["audit", "--session-dir", str(session), "--catalog", str(catalog), "--output", str(output)]
        monkeypatch.setattr("sys.argv", argv)
        assert cli.main() == 0
        assert json.loads(output.read_bytes())["status"] == "label_contract_valid"
        assert cli.main() == 2
        argv[-1] = str(session / "audit.local.json")
        assert cli.main() == 2
        assert not (session / "audit.local.json").exists()
