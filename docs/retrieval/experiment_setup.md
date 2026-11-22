# Experiment setup and bounded diagnostics

Use a dedicated Python 3.11 virtual environment in this checkout. The pinned
`requirements-experiments.txt` covers snapshot preparation, retrieval and tests.
Install with `.venv/Scripts/python -m pip install -r requirements-experiments.txt`
on Windows. Commands in `scripts/retrieval/` insert this checkout's `src` explicitly;
they do not depend on an editable package from another project. Run pytest with
`PYTHONPATH=src` in the same environment.

## Freeze inputs before retrieval

1. Run `freeze_semantic_snapshot.py --source FULL_GRAPH.json --output-dir NEW_REFERENCE`.
   This reads an existing JSON export twice with `ijson`, excludes geometry and
   unsupported nodes, and preserves induced semantic edges. Exact source hashes
   and file identity must agree before and after extraction. The inventory is
   recounted from records rather than copied from a summary. Duplicate IDs,
   dangling endpoints and malformed arrays fail validation. Retained semantic
   records remain in memory along with the full source ID set. Numeric parsing
   uses binary64 floats; the manifest records its conversion policy. This is
   a semantic extraction, not a new graph construction or database import.
2. Freeze observation byte hashes before consulting evaluation GT. Store a JSON
   object with `observations: [{query_id, sha256}, ...]`. Keep image-only annotation
   review separate from canonical Room identity and spatial registration review.
3. Run `prepare_diagnostic_bundle.py --manifest INPUT_MANIFEST.json
   --graph-json NEW_REFERENCE/snapshot.json --observation-freeze FREEZE.json
   --output-dir NEW_BUNDLE` as one shell command. Every case must match the frozen
   inventory. The bundle copies exact observation and snapshot bytes, records
   hashes and writes manifests with paths relative to the bundle. It does not
   promote annotation status or copy evaluator targets from the source manifest.

The four pilot strata are scoring objects with relations, scoring objects without
relations, context only, and empty. One lexicographically first query per present
stratum is chosen before outcomes. This is a functional coverage sample under the
existing proposed joint method's object definition, not a random accuracy sample.

## Run the diagnostic

Run the following as one command, substituting local paths:

```text
python scripts/retrieval/run_retrieval_diagnostics.py
  --manifest NEW_BUNDLE/pilot_manifest.json
  --graph-json NEW_BUNDLE/snapshot.json
  --config configs/joint_evidence_v1.yaml
  --output-dir NEW_RUN --max-queries 4
  --budget-seconds 180 --per-query-timeout-seconds 10
```

The runner accepts no GT targets and produces no accuracy metrics. It verifies
snapshot and observation hashes, uses the full Room candidate pool, preserves
all rankings and solver/failure states, and records one attempt per query/variant.
It records but does not execute warmup/repeat settings from the benchmark config.
Each owned subprocess has a hard timeout including startup and IPC. The overall
budget stops new work and bounds workers; synchronous preparation and final
writes cannot be preempted. Output directories must be new or empty.

`run_manifest.json` is checkpointed throughout execution. Read its status and
failure list, not merely the process exit code. Successful execution certifies
neither annotation quality nor localization correctness. Worker inference time
excludes startup/IPC and graph loading; it is a single diagnostic measurement,
not the repeated warm-latency benchmark. Preserve the raw run if a failure leads
to a repair, and use a new output directory for the next run.

## Gate the actual evaluation

After a new Unity capture session has stopped, run:

```text
python scripts/retrieval/audit_unity_capture.py --session-dir NEW_SESSION
  --catalog CANONICAL_CATALOG.json --output private/new_capture_audit.json
```

This evaluation-only audit requires schema 4, exact image/label pairing, explicit
canonical IDs, consistent snapshot/pool/catalog hashes and recorded review stamps.
It hashes images as opaque bytes, copies no poses, and never repairs unknown GT.
Exit codes are 0 for label-contract-valid, 1 for an invalid session with a saved
report, and 2 for setup/input failure. Choose a new Git-ignored output outside the
raw session. A valid contract still requires independent spatial and annotation
review; recorded stamps cannot prove geometry or absence of overlap.

Formal accuracy evaluation additionally needs exact canonical GT, reviewed
image observations, a verified graph/pool, and a frozen method/protocol. Track
capture sessions, Room coverage and split units explicitly; many frames from one
capture are not independent test scenes. Missing/unknown GT cannot be replaced
by filenames, predicted ranks, nearest-room guesses or an invented canonical ID.

`joint_evidence_v1_proposed` remains an implementation candidate. Its diagnostic
outputs do not establish adoption of manuscript definitions or replace the
legacy baseline. Keep method IDs, snapshots, configs and result directories
distinct. Local source paths, catalogs, captures, manuscript notes and results
belong in ignored directories.
