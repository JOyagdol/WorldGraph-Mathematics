# Offline reference input preparation

`reference_input_preparation_v1` prepares two independently testable input
components. It does not select a complete retrieval method or compute scores,
rankings, retrieval latency or metrics. Existing legacy and joint retrieval
paths retain their own behavior.

## Semantic hop scopes

`build_room_hop_subgraphs(snapshot, h=2)` extracts the induced subgraph around
every Room using undirected traversal of all supplied edges. The input must be
an explicit semantic snapshot containing only Building, BuildingPart, Room,
BoundarySurface, Opening, Furniture and Installation nodes or recognized
aliases. Unsupported nodes, duplicate identifiers and dangling endpoints fail
validation before extraction. Geometry is not silently removed.

The result preserves original edge direction, parallel records, self-loops and
stored metadata. It does not fill missing gamma, epsilon, provenance or edge
IDs. Shared parents or relation edges can bring other Rooms into a scope;
`membership.other_room_ids` reports them. A hop radius is not an ownership or
room-boundary filter. The existing `typed_room_paths_v1` adapter is separate.

## Reference-vocabulary anchors

`select_reference_anchors` keeps the unknown Room placeholder and selects
non-Room nodes whose normalized category occurs in the whole supplied snapshot
and whose weight is at least the threshold. Edges require both selected
endpoints and a weight at least the same threshold. The comparison is inclusive,
so zero-weight atoms remain when the threshold is zero. Exclusions retain their
reasons. No target Room or GT field participates in selection.

The preparation wrapper explicitly uses the existing observation and reference
normalizer versions and records them in its configuration. Furniture categories
can come from the reference adapter's existing semantic-name rules; they are not
derived from instance IDs. Containers contribute only explicit categories.
Cabinet and storage remain distinct. Generic opening does not silently expand
to door/window, and contradictory specific-opening categories are rejected.
These are declared input rules, not a claim that a final publication taxonomy
has been resolved. Reverse/inverse directed observation atoms remain separate;
exact duplicate directed atoms are rejected. Existing joint normalization is
unchanged.

## Offline CLI

```powershell
python scripts/retrieval/prepare_reference_inputs.py `
  --manifest data/input/local_manifest.json `
  --graph-json data/input/local_semantic_snapshot.json `
  --anchor-threshold 0.5 --hop-radius 2 `
  --output-dir data/output/reference_input_preparation/run_001
```

The threshold above illustrates syntax; it is not a tuned or recommended
experimental setting. The CLI has no database connection or retrieval option.
The manifest requires a nonempty `cases` list with unique `query_id` and
`ovg_path` fields. Observation paths are relative to the manifest directory
unless absolute. A canonical GT field is not required at this stage. Observed
node and relation records must satisfy the observation-only contract.

The new or empty output directory receives `preparation.json` only after every
input validates. Existing artifacts are never overwritten. It contains:

- `status: not_run`, `preparation_status: completed`, `method_version: null`;
- Room inventory, hop scopes, anchors and exclusion diagnostics;
- file/content hashes, component/config identities and code provenance;
- annotation declarations by source, precedence and conflicts, with explicit
  `annotation_verification_performed: false` and no Oracle certification;
- unresolved scoring domain, IDF population, matching, proof and protocol work.

Content hashes use UTF-8 canonical JSON with sorted object keys, no non-finite
numbers and compact separators. Record order is retained. File hashes identify
exact bytes separately. Hash agreement does not certify annotation, provenance
truth or experimental independence. Normalizer `scoring` flags are inherited
diagnostics only; this component does not adopt their domain for a score.

## Verification

Synthetic tests cover induced cross edges, shared-parent Room expansion, zero
and boundary thresholds, whole-snapshot vocabulary, GT-independent selection,
directed relation preservation, invalid input rejection, declaration conflicts
and output preservation. They are software checks, not localization results.

```powershell
python -m pytest tests/retrieval/test_room_scope.py `
  tests/retrieval/test_reference_anchors.py `
  tests/retrieval/test_reference_preparation.py -q
```
