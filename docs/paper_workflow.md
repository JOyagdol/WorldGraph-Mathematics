# Paper implementation and evidence workflow

The paper workspace inherits a functional CityGML 2.0 graph/retrieval framework. Its development target is a consistent manuscript backed by executed evidence, not automatic adoption of every inherited design.

| Track | Role | Evidence boundary |
|---|---|---|
| Legacy heuristic | Historical baseline and reproduction | Old reports retain their original score, query and timing definitions. |
| Manuscript requirements | Specifications selected for the paper | Record adopted definitions before implementing missing behavior; no method ID is finalized by this document. |
| `joint_evidence_v1_proposed` | Candidate joint correspondence and scoring | Existing unit tests validate contracts; dataset performance remains a separate experiment. |
| Annotation preparation | Observation construction and human review | Structural validity does not establish visual truth or Oracle certification. |

## Required evidence chain

For each claim, keep manuscript location, method/version, commit, resolved configuration, observation provenance and review status, canonical GT, immutable graph snapshot, candidate pool/hash, raw run, metric definition and interpretation. Archive failed/abstained/incomplete queries; do not silently drop them from the denominator. Preserve complete rankings and tie information.

The five implemented variants are full, objects-only, relations-only, uniform-category and Boolean metadata. Compare them on the same reviewed observations and complete candidate pool. Relation-only still needs node compatibility. Weight and geometric-threshold sensitivity are separate interventions and may require new graph snapshots.

Distinguish query-average from Room-macro aggregation, and closed-set single-GT ranking from cross-validation. A protocol name must describe the actual data/pool split. Keep the true Room in a closed-set candidate pool. Consecutive capture frames may be strongly dependent; repeated frames do not establish room or building generalization.

Record cold setup separately from warm online retrieval. Include normalization, candidate loading, matching/ranking and output boundaries explicitly, plus warmup/repeat/seed and environment. Tests, preparation time and historical stage profiles are not interchangeable with measured retrieval latency.

## Local author workflow

Private handoffs identify the manuscript original, all highlighted locations, legacy notes, annotation review page and original asset locations. Keep originals unchanged; use versioned edited notes and a versioned Word revision. Maintain an audit status per issue: open, implementation-ready, awaiting-run, awaiting-author, verified, or deferred with rationale.

Before publishing numerical summaries, reconcile Abstract, Methods, Results, tables/figures, Discussion and Conclusion. A corrected table without corresponding prose updates is incomplete. Missing author facts require author input; missing measurements require execution, not estimates presented as results.
