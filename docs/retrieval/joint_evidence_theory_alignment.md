# Theory alignment for the proposed joint evidence method

This note maps the implementation candidate to manuscript claims that need review. It records proof obligations, not completed proofs or empirical results. Read the [implementation contract](joint_evidence_v1.md) for executable scope and failure policies.

## Version policy

This is a review of the proposed joint method, not an instruction to rewrite the manuscript to match it. Requirements preserved from the manuscript must be implemented and tested as a separately identified method when they differ. Record an explicit method decision before changing equations, proofs or reported results. Legacy results do not validate the proposed method.

## Manuscript correspondence

| Manuscript location | Required alignment with the current implementation candidate |
|---|---|
| Definitions 1–2 | Separate the complete stored World Graph, geometry inventory and Room-scoped semantic retrieval graph. Define shared/fallback membership explicitly. |
| Equations (10)–(17) | Describe actual relation generation, degenerate geometry handling and metadata provenance. The retrieval adapter consumes stored relations; it does not prove geometric correctness. |
| Definitions 4–5 | Define observable aliases, repeated instances, confidence/visibility, typed contexts and anchor exclusion. GT-correspondence and complete-room diagnostics are stronger inputs than clean-label Oracle observations. |
| Equation (21) | Use `D_A = Furniture ∪ Installation ∪ Opening` anchors for the object contribution. BoundarySurface and Room are relation contexts, not separately weighted object evidence. |
| Equations (23)–(24) | Apply the same domain to candidate category frequency and the query-wide denominator. Count matched observed instances and edges once; use one denominator per query/variant across Rooms. Update examples using surface object weights. |
| Definitions 6–8 and Algorithm 1 | For the proposed joint version, specify an explicit partial injective joint objective; evaluate any manuscript-preserving alternative separately. State compatibility, directionality, best-edge selection, budget and termination conditions. |
| Propositions 2/4/5 and Remarks 1–2 | Reassess assumptions on exactness, deletion costs, normalization, incidence and ambiguity. Implementation does not automatically repair earlier proofs. |
| Proposition 3 | Include candidate count, reference extraction/cache policy and actual search complexity. Exact branch-and-bound remains combinatorial; a budgeted unfinished run is not an exact solution. |
| Tables 3–4 and 6 | Populate only from executed inventory, observations, full rankings, evidence and environment records. Display-name hints are not canonical GT IDs. |
| Table 7 | Compare speed only when input/output tasks, hardware, candidate scope and timed work match. Cached retrieval timing is not uncached pipeline timing. |
| Table 8 and Figure 6 | Use the five actual interventions and their measured records. Keep researcher-curated, verified Oracle, synthetic and automatic conditions separate. |
| Abstract, Discussion, Conclusion | Limit claims to demonstrated inputs, datasets, failure modes and uncertainty. Update numbers only from raw results. |

## Counterexamples and conditional claims

**Arbitrary mapping is not the joint optimum.** Two chairs may share a category while only one satisfies a relation to a table. The synthetic fixture additionally requires the same observed chair to satisfy a second directed relation. Matching categories first can lose an available complete embedding. Objects-only is allowed to optimize only its own objective; it is not a proof of the full objective's optimum.

**Counts are not graph incidence.** Equal node-category and edge-type counts do not specify which endpoints are connected. The fixture's two chair/table Rooms have identical category counts and the same relation types but different shared-endpoint structure. A count-signature equality does not establish indistinguishability for every observation.

**The score is not a general graph edit distance.** No independent GED optimization, edit algebra or complete cost correspondence is implemented. An analogy to deletion costs needs a precise equivalence between the chosen edit operations, injective mappings and this objective; node-only assignment does not supply that equivalence.

**Deletion stability is conditional.** Any bound must state fixed category weights and pool, nonnegative weights, exact matching, and deletion of every edge incident to a deleted observation node. Deleting observations changes `Z(A)`; a bound on a raw numerator does not automatically bound the normalized score. Recomputing anchors, frequencies or compatibility changes the problem again.

**Score stability is not rank stability.** Rank also depends on margins between candidates. Arbitrarily small score changes can reverse a near tie. Stable ID ordering makes output reproducible, not semantically unique; ties and best/worst tie ranks must remain visible.

**A complete semantic match need not score one.** Stored gamma/epsilon below one reduce relation contributions even with all endpoints matched. Confidence/evidence products are not calibrated probabilities. A Boolean-metadata intervention changes the scoring assumption rather than verifying the original metadata.

**Exactness is reported per candidate.** Objects-only has a polynomial weighted matching route. General joint search reports an upper bound and `optimality_proven`; a state-budget stop invalidates exact primary aggregation. No claim of a universally tractable exact matcher follows from small fixture success.

**Oracle is an input condition, not a performance guarantee.** Accurate labels do not remove map errors, repeated layouts, partial-view ambiguity or relation-definition mismatch. A validated clean-label result is neither an automatic system's real-world performance nor a general theoretical upper bound.

## Evidence still required

1. Keep the counterexample tests passing and inspect synthetic correspondence/edge evidence; these are implementation checks, not dataset experiments.
2. Verify canonical Room inventory, annotation scope, view provenance and the planned splits before real-data evaluation. Current E103/E102/E204 target fields are intentionally null.
3. Execute controlled full/objects-only and other paired comparisons, document ties/failures, and record the actual timing scope and hardware. No accuracy, latency, robustness or cross-building generalization result has been measured in this delivery.
