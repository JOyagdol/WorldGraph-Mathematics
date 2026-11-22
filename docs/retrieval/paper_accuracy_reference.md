# Reference-solver accuracy baseline

The earlier accuracy comparison uses the pre-optimization `exact_reference`
solver and the original semantic Room adapter. The current Table 8 experiment
has a [separate ranking specification](paper_table8.md). The baseline configuration is
[`configs/paper_accuracy_reference.yaml`](../../configs/paper_accuracy_reference.yaml);
the entry point is
[`scripts/retrieval/run_paper_accuracy.py`](../../scripts/retrieval/run_paper_accuracy.py).
This entry point rejects `exact_pruned_v2` and `exact_pruned_v3`. Its default does
not import geometric membership or select a repaired reference map. Its optional
space-group file applies previously declared physical-space unions to the
original candidates, with no access to query evidence or evaluation labels.

An explicit `--hosted-support-ledger` flag selects a separate relation-only
intervention. It calls `hosted_reference.scope_hosted_by` after building the
original candidates and optional unions. All candidate nodes, category counts
and non-HOSTED_BY edges remain unchanged; only unsupported existing HOSTED_BY
edges are omitted. The default command does not enable this intervention.

The original shared defaults were already `exact_reference`; the active paper
configuration now makes that choice explicit. The matcher/scorer source is
restored to the pre-optimization version, removing the v2/v3 search code and
dispatch. The original joint-search, object-matching, compatibility, score and
semantic-adapter behavior is retained.
There is no new reranking, weight adjustment or annotation certification.
Separate, versioned fast-solver and geometry-map experiments remain available
as historical evidence. Do not overwrite or reclassify their raw results.

## Execution

Use this self-contained synthetic fixture to check the entry point. Its inputs
are invented graphs, not measured image-localization evidence.

```powershell
python scripts/retrieval/run_paper_accuracy.py --manifest tests/fixtures/oracle/manifest.json --graph-json tests/fixtures/oracle/reference_graph.json --output-dir data/output/paper_accuracy_fixture
```

For a real evaluation, supply a local manifest, a frozen semantic snapshot and,
when applicable, an explicit group-definition JSON list. The manifest follows
the existing Oracle manifest schema and must name the complete candidate pool
after grouping. Its GT IDs already refer to the declared evaluation spaces;
the entry point does not silently remap labels.

```powershell
python scripts/retrieval/run_paper_accuracy.py --manifest configs/paper_manifest.local.json --graph-json data/output/frozen_reference/snapshot.json --space-groups configs/paper_space_groups.local.json --output-dir data/output/paper_accuracy_run
```

To reproduce the separately named HOSTED_BY intervention, use the same inputs
and supply the source-derived ledger produced by
`hosted_reference.build_hosted_support_ledger(snapshot, geometry)`. Freeze its
source geometry and policy before retrieval; this command does not create or
certify that geometric evidence. `scope_hosted_by` checks the ledger policy and
snapshot identity. The run records the ledger's byte and content hashes and
both the baseline and intervened candidate-content hashes.

```powershell
python scripts/retrieval/run_paper_accuracy.py --manifest configs/paper_manifest.local.json --graph-json data/output/frozen_reference/snapshot.json --space-groups configs/paper_space_groups.local.json --hosted-support-ledger data/output/frozen_reference/hosted_support_ledger.json --output-dir data/output/paper_accuracy_hosted_scope
```

Compare this new output with the default command's output on exactly the same
manifest. The flag does not change the solver, object weights, observations or
GT. Missing support is an unresolved reference relation, not evidence that a
physical opening or wall is absent. This is different from the older geometric
membership experiment, which also changed candidate node inventories.

Use `--prepare-only` with a new output directory to validate and record inputs
without inference. Preparation checks the snapshot, all observations and pool
membership; it does not certify annotations. Every invocation requires a new
or empty output directory.

## Interpretation and reproducibility

- Exactly one inference is made per query and configured variant. The default
  variants are Full and Objects-only. No warm repetitions or latency aggregate
  are collected; the shared matcher's internal stage timers are discarded.
  `timing` in the shared configuration schema is required to be
  `warmup: 0, repeat: 1`; it is not a measured-time claim for this entry point.
- The solver retains the original per-candidate `max_states: 100000` limit.
  This CLI does not add a wall-clock timeout. A separately bounded experiment
  harness must declare its timeout and preserve timed-out queries as failures.
- Full rankings, solver status, mappings and tie metadata are stored per query.
  GT is attached by evaluation only after inference. Input byte hashes,
  candidate-content and pool hashes, resolved configuration and code provenance
  are recorded with the run.
- Abstentions remain zero-hit outcomes, with the existing `N+1` MTR penalty.
  Exceptions and incomplete searches remain in the cohort and invalidate
  strict primary aggregates; the shared evaluator labels failure-count
  diagnostics separately. No tie, abstention or timeout is removed after seeing
  its outcome.
- A historical timeout-excluded subset and a complete input cohort answer
  different questions. Preserve each original denominator and exclusion rule;
  do not present a conditional subset as the new primary result.

Historical fast-solver reproduction requires the corresponding archived source
and frozen method configuration, with a new output directory. The current
matcher/scorer and both current CLIs reject `exact_pruned_v2` and
`exact_pruned_v3`; they never silently substitute `exact_reference`. Preserved
fast-run records are not evidence that accuracy improved. Geometric reference
reconstruction is a separate map intervention, not a solver rollback.
