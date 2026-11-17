# Joint evidence retrieval v1: implementation candidate

`joint_evidence_v1_proposed` is a reusable Room matching, scoring, evaluation and timing component for upcoming Oracle experiments. It does not implement perception, certify annotations, or complete the Oracle experiment design. The existing `legacy_heuristic` remains the default single-query method. The reference pipeline remains CityGML **2.0**.

At the initial implementation handoff (2026-11-02), tests had not been executed. After the user authorized Codex to run pytest, retrieval tests passed (135 tests) and the full repository suite passed (159 tests) in the existing Python 3.11.15 environment. One test's UTF-8 CSV/JSONL reads were made explicit for Windows compatibility. No real-data Neo4j experiment or measured retrieval accuracy/latency study has been run. Engineering defaults are not calibrated parameters. See [theory alignment](joint_evidence_theory_alignment.md) before using the method in a paper.

## Entry points and ownership

Paths below are relative to the repository root.

| Component | Current implementation |
|---|---|
| Strict observation normalization and anchor selection | `src/citygml_sg/ovg/validation/oracle_observation.py` |
| Room scope, normalization, metadata and read-only snapshot adapter | `src/citygml_sg/retrieval/reference_graph.py` |
| Partial injective joint correspondence | `src/citygml_sg/retrieval/graph_matching/joint_matcher.py` |
| Shared score and five interventions | `src/citygml_sg/retrieval/scoring/joint_score.py` |
| Full-pool metrics and paired query comparison | `src/citygml_sg/retrieval/evaluation/oracle_metrics.py` |
| Common experiment and timing harness | `src/citygml_sg/retrieval/oracle_runner.py` |
| Benchmark/preparation/inventory CLI | `scripts/retrieval/run_oracle_evaluation.py` |
| Measured-record table export | `src/citygml_sg/retrieval/reporting/ablation_table.py` |
| Method settings without connection credentials | `configs/joint_evidence_v1.yaml` |

`room_localization_queries.py --method joint_evidence_v1_proposed` uses the common observation/reference/matching path for a single query. Its console `--limit` does not truncate the stored ranking. `--method legacy_heuristic`, `profile_room_localization_stages.py`, `evaluate_room_retrieval_metrics.py`, and `run_room_retrieval_suite.py` retain their legacy roles; their historical outputs are not results of this method. `signature_similarity.py` remains a count helper, not an instance matcher.

The legacy `view_params.py` receives one bounded fix: when an explicit OVG contains no furniture or installation category, the corresponding default keyword list is cleared. Runs without an OVG retain defaults; explicit CLI overrides remain available. Other legacy heuristic assumptions are not silently redefined.

## Input and reference contracts

Observations preserve distinct aliases for repeated instances. Labels normalize to Room, Furniture, Installation, Opening and BoundarySurface. Refrigerator/freezer map to fridge, television/display to tv, couch to sofa, and pillar/기둥 to column. Sink, counter, cabinet and storage remain distinct. Wall and floor subtype aliases are canonicalized consistently with the reference adapter; original reference surface subtype is retained.

Each observation node has confidence and visibility in finite `[0,1]`, with `w(u)=confidence*visibility`. Legacy object `weight` is ignored with a warning. Relation weights are explicit finite `[0,1]`, or unit weight when omitted. Positive-weight nodes meeting the configured threshold become anchors; excluded nodes lose their incident edges.

Furniture, Installation and Opening form the object-scoring domain `D_A`. BoundarySurface and the unknown-Room placeholder are relation contexts with zero independent object contribution. A typed endpoint reuses a unique compatible explicit context; otherwise one canonical existential context is created. The living example's `FloorSurface` endpoint therefore reuses `floor_1`. Multiple compatible explicit floors require an explicit alias and produce an ambiguity error. Generic BoundarySurface is a label-level context wildcard; typed subtypes still constrain correspondence.

Observation objects/relations reject map instance IDs and GT Room fields. GT is read only from the evaluator manifest. Legacy query constraints, `min_object_matches`, `require_all_required_objects`, candidate limits and bonus weights are not inference parameters for this path. No Room is removed by a minimum-match filter.

All Room candidates, including empty Rooms, are retained. Furniture/installation membership follows explicit Room edges. Boundary scope uses direct Room boundaries, immediate Building/BuildingPart boundary fallback only when direct boundaries are absent, and the existing connected-opening/host-wall and one-step wall/floor adjacency paths. It never traverses a BuildingPart to collect sibling Room furniture. Shared nodes and fallback edge paths are recorded in `membership`; fallback does not fabricate a scored edge.

The adapter maps stored `confidence -> gamma` and `evidence_score -> epsilon`, preserving producer metadata and stable relationship IDs. Only typed base semantic edges use the documented `citygml_semantic_hierarchy_v1` unit-metadata policy. Missing inferred metadata stays null. When observed relation atoms are active, full, relations-only and uniform-category scoring require metadata across the candidate graphs. With no active observed relations, or for objects-only, missing inactive metadata is allowed and reported. Boolean metadata is an explicit intervention. Present invalid values are rejected in every variant.

Semantic prefetch excludes geometry payloads while collecting full node-type inventory. Full stored size, geometry count and the semantic candidate union are reported separately. Layered boundary collapse is not rerun without its required geometry and is recorded as such. Freeze the source during Neo4j collection or use an immutable exported JSON for repeat comparisons; a read transaction alone does not guarantee an immutable source under concurrent writes.

## Objective and five variants

For all `N` reference Rooms, `n(k)` counts Rooms containing category `k` in the same object-scoring label domain. Fix these weights for the query/variant:

```text
rho(k) = ln(1 + N / (1 + n(k)))
a(u) = w(u) * rho(category(u));  b(e') = gamma(e') * epsilon(e')
N_r(M) = alpha * sum_matched_nodes a(u)
       + beta * sum_matched_observed_edges w(e) * b(best compatible stored edge)
Z(A) = alpha * sum_all_anchor_nodes a(u) + beta * sum_all_anchor_edges w(e)
s(r) = max_M N_r(M) / Z(A), when Z(A) > 0
```

`M` is partial and injective; a relation scores only when its mapped endpoints have a compatible stored edge. Each observed edge scores once, choosing the largest metadata product when alternatives exist. The same query/variant denominator applies to every Room. Extra unobserved candidate objects incur no penalty and cannot multiply one observed object's contribution.

| Variant | Intervention; all use the same observations and complete candidate pool |
|---|---|
| `full` | Room IDF, object and relation terms, stored metadata |
| `objects_only` | `beta=0`; relation evidence cannot influence correspondence or tie-breaking |
| `relations_only` | `alpha=0`; label/subtype/category compatibility and injectivity remain |
| `uniform_category` | `rho=1`; observation weights and stored metadata remain |
| `boolean_metadata` | `gamma*epsilon=1` for existing compatible edges; absent edges remain absent |

The configurable relation map is identity plus observed ADJACENT_TO→stored TOUCHES/INTERSECTS and observed TOUCHES→stored INTERSECTS. It is not made symmetric. Configured symmetric relations allow endpoint reversal; ABOVE/BELOW use explicit inverse-direction handling.

Objects-only uses exact weighted bipartite matching. Other variants use joint branch-and-bound with a per-Room state budget. Reports expose the algorithm, states, upper bound, termination reason and `optimality_proven`. Budget exhaustion yields `solver_incomplete`, not a certified exact result.

## Execution commands

Run from the repository root in Python 3.10+ with `python -m pip install -e ".[dev]"`. Codex runs pytest checks and reports the command and result; other Python/Conda processing and benchmark/database experiments remain user-run unless separately authorized. Use a new or empty output directory for each invocation, or omit `--output-dir` for a timestamped directory.

Prepare the real-data template without connecting to Neo4j or computing retrieval metrics:

```powershell
python scripts/retrieval/run_oracle_evaluation.py --prepare-only --manifest docs/examples/oracle_evaluation_manifest.template.json --method-config configs/joint_evidence_v1.yaml --output-dir data/output/oracle_preparation
```

Export canonical Room IDs and full/semantic inventory from the current local database:

```powershell
python scripts/retrieval/run_oracle_evaluation.py --inventory-only --config configs/default.local.yaml --output-dir data/output/oracle_inventory
```

Copy the manifest template to ignored `configs/oracle_manifest.local.json`, map E103/E102/E204 display hints to canonical IDs from `candidate_pool.json`, and fill snapshot/pool identity and view provenance. These real target IDs remain null in the public template; Room count and annotation validity are not assumed.

```powershell
Copy-Item docs/examples/oracle_evaluation_manifest.template.json configs/oracle_manifest.local.json
```

After editing that local manifest, run the five-variant benchmark:

```powershell
python scripts/retrieval/run_oracle_evaluation.py --manifest configs/oracle_manifest.local.json --config configs/default.local.yaml --method-config configs/joint_evidence_v1.yaml --output-dir data/output/oracle_benchmark
```

For a fixed exported graph, replace `--config configs/default.local.yaml` with `--graph-json data/output/e_type_import.json`. The method config is separate from the ignored connection config.

Freeze database writes while collecting a reference snapshot, or use an immutable JSON export. A read transaction does not guarantee snapshot isolation under concurrent Neo4j changes. E-type exports are local generated artifacts and are not supplied by a fresh clone.

Run the self-contained synthetic fixture (three invented Rooms; no image or Oracle validation claim):

```powershell
python scripts/retrieval/run_oracle_evaluation.py --manifest tests/fixtures/oracle/manifest.json --graph-json tests/fixtures/oracle/reference_graph.json --method-config configs/joint_evidence_v1.yaml --output-dir data/output/oracle_synthetic_fixture_run --warmup 0 --repeat 1
```

Run the prepared retrieval tests:

```powershell
python -m pytest tests/retrieval -q
```

Validation on 2026-11-02: retrieval **135 passed**; full suite **159 passed**.
The executed command shape was `python -B -m pytest tests -q --tb=short -p no:cacheprovider --basetemp=<new writable temporary directory> --junitxml=<report.xml>` using the project's existing Python 3.11.15 environment. The retrieval run used `tests/retrieval` instead of `tests`. A unique workspace temporary directory avoided restricted system-temp access; UTF-8 output was read explicitly. JUnit reports are local generated artifacts under `data/output/test_reports/20261102/`. Passing synthetic/unit tests does not establish real Oracle accuracy or online performance.

## Evaluation, timing and artifacts

The current primary evaluator is closed-set, single-GT Room retrieval. Recall@k equals Hit@k only in this singleton setting. MRR uses the full-pool GT rank; MTR averages rank, with the explicit `N+1` convention for retrieval abstention. Generic set helpers preserve the distinction between multi-GT Hit and Recall. JSON recall values are fractions; table export displays percentages.

`Z=0` produces `abstained_no_anchor`; all-zero evidence produces `abstained_no_evidence`. Both count as zero Hit/Recall/RR, not accidental Room-ID successes. Missing GT in the pool, duplicate IDs, incomplete rankings, unstable order/tie metadata, repeated evidence changes and unfinished solvers prevent a valid complete primary table. Failures remain visible in diagnostics instead of silently shrinking the query denominator.

Order is rounded active score descending, rounded active relation numerator descending, then canonical Room ID ascending (12 decimal places by default). Top-score ties and best/worst tie ranks are reported; deterministic tie resolution is not evidence of physical distinguishability. Paired full-versus-objects-only output reports each query's rank change and improved/tied/worsened counts.

Reference nodes/edges and Room candidate construction are preloaded once per run. Primary `online_retrieval_ms` starts from a normalized in-memory observation and includes anchor selection, Room IDF computation, matching, scoring, sorting and evidence construction. It excludes schema/file loading, driver creation, connection preflight, graph fetch/candidate preparation, and report serialization. Those costs are recorded separately; cached online DB fetch time is zero. This is not an end-to-end uncached localization latency claim.

Default measurement records a first query, three warmups and 30 measured warm repeats per query/variant. Variant order is seeded. Raw phases are retained; query-level timing summaries distinguish sample standard deviation and nearest-rank p95. Dataset Mean Time averages per-query warm means, not extra queries created from repeats. One-repeat fixture runs are functional checks, not timing studies.

Outputs include `run_manifest.json`, `resolved_config.json`, `candidate_pool.json`, `graph_inventory.json`, `per_query/<query_id>/<variant>.json`, `timing_runs.jsonl`, `evaluated_queries.json`, `metrics_by_variant.json`, CSV/Markdown ablation tables, and failure analysis. Hashes identify observation bytes, config, graph, candidate pool and relevant code state. Different input/validation conditions are aggregated separately. Preparation exports `not_run`/null values; failed or missing variant rows remain incomplete.

Final observation counts, building/Room scope, annotation procedures, view validation, evaluation split and final parameter selection remain pending. Synthetic fixtures and researcher-curated examples cannot be relabelled `oracle_verified` by passing schema checks.
