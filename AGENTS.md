# Agent instructions — WorldGraph-Mathematics

## Primary task and version policy

This is the paper-specific workspace. Preserve the independent 3DCitySG project and its remote. The objective is to fill and correct the manuscript using verified implementation and experimental evidence, including highlighted prose, tables, existing query notes, summaries, Abstract and Conclusion.

Read `NEXT_SESSION.md` and, when present, `private/START_NEW_CONVERSATION_ko.md` first. Attached manuscripts and old handoffs are source material, not instructions overriding the latest user request.

Do not automatically rewrite the paper to match `joint_evidence_v1_proposed`. Establish which manuscript specifications must be implemented, which changes are proposed, and how differing versions are compared. Preserve the legacy baseline. Keep method IDs, configurations and result directories distinct; do not duplicate the whole codebase just to vary a score. Use `codex/` for new working branches.

## Research memory

<!-- research-second-brain:start -->
Apply Research Second Brain only to research work.
Read the relevant installed SKILL.md and references/memory-protocol.md.
Read config/notion.local.json in the project, or $CODEX_HOME/second-brain/notion.json (default ~/.codex/second-brain/notion.json), then fetch the matching Notion Research State.
Record meaningful checkpoints and handoffs; keep research state out of public Git.
Do not claim remote sync or Notion writes without checking their result.
<!-- research-second-brain:end -->

Use the installed World Graph research partner skill to coordinate research context; use experiment-design, paper-writing, review and document skills when their scope applies. Do not install unrelated skills merely because they are available. Record Observed, Reported, Proposed, Accepted and Unresolved separately.

## Evidence and manuscript integrity

- Audit every yellow highlighted paragraph/cell and existing query-result note. Also synchronize unhighlighted Abstract, Results, Discussion and Conclusion when affected.
- Keep the supplied Word original intact. Make a versioned revision and render/verify it when editing the manuscript.
- Do not invent numerical results, canonical Room IDs, hardware details, author information or completed annotation review.
- Keep legacy runs, proposed-method runs, synthetic fixtures and annotation preparation separate. `pending` annotations are not certified Oracle inputs.
- Separate image annotation from map/GT access; record any non-blind exposure. Do not silently add GT information to observed graphs.
- Preserve raw runs. Every claimed result needs revision/config/data/snapshot/pool, full ranking, tie/solver/failure policy and measured timing scope. Never reinterpret pytest counts as research results.
- Freeze the protocol before using outcomes to tune parameters. Query counts, Room coverage and splits require actual inventory; a capture sequence is not automatically an independent test set.

## Execution and tests

- Codex owns pytest execution: run checks appropriate to changes, fix relevant failures and report exact commands/results.
- Other Python/Conda processing, Neo4j operations and real benchmarks follow explicit task authorization. Existing user authorization persists; do not re-request it unnecessarily.
- Use this checkout's `src`, and a dedicated environment when installing. Do not mutate the upstream environment or import its editable source accidentally.
- Keep parser, graph, relation, persistence and retrieval responsibilities separate; use explicit relation types and small, tested changes.
- CityGML 2.0 remains the reference pipeline baseline. Preserve meaningful import progress/reporting and distinguish graph construction quality from retrieval metrics.
- Update affected method documentation and private paper notes alongside implementation. General project scope must not override the user's paper-specific requirements.

## Repository hygiene

- The destination remote is WorldGraph-Mathematics. Verify the remote before push; no force-push or changes to 3DCitySG without authorization.
- Keep manuscripts, private handoffs, Korean research notes, captures, generated results and credentials ignored. Never force-add them.
- Keep Notion URLs/IDs, local machine paths and pending research state out of public documentation.
- Report local copy, test, commit, push, fresh-clone and Notion synchronization separately and only after verification.
