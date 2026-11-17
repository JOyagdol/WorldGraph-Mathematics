# Starting the next paper-development session

1. Open this repository as the workspace; read `AGENTS.md`.
2. If available, read `private/START_NEW_CONVERSATION_ko.md`, the yellow-highlight audit and result-note inventory linked there. These private sources are not included in a fresh GitHub clone.
3. Load the matching Research State using the local connection configuration or global second-brain configuration. Verify current Git status, remote and method versions instead of assuming the migration report is still current.
4. Build a requirements-to-evidence matrix: manuscript location → required behavior → implementation/config → test → raw experiment → note/table/prose to update. Include highlighted text and affected unhighlighted summaries.
5. Decide which requirements preserve the manuscript and which belong to the proposed method. Keep separate method/config/run identifiers when definitions differ; the joint implementation is not automatically the paper baseline.
6. Reconcile historical query notes with their raw records. Preserve original notes and edit versioned copies. Do not transfer old scores or timings to a new method.
7. Implement verified gaps and run pytest. Then prepare reviewed observations, canonical GT, a frozen graph and candidate pool, and an explicit evaluation protocol before actual experiments.
8. Update manuscript and result notes only from traceable evidence. Keep unknowns marked as unresolved and checkpoint meaningful decisions in private research memory.

See [the experiment workflow](docs/paper_workflow.md) and [the method contract](docs/retrieval/joint_evidence_v1.md). A public clone supplies executable code and synthetic fixtures; it does not certify access to real data or completed paper results.
