# WorldGraph-Mathematics

Paper-focused development workspace derived from [3DCitySG](https://github.com/JOyagdol/3DCitySG). The upstream project remains independent. This repository supports evidence-backed manuscript completion and reproducible Room retrieval experiments using a CityGML **2.0** World Graph.

Start with [AGENTS.md](AGENTS.md), [NEXT_SESSION.md](NEXT_SESSION.md), and [the experiment workflow](docs/paper_workflow.md). On the author's workstation, `private/START_NEW_CONVERSATION_ko.md` contains the detailed handoff, manuscript audit, result-note inventory and local asset locations. Private files are intentionally absent from a GitHub clone.

## Current implementation status

- The inherited `legacy_heuristic` is a historical baseline with its own assumptions and outputs.
- `joint_evidence_v1_proposed` is an implemented candidate for part of Oracle Evaluation. It is **not automatically the method adopted by the manuscript**.
- Manuscript-preserving requirements and proposed improvements must have distinct method/config/run identities when their behavior differs.
- Unit tests and synthetic fixtures verify software contracts. They do not establish real-data accuracy, timing, annotation validity or theoretical claims.

The current working-tree snapshot includes previously uncommitted Oracle components. See [source provenance](docs/upstream_provenance.json); the upstream base commit alone does not contain these changes. The Python package name remains `citygml_sg` to preserve imports and scripts.

## Install and test

Python 3.10 or later is required. Use a separate virtual environment for this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest tests -q
```

Codex runs appropriate pytest checks when changing the code and reports the command and outcome. Actual database operations and research benchmarks follow the user's authorized scope. Do not repoint an existing upstream editable installation to this checkout.

An offline preparation command, which does not run retrieval or connect to a database:

```powershell
.\.venv\Scripts\python.exe scripts/retrieval/run_oracle_evaluation.py --manifest tests/fixtures/oracle/manifest.json --prepare-only --output-dir data/output/fixture_preparation
```

The output directory must be new or empty. The fixture is synthetic and must not be reported as a paper experiment.

## Implementation references

- [Joint method contract and execution guide](docs/retrieval/joint_evidence_v1.md)
- [Method and theory review obligations](docs/retrieval/joint_evidence_theory_alignment.md)
- [Retrieval documentation](docs/retrieval/README.md)
- [Observed View Graph schema](docs/schemas/observed_view_graph.schema.json)
- [Graph schema guide](docs/graph_schema.md)
- [Project configuration template](configs/default.yaml)

## Public and local files

Git contains code, synthetic fixtures, curated sample data and general instructions. Unpublished manuscripts, detailed research notes, real observations, generated results, credentials and Notion state remain ignored local files. Large original graph exports and capture images may be external read-only assets, listed in the author's private migration manifest; a public clone is not a complete paper dataset.

Never publish a result without its method ID, code revision, resolved configuration, observation/GT/snapshot provenance, candidate pool and raw run evidence.
