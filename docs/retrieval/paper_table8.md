# Table 8 room retrieval

The Table 8 entry point runs three variants of `shared_wall_or_single_opening_v1`. All use the same normalized observation, anchor selection and candidate pool. The map is built with the original semantic Room adapter, with any declared physical-space unions. A source-derived support ledger restricts stored opening-to-wall edges to the rooms supported by the source model. The entry point checks the ledger's snapshot identity and records the hashes of the ledger, snapshot, observations and candidate pool.

```powershell
python scripts/retrieval/run_paper_table8.py --manifest configs/paper_manifest.local.json --graph-json data/output/frozen_reference/snapshot.json --space-groups configs/paper_space_groups.local.json --hosted-support-ledger data/output/frozen_reference/hosted_support_ledger.json --output-dir data/output/paper_table8
```

The method activates opening-pattern evidence in two circumstances.

1. At least two distinct observed openings have an explicit `HOSTED_BY` relation to the same observed wall.
2. The complete set of scored objects contains a single opening, and the complete set of relations contains its one `HOSTED_BY` edge to a wall. Unscored wall, floor or ceiling context may also be present.

For an active observation with k hosted openings and a candidate with n hosted openings, the pattern score is the fraction of ordered samples of k distinct candidate openings that satisfy the observed opening types and the same-wall and different-wall constraints. A generic observed opening can match any opening. Each stored opening counts once. The denominator is the falling factorial (n)ₖ. A candidate with fewer than k eligible openings receives zero. This model assumes uniform sampling of verified openings and unit hosted-edge evidence.

| Variant | Ranking with active opening evidence | Behavior with inactive opening evidence |
|---|---|---|
| Full | Object-only score, exact opening-pattern fraction, joint score, joint relation numerator | Object-only ranking |
| Objects only | Object-only ranking | Object-only ranking |
| Relations only | Exact opening-pattern fraction, relation-only score, relation-only numerator | Abstention |

The relations-only variant uses opening and wall types to check relation endpoints and applies the common observation eligibility test. Its node score and node contribution are zero. Its secondary objective uses the relation-only matcher with object weight disabled. Complete absence of matched evidence produces abstention. This ablation measures the relationship component of the selected Full within its observation eligibility conditions.

The component objectives use `joint_evidence_v1_proposed` with the original `exact_reference` solver. The Full ranking compares object-only and joint scores from independently maximized correspondences. Scores and relation numerators are rounded to the configured precision for ranking. The opening fraction uses exact integer arithmetic. Canonical room ID resolves equal evidence keys. Each candidate report includes the four-part ranking key, tie size and best/worst tie rank. Hit@1 includes this deterministic last tie break. A unique correct first result additionally requires a singleton top evidence group.

The CLI makes one inference per query and variant. The strict evaluation retains all original queries. An abstention contributes zero to Hit and reciprocal rank and a target rank of candidate count plus one. A component budget failure invalidates the operational primary aggregate. Any dedicated timing experiment must record its component evaluator, preprocessing and preload boundaries, warmup, repetitions, wall deadline and failure policy. Where a separate exact certificate supplies full-cohort accuracy, the certificate and the operational solver have distinct execution identities. A latency value refers to the implementation and timing scope measured in that run.
