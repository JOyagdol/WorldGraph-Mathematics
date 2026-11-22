# Explicit evaluation spaces

`space_groups.py` supplies a separate, declared space representation while the
existing Room adapter and original snapshots remain unchanged. Group definitions
must be frozen independently of query predictions and must describe source
space semantics, not an accuracy-driven merge.

```python
groups = [{"space_id": "space_open_plan", "space_name": "Open-plan space",
           "member_room_ids": ["room_a", "room_b"],
           "provenance": "reviewed source-space definition"}]
spaces = group_room_candidates(room_candidates, groups)
label_mapping = room_to_space_map(room_candidates, groups)
```

The group ID may be new, or an explicitly selected member Room ID retained as a
representative ID. Reusing the minimum canonical member ID preserves that group's
relative deterministic tie position. It must not collide with any other source
node. The metadata distinguishes a representative ID from a new ID. Unlisted Rooms
remain singleton candidates with their original IDs and graph content. Explicit
groups require at least two distinct existing Rooms and cannot overlap.

## Graph union

The union preserves object instances and stored relations by global ID. Shared
records are included once; conflicting records are rejected. A shared window
therefore does not become two matchable windows. The union does not infer missing
installation ownership, new cross-object relations or unseen source edges.

The matcher binds the observation's unknown Room context to `candidate.room_id`.
Consequently member Room context nodes are replaced by one derived space Room
context. Existing incident edges retain their IDs, types, confidence and evidence
values; only their member-Room endpoints are projected to the group ID. Each
changed edge records its original endpoints and definition digest. This is a
declared quotient representation, not a claim that the source stored a new Room
or a new physical relation. Distinct original edges remain distinct even if
endpoint projection makes their triples equal; the matcher still selects one
best stored edge for each observed relation.

Union metadata preserves member names, source membership paths, normalization
versions and a canonical group-definition SHA-256. Group metadata describes
sharing in the new pool. Singleton membership metadata remains the historical
source metadata, so its `shared_node_ids` should not be interpreted as a fresh
inventory of sharing after grouping.

## Room-search projection baseline

`project_room_ranking_to_spaces(report, groups)` is a separate baseline. It ranks
spaces by the best original Room member under the original quantized score,
active relation contribution and canonical Room ID policy. It then recomputes
space ranks, tie intervals and pool hashes using the canonical space ID for final
ties. The original Room ranking and pool identity remain in `source_room_*`.

Original scores, category weights and category Room counts remain tied to the
original Room pool. No union matching or new space-level IDF is claimed for this
baseline. The graph-union alternative recomputes IDF on the grouped candidate
pool through the unchanged scorer. Comparisons should report both pool sizes and
the same group-level targets. A changed canonical ID can change deterministic
tie ordering; retain ties and report that policy rather than hiding it.

Projection never promotes a failure, abstention or incomplete solver result to a
completed result. Its computation belongs inside the baseline's measured online
scope; source graph grouping belongs in offline preparation. Graph union and
post-search projection are distinct interventions, not interchangeable ways to
claim a new retrieval result.
