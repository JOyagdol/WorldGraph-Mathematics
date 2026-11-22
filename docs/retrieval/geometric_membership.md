# Geometry based reference membership

`validated_prism_membership_v1` is an opt-in preparation policy for source
CityGML geometry. It is separate from the legacy `typed_room_paths_v1` adapter.
It changes the candidate graph's membership, not the matching objective or the
observation graph. This policy does not certify image annotations or Room labels.

## Geometry contract

The source parser reads each semantic object's own direct LoD geometry. It does
not use a Room's descendant furniture coordinates to define the Room. Unsupported
geometry, unresolved links, invalid rings, and invalid shells are recorded.

Rooms must validate as closed vertical prisms with a simple footprint. Object
containment uses the convex envelope of all source vertices and their full height
range. This is a sufficient, conservative containment certificate: it may leave
valid nonconvex objects unresolved. An axis-aligned box is only a broad-phase
filter; a box or centroid alone never establishes membership.

The initial engineering policy fixes geometry epsilon to `1e-6` metres and face
contact tolerance to `0.05` metres, corresponding to the existing spatial
intersection and touch defaults. These are declared preparation parameters,
not values calibrated against retrieval accuracy. Other source coordinate
systems and units require an explicit, validated conversion before use.

- Existing semantic Furniture ownership is preserved and checked against
  geometry. Disagreement is reported rather than silently reassigned.
- An Installation is admitted only when its full shape has a unique containing
  Room. Cross-boundary and multiple-Room assignments remain unresolved.
- Boundary surfaces and openings use overlapping polygon area near a Room face.
  Shared contact is allowed and recorded. Contact does not imply visibility.
- Every source Room remains a candidate, including Rooms with no admitted
  observation cues.

## Candidate construction

`propose_geometric_memberships` returns a ledger containing the source identity,
Room validity, per-node evidence, unresolved cases, and explicit allowed node
sets. `build_geometric_room_candidates` checks the ledger's policy and source
snapshot content hash, then normalizes only those allowed nodes.

The candidate builder preserves the original stored edges whose endpoints are
both admitted. It creates no Room membership edge or other scored relationship.
Container boundary fallback, surface-adjacency expansion, and opening expansion
are disabled in this opt-in path. In particular, admitting a shared wall does
not automatically admit every opening of that wall. The original adapter and
its defaults are unchanged.

Use the existing explicit space-union layer after candidate construction when
physical space grouping has been independently declared. Unioning candidate
graphs does not generate new geometric relationships.

## Evaluation and interpretation

Freeze the source, ledger, candidate bundle, observations, candidate IDs, solver,
and scoring configuration before comparison. Use the same candidate construction
for both objects-only and full scoring when measuring the relation effect.
Compare old and new membership separately from that within-map ablation, since
membership can change object inventories and corpus category weights.

Report failures, abstentions, score ties, unique first-place predictions, and
per-space results in addition to aggregate ranking metrics. A change in a tied
prediction may be hidden by canonical-ID tie ordering. Keep uncertain memberships
and annotation status visible; a more specific geometric map is not automatically
a more accurate retrieval system.
