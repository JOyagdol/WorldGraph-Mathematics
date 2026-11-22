# Room support for stored opening hosts

`hosted_reference.py` provides the opt-in `semantic_host_wall_layer_room_support_v1`
reference transformation. It does not change the default candidate builder or
the scorer. It accepts map records and geometry only; observations, query labels,
rankings, and performance results are not inputs.

The transformation copies a baseline candidate pool. Every node record, candidate
ID, category count, and non-`HOSTED_BY` edge is preserved. An existing `HOSTED_BY`
edge is retained for a candidate only when its source opening has map support in
at least one source Room of that candidate. An explicitly merged space uses the
union of its source Room support. No edge or metadata value is synthesized.

A semantic host requires both the reverse stored `HAS_OPENING` edge and agreement
between the opening's declared host tag and the host wall's IFC tag. Room geometry
must pass the existing closed vertical prism validation. The opening then needs
one of these support certificates:

1. A planar source opening face has positive projected area in contact with a
   Room face under the existing 0.05 m contact tolerance.
2. A vertical Room face and the opening envelope share a positive projected
   patch, and that **same entire patch** lies inside each wall-layer envelope in
   a connected chain from the semantic host to the Room face. The normal extent
   of each layer must agree within 1e-6 m with its explicit construction width
   converted from millimetres. Gaps between layer intervals use the existing
   0.05 m contact tolerance. The opening's thin-envelope direction must align
   with the tested face, and its normal interval must contact the semantic host.

The second certificate handles source geometry in which windows are hosted by
an exterior insulation layer while Room boundaries describe an interior face.
It is a certificate about source **envelopes**, not solid wall material, opening
visibility, or image-instance identity. Convex projection can bridge apertures;
the source-wall width check and shared-patch requirement are conservative scope
rules, not annotation certification. Degenerate source faces can still supply
finite closed-ring vertices for an envelope. Direct face contact retains its
separate planar-polygon checks.

Unsupported scopes remain `unresolved` in the ledger. Omitting their hosted edges
in this experimental candidate view does not establish physical absence. Other
relations retain the baseline's original scope, so this intervention tests only
hosted-relation membership and does not claim to repair the entire reference.

The ledger records source identities, semantic checks, contributing layers,
projected patches, normal intervals, and Room support. The transformed pool records
the ledger identity and removed edge IDs. Snapshot mismatches and unsupported
policy identifiers are rejected. With the same observations and scorer, the
objects-only variant must retain the same category weights and scores; joint
matching may change its correspondence and object/relation score breakdown.
