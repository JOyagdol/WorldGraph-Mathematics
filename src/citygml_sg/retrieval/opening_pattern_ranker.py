"""Room ranking with object evidence and observed opening-to-wall patterns."""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations
from math import perm

from citygml_sg.retrieval.evaluation.oracle_metrics import candidate_ids_hash
from citygml_sg.retrieval.graph_matching.joint_matcher import rank_rooms


METHOD_VERSION = "shared_wall_or_single_opening_v1"
VARIANTS = ("full", "objects_only", "relations_only")
ABSTENT = {"abstained_no_anchor", "abstained_no_evidence"}


def opening_pattern(graph):
    """Extract distinct opening instances and their wall identities."""
    nodes = {node["id"]: node for node in graph["nodes"]}
    hosts = {}
    for edge in graph["edges"]:
        if edge["type"] != "HOSTED_BY":
            continue
        opening = nodes[edge["source"]]
        wall = nodes[edge["target"]]
        if opening["label"] != "Opening" or wall["category"] != "wall":
            raise ValueError("An opening pattern requires Opening -> wall endpoints")
        if any(edge.get(field, 1) != 1 for field in ("weight", "gamma", "epsilon")):
            raise ValueError("Opening patterns require unit HOSTED_BY evidence")
        instance = (opening["category"], wall["id"])
        if hosts.setdefault(opening["id"], instance) != instance:
            raise ValueError("An opening has more than one host wall")
    return tuple(hosts.values())


def prepare_opening_patterns(candidates):
    patterns = {room["room_id"]: opening_pattern(room) for room in candidates}
    if len(patterns) != len(candidates) or not patterns:
        raise ValueError("A nonempty pool of unique candidate rooms is required")
    return patterns


def gate_reason(anchors, observed):
    if any(n >= 2 for n in Counter(wall for _, wall in observed).values()):
        return "shared_wall"
    scored = [node for node in anchors["nodes"] if node["scoring"]]
    edges = anchors["edges"]
    if (len(scored) == 1 and scored[0]["label"] == "Opening" and len(observed) == 1
            and len(edges) == 1 and edges[0]["source"] == scored[0]["id"]):
        return "complete_single_opening"
    return "closed"


def match_probability(observed, reference):
    """Exact proportion of ordered draws satisfying type and shared-wall constraints."""
    n, k = len(reference), len(observed)
    if k > n:
        return Fraction(0)
    matched = 0
    for sample in permutations(reference, k):
        if any(a != "opening" and a != b for (a, _), (b, _) in zip(observed, sample)):
            continue
        if any((observed[i][1] == observed[j][1]) != (sample[i][1] == sample[j][1])
               for i in range(k) for j in range(i)):
            continue
        matched += 1
    return Fraction(matched, perm(n, k))


def decode_rank_key(row):
    """Decode the exact JSON key. The canonical room ID is applied separately."""
    obj, ratio, auxiliary, relation = row["rank_key"]
    numerator, denominator = ratio
    if (type(numerator) != int or type(denominator) != int
            or not 0 <= numerator <= denominator or denominator < 1):
        raise ValueError("Invalid rational opening-pattern evidence")
    return obj, Fraction(numerator, denominator), auxiliary, relation


def rank_opening_evidence(anchors, candidates, *, config, variant, reference_patterns=None,
                          core_rank=rank_rooms):
    """Infer one Table 8 variant. Inputs contain normalized anchors and a frozen map.

    The component scorer may be replaced by an independently certified exact
    evaluator of the same objectives. A failed component leaves the combined
    ranking incomplete. Individual component scores remain inspectable.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported opening-pattern variant: {variant}")
    ids = sorted(room["room_id"] for room in candidates)
    if not ids:
        raise ValueError("Empty candidate pool")
    decimals = config.get("score_tie_decimals", 12)
    identity = {
        "variant": variant, "method_version": METHOD_VERSION,
        "candidate_ids": ids, "candidate_count": len(ids),
        "candidate_ids_hash": candidate_ids_hash(ids), "score_tie_decimals": decimals,
        "top_score_tie": False, "rank_key_policy": METHOD_VERSION,
        "tie_policy": "object_score,opening_fraction,auxiliary_score,relation_numerator descending then canonical room ID",
        "component_method_version": config["version"], "component_statuses": {},
    }
    observed = () if variant == "objects_only" else opening_pattern(anchors)
    gate = "disabled" if variant == "objects_only" else gate_reason(anchors, observed)
    identity["gate_reason"] = gate

    def no_ranking(status):
        return {**identity, "status": status, "ranking": [], "full_ranking": False}

    if variant == "relations_only" and gate == "closed":
        return no_ranking("abstained_no_evidence" if anchors["edges"] else "abstained_no_anchor")
    use_pattern = gate in ("shared_wall", "complete_single_opening")
    components = ["relations_only"] if variant == "relations_only" else ["objects_only"]
    if variant == "full" and use_pattern:
        components.append("full")
    reports = {}
    for component in components:
        report = core_rank(anchors, candidates, config=config, variant=component)
        identity["component_statuses"][component] = report["status"]
        if report["status"] not in ABSTENT | {"completed"}:
            result = no_ranking(report["status"])
            result.update(failure_stage=component, component_failure=report)
            return result
        reports[component] = report
    active = reports[components[-1]]
    if active["status"] in ABSTENT:
        return no_ranking(active["status"])
    if use_pattern:
        if reference_patterns is None:
            reference_patterns = prepare_opening_patterns(candidates)
        if set(reference_patterns) != set(ids):
            raise ValueError("Reference pattern pool differs from the candidate pool")
    obj_scores = {}
    if "objects_only" in reports and reports["objects_only"]["status"] == "completed":
        obj_scores = {row["room_id"]: row["score"] for row in reports["objects_only"]["ranking"]}
        if set(obj_scores) != set(ids):
            raise ValueError("Incomplete object score pool")
    rows = []
    for item in active["ranking"]:
        row = dict(item)
        room = row["room_id"]
        row["object_component_score"] = obj_scores.get(room, 0)
        obj = round(row["object_component_score"], decimals)
        probability = match_probability(observed, reference_patterns[room]) if use_pattern else Fraction(0)
        auxiliary = round(row["score"], decimals) if use_pattern else 0
        relation = round(row["relation_numerator"], decimals) if use_pattern else 0
        if variant == "relations_only" and row["object_numerator"] != 0:
            raise ValueError("Relations-only component used object evidence")
        row["rank_key"] = [obj, [probability.numerator, probability.denominator], auxiliary, relation]
        rows.append(row)
    if len(rows) != len(ids) or {r["room_id"] for r in rows} != set(ids):
        raise ValueError("Incomplete component score pool")
    rows.sort(key=lambda row: (tuple(-x for x in decode_rank_key(row)), row["room_id"]))
    groups = defaultdict(list)
    for index, row in enumerate(rows, 1):
        groups[decode_rank_key(row)].append(index)
    for index, row in enumerate(rows, 1):
        tied = groups[decode_rank_key(row)]
        row.update(rank=index, best_tie_rank=tied[0], worst_tie_rank=tied[-1], tie_size=len(tied))
    result = {**active, **identity, "status": "completed", "ranking": rows,
              "full_ranking": True, "top_score_tie": rows[0]["tie_size"] > 1}
    # A component timer cannot represent the combined retrieval call.
    result.pop("timings", None)
    return result
