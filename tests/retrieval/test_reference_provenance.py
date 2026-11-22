from copy import deepcopy

from citygml_sg.retrieval.reference_graph import build_room_candidates


def test_one_step_boundary_expansion_does_not_reuse_new_paths_of_existing_seeds():
    def node(node_id, node_type, **properties):
        return {"id": node_id, "type": node_type, "properties": properties}

    def edge(source, target, relation):
        return {"source": source, "target": target, "type": relation}

    snapshot = {
        "nodes": [node("p", "BuildingPart"), node("r", "Room"),
                  node("w1", "WallSurface"), node("f1", "FloorSurface"),
                  node("w2", "WallSurface"), node("f2", "FloorSurface"),
                  node("far_wall", "WallSurface"), node("d", "Door")],
        "edges": [edge("p", "r", "INTERIOR_ROOM"), edge("p", "w1", "BOUNDED_BY"),
                  edge("p", "f1", "BOUNDED_BY"), edge("p", "w2", "BOUNDED_BY"),
                  edge("w1", "f1", "ADJACENT_SURFACE"),
                  edge("f1", "w2", "ADJACENT_SURFACE"),
                  edge("w2", "f2", "ADJACENT_SURFACE"),
                  edge("f2", "far_wall", "ADJACENT_SURFACE"),
                  edge("w2", "d", "HAS_OPENING")],
    }
    original = deepcopy(snapshot)
    candidate = build_room_candidates(snapshot)[0]
    assert snapshot == original
    assert {node["id"] for node in candidate["nodes"]} == {"r", "w1", "f1", "w2", "f2", "d"}
    assert {(item["source"], item["target"], item["type"]) for item in candidate["edges"]} == {
        ("w1", "f1", "ADJACENT_SURFACE"), ("f1", "w2", "ADJACENT_SURFACE"),
        ("w2", "f2", "ADJACENT_SURFACE"), ("w2", "d", "HAS_OPENING"),
    }
    paths = candidate["membership"]["node_sources"]
    assert all(path["source"].count("_surface_adjacency") <= 1
               for node_paths in paths.values() for path in node_paths)
    # The newly reached floor is still one step from an original wall seed.
    assert {path["source"] for path in paths["f2"]} == {
        "container_boundary_fallback_surface_adjacency",
    }
    assert {len(path["edge_ids"]) for path in paths["f2"]} == {3}
    assert any(path["source"] == "container_boundary_fallback_surface_adjacency_opening"
               for path in paths["d"])
