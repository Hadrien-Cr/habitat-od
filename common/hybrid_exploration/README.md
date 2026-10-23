# Hybrid exploration: frontiers + per-object sector viewpoints

A baseline for object-centric exploration. Whenever it has no goal, the agent pools two kinds of candidate and
picks the one with the best **gain per travel time** as its goal:

- **Frontiers** (unexplored edges of the top-down map): constant gain `frontier_gain`.
- **Object viewpoints**: for every object, `n_sectors` points on a ring around its footprint, one per sector
  that `viewpoint_mode` still keeps. Constant gain `viewpoint_gain`, whatever the object's confidence.

```
u_j = gain_j / t_j        goal = argmax_j u_j
```

`t_j` is the point-turn completion time along the navmesh shortest path (`common/planning/frontier_paths.py`).
For a viewpoint it also includes the final turn to face the object, so it is the time until the agent stands there facing it (only that turn once it is already there).

The agent keeps its goal, without re-scoring, until it is done or no longer valid, and only then picks a new one:

- **Done:** a frontier once the agent is within `reach_m` of it, a viewpoint once its sector is inspected.
- **No longer valid:** a viewpoint that is no longer a candidate (its object is gone or below `confidence_thresh`, or the
  viewpoint left the navigable or revealed area), or any goal no navmesh path leads to.

A kept viewpoint is refreshed every step from its object's current points. Re-scoring every step instead let the
heading-dependent travel times flip the goal between two viewpoints on either side of the agent, turning it left and
right in place forever.

## Inputs

At step `t`, `HybridFrontierAgent.update_objects({object_id: ObjectBelief(confidence, points)})`
passes in the current objects `O_i`, each with a confidence `p_i` and a world-frame point cloud `pc_i`.
The explored map `E_t` and frontiers `F_t` are kept internally by `ExplorationMap`: each step reveals the agent's field of
view up to `MapConfig.visibility_dist_m` (3 m).
`act(agent_state, collided)` then returns a vanilla `move_forward` / `move_backward` / `move_left` / `move_right` / `turn_left` / `turn_right` / `stop` action;
`collided` says whether the previous step collided.

## Moving

The agent heads for the next waypoint of the navmesh path to its goal. Towards a frontier it turns to face the waypoint, then
steps forward. Towards a viewpoint, it snaps the waypoint's direction to the nearest yaw reachable by whole turn steps,
and considers the four yaws that follow that direction with a forward, backward, left or right step. It picks the yaw needing
the fewest turns at arrival to face the object from the viewpoint (ties: fewest turns now), turns to it, then steps.
It keeps that step, without turning, while the waypoint stays within one turn step of its direction: re-picking at half a turn
step would zigzag between two yaws as the waypoint's bearing drifts across their boundary.

## Object viewpoints

- **Ring:** centered on the xz bounding box of `pc_i`, with radius = footprint half-diagonal + standoff, where
  standoff = clip(`standoff_per_size` · V^(1/3), `min_standoff_m`, `max_standoff_m`) and V is the volume of `pc_i`'s bounding box.
  Sector `k` covers bearings `[k, k+1) * 2π / n_sectors`.
- **Filters**, all checked on the agent's own map, never the simulator mesh:
  - the viewpoint must be navigable;
  - it must lie in the revealed area.
- **Inspection:** checked every step for every viewpoint, goal or not: a sector counts as inspected once the agent is within
  `reach_m` of its viewpoint and facing the object (within one turn step). `viewpoint_mode` then decides which viewpoints are dropped:
  - `exhaustive` (default): only the reached one, so every viewpoint stays a candidate until it is reached;
  - `coverage`: also its neighbours within `coverage_angle_deg`, since they would show nearly the same side.

  Once the agent reaches its goal viewpoint, it counts as there until the viewpoint is inspected or the goal changes.
  The viewpoint drifts as the object's points grow, and without this latch the agent would flip between facing the
  object and walking back to it.
- **Termination:** the agent stops when there are no frontiers left and every object has no viewable,
  uninspected sector left, or once its last `max_successive_collisions` (3) steps all collided.

## Files

| file | contents |
|---|---|
| `hybrid_agent.py` | `HybridConfig`, `Decision` (candidates, times, scores and best candidate, at the last pick), `HybridFrontierAgent` (its kept `goal`) |
| `exploration_map.py` | `MapConfig`, `ExplorationMap`: navigable island map, revealed area, current field of view, frontiers, world/pixel conversion |
| `object_viewpoints.py` | `ObjectBelief`, `Candidate`, sector ring geometry, top-down line-of-sight check |
| `plotting.py` | Decision overlays: current field of view in yellow, object points with confidence, candidates colored by score and labelled with u*t above and t below, chosen goal ringed |

`map_tile()` returns a labelled tile of the current step's decision, ready for `common/utils/plot_utils.py::make_mosaic`.
Objects are kept as point clouds, not as an occupancy grid.

## Running

`tests/test_dummy_scene.py` runs the agent in exhaustive mode for up to 200 steps in a synthetic 8 × 12 m flat:
two 8 × 6 m rooms joined by a 1 m doorway, with ProcTHOR-hab objects at their own scale: a table with 4 bottles on it
and a chair 0.1 m off a wall in one, a bed turned sideways with a cushion lying on it in the other.
An oracle detector gives each visible object a confidence equal to its image coverage, max-fused over time; it only steers the agent.
`tests/testdump/test_dummy_scene/` is emptied when the test starts, then gets `decision.png` (the last step's decision), `object.png` (object
occupancy), `predictions_mosaic.png` and `object_tracks.csv` (every fused object's box at every step).
As in `habitat_embodied_sl/main.py`, the class-agnostic `Detector` segments every frame, its masks and region embeddings
are fused into a `ConceptGraphMemory` (`common/utils/concept_graph.py`), resolved once the run ends, filtered by
`ConceptGraphMemory.keep_foreground`, classified by `ConceptGraphMemory.classify`, scored against the placed objects, and
`retrieval_mosaic.png` shows one row per resolved object of the diverse views of it `common/utils/retrieval.py`'s
`retrieve_views` selects.

```bash
BASE_DIR=$(pwd) HABITAT_DATA=$HOME/habitat_data/data PYTHONPATH=. python -m pytest tests/test_dummy_scene.py -v
```

## Knobs

- `frontier_gain` trades exploring against inspecting. Below `viewpoint_gain`, the agent favours inspecting known objects over exploring;
  large values approach plain frontier exploration.
- `standoff_per_size`, `min_standoff_m` and `max_standoff_m` set how far the agent stands from an object, scaled with its size.
- `n_sectors` sets the maximum number of views per object, `coverage_angle_deg` how many of them coverage mode skips.

## Known simplifications

- Object gain depends on neither the viewpoint nor the confidence. Upgrades: the projected area of `pc_i` in the candidate camera,
  or "seen from closer than before".
- The map is 2D, so occlusion by low objects is overestimated, and objects under a table are treated as hidden.
- Paths still come from the ground-truth navmesh, as in `common/agents/frontier_agent.py`.
