# Hybrid exploration: frontiers + per-object sector viewpoints

A baseline for object-centric exploration. At each step the agent pools two kinds of goal and
goes to the one with the best **gain per travel time**:

- **Frontiers** (unexplored edges of the top-down map): constant gain `frontier_gain`.
- **Object viewpoints**: for every object, `n_sectors` points on a ring around its footprint, one per sector
  that `viewpoint_mode` still keeps. Constant gain `viewpoint_gain`, whatever the object's confidence.

```
u_j = gain_j / t_j        goal = argmax_j u_j
```

`t_j` is the point-turn completion time along the navmesh shortest path (`common/planning/frontier_paths.py`).
For a viewpoint it also includes the final turn to face the object, so it is the time until the agent stands there facing it (only that turn once it is already there).

## Inputs

At step `t`, `HybridFrontierAgent.update_objects({object_id: ObjectBelief(confidence, points)})`
passes in the current objects `O_i`, each with a confidence `p_i` and a world-frame point cloud `pc_i`.
The explored map `E_t` and frontiers `F_t` are kept internally by `ExplorationMap`.
`act(agent_state)` then returns a vanilla `move_forward` / `turn_left` / `turn_right` / `stop` action.

## Object viewpoints

- **Ring:** centered on the xz bounding box of `pc_i`, with radius = footprint half-diagonal + `standoff_m`.
  Sector `k` covers bearings `[k, k+1) * 2π / n_sectors`.
- **Filters**, all checked on the agent's own map, never the simulator mesh:
  - the viewpoint must be navigable;
  - it must lie in the revealed area;
  - the top-down line from it to the object must be free until it reaches the object's own footprint.
- **Inspection:** a sector counts as inspected once the agent is within `reach_m` of its viewpoint
  and facing the object (within one turn step). `viewpoint_mode` then decides which viewpoints are dropped:
  - `exhaustive` (default): only the reached one, so every viewpoint stays a candidate until it is reached;
  - `coverage`: also its neighbours within `coverage_angle_deg`, since they would show nearly the same side.

  Once the agent reaches its goal viewpoint, it counts as there until the viewpoint is inspected or the goal changes.
  The viewpoint drifts as the object's points grow, and without this latch the agent would flip between facing the
  object and walking back to it.
- **Termination:** the agent stops when there are no frontiers left and every object has no viewable,
  uninspected sector left.

## Files

| file | contents |
|---|---|
| `hybrid_agent.py` | `HybridConfig`, `Decision` (candidates, times, scores, goal), `HybridFrontierAgent` |
| `exploration_map.py` | `MapConfig`, `ExplorationMap`: navigable island map, revealed area, current field of view, frontiers, world/pixel conversion |
| `object_viewpoints.py` | `ObjectBelief`, `Candidate`, sector ring geometry, top-down line-of-sight check |
| `plotting.py` | Decision overlays: current field of view in yellow, object points with confidence, candidates colored by score and labelled with u*t above and t below, chosen goal ringed |

`map_tile()` and `object_map_tile()` return labelled tiles of the current step's decision and objects, ready for `common/utils/plot_utils.py::make_mosaic`.
`object_map_image()` draws the believed object occupancy: every tracked object point in white, the explored area green, everything else gray.
Objects are kept as point clouds, not as an occupancy grid.

## Running

`tests/test_dummy_scene.py` runs the agent in exhaustive mode for 100 steps in a synthetic 16 × 12 m flat:
4 rooms of different rectangular sizes, joined by 1 m doorways, with 5 HSSD-HAB objects per room (20 distinct chairs, couches and benches,
at their own scale), one of them 0.1 m off a wall and two of them 0.4 m apart.
An oracle detector gives each visible object a confidence equal to its image coverage, max-fused over time; it is only drawn, not scored.
`tests/testdump/test_dummy_scene/` is emptied when the test starts, then gets the last step's `decision.png` (decision map)
and `object.png` (object occupancy), and `observation_mosaic.png`, one rgb | semantic | depth tile per step.
As in `habitat_embodied_sl/main.py`, Detic (classifying against chair, couch and bench) segments every frame, its masks
and region embeddings are fused into a `ConceptGraphMemory` (`common/utils/concept_graph.py`), resolved once the run ends,
classified by `ConceptGraphMemory.classify` (`resolved_trial_mosaic.png`), and `retrieval_mosaic.png` shows one row per
resolved object of the diverse views of it `common/utils/retrieval.py`'s `retrieve_views` selects.

```bash
BASE_DIR=$(pwd) HABITAT_DATA=$HOME/habitat_data/data PYTHONPATH=. python -m pytest tests/test_dummy_scene.py -v
```

## Knobs

- `frontier_gain` trades exploring against inspecting. Below `viewpoint_gain`, the agent favours inspecting known objects over exploring;
  large values approach plain frontier exploration.
- `standoff_m` sets how close the agent stands to the object.
- `n_sectors` sets the maximum number of views per object, `coverage_angle_deg` how many of them coverage mode skips.

## Known simplifications

- Object gain depends on neither the viewpoint nor the confidence. Upgrades: the projected area of `pc_i` in the candidate camera,
  or "seen from closer than before".
- The map is 2D, so occlusion by low objects is overestimated, and objects under a table are treated as hidden.
- Paths still come from the ground-truth navmesh, as in `common/agents/frontier_agent.py`.
