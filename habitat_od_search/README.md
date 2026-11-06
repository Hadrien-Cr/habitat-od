# Object search on HSSD-rare

`main.py` evaluates `SearchAgent` (`search_agent.py`), a Detic-driven single-target ObjectNav agent, on IMPRINT's
HSSD-rare episodes. `enhanced_detector.py` is a baseline that replaces the agent's text query with example views
of the target object. The agent plans only from its own depth-built belief map, the shared `EstimatedMap` /
`DiscretePlanner` of `common/agents/base_exploration_agent.py` and `common/hybrid_exploration/`. Ground truth is
used only for scoring, unless an ablation flag (below) is turned on.

## Episodes (`hssd_rare_dataset.py`)

- Episodes are read from `third_party/IMPRINT/dataset/hssd_rare_eps/<scene_id>.json.gz` (habitat-lab's
  ObjectNavDatasetV1 shape), keeping each episode's own start position but snapping its start yaw to the nearest
  45° (`_snapped_start_rotation`), so every episode starts facing one of 8 axis-aligned headings instead of the
  dataset's arbitrary one. When `scenes` is empty, every saved scene is used (`available_scenes`).
- Each episode's `object_category` is a fine-grained HSSD instance string, such as
  `table, Metal 25" 2 Tier Accent Table`. Its main category is the text before the first comma (`main_category`).
  The main category is the search query, and its tier in `main_cat_split.json` (`easy`/`medium`/`hard`) is the
  episode's rarity tier.
- Goals are matched to simulator objects by handle (`SceneGoal.object_handle`), because IMPRINT's `object_id` is
  not habitat-sim's runtime id.

## Per episode (`main.py::run_episode`)

1. **Filter:** skip episodes whose start navmesh island is smaller than `min_island_area_m2` (10 m²), for example
   a balcony cut off from the house, or whose `episode_id` doesn't match the config's own `episode_id` when one is
   set (unset runs every episode of the scene -- set it, with `scenes` narrowed to one scene, to rerun a single
   episode, e.g. to inspect a run's images again under updated code). Stop after `max_episodes` episodes across all
   scenes (`null` runs every one).
2. **Reset:** the agent gets the episode's start pose. From then on it sees only poses relative to that start, its
   yaw snapped to 90° so the map grid stays on the world axes (`ref_state`, `ref_frame_state`), as habitat's episodic
   GPS + compass give them. The target embedding is the CLIP text
   embedding of the main category. Its belief map (`EstimatedMap`, fixed at `map_size_m`, 48 m) is centered on the
   scene's own navmesh extent (`pathfinder.get_bounds()`, computed once per scene) rather than the agent's start
   position, whenever that extent fits within `map_size_m` -- so a long episode doesn't run the agent past the
   map's fixed edge; otherwise (or for the other two agents, which don't pass navmesh bounds) it's centered on the
   agent's own start position, as before.
3. **Per step (`SearchAgent.act`):**
   - Update the belief map from depth: obstacles are depth points 0.2–0.88 m above the floor, plus the
     floor-plane crossings of no-hit downward rays (dropping any crossing whose own cluster covers fewer than
     `min_missing_floor_cluster_m`, 0.5 m, cells -- e.g. a single bad depth pixel from a hole in the scanned
     mesh). Then age the goal-score map by `score_decay` (0.9), within the agent's current field of view only --
     cells outside it keep their last score, since no new observation of them happened this step.
   - **Detect:** run `Detector.predict_target`, Detic's single-class zero-shot inference against the target
     embedding (classifier reset to [target, background]), with SAM masks when `--sam-checkpoint` is given.
     Boxes are kept above `--score-thresh` (0.4) and suppressed at `--nms-thresh` (0.5).
   - **Fuse:** if the best detection scores at least `goal_score_thresh` (0.5), back-project its mask (at
     `lift_image_size`, 512) into world points, drop the floor and the wall behind them, then run
     concept-graphs' own per-detection denoising (voxel-downsample at `downsample_voxel_size`, 0.025 m, then
     `main_cluster_mask`'s largest DBSCAN cluster -- `detection_dbscan_eps`/`detection_dbscan_min_points`,
     0.05 m/5 -- plus any smaller one attached within `dbscan_eps`, 0.1 m, up to
     `detection_max_attached_fraction`, 0.15, of it) before scattering the score onto the estimated map's own
     `goal_score` grid (`EstimatedMap.mark_goal`), kept as each cell's running max (and the 3D point that set
     it). The believed goal (`goal_mask()`) is every cell currently tied at that grid's own peak score, not a
     single frozen point -- so the planner can pick whichever of them is actually reachable.
   - **Stop** at `max_steps` (2000 in `eval_200_*`; the default config uses 500) or when stuck (no net
     translation over 25 steps, `BaseExplorationAgent.is_stuck`).
   - **Act:** start with a full 360° spin. Then the `DiscretePlanner` (cells of `resolution_m`, 0.05 m in
     `eval_200_*`; the default config uses 0.1 m) plans towards the believed goal cell once one exists, and
     towards the frontiers otherwise. It stops within 0.75 × `reach_m` of the goal, or when there is neither a
     goal nor a frontier left. With `agent.planner_debug` (off by default), each step also writes a
     `snapshot_planner_<timestep>.png` mosaic (traversable/goal/FMM-distance/frontier cells around the agent) to
     the episode's own `images/<run_index>_ep<episode_id>_<scene_id>/planner/`, alongside its own
     `images/<run_index>_ep<episode_id>_<scene_id>/snapshot_gif_*.png`.
   - Every agent's `act` (`common/agents/abstract_agent.py::Agent`) returns `(action, StopReason)`, the reason
     only set when `action` is `"stop"`: `stuck`, `collision` (the last `max_successive_collisions` steps all
     collided; `FrontierExplorationAgent`/`HybridFrontierAgent` only), `exploration_exhausted` (no goal and no
     frontier left), `max_steps`, or one of
     `DiscretePlanner`'s own two stop cases -- `goal_reached` (within its goal tolerance of a found goal) or
     `no_feasible_plan` (no heading gets closer to the goal, short of it; `DiscretePlanner.last_stop_reason`).
     It is kept on the episode's `EpisodeTrace`/`EpisodeResult` as `stop_reason`.
4. **Ablations:**
   - `agent.use_gt_detections`: the agent uses the goal objects' ground-truth `bbsgt` instances (area/visibility
     filtered, scored 1.0, `sense.py::target_instances`) instead of Detic's detections. This isolates recognition.

## Metrics (`evaluator.py`)

- **Success:** the final position is within `SUCCESS_DIST_M` (1 m), horizontally, of a goal object's mesh
  footprint (1000 surface samples, top-down).
- **SPL:** success × geodesic distance / max(path length, geodesic distance), using the episode's own
  `geodesic_distance`.
- **Distance to goal (dtg):** the final horizontal distance to the nearest goal footprint.
- **Failure category** (IMPRINT's failure appendix, narrowed down):

  | category | condition |
  |---|---|
  | `success` | success |
  | `exploration_failure` | no detection ever reached `goal_score_thresh`, and no goal was ever visible in a ground-truth frame |
  | `fn_recognition_failure` | no detection ever reached `goal_score_thresh`, but a goal was visible in some ground-truth frame |
  | `fp_recognition_failure` | the agent locked onto a believed goal more than 1 m from every real goal |
  | `tp_early_stop` | the agent locked onto a real goal (within 1 m) but stopped more than 1 m from its footprint |

## Outputs

`habitat_od_search/logs/<run_name>/`:

| file | contents |
|---|---|
| `metrics.jsonl` | one `EpisodeResult` per line, appended as each episode ends: scene, episode, category, rarity tier, success, SPL, steps, stuck, stop reason, path length, geodesic distance, dtg, failure category |
| `summary.md` | success / SPL / dtg / episode count per rarity tier and overall, followed by the share of each failure category; also printed |
| `images/<run_index>_ep<episode_id>_<scene_id>/` | with `save_visuals`: one 3-tile snapshot per step (Detic detections, the belief map cropped to 256 × 256 cells around the agent, ground truth with the goal footprints) and `episode.webp` |
| `images/<run_index>_ep<episode_id>_<scene_id>_last_snapshot.png` | the episode's last snapshot |
| `meshes/<scene_id>.glb` | with `save_mesh`: the scene's ground-truth mesh (stage + every placed object, ceiling stripped), written once per scene |

## Enhanced detector (`enhanced_detector.py`)

This runs on one target in one scene (`--scene`, and `--category`, which defaults to the rarest tier's first
single-instance category) and writes to `habitat_od_search/logs/enhanced_detector/<scene_id>/<category>/`:

1. **Query views:** every other object is removed and the navmesh is recomputed. The camera is aimed at the target
   from 8 bearings, 1.5 m outside its footprint, at 3 camera heights and 3 yaw offsets (±30°). Up to 8 views that
   pass the ground-truth area/visibility filters are kept. Each one's ground-truth box is embedded with
   `Detector.box_embeddings` and saved to `query_features.npz`, with crops in `query_crops/` and the mosaics
   `target_views.png` / `target_frames.png`.
2. **Collected boxes:** the full scene is restored, and 100 random poses are drawn within 3 m of 5 anchors (the
   target-class objects, or k-means centroids of them, topped up with farthest-point navigable points) and pointed
   at them, with up to ±45° jitter. Every class-agnostic `Detector.predict` box is embedded and saved to
   `collected_features.npz`, with crops in `collected_crops/`.
3. **Web images:** 8 Bing images are downloaded for the full category string. The largest `predict_target` box in
   each is embedded and saved to `scraped_features.npz` (crops in `scraped_crops/`).
4. **Matching:** `query_matches.png` shows the top 5 collected boxes for each query. The queries are each
   simulator view, each web image, the mean and max ensembles of both, and Detic's text embedding of the
   category.

## Running

```bash
export HABITAT_DATA=$HOME/habitat_data/data BASE_DIR=$(pwd) PYTHONNOUSERSITE=1
PYTHONPATH=. python habitat_od_search/main.py --config habitat_od_search/config/hssd_rare_eval.yaml \
    --config-file third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
    --checkpoint third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
PYTHONPATH=. python habitat_od_search/enhanced_detector.py --scene 102344094
```

Each run's config is a copy of `config/hssd_rare_eval.yaml` with its own `run_name`. The `eval_200_{gt,vanilla}`
configs in `logs/configs/` run 200 episodes with `max_steps` 2000 and `resolution_m` 0.05, with and without
`use_gt_detections`.
