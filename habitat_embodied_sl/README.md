# Retrieval datasets from semantic mapping with class-agnostic detection

Retrieval datasets are built in two stages, each with its own config, one scene per call each, so calls for different
scenes run in parallel, each on the GPU its caller's `CUDA_VISIBLE_DEVICES` names. Stage 1, `collect_train.py`, needs the
simulator and records each train scene raw (1.8) under `<raw_root>/<run_name>/<scene>/`, as its collection config
(`config/procthor8_collection.yaml`: image sizes, ground-truth filters, detector, fusion, budgets) says; a raw run's
records share that config, saved as `<raw_root>/<run_name>/config.yaml`. Stage 2, `build_dataset.py`, builds the scene's
shard of each of its variants' datasets, `datasets/<run_name>/`, from that record alone (steps 2 to 5, run once whatever
the number of variants), as its dataset config (`config/procthor8_build.yaml`: the raw run, the variant or a list of
them, resolve and retrieval settings) says, saved in full, with its one variant, as `datasets/<run_name>/config.yaml`;
each variant, and any other setting, is another dataset under its own `run_name`. It
maps each train scene: `common/hybrid_exploration`'s `HybridFrontierAgent` explores the scene in exhaustive mode, while a
class-agnostic `Detector` (`common/vision/detector.py`) segments every frame, and its detections are lifted to 3D and
fused into concept-graphs objects (`common/utils/concept_graph.py`). The build resolves, filters and classifies them and
retrieves diverse views of each object into its variant's train split. The test scenes skip the raw record:
`collect_test.py` collects them straight into their own COCO test set of fully annotated frames (step 6),
`datasets/<run_name>/test/`, as its test config (`config/procthor8_test.yaml`) says, shared by every variant's dataset.
Training configs name these splits as registered detectron2 datasets (`common/env_utils/dataset_registry.py`).

The target vocabulary (`object_params.vocab_name`, narrowed by `object_params.filter_classes`) is used only in the
classification step (4). Every earlier step is class-agnostic.

This outline describes the algorithm independently of the environment (`object_params.env_name`) and its vocabulary. Keep
environment- and dataset-specific details, such as how a vocabulary's class names are written or where its size tables
come from, out of it; they belong with the code that builds them (`common/env_utils/`).

## Algorithm outline

### 1. Per step: detect, lift, fuse, steer

- (1.1) **Observe.** Every sensor (rgb, depth, semantic) renders at `input_image_size` (1024) square. The step's rgb, its
  depth resized to `lift_image_size` (512) square by nearest neighbour, and the depth-sensor pose become a `Frame`,
  recorded (1.8) under a frame id, the step's index. Every depth-based step (lifting, raytracing) runs at
  that lift resolution; detection runs on the full-resolution rgb. Where no geometry renders (through windows,
  outdoors), the rgb is cleared to light blue (`SKY_RGB`, `common/env_utils/env_base.py`), not black.
- (1.2) **Detect** (`Detector.predict`) on the rgb, which Detic resizes to a `input_image_size` shortest edge and SAM to
  its own 1024 long side:
  - (1.2.1) Detic's CenterNet2 proposals are refined through every cascade stage's class-agnostic box regression.
  - (1.2.2) Each box is scored by its proposal objectness only. Boxes under `detector.score_thresh` (0.4, so a small object at
    the edge of the view, like an egg at 0.49, still counts) are dropped, then one class-agnostic NMS at `detector.nms_thresh`
    (0.5) runs, keeping Detic's top-k per image.
  - (1.2.3) Each box gets a region embedding: its last-stage box-head features projected into CLIP text space, normalized.
  - (1.2.4) Masks come from SAM (`vit_b`) prompted with the boxes, or from Detic's mask head when `detector.sam_checkpoint` is null,
    binarized at `detector.mask_thresh` (0.5).
  - (1.2.5) `without_parts` (drops masks lying ≥ 80% inside a larger one) exists but is currently disabled.
- (1.3) **Lift** (`predicted_objects`, `project_detections_to_objects`):
  - (1.3.1) Resize each mask to the frame's depth by nearest neighbour, the same pixels the depth was resized with.
  - (1.3.2) Drop detections whose mask covers less than `_MIN_LIFT_FRACTION` (5e-4) of the frame's pixels with valid depth.
  - (1.3.3) Back-project each mask's in-range depth pixels to world points.
  - (1.3.4) **Floor:** drop points at most 2 cm above the floor (agent height minus the navmesh floor offset).
  - (1.3.5) **Background walls:** peel up to 3 RANSAC planes (2 cm inlier distance). A plane is removed only if it is vertical
    (|normal y| ≤ 0.1) and its median camera distance is at least 10 cm beyond the 95th percentile of the other points'
    distances. Flat objects such as paintings hang only a few cm off their wall, so they stay.
- (1.4) **Fuse** (`add_processed`, into the list of fused objects):
  - (1.4.1) Skip detections with fewer than 10 points (`min_points_threshold`, lowered from concept-graphs' 16 for small objects).
  - (1.4.2) **Denoise the detection:** downsample to 2.5 cm voxels. Keep the largest tight DBSCAN cluster (eps 5 cm, min 5),
    plus any small clusters (< 15% of its size) within 10 cm of it. This keeps thin parts and drops mask spill onto a
    neighbouring object.
  - (1.4.3) **Match:** score the detection against each existing object. The spatial term is its 3D IoU with the part of that
    object visible in this frame (object points near the frame's downsampled depth cloud). The match score is that IoU
    plus the cosine similarity of their region embeddings, or 0 when the IoU is under 0.1. The detection merges into
    the best-scoring object, or becomes a new object if every score is 0.
  - (1.4.4) **Protect smaller objects:** before a merge or add, drop the detection's points within 2.5 cm of any smaller object
    other than the target, so a large object can't absorb what lies on it. A new object still needs ≥ 10 points.
  - (1.4.5) **Absorb covered parts** (`_covered_parts`, `_absorbed`): a smaller object at least half of whose points
    (`absorb_covered_fraction` 0.5) lie within 2.5 cm of a merged detection's is a part of the target object it was
    split into, not something lying on it, which its detections' masks rarely include (at most 0.3 of an object on a
    counter top, all of a fridge's side panel or the stickers on its door). Once the frame's detections are matched, each
    such part merges into the target.
  - (1.4.6) **Merge:** points are voxel-merged, each point keeping a count of the detections it appeared in. Region embedding
    and objectness are averaged, weighted by detection counts. Detected frames are unioned.
  - (1.4.7) **Merge overlapping objects:** a pair merges when overlap ratio > 0.5, embedding similarity > 0.5 and their
    sum > 1.2. Pairs merge greedily, highest sum first.
- (1.5) **Steer.** The step's lifted detections (1.3) also go to `agent.update_detections`, whose `ObjectTracker`
  (`common/hybrid_exploration/object_tracker.py`) denoises each as in 1.4.2 (detections under 10 points skipped) and
  matches it as in 1.4.3: it merges into the best-scoring tracked object, or is tracked as a new one if every score is 0
  and it has ≥ 10 points. Unlike the map, tracked objects are never merged with each other nor dropped, so each keeps
  the id its seen sectors are stored under. Every tracked object is an `ObjectBelief` of its points, with confidence
  min(detections, 5) / 5 (`_FULL_CONFIDENCE_DETECTIONS`); only objects of confidence ≥ 0.5
  (`HybridConfig.confidence_thresh`) get viewpoints. The agent sees its poses, and the tracked objects' points, in its reference frame (`ref_state`: the start
  position, with the start yaw snapped to 90° so the map grid stays on the world axes). Its belief map (`EstimatedMap`,
  0.1 m cells) marks as obstacles every
  depth point between 0.2 m (`MapConfig.max_climb_m`, habitat-sim's step height) and 0.88 m (`obstacle_height_m`) above
  the floor. Downward rays with no depth hit also count, at their floor-plane crossing within `visibility_dist_m`
  (`missing_floor_points`, e.g. the drop past a terrace edge), after opening the no-hit mask with a
  `missing_floor_opening_px` (5 × 5) kernel in the depth image -- a real void is a solid image region, unlike the
  isolated pixels and 1-px seam cracks of the scanned mesh. Then `agent.act` picks the action: belief-map-only, via
  a shared `DiscretePlanner` FMM pass over a goal map of every unseen object viewpoint (or, once none remain, every frontier cell; frontier
  fragments under 0.5 m, `MapConfig.min_frontier_size_m`, are dropped, as are cells within 0.5 m of the agent,
  `MapConfig.min_frontier_dist_m` -- unless that leaves no frontier at all, then every frontier cell is kept) -- no ground-truth navmesh pathfinding. That pass
  only covers the bounding box of the explored and goal cells and the agent, grown by 2 m (`MapConfig.planning_margin_m`,
  `EstimatedMap.planning_window`), not the whole map. Of the
  agent's 8 headings, it heads for the one whose single 0.25 m move (`DiscretePlannerConfig.step_size_m`) stays on
  traversable cells (obstacles dilated by a 3-cell disk, `DiscretePlannerConfig.obs_dilation_selem_radius`) and ends
  closest to the goal on that FMM distance (the smallest turn on ties). Once exploration ends, the last step's 3-tile
  (observation, belief map cropped to the 5 x 5 m around the agent, ground truth) snapshot is saved as
  the record's `exploration_snapshot.png`, which the build copies to `datasets/<run_name>/<scene>/`. Once the record is
  saved, every recorded step's detections (`load_detections`) and ground truth are read back and drawn on its rgb
  (`segmentation_tile`), tile for tile, to the record's `predictions_mosaic.jpg` and `ground_truth_mosaic.jpg`
  (`save_step_mosaics`, 5 tiles per row; split into `<name>_<part>.jpg` of at most 32 rows each).
- (1.6) **Ground truth:** the step's ground-truth instances come from the `bbsgt` sensor, filtered by
  `object_params.min_area_fraction` (box area ≥ 5e-4 of the frame's pixels) and `filter_low_visibility`/`min_mesh_visibility_fraction`
  (`mesh_visibility_fraction`: the sum of the mask's connected components' own bounding-box areas, over the object's largest
  box face projected at the mask's mean depth),
  and remapped to the kept classes. They don't affect the map; they are kept RLE-encoded, with their object id and
  visibility fraction, for recovery (5.3).
- (1.7) **Stop** when the agent returns `stop`: when its last 3 steps all collided (`HybridConfig.max_successive_collisions`)
  or it gets stuck (no net translation over the last 25 steps, positions within 0.1 m; `BaseExplorationAgent.is_stuck`,
  logged as a warning), or the planner finds no heading that gets closer to the goal -- or when
  `collection.steps_per_scene` (1000) runs out.
- (1.8) **Raw record** (`common/utils/raw_scene.py`): every step's frame, the floor height its detections were lifted
  above and its detections are written to `<raw_root>/<run_name>/<scene>/` as they come, and once exploration ends the
  fused objects, their config, the camera, the agent's map grid (`MapGrid`: explored and obstacle cells and the reference
  pose its frame is relative to), the annotated objects, their `ground_truth.json` (4.5) and their per-frame ground truth
  (1.6), the scene's mesh (what 4.5 tells observable objects with) and, for viewing only, its annotated objects' point
  cloud, then `scene.json`, which
  marks the record complete. Steps 2 to 5 (`build_dataset.py`'s `build_scene`) run on that record alone, without the
  simulator. The run's first call saves the collection config, its `fusion` completed with every `ConceptGraphConfig`
  default and its kept classes as `classes`, as
  `<raw_root>/<run_name>/config.yaml`, and a call with another config or other classes fails (`save_run_config`), so a
  run's records share one: another config is another run, under its own `run_name`. The build classifies among those
  recorded classes and indexes them, so records keep their class ids however `CLASS_LABELS_PROCTHOR100` changes later.

### 2. Resolve (`concept_graph.resolve`)

Resolve and foreground-filter parameters (`ResolveConfig`) are the build's, not the record's: the collection config's
`resolve` section overrides their defaults, and a rebuild applies them to existing raw records. The voxel size is the
recorded fusion one (`ConceptGraphConfig`, in `scene.json`).

- (2.1) Drop objects with fewer than 3 detections.
- (2.2) Drop each object's points that appeared in fewer than 2 detections.
- (2.3) Denoise every object (largest DBSCAN cluster, eps 10 cm, min 10; the original object is kept if fewer than 4 points survive).
- (2.4) Drop objects with fewer than 10 points.
- (2.5) **Shared points:** a point goes to whichever nearby object (within 2.5 cm) was detected there more often. So a part
  keeps its points only where it was detected more often than the whole it belongs to.
- (2.6) Drop objects with fewer than 10 points.
- (2.7) **Merge adjacent parts** (`_merge_adjacent`): parts of one object fused apart (a house plant and its pot, a
  counter's top and its cabinet) fuse into one object. Repeatedly merge the most similar pair that passes both of these:
  - (2.7.1) region-embedding similarity ≥ 0.5 (`adjacent_visual_sim_thresh`; such parts measure 0.54 to 0.56, an object
    on a counter top or a fridge door against it at most 0.25);
  - (2.7.2) some point of one lies within 5 cm of some point of the other.

  The result is the resolved objects, all of class `UNCLASSIFIED`.

### 3. Foreground filter (`concept_graph.keep_foreground`)

- (3.1) **Raytrace:** project every resolved object into every stored frame. Each depth pixel is back-projected and takes
  the object of the nearest resolved point within 2.5 cm. An object counts as visible in a frame if it gets ≥ 5e-4
  (`_MIN_RAYTRACED_FRACTION`) of the frame's pixels and its `mesh_visibility_fraction` is ≥ the collection config's `object_params.min_mesh_visibility_fraction` (0.2), the one the ground truth is filtered with (1.6).
- (3.2) **Detection rate:** the share of the frames showing the object in which it was actually detected.
  Objects under 0.3 are dropped, which removes spurious one-off detections.
- (3.3) **Foreground score:** detection rate × objectness.

### 4. Classify (`concept_graph.classify`)

- (4.1) For every view of an object (each frame it is raytraced visible in), take the region embedding of its raytraced
  mask's box, the mask resized to the rgb by nearest neighbour. Each frame needs one backbone pass.
- (4.2) Each view's class probabilities are the softmax of Detic's temperature-scaled cosine similarity between that
  embedding and the CLIP text embeddings of the kept classes, prompted as `a <class>`.
- (4.3) **Size mask:** each class has a range for volume, shortest edge and largest edge, read
  from its per-class size table (`class_size_ranges`) and widened to [0.5 · min, max / 0.5] (`_SIZE_MARGIN`). A class is plausible for an object when the axis-aligned extent of
  its points lies in that range; classes with no range are always plausible.
- (4.4) The object's class scores are the mean probabilities over its views × its foreground score, i.e.
  p(object) · p(class | object), zeroed for implausible classes. Its class is the argmax. Objects with no view, or with
  no plausible class, are dropped.
- (4.5) **Map files** (`common/utils/eval_utils.py`): the classified objects go to `<scene>/predictions.json`
  (`predictions_dataset`) and the scene's annotated objects (5.2) to `<scene>/ground_truth.json` (`ground_truth_dataset`,
  computed by `collect_train.py` before the environment closes, as its `navigable` needs the simulator, and recorded raw (1.8)
  for the build to write). Each file is a single-scene COCO dataset on the agent's own map grid (its
  `EstimatedMap`, in the reference frame), whose one image holds the grid's size, cell size and RLE masks. The prediction
  file's `navigable` is the explored-and-free cells and `explored` the explored ones. The ground-truth file's `navigable`
  is the ground-truth navmesh resampled onto that grid. Each annotation has its class, `object_id`, a world-axis-aligned
  3D box (`box3d`), its volume as `area`, and its top-down footprint as an RLE `segmentation`. A prediction's box is the
  extent of its points snapped outward to 2 cm, its score its highest class probability (every class's in `class_scores`), and its footprint its points'
  cells with gaps closed and holes filled. A ground truth's box is its mesh bounds and its footprint its mesh's silhouette.
  Only observable annotated objects go to `ground_truth.json` (`observable_ground_truth`): from camera height above every
  navmesh cell, one every 0.5 m (`navigable_viewpoints`), rays within the depth range (10 m) to 27 points of each
  object's box are cast against the scene's mesh (1.8), and an object none of whose rays first hits inside its box (2 cm
  tolerance) is dropped, like a closed fridge's contents. Independent of the agent's trajectory, so a scene's
  unexplored parts still count as missed.
  Both files are drawn by class to `<scene>/estimated_topdown.png` and `<scene>/ground_truth_topdown.png` (`save_topdowns`).
  For viewing in 3D (`build_dataset.py`'s `save_scene_maps`), the classified objects' points go to
  `<scene>/semantic_point_cloud.ply` and their boxes, as wireframes, to `<scene>/semantic_wireframes.glb`, the
  observable ground truth's boxes to `<scene>/ground_truth_wireframes.glb`, all colored by class, and the scene's mesh
  to `<scene>/ground_truth_mesh.glb`; the raw record keeps the annotated objects' point cloud (1.8).
- (4.6) **Map evaluation** (`evaluate_maps.py`, run once every scene is collected; `evaluate_scene_jsons`) reads only those
  files, every scene pooled:
  - (4.6.1) **3D box AP:** COCO AP by 3D box IoU, both boxes first grown to at least 5 cm per axis
    (`_MIN_MATCHED_EXTENT_M`), at IoU 0.25 to 0.95 in 0.05 steps. It reports AP (IoU 0.5:0.95), AP25 and AP50, AP25 per
    volume (small < 0.01 m³ ≤ medium < 0.25 m³ ≤ large), AP25 of each object submitted once per class among its 5 highest `class_scores`, scored by that class's (`AP25 top5`,
    also per volume; `AP25` is top-1), and per-class AP25, AP50 and AP.
  - (4.6.2) **TIDE error breakdown** (arXiv 2008.08115, `tide_daps`) at 3D box IoU ≥ 0.25 (foreground) of each
    object's top-1 class. Each object COCO matching leaves unmatched gets one error, checked in order: `Loc` (its class,
    IoU in [0.1, 0.25)), `Cls` (another class, IoU ≥ 0.25), `Dupe` (its class, IoU ≥ 0.25 with already matched ground
    truth), `Bkg` (IoU < 0.1 with every ground truth), else `Both`; unmatched ground truth no `Cls`/`Loc` error targets
    is `Miss`. Each type's `dAP25/<error>` is the AP25 gained once an oracle fixes all its errors (`Cls`: relabeled,
    `Loc`: given its target's box, either dropped if a higher-scored object or a match already takes that target;
    `Both`/`Dupe`/`Bkg`: dropped; `Miss`: that ground truth dropped), negative gains clamped to 0, and `dAP25/FP` and
    `dAP25/FN` that of dropping every unmatched object or ground truth, drawn to `tide.png`.
  - (4.6.3) **Class-agnostic matching:** each scene's objects, by descending score, greedily take the unmatched ground truth
    of highest 3D box IoU, if ≥ 0.25. It prints a per-scene table of every object's match and of the unmatched ground
    truth, and draws a confusion matrix (`confusion_matrix.png`; unmatched objects and ground truth fall in "background").
  - (4.6.4) All of 4.6.1's and 4.6.2's metrics but the per-class AP go to one flat dict, `metrics.json`.

### 5. Retrieval and train sets

- (5.1) **Retrieval** (`common/utils/retrieval.py::retrieve_resolved_views`): for each object, pick up to `retrieval.k` (10)
  views greedily so they cover its (resolved point, viewing angle) elements, the angle from the point to the camera
  (not the agent's heading). A view's gain is visibility^α × Σ (1 − γ) γ^(effective times already covered), with
  γ = 0.1 and α = 1; a point already covered by a chosen view lends credit to a new candidate's angle for it in
  proportion to a Gaussian kernel of std σ = π/4 over their angular distance, so nearby angles share credit instead of
  only exact repeats. Each view's raytraced mask is resized to the rgb by nearest neighbour and labelled with the
  object's class.
- (5.2) **Ground truth:** every placed scene object annotated with a kept class, boxed by its world mesh bounds. Resolved
  objects are matched to it class-agnostically, greedily by score, at 3D box IoU ≥ 0.25.
- (5.3) **Recovery** (`common/utils/ground_truth_recovery.py`) of the retrieved views. It needs only the ground truth's 3D
  boxes and its RLE-encoded masks (1.6), no mesh:
  - (5.3.1) `recover_drop_fp`: views of objects matching no ground truth are dropped.
  - (5.3.2) `recover_gt_classes`: views of matching objects take their ground truth's class.
  - (5.3.3) `recover_add_fn`: every unmatched ground truth (`missed_ground_truth`) given a class gets up to
    `retrieval.k` of its masks as views, under the negative object id -(its index + 1). They are picked
    greedily as in 5.1, with one element per view, the viewing angle of its camera around the box center, weighted by its
    visibility fraction^α. Other objects' views are left as they are, so an absorbed false negative's pixels stay in its
    absorber's masks too.
  - (5.3.4) `recover_gt_masks`: views of matching objects take their ground truth's mask in their frame, so its box too;
    a matching object's view in whose frame its ground truth shows no mask is dropped. Other objects' views are kept.
  - (5.3.5) `pseudo_classes`: classifies the unmatched ground truths as 4 classifies resolved objects, from all their
    masks (every frame of 1.6 showing them) and their box's extent in place of the points': mean view probabilities
    (`plausible_class_probs`, shared with 4.4) zeroed for the classes their size does not fit, argmax. One with no plausible
    class gets no class, so no views. No foreground score, which scales every class alike.
- (5.4) **Train sets:** each object-view train set derives from the retrieved views on its own (`build_dataset.py`'s
  `SceneRecovery`), named `V<i>_` and the fixes applied to the pseudo-labels (`drop_add`: the objects; `cls_box`: their
  labels):
  - (5.4.1) `V1_pseudo`: as classified (5.1);
  - (5.4.2) `V2_drop_add`, the objects fixed and their labels pseudo: 5.3.1, then 5.3.3 with 5.3.5's classes
    (`drop_add_views`);
  - (5.4.3) `V3_cls_box_drop_add`, the objects and their labels fixed: 5.3.1, 5.3.2, 5.3.4, then 5.3.3 with the ground
    truths' classes (`cls_box_drop_add_views`). Its added views are 5.4.2's, relabelled.

  A fourth, `V4_full`, is no object-view set: its dataset's `train/` holds every frame some object-view variant's view
  is in, every ground-truth instance of each annotated (`ground_truth_frame`), the same frames whichever variant it is
  built beside; `main.py` trains on them, and every run scores them as its train set.

### 6. Test set (`collect_test.py`, `evaluation_set.py`), per evaluation scene, no raw record

- (6.1) **Explore:** a vanilla `FrontierAgent`, revealing fog of war out to `frontier_agent.visibility_dist_m` (3.0),
  explores the whole scene: until it stops or `frontier_agent.steps` (1000) run out. Its final map and trajectory are
  saved as the test set's `<scene>/decision_frame.png`.
- (6.2) **Strided frames:** the frame of every `frame_stride`-th step (5, turns included) is kept if it shows some
  ground truth (filtered and remapped as in 1.6). No other selection: a scene contributes in proportion to its
  exploration, and every view the agent sees is as likely to be kept.
- (6.3) The kept frames go straight to the COCO test set, all their ground-truth instances as annotations, each keeping
  its simulator object id. One test set is shared by every variant's dataset; changing 6.1 or 6.2 means recollecting it
  (minutes per scene).

Fusion and tracking thresholds live in `ConceptGraphConfig` (overridden by the collection config's `fusion`), resolve and
foreground ones in `ResolveConfig` (overridden by the dataset config's `resolve`), each saved in full with its run, the lifting and size-mask constants (`_MIN_LIFT_FRACTION`,
`_SIZE_MARGIN`) at the top of `common/utils/concept_graph.py`, the steering one (`_FULL_CONFIDENCE_DETECTIONS`) at the top
of `common/hybrid_exploration/object_tracker.py`; floor, wall and raytracing
thresholds (`_MIN_RAYTRACED_FRACTION`) at the top of `common/utils/projection_utils.py`; image sizes, ground-truth filters,
the detector's, fusion overrides and budgets in the collection config; the variant, resolve and retrieval overrides in
the dataset config; the test exploration and frame stride in the test config.

## Outputs

Each dataset, `datasets/<run_name>/` (`<dataset_root>/<run_name>/`, both from its config, `datasets`), holds one split
(`common/utils/dataset_writers.py`): an object-view variant's `object_ds/` (`V1_pseudo`, `V2_drop_add`,
`V3_cls_box_drop_add`), `V4_full`'s `train/` of fully annotated frames, or a test set's `test/`; its frames are in its
`images/`, and the split folder holds its json, its mosaic, its class counts and a `scenes/` folder of per-scene shards.
`common/env_utils/dataset_registry.py` registers each split with detectron2 as `<run_name>_<split>`, e.g.
`procthor8_V2_drop_add_object_ds`, `procthor8_V4_full_train` or `procthor8_test_test`, the names training configs give
in `DATASETS.TRAIN` and `DATASETS.TEST`. A `collect_train.py` call whose scene already has a complete raw record skips it,
unless `--overwrite`; otherwise it first deletes the scene's record, then records it; `collect_train.py` writes nothing under
`datasets/`. A `build_dataset.py` or `collect_test.py` call first deletes what an earlier one wrote for its scene (its
`<scene>/` folder, shard, frames, retrieval mosaics and their parts, semantic top-down map), leaving the other scenes'
alone, then writes its scene's records to `<split>/scenes/<scene>.json`. Calls for different scenes touch different
files, so they run in parallel without locking. Once every scene is written, `merge.py` pools the dataset's shards, in
scene name order, into its json, mosaic and class counts; rerun it after rewriting a scene. A dataset then holds:

| file | contents |
|---|---|
| `config.yaml` | the dataset's config (a dataset config with the raw run's `classes`, or a test config with its kept `classes`), saved by its first call; a call with another config fails (`save_run_config`) |
| `images/<scene>_<frame>.jpg` | every frame of the split: the train scenes' frames some view is in, or the test frames |
| `object_ds/object_ds.json` | `info` (`categories`, the dataset's `config`), `frames` (`"<scene>_<frame>"`: height, width) and `retrieved_objects` (`"<scene>/<object>"`: `gt_class`, the variant's label, and `views`, each a frame and an RLE mask at the rgb's size); read by `common/utils/object_views_dataset.py`'s `ObjectViewsDataset` |
| `train/train.json`, `test/test.json` | COCO instance segmentation of the train frames (every frame some variant's view is in and some ground truth shows) (`V4_full`) or of the test frames (a test set), `info` the dataset's `config`, with the same ground truth (`sense.ground_truth_instances`): `images` (`file_name` relative to `images/`), `annotations` (RLE `segmentation`, `bbox`, `area`, `iscrowd`, plus `object_id`, the simulator object id), `categories` (ids 1..N, the kept classes in order); image and annotation ids start at 1 too. Loads with detectron2's `register_coco_instances(name, {}, "<run_name>/test/test.json", "<run_name>/images")` |
| `<scene>/exploration_snapshot.png`, `<scene>/decision_frame.png`, `<scene>/images/planner/` | a train scene's last exploration snapshot (1.5), copied from its raw record; a test scene's frontier exploration map and trajectory (6.1), and its planner's dump folder |
| `<scene>/predictions.json`, `<scene>/ground_truth.json`, `<scene>/estimated_topdown.png`, `<scene>/ground_truth_topdown.png` | each train scene's classified objects and its annotated objects as single-scene COCO datasets on the agent's map grid (4.5), and their top-down maps by class |
| `<scene>/semantic_point_cloud.ply`, `<scene>/semantic_wireframes.glb`, `<scene>/ground_truth_wireframes.glb`, `<scene>/ground_truth_mesh.glb` | each train scene's classified objects' points and box wireframes, its observable ground truth's box wireframes, colored by class, and its mesh (4.5) |
| `map_eval/metrics.json`, `map_eval/tide.png`, `map_eval/confusion_matrix.png` | `evaluate_maps.py`'s scores of every train scene's map, pooled (4.6) |
| `semantic_topdown_maps/<scene>.png` | each train scene's resolved objects the variant keeps, by their class in the variant, on the agent's top-down map; objects `recover_add_fn` adds have no points to draw |
| `retrieval_mosaics/<scene>.jpg` | each train scene's objects in the variant, one row of its retrieved views per object, tiles at half resolution (`_TILE_DOWNSCALE`); objects recovery relabelled (5.3.2) or added (5.3.3) are bordered red. Split into `<scene>_<part>.jpg` of at most 32 objects each (`MAX_MOSAIC_ROWS`) unless `--no-split` |
| `retrieval_mosaics/<scene>_first_views.jpg` | the same objects, one row of each object's first retrieved view (its highest-gain one, 5.1) at full resolution, bordered red as above |
| `object_ds/scenes/<scene>.json`, `train/scenes/<scene>.json`, `test/scenes/<scene>.json` | one scene's records of the split, which merging pools into its json: `frames` and `retrieved_objects` for an object-view set; `images` and `annotations` (each pointing at its image by its index in the shard, ids assigned when pooled) for the COCO sets |
| `object_ds/object_ds_mosaic.jpg`, `train/train_mosaic.jpg`, `test/test_mosaic.jpg` | 32 records of the split, read back from its json and drawn in random order, each its frame with its annotations' boxes and classes |
| `object_ds/object_ds_class_counts.png`, `train/train_class_counts.png`, `test/test_class_counts.png` | the split's annotation count per class (one per view for a variant), stacked by scene on a log scale, written when the split is merged; the static test sets (`nyu_depth_v2.py`, `sunrgbd.py`) get theirs too |

Each train scene's raw record, `<raw_root>/<run_name>/<scene>/` (`raw_root` from the collection config, `raw/` in the repo, gitignored;
about 0.3 GB per 1,000-step train scene), lossless so a build from it is a build from `collect_train.py`'s own memory:

| file | contents |
|---|---|
| `../config.yaml` | the collection config every record of the run was made with, its `fusion` completed with every `ConceptGraphConfig` default, and its kept `classes`, which the ground truth's class ids index (1.8); the build reads a train scene's fusion and visibility settings back from it |
| `frames/<frame>.png`, `frames/<frame>.npz` | each step's rgb, and its depth (at `lift_image_size`), depth-sensor pose and floor height (`floor_height`, what replaying its detections' fusion needs) |
| `detections/<frame>.npz` | each train step's detector instances: every field as an array, its masks' RLE counts (`mask_counts`, `mask_size`) |
| `fused_objects.npz` | the fused `ConceptGraphObject`s: points, per-point detection counts, region features, objectness, detection counts and detected frames, ragged fields concatenated with `point_offsets` / `frame_offsets` |
| `scene.json` | the scene name and its depth camera; written last, it marks the record complete |
| `ground_truth_objects.npz`, `ground_truth_views.json` | a train scene's annotated objects (class, mesh box, object id), and every recorded frame's ground truth (1.6: frame, object id, class, visibility, RLE mask) |
| `map_grid.npz` | a train scene's `MapGrid`, the grid `predictions.json`'s footprints and the top-down maps are drawn on |
| `ground_truth.json` | a train scene's annotated objects on that grid (4.5), its `navigable` the simulator's navmesh; the build writes it to `<scene>/ground_truth.json` |
| `ground_truth_mesh.glb` | a train scene's mesh (stage and placed objects, `scene_world_mesh`), which the build casts line-of-sight rays against to drop unobservable annotated objects from `ground_truth.json` (4.5) |
| `ground_truth_point_cloud.ply` | for viewing only, not read by the build: a train scene's annotated objects' surfaces sampled at 1,600 points/m², colored by class like `semantic_point_cloud.ply` |
| `exploration_snapshot.png`, `images/planner/` | the scene's last exploration snapshot (1.5), and the planner's dump folder |
| `predictions_mosaic.jpg`, `ground_truth_mosaic.jpg` | every recorded step's detector predictions and ground truth drawn on its rgb, tile for tile (1.5), split into `<name>_<part>.jpg` of at most 32 rows each |

## Files

| file | contents |
|---|---|
| `build_dataset.py` | stage 2: one train scene's build from its raw record (1.8) into each of its variants' datasets, without the simulator: `build_scene` (steps 2 to 5: resolve → foreground → size-masked classify, retrieval and recovery, once) and, for each variant, `write_variant` into its shard of the variant's object views, or of `V4_full`'s frames, with its `predictions.json`, top-downs and mosaics. The layout and the writers are `common/utils/dataset_writers.py`'s |
| `collect_train.py` | stage 1: one train scene's raw record (1.8): the exploration loop and `SceneMapper`, which records each step and fuses its detections (step 1) |
| `evaluate_maps.py` | scores every train scene's `predictions.json` against its `ground_truth.json`, pooled, into `map_eval/` (4.6) |
| `run.sh` | runs, one per GPU of `GPUS` at a time, longest first, the source and NORM baselines (`adapt.py`), Mean-Teacher and embodied SL on each variant (`main.py`), each scored on the procthor8 test and train sets and NYU Depth V2, into `outputs/procthor8_runs/<run>.log` |
| `merge.py` | pools a dataset's per-scene shards into its split's json, mosaic and class counts (`object_ds/object_ds.json`, `train/train.json` or `test/test.json`), once every scene's `build_dataset.py` or `collect_test.py` call is done |
| `collect_test.py`, `evaluation_set.py` | one test scene's frontier exploration (6.1), its strided frames (6.2) written straight into the test set's dataset (6.3), no raw record |
| `config/procthor8_collection.yaml` | the collection config (`run_name` `procthor8`, `raw_root`, `seed`, image sizes, `object_params`, `detector`, `fusion` overrides, `collection` budget): the first 16 ProcTHOR-Train houses of 2 to 3 rooms, 1000 steps each, ProcTHOR-Train-5, -6, -9, -11, -14, -26, -30, -32, -33, -36, -38, -40, -41, -44, -45 and -47; collected one scene per call (Running) |
| `config/procthor8_build.yaml`, `config/procthor8_test.yaml` | the dataset config of `procthor8_collection.yaml`'s raw run (`run_name` `procthor8_${variant}`, `raw_run`, `dataset_root`, `variant`, `resolve` and `retrieval` overrides), one dataset per variant; and the test config (`run_name` `procthor8_test`, `dataset_root`, `seed`, image size, `object_params`, `frontier_agent`, `frame_stride`) of its test set, ProcTHOR-Val-0 to -15 |
| `meta_arch.py` | `CustomFT_Detic`, Detic's `CustomRCNN` registered as a `META_ARCHITECTURE` for fine-tuning its ROI heads on a train variant's `ObjectViewsDataset`, on the proposals of a frozen teacher copy of the checkpoint's backbone and proposal generator; trained by `main.py` |
| `main.py` | the fine-tuning of `CustomFT_Detic` on the train split its config names in `DATASETS.TRAIN`, scored on its `DATASETS.TEST` (optimizer: Detic's, honoring `SOLVER.OPTIMIZER` and `SOLVER.BACKBONE_MULTIPLIER`; training frames resized and flipped as `INPUT.MIN_SIZE_TRAIN` and `INPUT.RANDOM_FLIP` say), into `logs/<--run-name, or the config-file stem>/`: `results.json` (last and best evaluation), `metrics.png`, `vocabulary_tsne.png`, `class_ap.png` (per-class AP50 at each evaluation, written in its cell), `vocabulary_similarity.png` (the classes' CLIP cosine similarity × 100, written in its cell), `class_table.csv` (per class: training objects and views, test instances, first, best and last AP50), `box_areas.png` (train and test box sizes), confusion matrices |
| `config/r50_detic.yaml` | `CustomFT_Detic` on Detic's R50 checkpoint (`Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth`), whose FrozenBN backbone the adaptation baselines adapt: the source model, base of every `r50_*` config, its `DATASETS.TEST` procthor8's test set (whose classes are the training classes), its `V4_full` train frames and NYU Depth V2 |
| `config/r50_procthor8_embodied_sl.yaml`, `config/r50_procthor8_embodied_sl_{V1_pseudo,V2_drop_add,V3_cls_box_drop_add,V4_full}.yaml` | the reference fine-tune: AdamW 1e-4, the student backbone at × 0.1, training frames at the test scale (800) and never flipped, seed 0, 1,000 iterations evaluated every 500; one config per variant dataset, naming its split in `DATASETS.TRAIN`. Object views are drawn class-balanced (`balanced_loader`), `V4_full`'s frames uniformly (detectron2's train loader) |
| `config/r50_procthor8_embodied_sl_V4_full_bgneg.yaml` | `V4_full` with Detic's own `DeticCascadeROIHeads`, which also train on background proposals, instead of `PositiveOnlyDeticCascadeROIHeads`: a training setting, sound only where every object of a frame is labelled |
| `mean_teacher.py`, `config/r50_procthor8_unsup_mean_teacher.yaml` | the Mean-Teacher baseline on the reference's settings: `main.py`'s training with its labels replaced by a moving-average teacher's detections (score ≥ 0.5) on the `V3_cls_box_drop_add` variant's frames, the student trained on a photometrically augmented copy, the teacher (momentum 0.999) evaluated and reported, into `logs/<--run-name, or the config-file stem>/` |
| `DATASETS.TEST` (`main.py`, `adapt.py`, `mean_teacher.py`) | the registered splits a run scores (`common/env_utils/dataset_registry.py`, `<run_name>_<split>`): `adapt.py` once, `main.py` and `mean_teacher.py` the first at every evaluation and every one at the last only (their initial scores are the source model's), each in its own vocabulary: the classifier is reset to the split's class names' CLIP embeddings, then back to the training classes, the first's. Logged as `<split>/bbox/...` in `metrics.json`, and under the first's results in `results.json` |
| `nyu_depth_v2.py` | NYU Depth V2's labeled subset (or `--frames` of it) as such a test set: one box per (class, instance) pair of a frame, `static_vocabulary.py`'s classes but wall, floor, ceiling and unknown (47 classes, 21,972 boxes of the 1,449 frames) |
| `sunrgbd.py` | SUN RGB-D's 2D boxes (`SUNRGBDMeta2DBB_v2.mat`), every frame but the NYU Depth V2 ones it includes (or `--frames` of it), as such a test set: each box as its filled rectangle, clipped to the image, `static_vocabulary.py`'s classes but wall, floor, ceiling, otherprop and unknown |
| `static_vocabulary.py` | a static test set's classes: raw names grouped by canonical spelling (lowercased, `_`/`-` as spaces, a trailing plural `s` dropped), each named by its most frequent spelling, kept with at least 100 objects (`MIN_OCCURRENCES`); the other classes' objects are left unannotated |
| `experiment_readme.md` | the plan of the embodied-SL experiments, their names, status and results |
| `adapt.py` | the adaptation baselines below: the source model adapted by `--method`, scored once on the test set into `logs/<--run-name, or the config-file stem and method>/` (`results.json`, confusion matrices, `adapt.yaml`); run as `r50_procthor8_source` and `r50_procthor8_unsup_norm` |
| `config/r50_procthor8_unsup_norm.yaml` | NORM's hyperparameters |

## Adaptation baselines (`adapt.py`)

Unsupervised baselines for the fine-tuning: no label, and no training step. Each starts from the source model,
`CustomFT_Detic` as `main.py` builds it from `--config-file` before any training step, and adapts only the R50
backbone's FrozenBatchNorm2d layers (`common/vision/test_time_adaptation.py`), in the student and its teacher copy alike,
but `source_layers` (the ResNet stem's `bottom_up.base.bn1`, as Yoo et al. 2024), kept at their source statistics.
The test set is then scored once, in scene order.

- **`source`**: no adaptation; one test frame at a time. It must match `main.py`'s iteration-0 AP on the same config.
- **`norm`** (NORM, Schneider et al. 2020, as in Yoo et al. 2024): each layer normalizes a test batch of B frames, at
  most `test_batch_size` (4) of one scene, by the source running statistics and the batch's own, the batch's weighted
  B / (`prior_strength` + B), `prior_strength` = 128. Nothing is kept from one batch to the next.

Each BatchNorm layer has per-channel affine γ, β and running statistics μ_s, σ²_s, loaded from the checkpoint (the
source statistics); x is its input on a batch, of shape (B, C, H, W), mean and var taken per channel over B, H, W.

```
norm(model, test frames):
    replace every FrozenBatchNorm2d but source_layers by an AdaptiveBatchNorm2d with its γ, β, μ_s, σ²_s
    batches ← test frames in scene order, consecutive frames of one scene, at most test_batch_size (4) per batch
    for batch in batches:
        w ← B / (prior_strength + B), B = |batch|, prior_strength = 128
        run inference on batch; at every AdaptiveBatchNorm2d, on its input x:
            μ ← (1 − w) · μ_s + w · mean(x)
            σ² ← (1 − w) · σ²_s + w · var_unbiased(x)
            y ← γ · (x − μ) / sqrt(σ² + ε) + β                             # μ_s, σ²_s never change
        score the batch's detections
```

## Running

```bash
export BASE_DIR=$(pwd) PYTHONNOUSERSITE=1  # HABITAT_DATA already exported, see INSTALL.MD
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python habitat_embodied_sl/collect_train.py <scene> habitat_embodied_sl/config/procthor8_collection.yaml
```

`<scene>` is a train scene; both the renderer and the models run on the GPU `CUDA_VISIBLE_DEVICES` names. Each scene is
its own call; calls for different scenes can run at once, e.g. one per GPU:

```bash
scenes=(ProcTHOR-Train-5 ProcTHOR-Train-6 ProcTHOR-Train-9)
for gpu in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. nohup python habitat_embodied_sl/collect_train.py ${scenes[$gpu - 1]} \
      habitat_embodied_sl/config/procthor8_collection.yaml > collect_${scenes[$gpu - 1]}.log 2>&1 &
done
```

Then build every recorded scene into each variant's dataset from its raw record (1.8), without the simulator, Detic
classifying on the `CUDA_VISIBLE_DEVICES` GPU; the variant, like any of the dataset config's keys, can be set on the
command line, and a list of them builds every listed variant's dataset from one resolve, classification and retrieval.
A scene can be rebuilt alone into the same dataset, one variant or several; a changed step 2 to 5 setting is another
dataset, with its own `run_name`:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python habitat_embodied_sl/build_dataset.py <scene> habitat_embodied_sl/config/procthor8_build.yaml \
    variant=[V1_pseudo,V2_drop_add,V3_cls_box_drop_add,V4_full]
```

The test scenes (ProcTHOR-Val-0 to -15) go straight into the test set, one call each (6):

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python habitat_embodied_sl/collect_test.py <scene> habitat_embodied_sl/config/procthor8_test.yaml
```

Once every scene of a dataset is written, pool its shards into its split's json, mosaic and class counts, and score the
train scenes' maps from their `predictions.json` and `ground_truth.json` (4.6, the same in every variant's dataset):

```bash
for dataset in procthor8_V1_pseudo procthor8_V2_drop_add procthor8_V3_cls_box_drop_add procthor8_V4_full procthor8_test; do
  PYTHONPATH=. python habitat_embodied_sl/merge.py datasets/$dataset
done
PYTHONPATH=. python habitat_embodied_sl/evaluate_maps.py datasets/procthor8_V3_cls_box_drop_add
```

`tests/test_collect.py` runs every stage end to end, collect, build of every variant, test collection, merge and map
evaluation, on one train scene (ProcTHOR-Train-0) and one test scene (ProcTHOR-Val-7) into `tests/testdump/test_collect/`
(its three configs as `collection.yaml`, `build.yaml` and `test.yaml`, raw records in `raw/`, datasets in `datasets/`),
printing that train scene's map metrics, and checks that rebuilding from the raw record writes the same shard:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -m pytest tests/test_collect.py -s
```

Once the datasets are merged, the R50 fine-tuning on one variant and one adaptation baseline (`--method source` or
`norm`) run with the training configs, whose `DATASETS` name the registered splits:

```bash
PYTHONPATH=. python habitat_embodied_sl/main.py --config-file habitat_embodied_sl/config/r50_procthor8_embodied_sl_V3_cls_box_drop_add.yaml
PYTHONPATH=. python habitat_embodied_sl/adapt.py --config-file habitat_embodied_sl/config/r50_detic.yaml --method norm \
    --run-name r50_procthor8_unsup_norm
```

The source baseline is the same `adapt.py` call with `--method source --run-name r50_procthor8_source`, and
Mean-Teacher `python habitat_embodied_sl/mean_teacher.py` with its defaults.
