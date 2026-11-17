# Retrieval datasets from semantic mapping with class-agnostic detection

`collect.py` builds retrieval datasets from a collection config (`config/procthor8_collection.yaml`), one scene per
call, so calls for different scenes run in parallel, each on its own GPU, into the same `datasets/<run_name>/`. Each
call's `split` argument (`train` or `test`) says where its scene goes; the config lists no scenes. It maps
each train scene: `common/hybrid_exploration`'s `HybridFrontierAgent` explores the scene in
exhaustive mode, while a class-agnostic `Detector` (`common/vision/detector.py`) segments every frame, and its detections are lifted to
3D and fused into a concept-graphs `ConceptGraphMemory` (`common/utils/concept_graph.py`). When exploration ends, the map
is resolved, filtered and classified, and diverse views of each object are retrieved into three object-view train sets.
The test scenes give a COCO test set of fully annotated frames shared by all three. A scene goes to train or test,
never both: a call refuses a scene whose shard is already in the other split.

The target vocabulary (`object_params.vocab_name`, narrowed by `object_params.filter_classes`) is used only in the
classification step (4). Every earlier step is class-agnostic.

This outline describes the algorithm independently of the environment (`object_params.env_name`) and its vocabulary. Keep
environment- and dataset-specific details, such as how a vocabulary's class names are written or where its size tables
come from, out of it; they belong with the code that builds them (`common/env_utils/`).

## Algorithm outline

### 1. Per step: detect, lift, fuse, steer

- (1.1) **Observe.** Every sensor (rgb, depth, semantic) renders at `--input-image-size` (1024) square. The step's rgb, its
  depth resized to `--lift-image-size` (512) square by nearest neighbour, and the depth-sensor pose become a `Frame`,
  stored in the `ImageMemory` under a frame id. Every depth-based step (lifting, raytracing) runs at
  that lift resolution; detection runs on the full-resolution rgb. Where no geometry renders (through windows,
  outdoors), the rgb is cleared to light blue (`SKY_RGB`, `common/env_utils/env_base.py`), not black.
- (1.2) **Detect** (`Detector.predict`) on the rgb, which Detic resizes to a `--input-image-size` shortest edge and SAM to
  its own 1024 long side:
  - (1.2.1) Detic's CenterNet2 proposals are refined through every cascade stage's class-agnostic box regression.
  - (1.2.2) Each box is scored by its proposal objectness only. Boxes under `--score-thresh` (0.5) are dropped, then one
    class-agnostic NMS at `--nms-thresh` (0.5) runs, keeping Detic's top-k per image.
  - (1.2.3) Each box gets a region embedding: its last-stage box-head features projected into CLIP text space, normalized.
  - (1.2.4) Masks come from SAM (`vit_b`) prompted with the boxes, or from Detic's mask head when `--sam-checkpoint` is unset,
    binarized at `--mask-thresh` (0.5).
  - (1.2.5) `without_parts` (drops masks lying ≥ 80% inside a larger one) exists but is currently disabled.
- (1.3) **Lift** (`predicted_objects`, `project_detections_to_objects`):
  - (1.3.1) Resize each mask to the frame's depth by nearest neighbour, the same pixels the depth was resized with.
  - (1.3.2) Drop detections whose mask covers less than `_MIN_LIFT_FRACTION` (5e-4) of the frame's pixels with valid depth.
  - (1.3.3) Back-project each mask's in-range depth pixels to world points.
  - (1.3.4) **Floor:** drop points at most 2 cm above the floor (agent height minus the navmesh floor offset).
  - (1.3.5) **Background walls:** peel up to 3 RANSAC planes (2 cm inlier distance). A plane is removed only if it is vertical
    (|normal y| ≤ 0.1) and its median camera distance is at least 10 cm beyond the 95th percentile of the other points'
    distances. Flat objects such as paintings hang only a few cm off their wall, so they stay.
- (1.4) **Fuse** (`ConceptGraphMemory.__add__`):
  - (1.4.1) Skip detections with fewer than 16 points.
  - (1.4.2) **Denoise the detection:** downsample to 2.5 cm voxels. Keep the largest tight DBSCAN cluster (eps 5 cm, min 5),
    plus any small clusters (< 15% of its size) within 10 cm of it. This keeps thin parts and drops mask spill onto a
    neighbouring object.
  - (1.4.3) **Match:** score the detection against each existing object. The spatial term is its 3D IoU with the part of that
    object visible in this frame (object points near the frame's downsampled depth cloud). The match score is that IoU
    plus the cosine similarity of their region embeddings, or 0 when the IoU is under 0.1. The detection merges into
    the best-scoring object, or becomes a new object if every score is 0.
  - (1.4.4) **Protect smaller objects:** before a merge or add, drop the detection's points within 2.5 cm of any smaller object
    other than the target, so a large object can't absorb what lies on it. A new object still needs ≥ 16 points.
  - (1.4.5) **Merge:** points are voxel-merged, each point keeping a count of the detections it appeared in. Region embedding
    and objectness are averaged, weighted by detection counts. Detected frames are unioned.
  - (1.4.6) **Merge overlapping objects:** a pair merges when overlap ratio > 0.5, embedding similarity > 0.5 and their
    sum > 1.2. Pairs merge greedily, highest sum first.
- (1.5) **Steer.** The step's lifted detections (1.3) also go to `agent.update_detections`, whose `ObjectTracker`
  (`common/hybrid_exploration/object_tracker.py`) denoises each as in 1.4.2 (detections under 16 points skipped) and
  matches it as in 1.4.3: it merges into the best-scoring tracked object, or is tracked as a new one if every score is 0
  and it has ≥ 16 points. Unlike the map, tracked objects are never merged with each other nor dropped, so each keeps
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
  closest to the goal on that FMM distance (the smallest turn on ties). The step's labelled detector-prediction tile
  (`segmentation_tile`) is kept and, once exploration ends, saved as one
  `datasets/<run_name>/<scene>/predictions_mosaic.jpg` (`save_predictions_mosaic`, 5 tiles per row; split into
  `predictions_mosaic_<part>.jpg` of at most 32 rows each unless `--no-split`), along with the last step's 3-tile
  (observation, belief map cropped to the 5 x 5 m around the agent, ground truth) snapshot,
  `datasets/<run_name>/<scene>/exploration_snapshot.png`.
- (1.6) **Ground truth:** the step's ground-truth instances come from the `bbsgt` sensor, filtered by
  `object_params.min_area_fraction` (box area ≥ 5e-4 of the frame's pixels) and `filter_low_visibility`/`min_mesh_visibility_fraction`
  (`mesh_visibility_fraction`: the sum of the mask's connected components' own bounding-box areas, over the object's largest
  box face projected at the mask's mean depth),
  and remapped to the kept classes. They don't affect the map; they are kept RLE-encoded, with their object id and
  visibility fraction, for recovery (5.3).
- (1.7) **Stop** when the agent returns `stop`: when its last 3 steps all collided (`HybridConfig.max_successive_collisions`)
  or it gets stuck (no net translation over the last 25 steps, positions within 0.1 m; `BaseExplorationAgent.is_stuck`,
  logged as a warning), or the planner finds no heading that gets closer to the goal -- or when
  `collection.steps_per_scene` (500) runs out.

### 2. Resolve (`ConceptGraphMemory.resolve`)

- (2.1) Drop objects with fewer than 3 detections.
- (2.2) Drop each object's points that appeared in fewer than 2 detections.
- (2.3) Denoise every object (largest DBSCAN cluster, eps 10 cm, min 10; the original object is kept if fewer than 4 points survive).
- (2.4) Drop objects with fewer than 16 points.
- (2.5) **Shared points:** a point goes to whichever nearby object (within 2.5 cm) was detected there more often. So a part
  keeps its points only where it was detected more often than the whole it belongs to.
- (2.6) Drop objects with fewer than 16 points.
- (2.7) **Merge adjacent parts** (`_merge_adjacent`): faces of one solid detected apart (a fridge's front and its side panel,
  never in one mask) fuse into one object. Repeatedly merge the most similar pair that passes both of these:
  - (2.7.1) region-embedding similarity ≥ 0.8;
  - (2.7.2) some point of one lies within 5 cm of some point of the other.

  The result is the resolved objects, all of class `UNCLASSIFIED`.

### 3. Foreground filter (`ConceptGraphMemory.keep_foreground`)

- (3.1) **Raytrace:** project every resolved object into every stored frame. Each depth pixel is back-projected and takes
  the object of the nearest resolved point within 2.5 cm. An object counts as visible in a frame if it gets ≥ 5e-4
  (`_MIN_RAYTRACED_FRACTION`) of the frame's pixels and its `mesh_visibility_fraction` is ≥ `object_params.min_mesh_visibility_fraction` (0.2).
- (3.2) **Detection rate:** the share of the frames showing the object in which it was actually detected.
  Objects under 0.3 are dropped, which removes spurious one-off detections.
- (3.3) **Foreground score:** detection rate × objectness.

### 4. Classify (`ConceptGraphMemory.classify`)

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
  written before the environment closes). Each file is a single-scene COCO dataset on the agent's own map grid (its
  `EstimatedMap`, in the reference frame), whose one image holds the grid's size, cell size and RLE masks. The prediction
  file's `navigable` is the explored-and-free cells and `explored` the explored ones. The ground-truth file's `navigable`
  is the ground-truth navmesh resampled onto that grid. Each annotation has its class, `object_id`, a world-axis-aligned
  3D box (`box3d`), its volume as `area`, and its top-down footprint as an RLE `segmentation`. A prediction's box is the
  extent of its points snapped outward to 2 cm, its score its highest class probability (every class's in `class_scores`), and its footprint its points'
  cells with gaps closed and holes filled. A ground truth's box is its mesh bounds and its footprint its mesh's silhouette.
  Both files are drawn by class to `<scene>/estimated_topdown.png` and `<scene>/ground_truth_topdown.png` (`save_topdowns`).
- (4.6) **Map evaluation** (`evaluate_maps.py`, run once every scene is collected; `evaluate_scene_jsons`) reads only those
  files, every scene pooled:
  - (4.6.1) **3D box AP:** COCO AP by 3D box IoU, both boxes first grown to at least 5 cm per axis
    (`_MIN_MATCHED_EXTENT_M`), at IoU 0.25 to 0.95 in 0.05 steps. It reports AP (IoU 0.5:0.95), AP25 and AP50, AP25 per
    volume (small < 0.01 m³ ≤ medium < 0.25 m³ ≤ large), AP25 of each object submitted once per class among its 5 highest `class_scores`, scored by that class's (`AP25 top5`,
    also per volume; `AP25` is top-1), AP25 with classes ignored (`AP25 agnostic`, also per volume), and per-class AP25,
    AP50 and AP.
  - (4.6.2) **Class-agnostic matching:** each scene's objects, by descending score, greedily take the unmatched ground truth
    of highest 3D box IoU, if ≥ 0.25. It reports the share of ground truth matched (`recall25 agnostic`) and of objects
    matching one (`precision25 agnostic`), each also per volume (the ground truth's, the object's), with a per-scene table of every object's match and of the unmatched ground truth,
    and a confusion matrix (`confusion_matrix.png`; unmatched objects and ground truth fall in "background").
  - (4.6.3) All of 4.6.1's and 4.6.2's metrics but the per-class AP go to one flat dict, `metrics.json`.

### 5. Retrieval and train sets

- (5.1) **Retrieval** (`common/utils/retrieval.py::retrieve_resolved_views`): for each object, pick up to `collection.k_retrieval` (10)
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
  - (5.3.3) `recover_add_fn`: every unmatched ground truth gets up to `collection.k_retrieval` of its masks as views, under
    a new, negative object id. They are picked greedily as in 5.1, with one element per view, the viewing angle of its
    camera around the box center, weighted by its visibility fraction^α. Other objects' views are left as they are, so an
    absorbed false negative's pixels stay in its absorber's masks too.
- (5.4) **Train sets:** the scene's views are added to three object-view train sets:
  - (5.4.1) `pseudo`: as classified (5.1);
  - (5.4.2) `gt`: after 5.3.2;
  - (5.4.3) `gt++`: after 5.3.1, 5.3.2 and 5.3.3.

### 6. Test set (`evaluation_set.py`), per evaluation scene

- (6.1) **Explore:** a vanilla `FrontierAgent`, revealing fog of war out to `frontier_agent.visibility_dist_m` (3.0),
  explores until `evaluation.downsampling` × `evaluation.frames_per_scene` (4 × 50) of its frames show some ground truth
  (filtered and remapped as in 1.6), it stops, or `frontier_agent.steps` (1000) run out.
- (6.2) **Diverse frames:** those frames are embedded by CLIP ViT-B/32 (unit-norm image embeddings), and
  `evaluation.frames_per_scene` of them are picked by farthest-point sampling (`k_centroid_greedy`) on cosine distance.
- (6.3) Each picked frame is written to the COCO test set with all its ground-truth instances as annotations, each
  keeping its simulator object id. One test set is shared by the three train sets.

Fusion and tracking thresholds live in `ConceptGraphConfig`, the lifting and size-mask constants (`_MIN_LIFT_FRACTION`,
`_SIZE_MARGIN`) at the top of `common/utils/concept_graph.py`, the steering one (`_FULL_CONFIDENCE_DETECTIONS`) at the top
of `common/hybrid_exploration/object_tracker.py`; floor, wall and raytracing
thresholds (`_MIN_RAYTRACED_FRACTION`) at the top of `common/utils/projection_utils.py`; budgets, scenes and ground-truth
filters in the collection config.

## Outputs

`datasets/<run_name>/` gets (`common/utils/dataset_writers.py`) one object-view train set per variant,
`<variant>/object_ds/` for `pseudo`, `gt` and `gt++`, and two COCO sets they share: `train/`, the
full ground truth of every frame some variant's view is in, and `test/`. Every frame of every split is in the shared
`images/`. Each split folder holds its json, its mosaic and a `scenes/` folder of per-scene shards. A call whose scene
already has all its shards in its split (the train set's and one per variant, or the test set's) skips it, unless
`--overwrite`. Otherwise it first deletes what an earlier run of its scene wrote (its `<scene>/` folder, shards, frames
and per-scene pngs and their parts), leaving the other scenes' alone, then writes its scene's records to
`<split>/scenes/<scene>.json`. Calls for different scenes touch different files, so they run in parallel without locking.
Once every scene's call has finished, `merge.py` pools each split's shards, in scene name order, into its json and mosaic;
rerun it after recollecting a scene. The dataset then holds:

| file | contents |
|---|---|
| `images/<scene>_<frame>.jpg` | every frame of every split: the train scenes' frames some view is in, and the test frames |
| `<variant>/object_ds/object_ds.json` | `info` (`categories`, `retrieval` parameters), `frames` (`"<scene>_<frame>"`: height, width) and `retrieved_objects` (`"<scene>/<object>"`: `gt_class`, the variant's label, and `views`, each a frame and an RLE mask at the rgb's size); read by `common/utils/object_views_dataset.py`'s `ObjectViewsDataset` |
| `train/train.json`, `test/test.json` | COCO instance segmentation of the train frames (every frame some variant's view is in and some ground truth shows; info: the `collection` config) and of the test frames (info: the `evaluation` config), with the same ground truth (`sense.ground_truth_instances`): `images` (`file_name` relative to `images/`), `annotations` (RLE `segmentation`, `bbox`, `area`, `iscrowd`, plus `object_id`, the simulator object id), `categories` (ids 1..N, the kept classes in order); image and annotation ids start at 1 too. Loads with detectron2's `register_coco_instances(name, {}, "<run_name>/test/test.json", "<run_name>/images")` |
| `<variant>/decision_frames/<scene>.png`, `test/decision_frames/<scene>.png`, `<scene>/` (exploration snapshot, predictions mosaic, planner dumps) | each collection scene's exploration map with the agent's trajectory and last decision once exploration ends, the same in every variant; each test scene's frontier exploration map and trajectory |
| `<scene>/predictions.json`, `<scene>/ground_truth.json`, `<scene>/estimated_topdown.png`, `<scene>/ground_truth_topdown.png` | each collection scene's classified objects and its annotated objects as single-scene COCO datasets on the agent's map grid (4.5), and their top-down maps by class |
| `map_eval/metrics.json`, `map_eval/confusion_matrix.png` | `evaluate_maps.py`'s scores of every collection scene's map, pooled (4.6) |
| `<variant>/semantic_topdown_maps/<scene>.png` | each collection scene's resolved objects the variant keeps, by their class in the variant, on the agent's top-down map; objects `recover_add_fn` adds have no points to draw |
| `<variant>/retrieval_mosaics/<scene>.jpg` | each collection scene's objects in the variant, one row of its retrieved views per object, tiles at half resolution (`_TILE_DOWNSCALE`); objects recovery relabelled (5.3.2) or added (5.3.3) are bordered red. Split into `<scene>_<part>.jpg` of at most 32 objects each (`MAX_MOSAIC_ROWS`) unless `--no-split` |
| `<variant>/retrieval_mosaics/<scene>_first_views.jpg` | the same objects, one row of each object's first retrieved view (its highest-gain one, 5.1) at full resolution, bordered red as above |
| `<variant>/object_ds/scenes/<scene>.json`, `train/scenes/<scene>.json`, `test/scenes/<scene>.json` | one scene's records of the split, which merging pools into its json: `frames` and `retrieved_objects` for an object-view set; `images` and `annotations` (each pointing at its image by its index in the shard, ids assigned when pooled) for the COCO sets |
| `<variant>/object_ds/object_ds_mosaic.jpg`, `train/train_mosaic.jpg`, `test/test_mosaic.jpg` | 32 records of the split, read back from its json and drawn in random order, each its frame with its annotations' boxes and classes |

## Files

| file | contents |
|---|---|
| `collect.py` | one scene's run, written to its own shard of each split: for a `train` scene, exploration loop and `SceneMapper`, resolve → foreground → size-masked classify, retrieval and recovery into the variants' object views, and its frames' full ground truth into the train set; for a `test` scene, its frames into the test set. The layout (`images/`, splits, variants) and the writers are `common/utils/dataset_writers.py`'s |
| `evaluate_maps.py` | scores every collection scene's `predictions.json` against its `ground_truth.json`, pooled, into `map_eval/` (4.6) |
| `run.sh` | runs one after another, on `GPU`, the source and NORM baselines (`adapt.py`), Mean-Teacher and embodied SL on each variant (`main.py`), each scored on the procthor8 test and train sets and NYU Depth V2, into `outputs/procthor8_runs/<run>.log` |
| `merge.py` | pools every split's per-scene shards into its json and mosaic (`train/train.json`, `test/test.json`, `<variant>/object_ds/object_ds.json`), once every scene's `collect.py` call is done |
| `evaluation_set.py` | a test scene's frontier exploration and CLIP-diverse frame selection |
| `config/procthor8_collection.yaml` | the collection config (`run_name` `procthor_medium`, `seed`, `object_params`, `collection` and `evaluation` budgets): the first 8 ProcTHOR-Train houses of 2 to 4 rooms, 1000 steps each, and ProcTHOR-Val-0 to -7 for the test set; train houses ProcTHOR-Train-5, -6, -9, -11, -14, -26, -30 and -32; collected one scene per call (Running) |
| `meta_arch.py` | `CustomFT_Detic`, Detic's `CustomRCNN` registered as a `META_ARCHITECTURE` for fine-tuning its ROI heads on a train variant's `ObjectViewsDataset`, on the proposals of a frozen teacher copy of the checkpoint's backbone and proposal generator; trained by `main.py` |
| `main.py` | the fine-tuning of `CustomFT_Detic` on the train variant its config names in `DATASETS.TRAIN` (optimizer: Detic's, honoring `SOLVER.OPTIMIZER` and `SOLVER.BACKBONE_MULTIPLIER`; training frames resized and flipped as `INPUT.MIN_SIZE_TRAIN` and `INPUT.RANDOM_FLIP` say), into `logs/<--run-name, or the config-file stem>/`: `results.json` (last and best evaluation), `metrics.png`, `vocabulary_tsne.png`, `class_ap.png` (per-class AP50 at each evaluation, written in its cell), `vocabulary_similarity.png` (the classes' CLIP cosine similarity × 100, written in its cell), `class_table.csv` (per class: training objects and views, test instances, first, best and last AP50), `box_areas.png` (train and test box sizes), confusion matrices |
| `config/r50_detic.yaml` | `CustomFT_Detic` on Detic's R50 checkpoint (`Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth`), whose FrozenBN backbone the adaptation baselines adapt: the source model, base of every `r50_*` config |
| `config/r50_procthor8_embodied_sl.yaml`, `config/r50_procthor8_embodied_sl_{pseudo,gt,gt++}.yaml` | the reference fine-tune: AdamW 1e-4, the student backbone at × 0.1, training frames at the test scale (800) and never flipped, seed 0, 1,000 iterations evaluated every 500; one config per train variant, named in `DATASETS.TRAIN` |
| `mean_teacher.py`, `config/r50_procthor8_unsup_mean_teacher.yaml` | the Mean-Teacher baseline on the reference's settings: `main.py`'s training with its labels replaced by a moving-average teacher's detections (score ≥ 0.5) on the `gt++` variant's frames, the student trained on a photometrically augmented copy, the teacher (momentum 0.999) evaluated and reported, into `logs/<--run-name, or the config-file stem>/` |
| `--static-test-dirs` (`main.py`, `adapt.py`, `mean_teacher.py`) | static COCO test sets (`<dir>/test.json`, images in `<dir>/../images/`) scored at every evaluation besides the collection's test set and its train set (`train/`, logged as `<run_name>_train/bbox/...`), which every run scores, each in its own vocabulary: the classifier is reset to the set's class names' CLIP embeddings, then back to the training classes. Logged as `<set>/bbox/...` in `metrics.json`, and under the collection test set's results in `results.json` |
| `common/env_utils/nyu_depth_v2.py` | NYU Depth V2's labeled subset (or `--frames` of it) as such a test set: one box per (class, instance) pair of a frame, the dataset's own class names but wall, floor, ceiling and unknown |
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
export HABITAT_DATA=$HOME/habitat_data/data BASE_DIR=$(pwd) PYTHONNOUSERSITE=1
PYTHONPATH=. python habitat_embodied_sl/collect.py <gpu> <split> <scene> habitat_embodied_sl/config/procthor8_collection.yaml \
    --config-file third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
    --checkpoint third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
    --sam-checkpoint data/sam/sam_vit_b_01ec64.pth
```

`<split>` is `train` (explore and map the scene into the train sets) or `test` (pick its test frames), `<scene>` the
scene, and `<gpu>` the index of the GPU both the renderer and the models run on. Each scene is its own call, so every
scene, test scenes included, needs one before the splits are complete; calls for different scenes can run at once, e.g.
one per GPU:

```bash
scenes=(ProcTHOR-Train-5 ProcTHOR-Train-6 ProcTHOR-Train-9 ProcTHOR-Train-11)
for gpu in 0 1 2 3; do
  PYTHONPATH=. nohup python habitat_embodied_sl/collect.py $gpu train ${scenes[$gpu]} habitat_embodied_sl/config/procthor8_collection.yaml \
      --sam-checkpoint data/sam/sam_vit_b_01ec64.pth > collect_${scenes[$gpu]}.log 2>&1 &
done
```

Once every scene's call has finished, pool the shards into each split's json and mosaic:

```bash
PYTHONPATH=. python habitat_embodied_sl/merge.py habitat_embodied_sl/config/procthor8_collection.yaml
```

Then score the train scenes' maps from their `predictions.json` and `ground_truth.json` (4.6):

```bash
PYTHONPATH=. python habitat_embodied_sl/evaluate_maps.py habitat_embodied_sl/config/procthor8_collection.yaml
```

Defaults: `--input-image-size 1024` (sensor resolution and Detic's shortest edge), `--lift-image-size 512` (depth and mask resolution for lifting and raytracing).

Once the collection config's splits are complete, the R50 fine-tuning and one adaptation baseline (`--method source`
or `norm`) run with:

```bash
PYTHONPATH=. python habitat_embodied_sl/main.py --config habitat_embodied_sl/config/procthor8_collection.yaml \
    --config-file habitat_embodied_sl/config/r50_procthor8_embodied_sl_gt++.yaml
PYTHONPATH=. python habitat_embodied_sl/adapt.py --config habitat_embodied_sl/config/procthor8_collection.yaml \
    --config-file habitat_embodied_sl/config/r50_detic.yaml --method norm --run-name r50_procthor8_unsup_norm
```

The source baseline is the same `adapt.py` call with `--method source --run-name r50_procthor8_source`, and
Mean-Teacher `python habitat_embodied_sl/mean_teacher.py` with its defaults.
