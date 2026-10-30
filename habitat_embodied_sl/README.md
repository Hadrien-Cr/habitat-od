# Retrieval datasets from semantic mapping with class-agnostic detection

`main.py` builds retrieval datasets from a collection config (`--config`, e.g. `config/collection_procthor.yaml`). It maps
every collection scene (`collection.scenes`): `common/hybrid_exploration`'s `HybridFrontierAgent` explores the scene in
exhaustive mode, while a class-agnostic `Detector` (`common/vision/detector.py`) segments every frame, and its detections are lifted to
3D and fused into a concept-graphs `ConceptGraphMemory` (`common/utils/concept_graph.py`). When exploration ends, the map
is resolved, filtered and classified, and diverse views of each object are retrieved into three object-view train sets.
The evaluation scenes (`evaluation.scenes`) give a COCO test set of fully annotated frames shared by all three.

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
  that lift resolution; detection runs on the full-resolution rgb.
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
  - (1.3.2) Drop detections whose mask covers less than `_MIN_LIFT_FRACTION` (2e-4) of the frame's pixels with valid depth.
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
- (1.5) **Steer.** Every fused object is passed to `agent.update_objects` as an `ObjectBelief` of its points, with
  confidence min(detections, 5) / 5. Then `agent.act` picks the action: belief-map-only, via a shared `DiscretePlanner`
  FMM pass over a goal map of every unseen object viewpoint (or, once none remain, every frontier cell) -- no
  ground-truth navmesh pathfinding. The step's labelled detector-prediction tile
  (`segmentation_tile`) is kept and, once exploration ends, saved both individually
  (`datadump/<run_name>/<scene>/prediction_frames/step_<step>.png`) and as one `predictions_mosaic.png`
  (`save_exploration_visuals`, which also writes that scene's final `decision.png`/`object.png`). Each step also saves a
  3-tile (observation, belief map, ground truth) snapshot to
  `datadump/<run_name>/<scene>/images/exploration_snapshots/snapshot_<step>.png`, assembled into `exploration.webp`
  (lossless animated WebP, not GIF -- GIF's 256-color palette cap crowds out or merges rare colors like the belief
  map's green explored tint) once exploration ends.
- (1.6) **Ground truth:** the step's ground-truth instances come from the `bbsgt` sensor, filtered by
  `object_params.min_area_fraction` (min(2 × mask area, box area) ≥ 2e-4 of the frame's pixels) and `filter_low_visibility`/`min_mesh_visibility_fraction`
  (`mesh_visibility_fraction`: the sum of the mask's connected components' own bounding-box areas, over the object's largest
  box face projected at the mask's mean depth),
  and remapped to the kept classes. They don't affect the map; they are kept RLE-encoded, with their object id and
  visibility fraction, for recovery (5.3).
- (1.7) **Stop** when the agent returns `stop`, its last 3 steps all collided (`HybridConfig.max_successive_collisions`),
  it gets stuck (no net translation over the last 15 steps, logged as a warning) or `collection.steps_per_scene` (500)
  runs out.

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
  the object of the nearest resolved point within 2.5 cm. An object counts as visible in a frame if it gets ≥ 2e-4
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

### 5. Retrieval and train sets

- (5.1) **Retrieval** (`common/utils/retrieval.py::retrieve_views`): for each object, pick up to `collection.k_retrieval` (10)
  views greedily so they cover its (resolved point, viewing angle) elements, the angle from the point to the camera
  (not the agent's heading). A view's gain is visibility^α × Σ (1 − γ) γ^(effective times already covered), with
  γ = 0.5 and α = 1; a point already covered by a chosen view lends credit to a new candidate's angle for it in
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
  - (5.4.2) `pseudo_gt_classes`: after 5.3.2;
  - (5.4.3) `recovered`: after 5.3.1, 5.3.2 and 5.3.3.

### 6. Test set (`evaluation_set.py`), per evaluation scene

- (6.1) **Explore:** a vanilla `FrontierAgent`, revealing fog of war out to `frontier_agent.visibility_dist_m` (3.0),
  explores until `evaluation.downsampling` × `evaluation.frames_per_scene` (4 × 50) of its frames show some ground truth
  (filtered and remapped as in 1.6), it stops, or `frontier_agent.steps` (1000) run out.
- (6.2) **Diverse frames:** those frames are embedded by CLIP ViT-B/32 (unit-norm image embeddings), and
  `evaluation.frames_per_scene` of them are picked by farthest-point sampling (`k_centroid_greedy`) on cosine distance.
- (6.3) Each picked frame is written to the COCO test set with all its ground-truth instances as annotations, each
  keeping its simulator object id. One test set is shared by the three train sets.

Fusion thresholds live in `ConceptGraphConfig` and the lifting, steering and size-mask constants (`_MIN_LIFT_FRACTION`,
`_FULL_CONFIDENCE_DETECTIONS`, `_SIZE_MARGIN`) at the top of `common/utils/concept_graph.py`; floor, wall and raytracing
thresholds (`_MIN_RAYTRACED_FRACTION`) at the top of `common/utils/projection_utils.py`; budgets, scenes and ground-truth
filters in the collection config.

## Outputs

`datasets/<run_name>/` is emptied at start, then gets (`common/utils/dataset_writers.py`) one train set per variant,
`<variant>/train/` for `pseudo`, `pseudo_gt_classes` and `recovered`, and the test set they share, `test/`. Each split
folder holds its json, its mosaic and an `images/` folder:

| file | contents |
|---|---|
| `<variant>/train/images/<scene>_<frame>.jpg` | every frame some view of the train set is in |
| `<variant>/train/train.json` | `info` (`categories`, `retrieval` parameters), `frames` (`"<scene>_<frame>"`: height, width) and `retrieved_objects` (`"<scene>/<object>"`: `gt_class`, the variant's label, and `views`, each a frame and an RLE mask at the rgb's size); read by `common/utils/object_views_dataset.py`'s `ObjectViewsDataset` |
| `test/images/<scene>_<frame>.jpg`, `test/test.json` | COCO instance segmentation of the test frames: `images` (`file_name` relative to `images/`), `annotations` (RLE `segmentation`, `bbox`, `area`, `iscrowd`, plus `object_id`, the simulator object id), `categories` (ids 1..N, the kept classes in order), `info` (the `evaluation` config); image and annotation ids start at 1 too. Loads with detectron2's `register_coco_instances(name, {}, "<run_name>/test/test.json", "<run_name>/test/images")` |
| `<variant>/decision_frames/<scene>.png`, `test/decision_frames/<scene>.png` | each collection scene's exploration map with the agent's trajectory and last decision once exploration ends, the same in every variant; each test scene's frontier exploration map and trajectory |
| `<variant>/semantic_topdown_maps/<scene>.png` | each collection scene's resolved objects the variant keeps, by their class in the variant, on the agent's top-down map; objects `recover_add_fn` adds have no points to draw |
| `<variant>/retrieval_mosaics/<scene>.png` | each collection scene's objects in the variant, one row of its retrieved views per object, tiles at half resolution (`_TILE_DOWNSCALE`); objects recovery relabelled (5.3.2) or added (5.3.3) are bordered red |
| `<variant>/train/train_mosaic.png`, `test/test_mosaic.png` | 32 records of the split, read back from its json and drawn in random order, each its frame with its annotations' boxes and classes |

## Files

| file | contents |
|---|---|
| `main.py` | the full run: per collection scene, exploration loop and `SceneMapper`, resolve → foreground → size-masked classify, retrieval and recovery into the train sets; then the test sets |
| `evaluation_set.py` | a test scene's frontier exploration and CLIP-diverse frame selection |
| `config/collection_procthor.yaml` | collection config: `run_name`, `seed`, `object_params`, `collection` and `evaluation` scenes and budgets |
| `meta_arch.py` | `CustomFT_Detic`, Detic's `CustomRCNN` registered as a `META_ARCHITECTURE` for fine-tuning on a train variant's `ObjectViewsDataset`; trained by the root `train.py` |

## Running

```bash
export HABITAT_DATA=$HOME/habitat_data/data BASE_DIR=$(pwd) PYTHONNOUSERSITE=1
PYTHONPATH=. python habitat_embodied_sl/main.py --config habitat_embodied_sl/config/collection_procthor.yaml \
    --config-file third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
    --checkpoint third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
    --sam-checkpoint data/sam/sam_vit_b_01ec64.pth
```

Defaults: `--input-image-size 1024` (sensor resolution and Detic's shortest edge), `--lift-image-size 512` (depth and mask resolution for lifting and raytracing).
