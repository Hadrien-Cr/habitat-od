"""Builds retrieval datasets from a collection config (see habitat_embodied_sl/config/procthor8_collection.yaml and
habitat_embodied_sl/README.md), one scene per call, so one call per scene can run in parallel, each on its own GPU, all
writing to the config's datasets/<run_name>/. Each call's split argument says which split its scene goes to; a scene goes
to train or test, never both. In a train scene, common/hybrid_exploration's HybridFrontierAgent
explores in exhaustive mode for up to collection.steps_per_scene steps (or until it gets stuck -- no net translation
over BaseExplorationAgent.is_stuck's window, which stops it, logged as a warning) while a Detector (Detic's proposals scored by
objectness alone, SAM masks) segments every frame; its masks and region embeddings are fused into a concept-graphs
ConceptGraphMemory (common/utils/concept_graph.py), while the agent's own ObjectTracker
(common/hybrid_exploration/object_tracker.py) tracks the same detections into the objects it inspects. The last step's 3-tile
(observation, belief map, ground truth) snapshot is saved to datasets/<run_name>/<scene>/exploration_snapshot.png and
every step's predictions to predictions_mosaic.jpg there. The map is then resolved, its
foreground objects kept and classified (ConceptGraphMemory.classify): its objects go to
datasets/<run_name>/<scene>/predictions.json and the scene's annotated ones to ground_truth.json there (eval_utils.py's
single-scene COCO datasets: 3D boxes and top-down footprints on the agent's map grid), drawn by class to
estimated_topdown.png and ground_truth_topdown.png there and scored, every scene pooled, by
habitat_embodied_sl/evaluate_maps.py. Up to collection.k_retrieval diverse views of each object are then retrieved
(common/utils/retrieval.py). Those views make three object-view train sets under
datasets/<run_name>/<variant>/object_ds/: "pseudo" with the classified labels, "gt" with matched objects taking
their ground-truth class (recover_gt_classes), and "gt++" with false positives dropped and false negatives' views
added too (recover_drop_fp, recover_add_fn; common/utils/ground_truth_recovery.py). Every frame some variant's view is
in also goes, with its full ground truth as the test set's (ground_truth_frame), to the COCO train set
datasets/<run_name>/train/train.json, which main.py scores like the test set. Their shared COCO test set,
datasets/<run_name>/test/test.json, pools every test scene's diverse, fully annotated frames
(habitat_embodied_sl/evaluation_set.py); a test scene's call writes those. All frames go to the shared
datasets/<run_name>/images/. A call whose scene already has all its shards in its split (the train set's and one per
variant, or the test set's) skips it unless --overwrite. Otherwise it first deletes what
an earlier run of its scene wrote, then writes the scene's records to its own shard of each split
(common/utils/dataset_writers.py). Once every scene's call is done, habitat_embodied_sl/merge.py pools each split's shards
into its json and mosaic. Each train scene gets a retrieval mosaic per variant, <variant>/retrieval_mosaics/<scene>.jpg,
one row of views per object at half resolution, its objects relabelled or added by recovery bordered red, split into
<scene>_<part>.jpg of at most 32 objects each unless --no-split; <scene>_first_views.jpg there too, one row of each
object's first view; and a semantic top-down map per variant, <variant>/semantic_topdown_maps/<scene>.png, of its
resolved objects by their class in the variant. predictions_mosaic.jpg is split the same way, 32 rows per part.

Usage (Detic's Swin-B checkpoint from https://github.com/facebookresearch/Detic's model zoo; Detic classifies
against the CLIP text embeddings of the config's vocab_name/filter_classes class names):
  export HABITAT_DATA=$HOME/habitat_data/data BASE_DIR=$(pwd) PYTHONNOUSERSITE=1
  PYTHONPATH=. python habitat_embodied_sl/collect.py <gpu> {train,test} <scene> habitat_embodied_sl/config/procthor8_collection.yaml \
      --config-file third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
      --checkpoint third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
      --sam-checkpoint data/sam/sam_vit_b_01ec64.pth
  PYTHONPATH=. python habitat_embodied_sl/merge.py habitat_embodied_sl/config/procthor8_collection.yaml  # once all scenes are done
"""
import argparse
import copy
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from detectron2.structures import Instances
from habitat.tasks.nav.nav import TopDownMap
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from common import constants
from common.hybrid_exploration.frontier_exp_agent import exploration_snapshot, save_predictions_mosaic
from common.hybrid_exploration.hybrid_agent import HybridFrontierAgent
from common.utils.coco_utils import write_json
from common.utils.eval_utils import (
    GROUND_TRUTH_JSON, PREDICTIONS_JSON, GroundTruthObjects, ground_truth_dataset, matched_ground_truth, predictions_dataset, save_topdowns,
    semantic_topdown,
)
from common.env_utils.env_base import ExplorationEnv, build_agent, build_env
from common.env_utils.object_annotations import PROCTHOR100_SIZES_PATH, class_size_ranges, kept_classes_from_args, resolve_classes
from common.env_utils.sense import ground_truth_instances
from common.env_utils.vocab_constants import make_colors
from common.utils.ground_truth_recovery import (
    GroundTruthAnnotation, GroundTruthView, encoded_views, ground_truth_frame, recover_add_fn, recover_drop_fp, recover_gt_classes,
    recovered_objects,
)
from common.utils.image_memory import Frame
from common.utils.plot_utils import MAX_MOSAIC_ROWS, MOSAIC_JPEG_QUALITY, PNG_COMPRESS_LEVEL, save_mosaic_parts
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import DepthCamera, nearest_resized
from common.utils.dataset_writers import SCENES_DIR, VARIANTS, dataset_writers, scene_shard_path
from common.utils.retrieval import (
    RetrievedView, SegmentationStyle, ViewSelection, first_views_mosaic, objects_views_mosaics, retrieve_resolved_views, retrieved_views,
    segmentation_tile,
)
from common.vision.detic import Vocabulary
from habitat_embodied_sl.evaluation_set import ClipImageEncoder, evaluation_frames
from common.utils.concept_graph import ConceptGraphMemory, object_size_ranges, predicted_objects, processed_frame
from common.vision.detector import Detector

_SPLIT_WRITERS = {"train": ("train", *VARIANTS), "test": ("test",)}  # dataset_writers' keys of each split's shards
_TILE_DOWNSCALE = 2


def observe_frame(obs: dict, env: ExplorationEnv, lift_image_size: int) -> Frame:
    """obs' rgb, depth resized to lift_image_size x lift_image_size and depth-sensor pose."""
    rgb = obs["rgb"][:, :, :3]
    pose = sensor_pose_matrix(env.get_agent_state().sensor_states["depth"])
    return Frame(rgb, nearest_resized(obs["depth"].squeeze(), (lift_image_size, lift_image_size)).numpy(), pose)


def mosaic_max_rows(args: argparse.Namespace) -> Optional[int]:
    return None if args.no_split else MAX_MOSAIC_ROWS


def kept_vocabulary(object_params: DictConfig) -> Vocabulary:
    """Vocabulary of object_params' kept classes."""
    return Vocabulary(f"{object_params.vocab_name}_kept", tuple(kept_classes_from_args(object_params)))


def build_detector(args: argparse.Namespace, object_params: DictConfig) -> Detector:
    """Detector classifying among object_params' kept classes."""
    return Detector(
        args.config_file, args.checkpoint, kept_vocabulary(object_params), args.sam_checkpoint,
        score_thresh=args.score_thresh, nms_thresh=args.nms_thresh, mask_thresh=args.mask_thresh, input_size=args.input_image_size,
    )


@dataclass
class SceneMapper:
    """Fuses each frame the agent observes into fused and feeds its detections to the agent's tracker, keeping its
    RLE-encoded ground truth in ground_truth_views."""

    agent: HybridFrontierAgent
    detector: Detector
    camera: DepthCamera
    vocabulary: list
    floor_offset: float
    lift_image_size: int
    fused: ConceptGraphMemory = field(default_factory=ConceptGraphMemory)
    ground_truth_views: List[GroundTruthView] = field(default_factory=list)

    def observe(self, obs: dict, env: ExplorationEnv) -> Tuple[Frame, Instances]:
        frame = observe_frame(obs, env, self.lift_image_size)
        frame_id = self.fused.image_memory.add_frame(frame)
        instances = self.detector.predict(frame.rgb)
        floor_height = env.get_agent_state().position[1] - self.floor_offset
        detections = processed_frame(predicted_objects(frame_id, frame, instances, self.camera, floor_height), self.fused.config)
        self.fused = self.fused.add_processed(detections)
        self.agent.update_detections(detections)
        self.ground_truth_views += encoded_views(frame_id, ground_truth_instances(obs, self.vocabulary, self.detector.classes))
        return frame, instances


def explore(env: ExplorationEnv, mapper: SceneMapper, obs: dict, steps: int, style: SegmentationStyle, ground_truth, out_dir: Path) -> Tuple[list, bool]:
    """Has mapper observe every frame from obs on until its agent stops (also after too many successive collisions),
    gets stuck or steps run out, saving the last step's 3-tile (observation, belief map, ground truth) snapshot to
    out_dir/exploration_snapshot.png and returning (each step's labelled detector-prediction tile (segmentation_tile),
    whether the agent got stuck)."""
    collided = False
    prediction_tiles = []
    trajectory_px: list = []
    for step in tqdm(range(steps), desc="mapping", ncols=80):
        state = env.get_agent_state()
        heading = float(TopDownMap.get_polar_angle(state))
        trajectory_px.append(ground_truth.to_pixel(np.array(state.position)))
        ground_truth.reveal(np.array(state.position), heading)
        frame, instances = mapper.observe(obs, env)
        prediction_tiles.append(segmentation_tile(frame.rgb, instances, style, f"step{step:04d}"))
        action, _ = mapper.agent.act(env.get_agent_state(), obs["depth"], collided)
        if action == "stop":
            break
        obs, _, _, _ = env.step(action)
        collided = bool(np.any(obs["agent_collision_sensor"]))
    snapshot = exploration_snapshot(obs, mapper.agent, ground_truth, trajectory_px)
    if snapshot is not None:
        snapshot.save(out_dir / "exploration_snapshot.png", compress_level=PNG_COMPRESS_LEVEL)
    return prediction_tiles, mapper.agent.is_stuck()


def map_scene(scene: str, cfg: DictConfig, detector: Detector, style: SegmentationStyle, args: argparse.Namespace) -> Tuple[SceneMapper, GroundTruthObjects]:
    """(SceneMapper of the scene's exploration, the scene's annotated objects of detector's classes), saving every
    step's detector predictions as one mosaic (save_predictions_mosaic), the last step's 3-tile snapshot and those
    objects' ground_truth.json (ground_truth_dataset, on the agent's map grid) to datasets/<run_name>/<scene>/."""
    env, habitat_cfg = build_env(scene, cfg.seed, cfg.collection.steps_per_scene, cfg.object_params, args.input_image_size)
    scene_dump_dir = constants.DATASET_ROOT / cfg.run_name / scene
    agent = build_agent("hybrid", np.random.default_rng(cfg.seed), env, habitat_cfg, dump_location=str(scene_dump_dir), exp_name="planner")
    obs = env.reset(env.episodes[0], rng=agent.rng)
    floor_offset = env.navmesh_floor_offset()
    agent.reset(env.get_agent_state(), env.get_agent_state().position[1] - floor_offset)
    depth_cfg = habitat_cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    camera = DepthCamera(depth_cfg.hfov, depth_cfg.min_depth, depth_cfg.max_depth)
    vocabulary = resolve_classes(cfg.object_params.env_name, cfg.object_params.vocab_name)
    mapper = SceneMapper(agent, detector, camera, vocabulary, floor_offset, args.lift_image_size)
    gt_meshes = env.get_gt_meshes(detector.classes)
    map_config = agent.config.map_config
    ground_truth = env.gt_map_tracker(map_config.fov_deg, map_config.visibility_dist_m)
    ground_truth.set_object_footprints(gt_meshes)
    scene_dump_dir.mkdir(parents=True, exist_ok=True)
    prediction_tiles, stuck = explore(env, mapper, obs, cfg.collection.steps_per_scene, style, ground_truth, scene_dump_dir)
    if stuck:
        print(f"Warning: agent got stuck in {scene} (no net translation)")
    save_predictions_mosaic(prediction_tiles, scene_dump_dir, mosaic_max_rows(args))
    write_json(scene_dump_dir / GROUND_TRUTH_JSON, ground_truth_dataset(scene, gt_meshes, agent, env.get_pathfinder(), detector.classes))
    env.close()
    return mapper, GroundTruthObjects.from_meshes(gt_meshes)


def resolve_and_classify(mapper: SceneMapper, min_mesh_visibility_fraction: float) -> None:
    """Resolves mapper's objects, keeps the foreground ones then classifies them among the classes their size fits."""
    size_ranges = object_size_ranges(mapper.detector.classes, class_size_ranges(PROCTHOR100_SIZES_PATH))
    mapper.fused.resolve()
    mapper.fused.keep_foreground(mapper.camera, min_mesh_visibility_fraction)
    mapper.fused.classify(mapper.detector, mapper.camera, min_mesh_visibility_fraction, size_ranges)


def scene_variants(mapper: SceneMapper, ground_truth: GroundTruthObjects, cfg: DictConfig) -> Dict[str, List[RetrievedView]]:
    """{variant: views} of the views retrieved of mapper's classified objects: as classified ("pseudo"), matches taking
    their ground-truth class ("gt"), and also without false positives and with false negatives' views ("gt++")."""
    resolved, memory = mapper.fused.resolved_objects, mapper.fused.image_memory
    selection = ViewSelection(k=cfg.collection.k_retrieval)
    retrieved = retrieve_resolved_views(resolved, memory, mapper.camera, cfg.object_params.min_mesh_visibility_fraction, selection)
    pseudo = retrieved_views(resolved, memory, *retrieved)
    matches = matched_ground_truth(resolved, ground_truth)
    gt_classes = recover_gt_classes(pseudo, matches, ground_truth)
    annotation = GroundTruthAnnotation(ground_truth, mapper.ground_truth_views)
    recovered = recover_add_fn(recover_drop_fp(gt_classes, matches), annotation, matches, memory, selection)
    return dict(zip(VARIANTS, (pseudo, gt_classes, recovered)))


def save_retrieval_mosaics(scene: str, scene_views: Dict[str, List[RetrievedView]], style: SegmentationStyle, root: Path, max_rows: Optional[int]) -> None:
    """root/<variant>/retrieval_mosaics/<scene>.jpg of every variant's scene_views (scene_variants): one row per object of
    its views, tiles downscaled by _TILE_DOWNSCALE, bordered red when recovery relabelled or added the object, split into
    parts of at most max_rows objects (save_mosaic_parts); and <scene>_first_views.jpg, one row of each object's first view."""
    for variant, views in scene_views.items():
        highlighted = recovered_objects(scene_views[VARIANTS[0]], views)
        out_dir = root / variant / "retrieval_mosaics"
        save_mosaic_parts(objects_views_mosaics(views, style, highlighted, _TILE_DOWNSCALE, max_rows), out_dir / f"{scene}.jpg")
        first_views = first_views_mosaic(views, style, highlighted)
        if first_views is not None:
            first_views.save(out_dir / f"{scene}_first_views.jpg", quality=MOSAIC_JPEG_QUALITY)


def save_semantic_topdowns(scene: str, scene_views: Dict[str, List[RetrievedView]], mapper: SceneMapper, style: SegmentationStyle, root: Path) -> None:
    """root/<variant>/semantic_topdown_maps/<scene>.png of every variant's scene_views (scene_variants): the resolved objects
    its views show, by their class there, on the agent's map; recover_add_fn's objects have no points to draw."""
    for variant, views in scene_views.items():
        object_classes = {view.object_id: view.class_id for view in views if view.object_id >= 0}
        resolved = copy.deepcopy(mapper.fused.resolved_objects).relabeled(object_classes)
        out_path = root / variant / "semantic_topdown_maps" / f"{scene}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        semantic_topdown(predictions_dataset(scene, resolved, mapper.agent, style.classes), style).save(out_path, compress_level=PNG_COMPRESS_LEVEL)


def train_frames(scene_views: Dict[str, List[RetrievedView]], ground_truth_views: List[GroundTruthView]) -> List[Tuple[int, np.ndarray, Instances]]:
    """(frame id, rgb, ground truth) of every frame some variant's view is in and some ground truth shows, as the test set's."""
    frames = {view.frame_id: view.rgb for views in scene_views.values() for view in views}
    frame_views = defaultdict(list)
    for view in ground_truth_views:
        frame_views[view.frame_id].append(view)
    return [(frame_id, frames[frame_id], ground_truth_frame(frame_views[frame_id], frames[frame_id].shape[:2])) for frame_id in sorted(frames) if frame_views[frame_id]]


def flush_scene(root: Path, scene: str) -> None:
    """Deletes what an earlier run of the scene wrote under root: its own folder, and in every split its shard, its frames
    and its per-scene images (frames, retrieval mosaics and their parts, semantic top-down maps, decision frames)."""
    shutil.rmtree(root / scene, ignore_errors=True)
    patterns = (f"**/{SCENES_DIR}/{scene}.json", *(f"**/{scene}{suffix}.{extension}" for suffix in ("", "_*") for extension in ("png", "jpg")))
    for path in [path for pattern in patterns for path in root.glob(pattern)]:
        path.unlink()


def collect_scene(scene: str, cfg: DictConfig, writers: dict, args: argparse.Namespace) -> None:
    """Maps the collection scene into its shard of every variant's object views and of the train set (its frames' full
    ground truth); habitat_embodied_sl/merge.py pools the shards."""
    detector = build_detector(args, cfg.object_params)
    style = SegmentationStyle(detector.classes, make_colors(len(detector.classes), seed=0, ctype=0))
    mapper, ground_truth = map_scene(scene, cfg, detector, style, args)
    resolve_and_classify(mapper, cfg.object_params.min_mesh_visibility_fraction)
    root = constants.DATASET_ROOT / cfg.run_name
    write_json(root / scene / PREDICTIONS_JSON, predictions_dataset(scene, mapper.fused.resolved_objects, mapper.agent, detector.classes))
    save_topdowns(root / scene, style)
    scene_views = scene_variants(mapper, ground_truth, cfg)
    save_retrieval_mosaics(scene, scene_views, style, root, mosaic_max_rows(args))
    save_semantic_topdowns(scene, scene_views, mapper, style, root)
    for variant, views in scene_views.items():
        writers[variant].write_scene(scene, views)
    writers["train"].write_scene(scene, train_frames(scene_views, mapper.ground_truth_views))


def test_scene(scene: str, cfg: DictConfig, writer, args: argparse.Namespace) -> None:
    """The evaluation scene's diverse frames into its shard of the shared test set, datasets/<run_name>/test/."""
    writer.write_scene(scene, evaluation_frames(scene, cfg, writer.categories, args.input_image_size, ClipImageEncoder("cuda")))


def scene_shards(writers: dict, scene: str, split: str) -> List[Path]:
    """The scene's shards already written to the split: the train set's and every variant's object views', or the test set's."""
    paths = [scene_shard_path(writers[key].split_dir, scene) for key in _SPLIT_WRITERS[split]]
    return [path for path in paths if path.exists()]


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    root = constants.DATASET_ROOT / cfg.run_name
    writers = dataset_writers(cfg, list(kept_vocabulary(cfg.object_params).classes))
    leaked = scene_shards(writers, args.scene, "test" if args.split == "train" else "train")
    if leaked:
        raise ValueError(f"{args.scene} is already in {leaked[0]}: a scene goes to train or test, never both")
    if not args.overwrite and len(scene_shards(writers, args.scene, args.split)) == len(_SPLIT_WRITERS[args.split]):
        print(f"{args.scene} already written to {root}, skipped (--overwrite recollects it)")
        return
    flush_scene(root, args.scene)
    if args.split == "train":
        collect_scene(args.scene, cfg, writers, args)
    else:
        test_scene(args.scene, cfg, writers["test"], args)
    print(f"{args.scene} written to {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gpu", help="Index of the GPU both habitat-sim's renderer and the models run on.")
    parser.add_argument("split", choices=["train", "test"], help="Explore and map the scene into the train sets, or pick its test frames.")
    parser.add_argument("scene", help="Scene to collect, e.g. ProcTHOR-Train-5 or ProcTHOR-Val-0.")
    parser.add_argument("config", help="Collection config: run_name, budgets, object_params.")
    parser.add_argument("--config-file", default="third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    parser.add_argument("--checkpoint", default="third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--input-image-size", type=int, default=1024, help="Side of the rendered frames, Detic's and SAM's input.")
    parser.add_argument("--lift-image-size", type=int, default=512, help="Side depth and masks are resized to for lifting and raytracing.")
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--no-split", action="store_true", help="Save each mosaic whole instead of in parts of at most 32 rows.")
    parser.add_argument("--overwrite", action="store_true", help="Recollect the scene even if all its shards are already written.")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # read when CUDA first initializes, which none of the imports above does
    main(args)
