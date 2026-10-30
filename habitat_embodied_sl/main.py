"""Builds retrieval datasets from a collection config (--config, see habitat_embodied_sl/config/collection_procthor.yaml
and habitat_embodied_sl/README.md). In each collection.scenes scene, common/hybrid_exploration's HybridFrontierAgent
explores in exhaustive mode for up to collection.steps_per_scene steps (or until it gets stuck -- no net translation
over _STUCK_WINDOW_STEPS consecutive steps, logged as a warning) while a Detector (Detic's proposals scored by
objectness alone, SAM masks) segments every frame; its masks and region embeddings are fused into a concept-graphs
ConceptGraphMemory (common/utils/concept_graph.py) whose objects are the agent's beliefs. Each step's 3-tile
(observation, belief map, ground truth) snapshot is saved to datasets/<run_name>/<scene>/images/exploration_snapshots/
and assembled into exploration.webp there. The map is then resolved, its
foreground objects kept and classified (ConceptGraphMemory.classify), and up to collection.k_retrieval diverse views of
each retrieved (common/utils/retrieval.py). Those views make three object-view train sets under
datasets/<run_name>/<variant>/train/: "pseudo" with the classified labels, "pseudo_gt_classes" with matched objects taking
their ground-truth class (recover_gt_classes), and "recovered" with false positives dropped and false negatives' views
added too (recover_drop_fp, recover_add_fn; common/utils/ground_truth_recovery.py). Their shared COCO test set,
datasets/<run_name>/test/test.json, pools every evaluation.scenes scene's diverse, fully annotated frames
(habitat_embodied_sl/evaluation_set.py). Each split is written with a scrambled mosaic by common/utils/dataset_writers.py,
and each collection scene gets a retrieval mosaic per variant, <variant>/retrieval_mosaics/<scene>.png, one row of
views per object at half resolution, its objects relabelled or added by recovery bordered red, and a semantic top-down
map per variant, <variant>/semantic_topdown_maps/<scene>.png, of its resolved objects by their class in the variant.

Usage (Detic's Swin-B checkpoint from https://github.com/facebookresearch/Detic's model zoo; Detic classifies
against the CLIP text embeddings of the config's vocab_name/filter_classes class names):
  export HABITAT_DATA=$HOME/habitat_data/data BASE_DIR=$(pwd) PYTHONNOUSERSITE=1
  PYTHONPATH=. python habitat_embodied_sl/main.py --config habitat_embodied_sl/config/collection_procthor.yaml \
      --config-file third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
      --checkpoint third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
      --sam-checkpoint data/sam/sam_vit_b_01ec64.pth
"""
import argparse
import copy
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from detectron2.structures import Instances
from habitat.tasks.nav.nav import TopDownMap
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from common import constants
from common.hybrid_exploration.frontier_exp_agent import exploration_snapshot, save_exploration_visuals
from common.hybrid_exploration.hybrid_agent import HybridFrontierAgent
from common.utils.eval_utils import GroundTruthObjects, matched_ground_truth
from common.env_utils.env_base import ExplorationEnv, build_agent, build_env
from common.env_utils.object_annotations import PROCTHOR100_SIZES_PATH, class_size_ranges, kept_classes_from_args, resolve_classes
from common.env_utils.sense import ground_truth_instances
from common.env_utils.vocab_constants import make_colors
from common.utils.ground_truth_recovery import (
    GroundTruthAnnotation, GroundTruthView, encoded_views, recover_add_fn, recover_drop_fp, recover_gt_classes, recovered_objects,
)
from common.utils.image_memory import Frame
from common.utils.plot_utils import save_gif
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import DepthCamera, estimated_semantic_topdown, nearest_resized
from common.utils.dataset_writers import CocoFramesWriter, ObjectViewsWriter
from common.utils.retrieval import (
    RetrievedView, SegmentationStyle, ViewSelection, objects_views_mosaic, retrieve_resolved_views, retrieved_views, segmentation_tile,
)
from common.vision.detic import Vocabulary
from habitat_embodied_sl.evaluation_set import ClipImageEncoder, evaluation_frames
from common.utils.concept_graph import ConceptGraphMemory, object_beliefs, object_size_ranges, predicted_objects
from common.vision.detector import Detector

_VARIANTS = ("pseudo", "pseudo_gt_classes", "recovered")
_TILE_DOWNSCALE = 2
_GIF_FPS = 6
_STUCK_WINDOW_STEPS = 15
_STUCK_RADIUS_M = 0.3


def _is_stuck(trajectory: list, window: int, radius_m: float) -> bool:
    """True once the agent's last window positions all lie within radius_m of each other -- no net
    translation, even while its heading keeps oscillating between turns."""
    if len(trajectory) < window:
        return False
    recent = np.stack([position for position, _ in trajectory[-window:]])
    return bool(np.all(np.linalg.norm(recent - recent[0], axis=1) < radius_m))


def observe_frame(obs: dict, env: ExplorationEnv, lift_image_size: int) -> Frame:
    """obs' rgb, depth resized to lift_image_size x lift_image_size and depth-sensor pose."""
    rgb = obs["rgb"][:, :, :3]
    pose = sensor_pose_matrix(env.get_agent_state().sensor_states["depth"])
    return Frame(rgb, nearest_resized(obs["depth"].squeeze(), (lift_image_size, lift_image_size)).numpy(), pose)


def kept_vocabulary(object_params: DictConfig) -> Vocabulary:
    """Vocabulary of object_params' kept classes."""
    return Vocabulary(f"{object_params.vocab_name}_kept", tuple(kept_classes_from_args(object_params)))


def build_detector(args: argparse.Namespace, object_params: DictConfig) -> Detector:
    """Detector classifying among object_params' kept classes."""
    return Detector(
        args.config_file, args.checkpoint, kept_vocabulary(object_params), args.sam_checkpoint,
        score_thresh=args.score_thresh, nms_thresh=args.nms_thresh, mask_thresh=args.mask_thresh, device=args.device, input_size=args.input_image_size,
    )


@dataclass
class SceneMapper:
    """Fuses each frame the agent observes into fused, keeping its RLE-encoded ground truth in ground_truth_views."""

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
        self.fused = self.fused + predicted_objects(frame_id, frame, instances, self.camera, floor_height)
        self.ground_truth_views += encoded_views(frame_id, ground_truth_instances(obs, self.vocabulary, self.detector.classes))
        return frame, instances


def explore(env: ExplorationEnv, mapper: SceneMapper, obs: dict, steps: int, style: SegmentationStyle, ground_truth, images_dir: Path) -> Tuple[list, bool]:
    """Has mapper observe every frame from obs on until its agent stops (also after too many successive collisions),
    gets stuck or steps run out, saving a 3-tile (observation, belief map, ground truth) snapshot per step to
    images_dir/snapshot_<step>.png and returning (each step's labelled detector-prediction tile (segmentation_tile),
    whether the agent got stuck)."""
    collided, stuck = False, False
    prediction_tiles = []
    trajectory_px: list = []
    for step in tqdm(range(steps), desc="mapping", ncols=80):
        state = env.get_agent_state()
        heading = float(TopDownMap.get_polar_angle(state))
        trajectory_px.append(ground_truth.to_pixel(np.array(state.position)))
        ground_truth.reveal(np.array(state.position), heading)
        frame, instances = mapper.observe(obs, env)
        prediction_tiles.append(segmentation_tile(frame.rgb, instances, style, f"step{step:04d}"))
        mapper.agent.update_objects(object_beliefs(mapper.fused))
        action = mapper.agent.act(env.get_agent_state(), obs["depth"], collided)
        mosaic = exploration_snapshot(obs, mapper.agent, ground_truth, trajectory_px)
        if mosaic is not None:
            mosaic.save(images_dir / f"snapshot_{step:04d}.png")
        stuck = _is_stuck(mapper.agent.trajectory, _STUCK_WINDOW_STEPS, _STUCK_RADIUS_M)
        if action == "stop" or stuck:
            break
        obs, _, _, _ = env.step(action)
        collided = bool(np.any(obs["agent_collision_sensor"]))
    return prediction_tiles, stuck


def map_scene(scene: str, cfg: DictConfig, detector: Detector, style: SegmentationStyle, args: argparse.Namespace) -> Tuple[SceneMapper, GroundTruthObjects]:
    """(SceneMapper of the scene's exploration, the scene's annotated objects of detector's classes), saving every
    step's detector predictions (save_exploration_visuals) and every step's 3-tile snapshot assembled into
    exploration.webp (save_gif) to datasets/<run_name>/<scene>/."""
    env, habitat_cfg = build_env(scene, cfg.seed, cfg.collection.steps_per_scene, cfg.object_params, args.input_image_size)
    agent = build_agent("hybrid", np.random.default_rng(cfg.seed), env, habitat_cfg)
    obs = env.reset(env.episodes[0], rng=agent.rng)
    floor_offset = env.navmesh_floor_offset()
    agent.reset(np.array(env.get_agent_state().position), env.get_agent_state().position[1] - floor_offset)
    depth_cfg = habitat_cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    camera = DepthCamera(depth_cfg.hfov, depth_cfg.min_depth, depth_cfg.max_depth)
    vocabulary = resolve_classes(cfg.object_params.env_name, cfg.object_params.vocab_name)
    mapper = SceneMapper(agent, detector, camera, vocabulary, floor_offset, args.lift_image_size)
    gt_meshes = env.get_gt_meshes(detector.classes)
    map_config = agent.config.map_config
    ground_truth = env.gt_map_tracker(map_config.fov_deg, map_config.visibility_dist_m)
    ground_truth.set_object_footprints(gt_meshes)
    scene_dump_dir = constants.DATASET_ROOT / cfg.run_name / scene
    images_dir = scene_dump_dir / "images" / "exploration_snapshots"
    images_dir.mkdir(parents=True, exist_ok=True)
    prediction_tiles, stuck = explore(env, mapper, obs, cfg.collection.steps_per_scene, style, ground_truth, images_dir)
    if stuck:
        print(f"Warning: agent got stuck in {scene} (no net translation over the last {_STUCK_WINDOW_STEPS} steps)")
    save_gif(images_dir, "exploration.webp", fps=_GIF_FPS)
    save_exploration_visuals(agent, prediction_tiles, scene_dump_dir)
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
    their ground-truth class ("pseudo_gt_classes"), and also without false positives and with false negatives' views ("recovered")."""
    resolved, memory = mapper.fused.resolved_objects, mapper.fused.image_memory
    selection = ViewSelection(k=cfg.collection.k_retrieval)
    retrieved = retrieve_resolved_views(mapper.fused, mapper.camera, cfg.object_params.min_mesh_visibility_fraction, selection)
    pseudo = retrieved_views(resolved, memory, *retrieved)
    matches = matched_ground_truth(resolved, ground_truth)
    gt_classes = recover_gt_classes(pseudo, matches, ground_truth)
    annotation = GroundTruthAnnotation(ground_truth, mapper.ground_truth_views)
    recovered = recover_add_fn(recover_drop_fp(gt_classes, matches), annotation, matches, memory, selection)
    return dict(zip(_VARIANTS, (pseudo, gt_classes, recovered)))


def save_retrieval_mosaics(scene: str, scene_views: Dict[str, List[RetrievedView]], style: SegmentationStyle, root: Path) -> None:
    """root/<variant>/retrieval_mosaics/<scene>.png of every variant's scene_views (scene_variants): one row per object of
    its views, tiles downscaled by _TILE_DOWNSCALE, bordered red when recovery relabelled or added the object."""
    for variant, views in scene_views.items():
        mosaic = objects_views_mosaic(views, style, recovered_objects(scene_views[_VARIANTS[0]], views), _TILE_DOWNSCALE)
        if mosaic is None:
            continue
        out_path = root / variant / "retrieval_mosaics" / f"{scene}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mosaic.save(out_path)


def save_semantic_topdowns(scene: str, scene_views: Dict[str, List[RetrievedView]], mapper: SceneMapper, style: SegmentationStyle, root: Path) -> None:
    """root/<variant>/semantic_topdown_maps/<scene>.png of every variant's scene_views (scene_variants): the resolved objects
    its views show, by their class there, on the agent's map; recover_add_fn's objects have no points to draw."""
    for variant, views in scene_views.items():
        object_classes = {view.object_id: view.class_id for view in views if view.object_id >= 0}
        resolved = copy.deepcopy(mapper.fused.resolved_objects).relabeled(object_classes)
        out_path = root / variant / "semantic_topdown_maps" / f"{scene}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), estimated_semantic_topdown(resolved, mapper.agent.estimated_map, style.classes, style.colors)[:, :, ::-1])


def dataset_writers(cfg: DictConfig, classes: list) -> Tuple[Dict[str, ObjectViewsWriter], CocoFramesWriter]:
    """({variant: its train split's writer, datasets/<run_name>/<variant>/train/}, the shared test split's writer,
    datasets/<run_name>/test/), emptied."""
    root = constants.DATASET_ROOT / cfg.run_name
    shutil.rmtree(root, ignore_errors=True)
    retrieval = {**asdict(ViewSelection(k=cfg.collection.k_retrieval)), "min_mesh_visibility_fraction": cfg.object_params.min_mesh_visibility_fraction}
    train_writers = {variant: ObjectViewsWriter(root / variant / "train", {"categories": classes, "retrieval": retrieval}) for variant in _VARIANTS}
    info = {"evaluation": OmegaConf.to_container(cfg.evaluation), "frontier_agent": OmegaConf.to_container(cfg.frontier_agent)}
    return train_writers, CocoFramesWriter(root / "test", info, classes)


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    detector = build_detector(args, cfg.object_params)
    train_writers, test_writer = dataset_writers(cfg, detector.classes)
    style = SegmentationStyle(detector.classes, make_colors(len(detector.classes), seed=0, ctype=0))
    for scene in cfg.collection.scenes:
        mapper, ground_truth = map_scene(scene, cfg, detector, style, args)
        resolve_and_classify(mapper, cfg.object_params.min_mesh_visibility_fraction)
        scene_views = scene_variants(mapper, ground_truth, cfg)
        save_retrieval_mosaics(scene, scene_views, style, constants.DATASET_ROOT / cfg.run_name)
        save_semantic_topdowns(scene, scene_views, mapper, style, constants.DATASET_ROOT / cfg.run_name)
        for variant, views in scene_views.items():
            train_writers[variant].add_scene(scene, views)
    encoder = ClipImageEncoder(args.device)
    for scene in cfg.evaluation.scenes:
        test_writer.add_scene(scene, evaluation_frames(scene, cfg, detector.classes, args.input_image_size, encoder))
    rng = np.random.default_rng(cfg.seed)
    for writer in [*train_writers.values(), test_writer]:
        writer.write(rng)
    print(f"Train sets {', '.join(_VARIANTS)} and their test set written to {constants.DATASET_ROOT / cfg.run_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="habitat_embodied_sl/config/collection_procthor.yaml", help="Collection config: scenes, budgets, object_params.")
    parser.add_argument("--config-file", default="third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    parser.add_argument("--checkpoint", default="third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--input-image-size", type=int, default=1024, help="Side of the rendered frames, Detic's and SAM's input.")
    parser.add_argument("--lift-image-size", type=int, default=512, help="Side depth and masks are resized to for lifting and raytracing.")
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
