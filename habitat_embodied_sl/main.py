"""Builds retrieval datasets from a collection config (--config, see habitat_embodied_sl/config/collection_procthor.yaml
and habitat_embodied_sl/README.md). In each collection.scenes scene, common/hybrid_exploration's HybridFrontierAgent
explores in exhaustive mode for up to collection.steps_per_scene steps while a Detector (Detic's proposals scored by
objectness alone, SAM masks) segments every frame; its masks and region embeddings are fused into a concept-graphs
ConceptGraphMemory (common/utils/concept_graph.py) whose objects are the agent's beliefs. The map is then resolved, its
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
import trimesh
from detectron2.structures import Instances
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from common.hybrid_exploration.hybrid_agent import HybridFrontierAgent
from common.utils.eval_utils import GroundTruthObjects, matched_ground_truth
from common.env_utils.env_base import ExplorationEnv
from common.env_utils.object_annotations import PROCTHOR_NATIVE_SIZES_PATH, class_size_ranges, get_obj_from_id, object_world_mesh, resolve_classes
from common.env_utils.sense import ground_truth_instances
from common.env_utils.vocab_constants import make_colors
from common.utils.ground_truth_recovery import (
    GroundTruthAnnotation, GroundTruthView, encoded_views, recover_add_fn, recover_drop_fp, recover_gt_classes, recovered_objects,
)
from common.utils.image_memory import Frame
from common.utils.plot_utils import save_decision_map
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import DepthCamera, nearest_resized, semantic_topdown
from common.utils.dataset_writers import CocoFramesWriter, ObjectViewsWriter
from common.utils.retrieval import (
    RetrievedView, SegmentationStyle, ViewSelection, objects_views_mosaic, retrieve_views, retrieved_views, save_prediction_frame,
)
from common.vision.detic import Vocabulary
from habitat_embodied_al import constants
from habitat_embodied_al.reproduce.main import build_agent, build_env, kept_classes_from_args
from habitat_embodied_sl.evaluation_set import ClipImageEncoder, evaluation_frames
from common.utils.concept_graph import ConceptGraphMemory, object_beliefs, object_size_ranges, predicted_objects
from detector import Detector

_VARIANTS = ("pseudo", "pseudo_gt_classes", "recovered")
_TILE_DOWNSCALE = 2


def ground_truth_meshes(env: ExplorationEnv, classes: list) -> List[Tuple[int, int, trimesh.Trimesh]]:
    """(object id, class, world mesh) of every scene object the ground-truth sensor annotates with one of classes."""
    annotation, class_ids = env.get_object_annotation(), {name: idx for idx, name in enumerate(classes)}
    names = {object_id: annotation.classes[class_id] for object_id, class_id in annotation.obj_id_to_class_id.items()}
    objects = [
        (object_id, class_ids[name], get_obj_from_id(env.get_sim(), object_id)) for object_id, name in names.items() if name in class_ids
    ]
    meshes = [(object_id, class_id, object_world_mesh(obj)) for object_id, class_id, obj in objects if obj is not None]
    return [(object_id, class_id, mesh) for object_id, class_id, mesh in meshes if mesh is not None]


def retrieve_resolved_views(fused: ConceptGraphMemory, camera: DepthCamera, min_mesh_visibility_fraction: float, selection: ViewSelection) -> Tuple[dict, Dict[int, List[int]]]:
    """(frames raytraced with fused's resolved objects, the views retrieve_views selects of each)."""
    resolved, memory = fused.resolved_objects, fused.image_memory
    raytraced = memory.raytrace_frames(resolved, camera, min_mesh_visibility_fraction)
    return raytraced, retrieve_views(resolved, memory, raytraced, camera, selection)


def observe_frame(obs: dict, env: ExplorationEnv, lift_image_size: int) -> Frame:
    """obs' rgb, depth resized to lift_image_size x lift_image_size and depth-sensor pose."""
    rgb = obs["rgb"][:, :, :3]
    pose = sensor_pose_matrix(env.get_agent_state().sensor_states["depth"])
    return Frame(rgb, nearest_resized(obs["depth"].squeeze(), (lift_image_size, lift_image_size)).numpy(), pose)


def build_detector(args: argparse.Namespace, object_params: DictConfig) -> Detector:
    """Detector classifying among object_params' kept classes."""
    vocabulary = Vocabulary(f"{object_params.vocab_name}_kept", tuple(kept_classes_from_args(object_params)))
    return Detector(
        args.config_file, args.checkpoint, vocabulary, args.sam_checkpoint,
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


def explore(
    env: ExplorationEnv, mapper: SceneMapper, obs: dict, steps: int, style: SegmentationStyle, decision_frames_dir: Path, prediction_frames_dir: Path,
) -> None:
    """Has mapper observe every frame from obs on until its agent stops (also after too many successive collisions) or
    steps run out, saving each step's decision map to decision_frames_dir/step_<step>.png and its detector predictions
    to prediction_frames_dir/step_<step>.png."""
    collided = False
    for step in tqdm(range(steps), desc="mapping", ncols=80):
        frame, instances = mapper.observe(obs, env)
        mapper.agent.update_objects(object_beliefs(mapper.fused))
        action = mapper.agent.act(env.get_agent_state(), collided)
        save_decision_map(mapper.agent, decision_frames_dir / f"step_{step:04d}.png")
        save_prediction_frame(frame.rgb, instances, style, f"step{step:04d}", prediction_frames_dir / f"step_{step:04d}.png")
        if action == "stop":
            break
        obs, _, _, _ = env.step(action)
        collided = bool(np.any(obs["agent_collision_sensor"]))


def map_scene(scene: str, cfg: DictConfig, detector: Detector, style: SegmentationStyle, args: argparse.Namespace) -> Tuple[SceneMapper, GroundTruthObjects]:
    """(SceneMapper of the scene's exploration, the scene's annotated objects of detector's classes), saving the agent's
    final decision map as datasets/<run_name>/<variant>/decision_frames/<scene>.png in every variant's folder, and every
    step's decision map and detector predictions as datadump/<run_name>/<scene>/decision_frames/step_<step>.png and
    .../prediction_frames/step_<step>.png during exploration."""
    env, habitat_cfg = build_env(scene, cfg.seed, cfg.collection.steps_per_scene, cfg.object_params, args.input_image_size)
    agent = build_agent("hybrid", np.random.default_rng(cfg.seed), env, habitat_cfg)
    obs = env.reset(env.episodes[0], rng=agent.rng)
    depth_cfg = habitat_cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    camera = DepthCamera(depth_cfg.hfov, depth_cfg.min_depth, depth_cfg.max_depth)
    vocabulary = resolve_classes(cfg.object_params.env_name, cfg.object_params.vocab_name)
    mapper = SceneMapper(agent, detector, camera, vocabulary, env.navmesh_floor_offset(), args.lift_image_size)
    scene_dump_dir = constants.DATADUMP_ROOT / cfg.run_name / scene
    explore(env, mapper, obs, cfg.collection.steps_per_scene, style, scene_dump_dir / "decision_frames", scene_dump_dir / "prediction_frames")
    for variant in _VARIANTS:
        save_decision_map(agent, constants.DATASET_ROOT / cfg.run_name / variant / "decision_frames" / f"{scene}.png")
    ground_truth = GroundTruthObjects.from_meshes(ground_truth_meshes(env, detector.classes))
    env.close()
    return mapper, ground_truth


def resolve_and_classify(mapper: SceneMapper, min_mesh_visibility_fraction: float) -> None:
    """Resolves mapper's objects, keeps the foreground ones then classifies them among the classes their size fits."""
    size_ranges = object_size_ranges(mapper.detector.classes, class_size_ranges(PROCTHOR_NATIVE_SIZES_PATH))
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
        cv2.imwrite(str(out_path), semantic_topdown(resolved, mapper.agent, style.classes, style.colors)[:, :, ::-1])


def dataset_writers(cfg: DictConfig, classes: list) -> Tuple[Dict[str, ObjectViewsWriter], CocoFramesWriter]:
    """({variant: its train split's writer, datasets/<run_name>/<variant>/train/}, the shared test split's writer,
    datasets/<run_name>/test/), emptied."""
    root = constants.DATASET_ROOT / cfg.run_name
    shutil.rmtree(root, ignore_errors=True)
    retrieval = {**asdict(ViewSelection(k=cfg.collection.k_retrieval)), "min_mesh_visibility_fraction": cfg.object_params.min_mesh_visibility_fraction}
    train_writers = {variant: ObjectViewsWriter(root / variant / "train", {"categories": classes, "retrieval": retrieval}) for variant in _VARIANTS}
    return train_writers, CocoFramesWriter(root / "test", {"evaluation": OmegaConf.to_container(cfg.evaluation)}, classes)


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


def argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    return parser


if __name__ == "__main__":
    main(argument_parser(__doc__).parse_args())
