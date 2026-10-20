"""Semantic mapping run: common/hybrid_exploration's HybridFrontierAgent explores the HSSD-HAB scene --scene-name in
exhaustive mode while a Detector (Detic boxes, SAM masks) segments every frame; its predicted masks and region
embeddings are fused into a concept-graphs ConceptGraphMemory (common/utils/concept_graph.py), whose objects
are the agent's beliefs (object_beliefs: confidence linear in detections, full at _FULL_CONFIDENCE_DETECTIONS).
Once the exploration ends it is resolved, carved by SAM masks of a few diverse views of each object (with a SAM
checkpoint, ConceptGraphMemory.refine), then each object classified by Detic from the ensembled region embeddings
of its raytraced masks in every frame showing it (ConceptGraphMemory.classify). Saves, as tests/test_dummy_scene.py,
every step's decision map as decision.gif and predictions as predictions_mosaic.png, the last step's believed object
occupancy to object.png, mosaics of every step's ground truth and region crops, then from the resolved map: its
point cloud and its objects' box wireframes (semantic_wireframes.glb, one node per object), its semantic top-down map on the agent's top-down grid, a retrieval mosaic (one row per resolved object,
of the --k-retrieval diverse remembered views common/utils/retrieval.py's retrieve_views selects) and its first
column as the single row retrieval_first_views.png, plus
the ground-truth top-down map of the scene's annotated object mesh silhouettes and ground_truth_mesh.glb merging
every placed object of the scene, without its stage.

Usage (Detic's Swin-B checkpoint from https://github.com/facebookresearch/Detic's model zoo; Detic classifies
against the CLIP text embeddings of the --vocab-name/--filter-classes class names):
  export HABITAT_DATA=$HOME/habitat_data/data BASE_DIR=$(pwd) PYTHONNOUSERSITE=1
  PYTHONPATH=. python habitat_embodied_sl/main.py \
      --config-file third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
      --checkpoint third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
      --sam-checkpoint data/sam/sam_vit_b_01ec64.pth \
      --scene-name 102344022 --steps 500
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import trimesh
from scipy import ndimage
from detectron2.structures import Instances
from tqdm import tqdm

from common.hybrid_exploration.hybrid_agent import HybridFrontierAgent
from common.hybrid_exploration.object_viewpoints import ObjectBelief
from common.utils.concept_graph import ConceptGraphMemory, FrameDetections, project_detections_to_objects
from common.utils.eval_utils import GroundTruthObjects, evaluate_scene, save_resolved_wireframes, scene_confusion_matrix
from common.env_utils.env_base import ExplorationEnv
from common.env_utils.object_annotations import get_obj_from_id, object_world_mesh, objects_world_mesh, resolve_classes
from common.env_utils.sense import bbs_instances_as_gt, keep_valid_instances
from common.env_utils.vocab_constants import make_colors
from common.utils.image_memory import Frame
from common.utils.plot_utils import box_crop, make_mosaic, plot_semantic_2d_map, save_gif
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import DepthCamera, ResolvedObjects, save_resolved_ply
from common.utils.sam_refinement import RefinementConfig
from common.utils.retrieval import SegmentationStyle, ViewSelection, first_views_mosaic, retrieve_views, segmentation_tile, views_mosaic
from detector import Detector
from habitat_embodied_al.reproduce.main import add_object_args, build_agent, build_env, kept_classes_from_args, object_params_from_args

_MAP_SCALE = 8
_FOOTPRINT_CLOSING_ITERATIONS = 2
_CROP_TILE_SIZE = 320
_FULL_CONFIDENCE_DETECTIONS = 5
_GIF_FRAME_MS = 300
_RETRIEVAL_K = 10


@dataclass
class MappingProgress:
    prediction_tiles: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    ground_truth_tiles: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    decision_tiles: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    crop_rows: List[List[Tuple[str, np.ndarray]]] = field(default_factory=list)


def depth_camera(habitat_cfg) -> DepthCamera:
    depth_cfg = habitat_cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    return DepthCamera(depth_cfg.hfov, depth_cfg.min_depth, depth_cfg.max_depth)


def predicted_objects(frame: Frame, instances: Instances, camera: DepthCamera, floor_height: float) -> FrameDetections:
    """One ConceptGraphObject per prediction on frame, with its Detic region embedding."""
    if len(instances) == 0:
        return FrameDetections([], np.empty((0, 3)))
    return project_detections_to_objects(frame.depth, frame.pose, instances, camera, floor_height)


def object_beliefs(fused: ConceptGraphMemory) -> Dict[int, ObjectBelief]:
    """Every fused object's points, its confidence growing linearly with its detections up to _FULL_CONFIDENCE_DETECTIONS."""
    return {
        object_id: ObjectBelief(min(obj.object_num_dets, _FULL_CONFIDENCE_DETECTIONS) / _FULL_CONFIDENCE_DETECTIONS, obj.points)
        for object_id, obj in enumerate(fused.objects)
    }


def ground_truth_instances(obs: dict, vocabulary: list, classes: list) -> Instances:
    """obs' valid ground-truth instances of classes, their class ids remapped from the vocabulary's to classes'."""
    instances = bbs_instances_as_gt(keep_valid_instances(obs["bbsgt"]["instances"]))
    class_ids = {name: idx for idx, name in enumerate(classes)}
    names = [vocabulary[class_id] for class_id in instances.gt_classes.tolist()]
    kept = [idx for idx, name in enumerate(names) if name in class_ids]
    instances = instances[kept]
    instances.gt_classes = torch.tensor([class_ids[names[idx]] for idx in kept], dtype=torch.long)
    instances.gt_masks = instances.gt_masks.reshape(len(instances), *instances.image_size)
    return instances


def crop_tiles(frame: Frame, instances: Instances, detector: Detector) -> List[Tuple[str, np.ndarray]]:
    """Each prediction's box crop, labelled with Detic's class, with its region embedding's top classes and their probabilities drawn on it."""
    if len(instances) == 0:
        return []
    class_probs = detector.region_class_probs(instances.region_features)
    return [
        (f"detic {detector.class_name(detic_class)}", detector.draw_top_classes(probs, box_crop(frame.rgb, box, _CROP_TILE_SIZE)))
        for box, probs, detic_class in zip(instances.pred_boxes.tensor.int().tolist(), class_probs, instances.pred_classes.tolist())
    ]


def filled_footprint(pixels: np.ndarray, shape: tuple) -> np.ndarray:
    """Top-down mask of (N, 2) map pixels, with gaps between them closed and holes filled."""
    mask = np.zeros(shape, dtype=bool)
    mask[pixels[:, 0], pixels[:, 1]] = True
    return ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=_FOOTPRINT_CLOSING_ITERATIONS) | mask)


def object_footprints(fused: ResolvedObjects, agent: HybridFrontierAgent) -> List[Tuple[int, np.ndarray]]:
    """(class, filled top-down footprint) of every resolved object on the agent's map."""



def semantic_topdown(fused: ResolvedObjects, agent: HybridFrontierAgent, classes: list, colors: list) -> np.ndarray:
    """RGB top-down map of fused's resolved objects, by class, over the agent's navigable map."""
    pixels = agent.to_pixels(fused.resolved_points)
    inside = ((pixels >= 0) & (pixels < np.array(agent.top_down_map.shape))).all(axis=1)
    pixels, infos = pixels[inside], fused.resolved_infos[inside]
    object_classes = dict(zip(infos[:, 1], infos[:, 0]))
    object_footprints = [
        (object_classes[object_id], filled_footprint(pixels[infos[:, 1] == object_id], agent.top_down_map.shape))
        for object_id in object_classes
    ]

    image = plot_semantic_2d_map(
        agent.top_down_map, object_footprints, dict(enumerate(colors)), dict(enumerate(classes)),
        meters_per_grid_pixel=agent.meters_per_pixel(), scale=_MAP_SCALE,
    )
    return np.array(image)


def mesh_footprint(mesh: trimesh.Trimesh, agent: HybridFrontierAgent) -> np.ndarray:
    """Top-down silhouette of mesh on the agent's map: the union of its triangles' floor projections."""
    triangles = np.unique(agent.to_pixels(mesh.vertices)[mesh.faces][:, :, ::-1].astype(np.int32), axis=0)
    mask = np.zeros(agent.top_down_map.shape, dtype=np.uint8)
    # one call per triangle: a multi-polygon fillPoly fills by parity, leaving overlaps empty
    for triangle in triangles:
        cv2.fillConvexPoly(mask, triangle, 1)
    return mask > 0


def ground_truth_meshes(env: ExplorationEnv, classes: list) -> List[Tuple[int, trimesh.Trimesh]]:
    """(class, world mesh) of every scene object annotated with one of classes."""
    class_ids = {name: idx for idx, name in enumerate(classes)}
    objects = [
        (class_ids[info["class_name"]], get_obj_from_id(env.get_sim(), info["object_id"]))
        for info in env.get_object_occupancy().list_object_info if info["class_name"] in class_ids
    ]
    meshes = [(class_id, object_world_mesh(obj)) for class_id, obj in objects if obj is not None]
    return [(class_id, mesh) for class_id, mesh in meshes if mesh is not None]


def ground_truth_topdown(meshes: List[Tuple[int, trimesh.Trimesh]], agent: HybridFrontierAgent, classes: list, colors: list) -> np.ndarray:
    """RGB top-down map of the silhouette of every (class, world mesh) of meshes (ground_truth_meshes)."""
    footprints = [(class_id, mesh_footprint(mesh, agent)) for class_id, mesh in meshes]
    image = plot_semantic_2d_map(
        agent.top_down_map, footprints, dict(enumerate(colors)), dict(enumerate(classes)),
        meters_per_grid_pixel=agent.meters_per_pixel(), scale=_MAP_SCALE,
    )
    return np.array(image)


def save_exploration_visuals(agent: HybridFrontierAgent, decision_tiles: list, prediction_tiles: list, out_dir: Path) -> None:
    """decision.gif and predictions_mosaic.png of every step, and object.png of the last step."""
    save_gif(decision_tiles, str(out_dir / "decision.gif"), _GIF_FRAME_MS)
    make_mosaic(prediction_tiles, N_cols=5).save(out_dir / "predictions_mosaic.png")
    make_mosaic([agent.object_map_tile()], N_cols=1).save(out_dir / "object.png")


def save_retrieval_mosaics(fused: ConceptGraphMemory, camera: DepthCamera, style: SegmentationStyle, args: argparse.Namespace, out_dir: Path) -> None:
    """retrieval_mosaic.png, one row per resolved object, largest first, of the --k-retrieval views retrieve_views selects,
    and retrieval_first_views.png, its first column as a single row."""
    resolved, memory = fused.resolved_objects, fused.image_memory
    raytraced = memory.raytrace_frames(resolved, camera, args.min_visibility_fraction)
    object_frames = retrieve_views(resolved, memory, raytraced, camera, ViewSelection(k=args.k_retrieval))
    if not any(object_frames.values()):
        return
    views_mosaic(memory, raytraced, style, object_frames).save(out_dir / "retrieval_mosaic.png")
    first_views_mosaic(memory, raytraced, style, object_frames).save(out_dir / "retrieval_first_views.png")


def observe_frame(obs: dict, env: ExplorationEnv) -> Frame:
    """obs' rgb, depth and depth-sensor pose."""
    rgb = obs["rgb"][:, :, :3]
    pose = sensor_pose_matrix(env.get_agent_state().sensor_states["depth"])
    return Frame(rgb, obs["depth"].squeeze(), pose)


def main(args: argparse.Namespace) -> None:
    detector = Detector(
        args.config_file, 
        args.checkpoint, 
        kept_classes_from_args(args), 
        args.sam_checkpoint,
        agnostic_nms_thresh=args.agnostic_nms_thresh, 
        score_thresh=args.score_thresh, 
        mask_thresh=args.mask_thresh,
        objectness_weight=args.objectness_weight,
        device=args.device
    )
    
    env, habitat_cfg = build_env(args.scene_name, args.seed, args.steps, object_params_from_args(args))
    agent = build_agent("hybrid", np.random.default_rng(args.seed), env, habitat_cfg)
    

    colors = make_colors(len(detector.classes), seed=0, ctype=0)
    style = SegmentationStyle(detector.classes, colors)
    vocabulary = resolve_classes(args.env_name, args.vocab_name)
    obs = env.reset(env.episodes[0], rng=agent.rng)

    fused, progress = ConceptGraphMemory(), MappingProgress()
    camera = depth_camera(habitat_cfg)
    out_dir = Path(args.out_dir) if args.out_dir else Path("datadump/semantic_map") / f"{args.scene_name}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)


    for step in tqdm(range(args.steps), desc="mapping", ncols=80):
        # print(f"Step {step}: agent at {env.get_agent_state().position}")

        frame = observe_frame(obs, env)
        fused.image_memory.add_frame(frame)
        instances = detector.predict(frame.rgb)
        fused = fused + predicted_objects(frame, instances, camera, env.get_agent_state().position[1])
        agent.update_objects(object_beliefs(fused))
        action = agent.act(env.get_agent_state())
        label = f"step{step:03d}"
        progress.prediction_tiles.append(segmentation_tile(frame.rgb, instances, style, label))
        progress.ground_truth_tiles.append(segmentation_tile(frame.rgb, ground_truth_instances(obs, vocabulary, detector.classes), style, label))
        progress.decision_tiles.append(agent.map_tile())
        progress.crop_rows.append(crop_tiles(frame, instances, detector))

        if action == "stop":
            break
        obs, _, _, _ = env.step(action)

    meshes = ground_truth_meshes(env, detector.classes)
    ground_truth = ground_truth_topdown(meshes, agent, detector.classes, colors)
    objects_world_mesh(env.get_sim().get_rigid_object_manager().get_objects_by_handle_substring().values()).export(out_dir / "ground_truth_mesh.glb")
    env.close()

    print("Resolving...")
    fused.resolve()
    if detector.masker is not None:
        fused.refine(detector.masker, camera, RefinementConfig(min_visibility_fraction=args.min_visibility_fraction))
    fused.classify(detector, camera, args.min_visibility_fraction)

    print("Saving resolved ply...")
    save_resolved_ply(fused.resolved_objects, colors, out_dir / "semantic_point_cloud.ply")
    save_resolved_wireframes(fused.resolved_objects, detector.classes, colors, out_dir / "semantic_wireframes.glb")

    print("Saving topdown maps...")
    cv2.imwrite(str(out_dir / "semantic_topdown.png"), semantic_topdown(fused.resolved_objects, agent, detector.classes, colors)[:, :, ::-1])
    cv2.imwrite(str(out_dir / "ground_truth_topdown.png"), ground_truth[:, :, ::-1])
    #save_retrieval_mosaics(fused, camera, style, args, out_dir)

    print("Evaluating...")
    ground_truth_objects = GroundTruthObjects.from_meshes(meshes)
    evaluate_scene(fused.resolved_objects, ground_truth_objects, detector.classes)
    scene_confusion_matrix(fused.resolved_objects, ground_truth_objects, detector.classes, out_dir / "confusion_matrix.png")

    print(f"{len(progress.decision_tiles)} steps mapped, mosaics and final semantic_point_cloud.ply and semantic_wireframes.glb saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", default="third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    parser.add_argument("--checkpoint", default = "third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--scene-name", default="102344022")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agnostic-nms-thresh", type=float, default=0.7)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--objectness-weight", type=float, default=0.75)
    parser.add_argument("--k-retrieval", type=int, default=_RETRIEVAL_K)
    parser.add_argument("--device", default="cuda")
    
    add_object_args(parser)
    parser.set_defaults(env_name="HSSD-HAB", vocab_name="COCO80", filter_classes=["chair", "couch", "potted plant", "bed", "toilet", "tv"])
    main(parser.parse_args())
