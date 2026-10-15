"""Semantic mapping run: common/hybrid_exploration's HybridFrontierAgent explores the HSSD-HAB scene --scene-name in
exhaustive mode while a Detector (Detic boxes, SAM masks) segments every frame; its predicted masks and region
embeddings are fused into a concept-graphs ConceptGraphObjectSet (common/utils/concept_graph.py), whose objects
are the agent's beliefs (object_beliefs: confidence linear in detections, full at _FULL_CONFIDENCE_DETECTIONS).
Once the exploration ends it is resolved, then each object classified by Detic from the ensembled region embeddings
of its raytraced masks in every frame showing it (Detector.classify_objects). Saves, as tests/test_dummy_scene.py,
every step's decision map and predictions as decision.gif and predictions.gif, the last step's believed object
occupancy to object.png, mosaics of every step's ground truth and region crops, then from the resolved map: its
point cloud, its semantic top-down map on the agent's top-down grid, a retrieval mosaic (one row per resolved object,
of its --k-retrieval largest remembered views) and its greedy_retrieval_mosaic.png counterpart (greedy_object_frames), plus
the ground-truth top-down map of the scene's annotated object mesh silhouettes and ground_truth_mesh.glb merging
every placed object of the scene, without its stage.

Usage (Detic's Swin-B checkpoint from https://github.com/facebookresearch/Detic's model zoo; Detic classifies
against the CLIP text embeddings of the --vocab-name/--filter-classes class names):
  export HABITAT_DATA=$HOME/habitat_data/data BASE_DIR=$(pwd) PYTHONNOUSERSITE=1
  PYTHONPATH=. python habitat_embodied_sl/main.py \
      --config-file third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
      --checkpoint third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
      --sam-checkpoint data/sam/sam_vit_b_01ec64.pth \
      --scene-name 102816756 --steps 200
"""
import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch
import trimesh
from scipy import ndimage
from detectron2.structures import Instances
from tqdm import tqdm

from common.hybrid_exploration.hybrid_agent import HybridFrontierAgent
from common.hybrid_exploration.object_viewpoints import ObjectBelief
from common.utils.concept_graph import ConceptGraphObject, ConceptGraphObjectSet, project_detections_to_objects
from common.env_utils.env_base import ExplorationEnv
from common.env_utils.object_annotations import get_obj_from_id, object_world_mesh, objects_world_mesh, resolve_classes
from common.env_utils.sense import bbs_instances_as_gt, keep_valid_instances
from common.env_utils.vocab_constants import make_colors
from common.utils.midas import Midas
from common.utils.image_memory import Frame, ImageMemory
from common.utils.plot_utils import box_crop, make_mosaic, plot_semantic_2d_map, save_gif
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import DepthCamera, save_resolved_ply
from common.utils.retrieval import ROW_FRAMES, SegmentationStyle, objects_by_size, retrieval_mosaic, segmentation_tile, views_mosaic
from common.utils.view_selection import ViewSelection, above_mean_quality, object_view_geometry, select_views
from detector import Detector
from habitat_embodied_al.reproduce.main import add_object_args, build_agent, build_env, kept_classes_from_args, object_params_from_args

_MAP_SCALE = 8
_FOOTPRINT_CLOSING_ITERATIONS = 2
_CROP_TILE_SIZE = 320
_FULL_CONFIDENCE_DETECTIONS = 5
_GIF_FRAME_MS = 300
_VIEW_SELECTION = ViewSelection(k=ROW_FRAMES, diversity_weight=1.0, bandwidth=np.pi / 4, quality_filter=above_mean_quality)


@dataclass
class MappingProgress:
    prediction_tiles: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    ground_truth_tiles: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    decision_tiles: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    crop_rows: List[List[Tuple[str, np.ndarray]]] = field(default_factory=list)


def depth_camera(habitat_cfg) -> DepthCamera:
    depth_cfg = habitat_cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    return DepthCamera(depth_cfg.hfov, depth_cfg.min_depth, depth_cfg.max_depth)


def predicted_objects(frame: Frame, instances: Instances, camera: DepthCamera) -> List[ConceptGraphObject]:
    """One ConceptGraphObject per prediction on frame, with its Detic region embedding."""
    if len(instances) == 0:
        return []
    return project_detections_to_objects(frame.depth, frame.pose, instances, camera.hfov, camera.min_depth, camera.max_depth)


def object_beliefs(fused: ConceptGraphObjectSet) -> Dict[int, ObjectBelief]:
    """Every fused object's points, its confidence growing linearly with its detections up to _FULL_CONFIDENCE_DETECTIONS."""
    return {
        object_id: ObjectBelief(min(obj.num_detections, _FULL_CONFIDENCE_DETECTIONS) / _FULL_CONFIDENCE_DETECTIONS, obj.points)
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


def object_footprints(fused: ConceptGraphObjectSet, agent: HybridFrontierAgent) -> List[Tuple[int, np.ndarray]]:
    """(class, filled top-down footprint) of every resolved object on the agent's map."""



def semantic_topdown(fused: ConceptGraphObjectSet, agent: HybridFrontierAgent, classes: list, colors: list) -> np.ndarray:
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


def depth_estimator(depth_source: str, camera: DepthCamera, device: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """(rgb, sensor depth) -> depth: the sensor's own, or MiDaS' fitted to its unclipped pixels."""
    if depth_source == "sensor":
        return lambda rgb, depth: depth
    midas = Midas(device)
    return lambda rgb, depth: midas.depth_estimate(rgb, np.where((depth > camera.min_depth) & (depth < camera.max_depth), depth, 0.0))[0]


def greedy_object_frames(fused: ConceptGraphObjectSet, memory: ImageMemory, raytraced: dict, num_frames: int, hfov: float) -> Dict[int, List[int]]:
    """{object id: num_frames frames _VIEW_SELECTION selects} of every resolved object of fused, largest first."""
    selection = replace(_VIEW_SELECTION, k=num_frames)
    return {
        object_id: select_views(object_view_geometry(fused, memory, raytraced, object_id, hfov), selection)
        for object_id in objects_by_size(fused)
    }


def save_exploration_gifs(agent: HybridFrontierAgent, decision_tiles: list, prediction_tiles: list, out_dir: Path) -> None:
    """decision.gif and predictions.gif of every step, and object.png of the last step."""
    save_gif(decision_tiles, str(out_dir / "decision.gif"), _GIF_FRAME_MS)
    save_gif(prediction_tiles, str(out_dir / "predictions.gif"), _GIF_FRAME_MS)
    make_mosaic([agent.object_map_tile()], N_cols=1).save(out_dir / "object.png")


def retrieval_mosaics(fused: ConceptGraphObjectSet, memory: ImageMemory, camera: DepthCamera, style: SegmentationStyle, args: argparse.Namespace) -> dict:
    """{file name: mosaic or None} of retrieval_mosaic.png and greedy_retrieval_mosaic.png, one row per resolved object of --k-retrieval frames."""
    raytraced = memory.raytrace_frames(fused, camera, args.min_visibility_fraction)
    object_frames = greedy_object_frames(fused, memory, raytraced, args.k_retrieval, camera.hfov)
    return {
        "retrieval_mosaic.png": retrieval_mosaic(memory, raytraced, style, objects_by_size(fused), args.k_retrieval),
        "greedy_retrieval_mosaic.png": views_mosaic(memory, raytraced, style, object_frames),
    }


def observe_frame(obs: dict, env: ExplorationEnv, estimate_depth: Callable) -> Frame:
    """obs' rgb, estimated depth and depth-sensor pose."""
    rgb = obs["rgb"][:, :, :3]
    pose = sensor_pose_matrix(env.get_agent_state().sensor_states["depth"])
    return Frame(rgb, estimate_depth(rgb, obs["depth"].squeeze()), pose)


def main(args: argparse.Namespace) -> None:
    detector = Detector(
        args.config_file, 
        args.checkpoint, 
        kept_classes_from_args(args), 
        args.sam_checkpoint,
        agnostic_nms_thresh=args.agnostic_nms_thresh, 
        score_thresh=args.score_thresh, 
        device=args.device
    )
    
    env, habitat_cfg = build_env(args.scene_name, args.seed, args.steps, object_params_from_args(args))
    agent = build_agent("hybrid", np.random.default_rng(args.seed), env, habitat_cfg)
    

    colors = make_colors(len(detector.classes), seed=0, ctype=0)
    style = SegmentationStyle(detector.classes, colors)
    vocabulary = resolve_classes(args.env_name, args.vocab_name)
    obs = env.reset(env.episodes[0], rng=agent.rng)

    fused = ConceptGraphObjectSet()
    progress, memory = MappingProgress(), ImageMemory()
    camera = depth_camera(habitat_cfg)
    estimate_depth = depth_estimator(args.depth_source, camera, args.device)
    out_dir = Path(args.out_dir) if args.out_dir else Path("datadump/semantic_map") / f"{args.scene_name}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)


    for step in tqdm(range(args.steps), desc="mapping", ncols=80):
        # print(f"Step {step}: agent at {env.get_agent_state().position}")

        frame = observe_frame(obs, env, estimate_depth)
        memory.add_frame(frame)
        instances = detector.predict(frame.rgb)
        fused = fused + predicted_objects(frame, instances, camera)
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

    ground_truth = ground_truth_topdown(ground_truth_meshes(env, detector.classes), agent, detector.classes, colors)
    objects_world_mesh(env.get_sim().get_rigid_object_manager().get_objects_by_handle_substring().values()).export(out_dir / "ground_truth_mesh.glb")
    env.close()
    fused.resolve()
    fused = detector.classify_resolved(fused, memory, camera, args.min_visibility_fraction)

    save_exploration_gifs(agent, progress.decision_tiles, progress.prediction_tiles, out_dir)
    make_mosaic(progress.ground_truth_tiles, N_cols=5).save(out_dir / "ground_truth_mosaic.png")
    make_mosaic([tile for row in progress.crop_rows for tile in row], row_sizes=[len(row) for row in progress.crop_rows]).save(out_dir / "region_crops_mosaic.png")
    save_resolved_ply(fused, colors, out_dir / "semantic_point_cloud.ply")
    cv2.imwrite(str(out_dir / "semantic_topdown.png"), semantic_topdown(fused, agent, detector.classes, colors)[:, :, ::-1])
    cv2.imwrite(str(out_dir / "ground_truth_topdown.png"), ground_truth[:, :, ::-1])
    for name, mosaic in retrieval_mosaics(fused, memory, camera, style, args).items():
        if mosaic is not None:
            mosaic.save(out_dir / name)

    print(f"{len(progress.decision_tiles)} steps mapped, mosaics and final semantic_point_cloud.ply saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", default="third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    parser.add_argument("--checkpoint", default = "third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
    parser.add_argument("--sam-checkpoint", default="data/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("--scene-name", default="102816756")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agnostic-nms-thresh", type=float, default=0.7)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--depth-source", choices=["sensor", "midas"], default="sensor")
    parser.add_argument("--k-retrieval", type=int, default=ROW_FRAMES, help="Frames per retrieval mosaic row, one row per resolved object.")
    parser.add_argument("--device", default="cuda")
    
    add_object_args(parser)
    parser.set_defaults(env_name="HSSD-HAB", vocab_name="HSSD80", filter_classes=[])
    main(parser.parse_args())
