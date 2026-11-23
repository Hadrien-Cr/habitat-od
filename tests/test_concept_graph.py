"""
Standalone viewer for common/utils/concept_graph.py's fusion and resolution.

Runs against procthor_scene.py's ProcTHOR-hab scene (ProcTHOR-Train-0 unless --scene-id says otherwise) via
ObjectDetectorGTSensor on a raw habitat_sim.Simulator (no ExplorationEnv/Hydra config), driven by a FrontierExplorationAgent
(common/hybrid_exploration/frontier_exp_agent.py) with vanilla move_forward/turn_left/turn_right actions from a
random start pose. Each fused frame's gt_instances (GT masks and depth) get the Detic region embedding of their box, are
projected with project_detections_to_objects and fused with add_processed; resolve then runs concept-graphs'
filter/denoise, keep_foreground keeps the objects detected in enough of the frames showing them and classify classifies
each object from its raytraced masks in every explored frame.

Write: fuses every other frame of a 2 * --num-frames step exploration run, resolves the map once,
and saves <stem>.ply, a merged .glb of the scene (every placed rigid object, each loaded from its
source asset and transformed by its runtime scale/pose - habitat-sim can't export an assembled
scene) and an input_mosaic.png of every fused frame's GT segmentation.

Read: raytraces the resolved objects in each held-out frame (the ones in between, never fused) with
ResolvedObjects.visible_objects, with the GT sensor's own min_mesh_visibility_fraction, tiled into a
backprojected_mosaic.png, showing how well the map generalizes to unseen views.
Retrieve: for each of the --retrieval-objects largest resolved objects, common/utils/retrieval.py's
retrieve_resolved_views selects diverse frames among every exploration frame (fused and held-out) it appears in, tiled
one object per row into a retrieval_mosaic.png.
Finally saves the FrontierExplorationAgent's exploration_map.png (free/explored area, trajectory)
and an exploration_mosaic.png of that map at every step.

Has no pytest test functions - run it directly. Requires HABITAT_DATA pointing at real ai2thor-hab data
(see INSTALL.MD).
"""

import argparse
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple

import cv2
import habitat_sim
import numpy as np
import torch
from detectron2.structures import Boxes, Instances

from habitat.config.default_structured_configs import ObjectDetectorGTSensorConfig
from common.env_utils.object_annotations import resolve_classes, scene_world_mesh
from common.env_utils.object_detector_sensors import ObjectDetectorGTSensor
from common.env_utils.sense import bbs_instances_as_gt, keep_valid_instances
from common.env_utils.vocab_constants import make_colors
from common.hybrid_exploration.exploration_map import MapConfig
from common.hybrid_exploration.frontier_exp_agent import DiscretePlannerConfig, FrontierExplorationAgent, FrontierExplorationConfig
from common.utils.concept_graph import (
    ConceptGraphConfig, ConceptGraphObject, FrameDetections, add_processed, classify, keep_foreground, processed_frame, project_detections_to_objects,
    ResolveConfig, resolve,
)
from common.utils.image_memory import Frame, object_views, raytrace_frames
from common.utils.plot_utils import make_mosaic
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import ResolvedObjects, save_resolved_ply
from common.utils.retrieval import SegmentationStyle, ViewSelection, instances_from_masks, largest_objects, retrieve_resolved_views, segmentation_tile, views_mosaic
from common.vision.detic import Vocabulary
from common.vision.detector import Detector
from procthor_scene import CAMERA_HEIGHT, DATASET_CONFIG, DEPTH_CAMERA, ENV_NAME, HFOV_DEG, SCENE_ID, TURN_ANGLE_DEG, VOCAB_NAME, floor_height, floor_island, procthor_sim

_MIN_MESH_VISIBILITY_FRACTION = 0.2


def _random_agent_state(sim, rng, island_index: int = -1) -> habitat_sim.AgentState:
    state = habitat_sim.AgentState()
    state.position = sim.pathfinder.get_random_navigable_point(island_index=island_index)
    yaw = rng.uniform(0, 2 * np.pi)
    state.rotation = np.quaternion(np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)
    return state


def _frame_gt_instances(sensor: ObjectDetectorGTSensor, obs: dict, classes_keep: set) -> Instances:
    """The frame's valid gt_instances, without those whose class isn't in classes_keep when it isn't empty."""
    instances = sensor.decompose_frame(obs["semantic"], obs["depth"])["instances"]
    gt_instances = keep_valid_instances(bbs_instances_as_gt(instances))
    if not classes_keep or not len(gt_instances):
        return gt_instances
    classes = sensor.get_classes()
    return gt_instances[[i for i, c in enumerate(gt_instances.gt_classes.tolist()) if classes[c] in classes_keep]]


@dataclass
class _Viewer:
    sim: habitat_sim.Simulator
    sensor: ObjectDetectorGTSensor
    colors: list
    out_dir: str
    out_stem: str
    rng: np.random.Generator
    island_index: int
    explorer: FrontierExplorationAgent
    map_tiles: list = field(default_factory=list)


def viewer_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default=SCENE_ID, help="ProcTHOR-hab scene id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=100, help="exploration frames fused into the map, and as many interleaved held-out views backprojected from it")
    parser.add_argument("--classes-keep", nargs="*", default=[], help="drop gt_instances whose class isn't in this list; empty means keep every class")
    parser.add_argument("--retrieval-objects", type=int, default=10, help="largest resolved objects whose retrieved frames are saved as retrieval_mosaic.png rows")
    parser.add_argument("--out", default=None, help="output .ply path (default: tests/testdump/test_concept_graph/<scene_id>.ply); a matching .glb merged scene mesh is saved alongside it")
    parser.add_argument("--config-file", required=True, help="Detic config whose region embeddings are fused")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sam-checkpoint", default=os.path.join(os.environ.get("HABITAT_DATA", ""), "sam", "sam_vit_b_01ec64.pth"))
    return parser


@torch.no_grad()
def frame_objects(detector: Detector, frame_id: int, frame: Frame, gt_instances: Instances) -> FrameDetections:
    features, _, transform = detector._backbone(frame.rgb)
    boxes = Boxes(torch.as_tensor(transform.apply_box(gt_instances.gt_boxes.tensor.numpy()), dtype=torch.float32, device=detector._device))
    gt_instances.region_features = detector._region_embeddings(features, boxes).cpu()
    gt_instances.scores = torch.ones(len(gt_instances))
    return project_detections_to_objects(frame_id, frame, gt_instances, DEPTH_CAMERA, frame.pose[1, 3] - CAMERA_HEIGHT)


def resolve_and_classify(detector: Detector, objects: List[ConceptGraphObject], frames: Dict[int, Frame]) -> ResolvedObjects:
    config = ResolveConfig()
    resolved, sources = resolve(objects, ConceptGraphConfig.downsample_voxel_size, config)
    scores = keep_foreground(resolved, sources, raytrace_frames(resolved, frames, DEPTH_CAMERA, _MIN_MESH_VISIBILITY_FRACTION), config.min_detection_rate)
    classify(resolved, scores, object_views(resolved, frames, DEPTH_CAMERA, _MIN_MESH_VISIBILITY_FRACTION), detector, {})
    return resolved


def run_viewer(args: argparse.Namespace, detector: Detector) -> Tuple[_Viewer, ResolvedObjects, Dict[int, Frame]]:
    """Fuses every other frame of an exploration run, resolves and classifies the map, saves it, backprojects it into the
    held-out frames and returns it with every explored frame."""
    if not os.path.exists(DATASET_CONFIG):
        raise SystemExit(f"no ProcTHOR-hab data at {DATASET_CONFIG} (check HABITAT_DATA)")
    viewer = _start_viewer(args)
    try:
        run_st = time.time()
        objects, frames, held_out = _fuse_frames(viewer, args, detector)
        resolved = resolve_and_classify(detector, objects, frames)
        if len(resolved.resolved_points) == 0:
            raise SystemExit("none of the explored frames produced anything to resolve")
        print(f"{args.scene_id}: fused and resolved in {time.time() - run_st:.3f}s, {len(resolved.resolved_points)} resolved points")
        _save_map(viewer, args, resolved)
        _save_backprojections(viewer, resolved, frames, held_out)
        cv2.imwrite(os.path.join(viewer.out_dir, f"{viewer.out_stem}_exploration_map.png"), viewer.explorer.map_image())
        make_mosaic(viewer.map_tiles, N_cols=5).save(os.path.join(viewer.out_dir, f"{viewer.out_stem}_exploration_mosaic.png"))
        return viewer, resolved, frames
    finally:
        viewer.sim.close()


def _fuse_frames(viewer: _Viewer, args: argparse.Namespace, detector: Detector) -> Tuple[List[ConceptGraphObject], Dict[int, Frame], List[int]]:
    """(objects fused from every other frame, every frame, the ids of the frames held out), saving input_mosaic.png."""
    input_tiles, objects, frames, held_out, config = [], [], {}, [], ConceptGraphConfig()
    for step, (obs, agent_state) in enumerate(exploration_frames(viewer, 2 * args.num_frames)):
        frame_id = len(frames)
        frames[frame_id] = Frame(*_frame_geometry(obs, agent_state))
        gt_instances = _frame_gt_instances(viewer.sensor, obs, set(args.classes_keep))
        if step % 2:
            held_out.append(frame_id)
            continue
        input_tiles.append(_segmentation_tile(viewer, frames[frame_id].rgb, gt_instances, f"frame{frame_id:03d}"))
        if len(gt_instances):
            st = time.time()
            objects = add_processed(objects, processed_frame(frame_objects(detector, frame_id, frames[frame_id], gt_instances), config), config)
            print(f"frame {frame_id:03d}: inserted {len(gt_instances)} instances in {time.time() - st:.3f}s")
    make_mosaic(input_tiles, N_cols=4).save(os.path.join(viewer.out_dir, f"{viewer.out_stem}_input_mosaic.png"))
    return objects, frames, held_out


def _save_map(viewer: _Viewer, args: argparse.Namespace, resolved: ResolvedObjects) -> None:
    out_path = args.out or os.path.join(viewer.out_dir, f"{viewer.out_stem}.ply")
    save_resolved_ply(resolved, viewer.colors, out_path)
    print(f"saved to {out_path}")
    mesh_st = time.time()
    mesh_path = os.path.splitext(out_path)[0] + ".glb"
    scene_mesh = scene_world_mesh(viewer.sim)
    if not scene_mesh.is_empty:
        scene_mesh.export(mesh_path)
    print(f"saved merged scene mesh to {mesh_path} in {time.time() - mesh_st:.3f}s")


def _save_backprojections(viewer: _Viewer, resolved: ResolvedObjects, frames: Dict[int, Frame], held_out: List[int]) -> None:
    """backprojected_mosaic.png of the resolved objects raytraced in every held-out frame."""
    tiles, backproj_st = [], time.time()
    for i, frame_id in enumerate(held_out):
        frame, st = frames[frame_id], time.time()
        masks, _, classes = resolved.visible_objects(
            frame.depth, frame.pose, DEPTH_CAMERA.hfov, DEPTH_CAMERA.min_depth, DEPTH_CAMERA.max_depth,
            min_mesh_visibility_fraction=viewer.sensor.min_mesh_visibility_fraction,
        )
        if len(masks) == 0:
            print(f"backprojected frame {i:02d}: no resolved points in range, skipped ({time.time() - st:.3f}s)")
            continue
        tiles.append(_segmentation_tile(viewer, frame.rgb, instances_from_masks(masks, classes, frame.depth.shape), f"backprojected{i:02d}"))
        print(f"backprojected frame {i:02d}: raytraced in {time.time() - st:.3f}s")
    make_mosaic(tiles, N_cols=4).save(os.path.join(viewer.out_dir, f"{viewer.out_stem}_backprojected_mosaic.png"))
    print(f"backprojection pass: {time.time() - backproj_st:.3f}s total")


def _start_viewer(args: argparse.Namespace) -> _Viewer:
    default_dir = os.path.join(os.path.dirname(__file__), "testdump", "test_concept_graph")
    shutil.rmtree(default_dir, ignore_errors=True)
    out_dir = (os.path.dirname(args.out) or ".") if args.out else default_dir
    out_stem = os.path.splitext(os.path.basename(args.out))[0] if args.out else args.scene_id
    os.makedirs(out_dir, exist_ok=True)

    sim, annotation = procthor_sim(args.scene_id)
    sensor = ObjectDetectorGTSensor(sim, ObjectDetectorGTSensorConfig(
        env_name=ENV_NAME, vocab_name=VOCAB_NAME,
        min_area_fraction=5e-4, filter_low_visibility=True, min_mesh_visibility_fraction=0.2,
        filter_classes=[],
    ))
    sensor.annotation = annotation
    colors = make_colors(len(sensor.get_classes()), seed=0, ctype=0)
    rng = np.random.default_rng(args.seed)
    config = FrontierExplorationConfig(MapConfig(HFOV_DEG), DiscretePlannerConfig(TURN_ANGLE_DEG))
    explorer = FrontierExplorationAgent(config, DEPTH_CAMERA, rng, out_dir, out_stem)
    return _Viewer(sim, sensor, colors, out_dir, out_stem, rng, floor_island(sim.pathfinder), explorer)


def exploration_frames(viewer: _Viewer, num_steps: int) -> Iterator[Tuple[dict, habitat_sim.AgentState]]:
    """Observations and agent states along a FrontierExplorationAgent run from a random start, until it stops or num_steps frames."""
    sim_agent = viewer.sim.get_agent(0)
    start = _random_agent_state(viewer.sim, viewer.rng, viewer.island_index)
    sim_agent.set_state(start)
    viewer.explorer.reset(sim_agent.get_state(), floor_height(viewer.sim, np.array(start.position)))
    obs = viewer.sim.get_sensor_observations()
    for _ in range(num_steps):
        yield obs, sim_agent.get_state()
        action, _ = viewer.explorer.act(sim_agent.get_state(), obs["depth"])
        viewer.map_tiles.append(viewer.explorer.map_tile())
        if action == "stop":
            return
        obs = viewer.sim.step(action)


def _frame_geometry(obs: dict, agent_state: habitat_sim.AgentState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rgb, depth and camera-to-world pose of one observation."""
    return obs["rgb"][:, :, :3].copy(), obs["depth"].squeeze(), sensor_pose_matrix(agent_state.sensor_states["depth"])


def _segmentation_tile(viewer: _Viewer, rgb: np.ndarray, instances: Instances, label: str) -> Tuple[str, np.ndarray]:
    return segmentation_tile(rgb, instances, _style(viewer), label)


def _style(viewer: _Viewer) -> SegmentationStyle:
    return SegmentationStyle(viewer.sensor.get_classes(), viewer.colors)


def save_retrieval_mosaic(viewer: _Viewer, resolved: ResolvedObjects, frames: Dict[int, Frame], num_objects: int) -> None:
    """One mosaic row per largest resolved object, of the frames retrieve_resolved_views selects."""
    raytraced, object_frames = retrieve_resolved_views(resolved, frames, DEPTH_CAMERA, viewer.sensor.min_mesh_visibility_fraction, ViewSelection())
    mosaic = views_mosaic(frames, raytraced, _style(viewer), {object_id: object_frames[object_id] for object_id in largest_objects(resolved, num_objects)})
    if mosaic is not None:
        mosaic.save(os.path.join(viewer.out_dir, f"{viewer.out_stem}_retrieval_mosaic.png"))


if __name__ == "__main__":
    args = viewer_arg_parser().parse_args()
    vocabulary = Vocabulary(VOCAB_NAME, tuple(resolve_classes(ENV_NAME, VOCAB_NAME)))
    viewer, resolved, frames = run_viewer(args, Detector(args.config_file, args.checkpoint, vocabulary, args.sam_checkpoint))
    save_retrieval_mosaic(viewer, resolved, frames, args.retrieval_objects)
