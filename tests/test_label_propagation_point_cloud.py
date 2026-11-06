"""
Standalone viewer for common/utils/label_propagation_point_cloud.py::LabelPropagationPointCloud.

Runs against procthor_scene.py's ProcTHOR-hab scene (ProcTHOR-Train-0 unless --scene-id says otherwise) via
ObjectDetectorGTSensor on a raw habitat_sim.Simulator (no ExplorationEnv/Hydra config), driven by a FrontierExplorationAgent
(common/hybrid_exploration/frontier_exp_agent.py) with vanilla move_forward/turn_left/turn_right actions from a
random start pose. Each fused frame's gt_instances are projected with project_semantic_masks_to_3d,
fused via LabelPropagationPointCloud.__add__, and .resolve() voxelizes them and resolves per-voxel
class/object-id conflicts. Also hosts the write/read harness (run_viewer) and the retrieval mosaic
(save_retrieval_mosaic) that test_concept_graph.py reuses.

Write: fuses every other frame of a 2 * --num-frames step exploration run, resolves the map once,
and saves <stem>.ply, a merged .glb of the scene (every placed rigid object, each loaded from its
source asset and transformed by its runtime scale/pose - habitat-sim can't export an assembled
scene) and an input_mosaic.png of every fused frame's GT segmentation.

Read: queries each held-out frame (the ones in between, never fused) through .depth_raytracing, with the
GT sensor's own min_mesh_visibility_fraction, tiled into a backprojected_mosaic.png, showing how well the map
generalizes to unseen views.
Retrieve: every exploration frame (fused and held-out) is kept in an ImageMemory; for each of the
--retrieval-objects largest resolved objects, common/utils/retrieval.py's retrieve_views selects diverse frames it
appears in, tiled one object per row into a retrieval_mosaic.png.
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
from typing import Callable, Iterator, List, Tuple

import cv2
import habitat_sim
import numpy as np
import torch
from detectron2.structures import Instances

from habitat.config.default_structured_configs import ObjectDetectorGTSensorConfig
from common.hybrid_exploration.exploration_map import MapConfig
from common.hybrid_exploration.frontier_exp_agent import DiscretePlannerConfig, FrontierExplorationAgent, FrontierExplorationConfig
from common.env_utils.object_annotations import scene_world_mesh
from common.env_utils.object_detector_sensors import ObjectDetectorGTSensor
from common.env_utils.sense import bbs_instances_as_gt, keep_valid_instances
from common.env_utils.vocab_constants import make_colors
from common.utils.image_memory import Frame, ImageMemory
from common.utils.plot_utils import make_mosaic
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import ResolvedObjects, save_resolved_ply
from common.utils.retrieval import SegmentationStyle, ViewSelection, instances_from_masks, largest_objects, retrieve_views, segmentation_tile, views_mosaic
from common.utils.label_propagation_point_cloud import LabelPropagationPointCloud, project_semantic_masks_to_3d
from procthor_scene import DATASET_CONFIG, DEPTH_CAMERA, ENV_NAME, HFOV_DEG, SCENE_ID, TURN_ANGLE_DEG, VOCAB_NAME, floor_height, floor_island, procthor_sim

def _random_agent_state(sim, rng, island_index: int = -1) -> habitat_sim.AgentState:
    state = habitat_sim.AgentState()
    state.position = sim.pathfinder.get_random_navigable_point(island_index=island_index)
    yaw = rng.uniform(0, 2 * np.pi)
    state.rotation = np.quaternion(np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)
    return state


def _frame_gt_instances(sensor: ObjectDetectorGTSensor, obs: dict, classes_keep: set) -> Instances:
    """gt_instances with a synthetic one-hot gt_logits field - both maps fuse per-instance class
    logits, which ground truth doesn't have; one-hot over the true class is the least assuming
    stand-in for a model's confidence scores."""
    instances = sensor.decompose_frame(obs["semantic"], obs["depth"])["instances"]
    gt_instances = keep_valid_instances(bbs_instances_as_gt(instances))

    if classes_keep and len(gt_instances):
        classes = sensor.get_classes()
        keep_idx = [i for i, c in enumerate(gt_instances.gt_classes.tolist()) if classes[c] in classes_keep]
        gt_instances = gt_instances[keep_idx]

    if len(gt_instances) == 0:
        return gt_instances

    num_classes = len(sensor.get_classes())
    one_hot = torch.zeros(len(gt_instances), num_classes)
    one_hot[torch.arange(len(gt_instances)), gt_instances.gt_classes] = 1.0
    gt_instances.gt_logits = one_hot
    return gt_instances


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


def viewer_arg_parser(description: str, map_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--scene-id", default=SCENE_ID, help="ProcTHOR-hab scene id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=100, help=f"exploration frames fused into the {map_name}, and as many interleaved held-out views backprojected from it")
    parser.add_argument("--classes-keep", nargs="*", default=[], help="drop gt_instances whose class isn't in this list; empty means keep every class")
    parser.add_argument("--retrieval-objects", type=int, default=10, help="largest resolved objects whose retrieved frames are saved as retrieval_mosaic.png rows")
    parser.add_argument("--out", default=None, help="output .ply path (default: tests/testdump/<test file>/<scene_id>.ply); a matching .glb merged scene mesh is saved alongside it")
    return parser


def run_viewer(args: argparse.Namespace, out_name: str, fused, frame_map: Callable, resolve: Callable) -> Tuple[_Viewer, object, ImageMemory]:
    """Fuses frame_map(frame_id, frame, gt_instances) pieces of an exploration run into `fused`, resolves it with resolve(fused, memory),
    saves it, backprojects it and returns it with every explored frame."""
    
    if not os.path.exists(DATASET_CONFIG):
        raise SystemExit(f"no ProcTHOR-hab data at {DATASET_CONFIG} (check HABITAT_DATA)")
    viewer = _start_viewer(args, out_name)
    
    try:
        run_st = time.time()

        # collection
        input_tiles, memory, held_out = [], ImageMemory(), []

        for step, (obs, agent_state) in enumerate(exploration_frames(viewer, 2 * args.num_frames)):
            frame = Frame(*_frame_geometry(obs, agent_state))
            gt_instances = _frame_gt_instances(viewer.sensor, obs, set(args.classes_keep))
            frame_id = memory.add_frame(frame)
            if step % 2:
                held_out.append(frame_id)
                continue

            input_tiles.append(_segmentation_tile(viewer, frame.rgb, gt_instances, f"frame{frame_id:03d}"))
            if len(gt_instances):
                st = time.time()
                fused = fused + frame_map(frame_id, frame, gt_instances)
                print(f"frame {frame_id:03d}: inserted {len(gt_instances)} instances in {time.time() - st:.3f}s")

        make_mosaic(input_tiles, N_cols=4).save(os.path.join(viewer.out_dir, f"{viewer.out_stem}_input_mosaic.png"))

        # resolve
        fused = resolve(fused, memory)
        if len(fused.resolved_points) == 0:
            raise SystemExit("none of the explored frames produced anything to resolve")
        print(f"{args.scene_id}: fused and resolved in {time.time() - run_st:.3f}s, {len(fused.resolved_points)} resolved points")
        out_path = args.out or os.path.join(viewer.out_dir, f"{viewer.out_stem}.ply")
        save_resolved_ply(fused, viewer.colors, out_path)
        print(f"saved to {out_path}")

        mesh_st = time.time()
        mesh_path = os.path.splitext(out_path)[0] + ".glb"
        scene_mesh = scene_world_mesh(viewer.sim)
        if not scene_mesh.is_empty:
            scene_mesh.export(mesh_path)
        print(f"saved merged scene mesh to {mesh_path} in {time.time() - mesh_st:.3f}s")

        # backprojection
        backprojected_tiles = []
        backproj_st = time.time()
        for i, frame_id in enumerate(held_out):
            frame = memory.get_frame(frame_id)
            st = time.time()
            resolved_masks, _, resolved_classes, _ = fused.depth_raytracing(
                frame.depth, frame.pose, hfov=DEPTH_CAMERA.hfov, min_depth=DEPTH_CAMERA.min_depth, max_depth=DEPTH_CAMERA.max_depth, min_mesh_visibility_fraction=viewer.sensor.min_mesh_visibility_fraction,
            )
            if len(resolved_masks) == 0:
                print(f"backprojected frame {i:02d}: no resolved points in range, skipped ({time.time() - st:.3f}s)")
                continue
            back_instances = instances_from_masks(resolved_masks, resolved_classes, frame.depth.shape)
            backprojected_tiles.append(_segmentation_tile(viewer, frame.rgb, back_instances, f"backprojected{i:02d}"))
            print(f"backprojected frame {i:02d}: raytraced in {time.time() - st:.3f}s")

        make_mosaic(backprojected_tiles, N_cols=4).save(os.path.join(viewer.out_dir, f"{viewer.out_stem}_backprojected_mosaic.png"))
        print(f"backprojection pass: {time.time() - backproj_st:.3f}s total")

        cv2.imwrite(os.path.join(viewer.out_dir, f"{viewer.out_stem}_exploration_map.png"), viewer.explorer.map_image())
        make_mosaic(viewer.map_tiles, N_cols=5).save(os.path.join(viewer.out_dir, f"{viewer.out_stem}_exploration_mosaic.png"))
        return viewer, fused, memory
    
    finally:
        viewer.sim.close()


def _start_viewer(args: argparse.Namespace, out_name: str) -> _Viewer:
    default_dir = os.path.join(os.path.dirname(__file__), "testdump", out_name)
    shutil.rmtree(default_dir, ignore_errors=True)
    out_dir = (os.path.dirname(args.out) or ".") if args.out else default_dir
    out_stem = os.path.splitext(os.path.basename(args.out))[0] if args.out else args.scene_id
    os.makedirs(out_dir, exist_ok=True)

    sim, annotation = procthor_sim(args.scene_id)
    sensor = ObjectDetectorGTSensor(sim, ObjectDetectorGTSensorConfig(
        env_name=ENV_NAME, vocab_name=VOCAB_NAME,
        min_area_fraction=2e-4, filter_low_visibility=True, min_mesh_visibility_fraction=0.2,
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


def save_retrieval_mosaic(viewer: _Viewer, fused: ResolvedObjects, memory: ImageMemory, num_objects: int) -> None:
    """One mosaic row per largest resolved object, of the frames retrieve_views selects."""
    camera = DEPTH_CAMERA
    raytraced = memory.raytrace_frames(fused, camera, viewer.sensor.min_mesh_visibility_fraction)
    object_frames = retrieve_views(fused, memory, raytraced, camera, ViewSelection())
    mosaic = views_mosaic(memory, raytraced, _style(viewer), {object_id: object_frames[object_id] for object_id in largest_objects(fused, num_objects)})
    if mosaic is not None:
        mosaic.save(os.path.join(viewer.out_dir, f"{viewer.out_stem}_retrieval_mosaic.png"))


def resolve_point_cloud(fused: LabelPropagationPointCloud, memory: ImageMemory) -> LabelPropagationPointCloud:
    fused.resolve()
    return fused


def frame_point_cloud(frame_id: int, frame: Frame, gt_instances: Instances) -> LabelPropagationPointCloud:
    return project_semantic_masks_to_3d(frame, gt_instances, DEPTH_CAMERA)


if __name__ == "__main__":
    args = viewer_arg_parser(__doc__, "LabelPropagationPointCloud").parse_args()
    viewer, fused, memory = run_viewer(args, "test_label_propagation_point_cloud", LabelPropagationPointCloud(), frame_point_cloud, resolve_point_cloud)
    save_retrieval_mosaic(viewer, fused, memory, args.retrieval_objects)
