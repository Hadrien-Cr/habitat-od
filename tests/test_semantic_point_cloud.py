"""
Standalone viewer for common/utils/semantic_point_cloud.py::SemanticPointCloud.

Runs against a real single-room ProcTHOR-hab scene via ObjectDetectorGTSensor on a raw
habitat_sim.Simulator (no ExplorationEnv/Hydra config), driven by a FrontierAgent
(common/agents/frontier_agent.py) with vanilla move_forward/turn_left/turn_right actions from a
random start pose. Each fused frame's gt_instances are projected with project_semantic_masks_to_3d,
fused via SemanticPointCloud.__add__, and .resolve() voxelizes them and resolves per-voxel
class/object-id conflicts. Also hosts the write/read harness (run_viewer) and the retrieval mosaic
(save_retrieval_mosaic) that test_concept_graph.py reuses.

Write: fuses every other frame of a 2 * --num-frames step exploration run, resolves the map once,
and saves <stem>.ply, a merged .glb of the scene (every placed rigid object, each loaded from its
source asset and transformed by its runtime scale/pose - habitat-sim can't export an assembled
scene) and an input_mosaic.png of every fused frame's GT segmentation.

Read: queries each held-out frame (the ones in between, never fused) through .depth_raytracing, with the
GT sensor's own min_visibility_fraction, tiled into a backprojected_mosaic.png, showing how well the map
generalizes to unseen views.
Retrieve: every exploration frame (fused and held-out) is kept in an ImageMemory; for each of the
--retrieval-objects largest resolved objects, ImageMemory.frames_showing lists the frames it appears in, tiled one
object per row into a retrieval_mosaic.png - the matched instance where it was fused, its raytraced mask where it is
only visible (held-out frames, or filtered out of GT).
Finally saves the FrontierAgent's exploration_map.png (navigable map, revealed area, trajectory,
circled closest frontier) and an exploration_mosaic.png of that map at every step.

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
import magnum as mn
import numpy as np
import torch
import trimesh
from detectron2.structures import Instances

from habitat.config.default_structured_configs import ObjectDetectorGTSensorConfig
from common.agents.frontier_agent import FrontierAgent, FrontierAgentConfig
from common.env_utils.object_annotations import object_world_mesh
from common.env_utils.object_detector_sensors import ObjectDetectorGTSensor
from common.env_utils.sense import bbs_instances_as_gt, keep_valid_instances
from common.env_utils.vocab_constants import make_colors
from common.utils.image_memory import Frame, ImageMemory
from common.utils.plot_utils import make_mosaic
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import DepthCamera, ResolvedObjects, save_resolved_ply
from common.utils.retrieval import SegmentationStyle, instances_from_masks, retrieval_mosaic, segmentation_tile
from common.utils.semantic_point_cloud import SemanticPointCloud, project_semantic_masks_to_3d

HABITAT_DATA = os.environ.get("HABITAT_DATA")
_RESOLUTION = [640, 640]
HFOV_DEG = 90
MIN_DEPTH = 0.5
MAX_DEPTH = 15.0
_AGENT_HEIGHT = 0.88  # matches common/config/hssd-hab/default.yaml's agent height
_FORWARD_STEP_M = 0.25
_TURN_ANGLE_DEG = 30.0  # matches common/config/hssd-hab/default.yaml's turn_angle
_ENV_NAME = "ProcTHOR-hab"
_VOCAB_NAME = "ProcTHOR-native"  # fully data-driven - see object_annotations.py
_DATASET_CONFIG = f"{HABITAT_DATA}/scene_datasets/ai2thor-hab/ai2thor-hab/ai2thor-hab.scene_dataset_config.json" if HABITAT_DATA else None
_DEFAULT_SCENE_ID = "ProcTHOR-Train-9632"  # same one test_object_detector.py uses


def _build_sim(dataset_config: str, scene_id: str) -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = dataset_config
    backend_cfg.scene_id = scene_id
    backend_cfg.enable_physics = True

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = _RESOLUTION
    rgb_spec.hfov = mn.Deg(HFOV_DEG)
    rgb_spec.position = [0.0, _AGENT_HEIGHT, 0.0]

    semantic_spec = habitat_sim.CameraSensorSpec()
    semantic_spec.uuid = "semantic"
    semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_spec.resolution = _RESOLUTION
    semantic_spec.hfov = rgb_spec.hfov
    semantic_spec.position = rgb_spec.position

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = _RESOLUTION
    depth_spec.hfov = rgb_spec.hfov
    depth_spec.position = rgb_spec.position
    depth_spec.depth_min = MIN_DEPTH
    depth_spec.depth_max = MAX_DEPTH

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, semantic_spec, depth_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=_FORWARD_STEP_M)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=_TURN_ANGLE_DEG)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=_TURN_ANGLE_DEG)),
    }

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    if not sim.pathfinder.is_loaded:  # ProcTHOR-hab doesn't ship a precomputed .navmesh
        nav_settings = habitat_sim.NavMeshSettings()
        nav_settings.set_defaults()
        nav_settings.include_static_objects = True
        sim.recompute_navmesh(sim.pathfinder, nav_settings)
    return sim


def _floor_island(pathfinder: habitat_sim.PathFinder) -> int:
    """Recomputed ProcTHOR-hab navmeshes (see _build_sim) also cover the open-top houses' roof and
    furniture tops, and the roof can outsize the furnished floor, so the floor is picked as the lowest island."""
    return min(range(pathfinder.num_islands), key=lambda island: pathfinder.get_random_navigable_point(island_index=island)[1])


def _random_agent_state(sim, rng, island_index: int = -1) -> habitat_sim.AgentState:
    state = habitat_sim.AgentState()
    state.position = sim.pathfinder.get_random_navigable_point(island_index=island_index)
    yaw = rng.uniform(0, 2 * np.pi)
    state.rotation = np.quaternion(np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)
    return state


def _export_scene_mesh(sim: habitat_sim.Simulator, out_path: str) -> None:
    """Merges the stage + every placed rigid object's own source mesh (scaled and transformed
    to its runtime world pose) into one .glb - habitat-sim has no built-in "export the
    assembled scene" call, so this reassembles it piece by piece. Object-level scale
    (obj.scale, from the object's template) is separate from obj.transformation (rotation +
    translation only - rigid-body transforms don't carry scale), so both must be applied."""
    meshes = []

    stage_path = sim.get_stage_initialization_template().render_asset_fullpath
    # if stage_path and os.path.exists(stage_path):
    #     meshes.append(trimesh.load(stage_path, force="mesh"))

    rom = sim.get_rigid_object_manager()
    for handle in rom.get_object_handles():
        mesh = object_world_mesh(rom.get_object_by_handle(handle))
        if mesh is not None:
            meshes.append(mesh)

    if meshes:
        trimesh.util.concatenate(meshes).export(out_path)


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
    explorer: FrontierAgent
    map_tiles: list = field(default_factory=list)


def viewer_arg_parser(description: str, map_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--scene-id", default=_DEFAULT_SCENE_ID, help="ProcTHOR-hab scene id (pick a known single-room one from your data if that matters)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=100, help=f"exploration frames fused into the {map_name}, and as many interleaved held-out views backprojected from it")
    parser.add_argument("--classes-keep", nargs="*", default=[], help="drop gt_instances whose class isn't in this list; empty means keep every class")
    parser.add_argument("--retrieval-objects", type=int, default=10, help="largest resolved objects whose retrieved frames are saved as retrieval_mosaic.png rows")
    parser.add_argument("--out", default=None, help="output .ply path (default: tests/testdump/<test file>/<scene_id>.ply); a matching .glb merged scene mesh is saved alongside it")
    return parser


def run_viewer(args: argparse.Namespace, out_name: str, fused, frame_map: Callable, resolve: Callable) -> Tuple[_Viewer, object, ImageMemory]:
    """Fuses frame_map(rgb, depth, pose, gt_instances) pieces of an exploration run into `fused`, resolves it with resolve(fused, memory),
    saves it, backprojects it and returns it with every explored frame."""
    
    if not _DATASET_CONFIG or not os.path.exists(_DATASET_CONFIG):
        raise SystemExit(f"no ProcTHOR-hab data at {_DATASET_CONFIG} (check HABITAT_DATA)")
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
                fused = fused + frame_map(frame.rgb, frame.depth, frame.pose, gt_instances)
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
        _export_scene_mesh(viewer.sim, mesh_path)
        print(f"saved merged scene mesh to {mesh_path} in {time.time() - mesh_st:.3f}s")

        # backprojection
        backprojected_tiles = []
        backproj_st = time.time()
        for i, frame_id in enumerate(held_out):
            frame = memory.get_frame(frame_id)
            st = time.time()
            resolved_masks, _, resolved_classes, _ = fused.depth_raytracing(
                frame.depth, frame.pose, hfov=HFOV_DEG, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_visibility_fraction=viewer.sensor.min_visibility_fraction,
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

    sim = _build_sim(_DATASET_CONFIG, args.scene_id)
    sensor = ObjectDetectorGTSensor(sim, ObjectDetectorGTSensorConfig(
        env_name=_ENV_NAME, vocab_name=_VOCAB_NAME,
        area_thr=250, filter_low_visibility=True, min_visibility_fraction=0.2,
        filter_classes=[],
    ))
    sensor.setup_semantic_labels()
    colors = make_colors(len(sensor.get_classes()), seed=0, ctype=0)
    rng = np.random.default_rng(args.seed)
    explorer = FrontierAgent(sim.pathfinder, FrontierAgentConfig(HFOV_DEG, _TURN_ANGLE_DEG), rng)
    return _Viewer(sim, sensor, colors, out_dir, out_stem, rng, _floor_island(sim.pathfinder), explorer)


def exploration_frames(viewer: _Viewer, num_steps: int) -> Iterator[Tuple[dict, habitat_sim.AgentState]]:
    """Observations and agent states along a FrontierAgent run from a random start, until it stops or num_steps frames."""
    sim_agent = viewer.sim.get_agent(0)
    sim_agent.set_state(_random_agent_state(viewer.sim, viewer.rng, viewer.island_index))
    obs = viewer.sim.get_sensor_observations()
    for _ in range(num_steps):
        yield obs, sim_agent.get_state()
        action = viewer.explorer.act(sim_agent.get_state())
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
    """One mosaic row per largest resolved object, of every frame ImageMemory.frames_showing retrieves it in."""
    raytraced = memory.raytrace_frames(fused, DepthCamera(HFOV_DEG, MIN_DEPTH, MAX_DEPTH), viewer.sensor.min_visibility_fraction)
    mosaic = retrieval_mosaic(fused, memory, raytraced, _style(viewer), num_objects)
    if mosaic is not None:
        mosaic.save(os.path.join(viewer.out_dir, f"{viewer.out_stem}_retrieval_mosaic.png"))


def resolve_point_cloud(fused: SemanticPointCloud, memory: ImageMemory) -> SemanticPointCloud:
    fused.resolve()
    return fused


def frame_point_cloud(rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray, gt_instances: Instances) -> SemanticPointCloud:
    infos = [{"id_object": int(info["object_id"])} for info in gt_instances.infos]
    return project_semantic_masks_to_3d(depth, pose, gt_instances, infos, hfov=HFOV_DEG, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)


if __name__ == "__main__":
    args = viewer_arg_parser(__doc__, "SemanticPointCloud").parse_args()
    viewer, fused, memory = run_viewer(args, "test_semantic_point_cloud", SemanticPointCloud(), frame_point_cloud, resolve_point_cloud)
    save_retrieval_mosaic(viewer, fused, memory, args.retrieval_objects)
