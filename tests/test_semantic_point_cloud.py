"""
Standalone viewer for common/utils/semantic_point_cloud.py::SemanticPointCloud.

Runs against a real single-room ProcTHOR-hab scene via ObjectDetectorGTSensor on a raw
habitat_sim.Simulator (no ExplorationEnv/Hydra config). Each frame's gt_instances are projected
with project_semantic_masks_to_3d, fused via SemanticPointCloud.__add__, and .resolve() voxelizes
them and resolves per-voxel class/object-id conflicts. Also hosts the write/read harness
(run_viewer) that concept_graph_viewer.py reuses.

Write: samples --num-frames navigable poses (all but the first within --radius of it, so they
overlap), fuses each frame's map piece and calls .resolve() after every frame, saving one
cumulative .ply per step (<stem>_step1.ply, <stem>_step1:2.ply, ...), the final <stem>.ply, a
merged .glb of the scene (stage + every placed rigid object, each loaded from its source asset and
transformed by its runtime scale/pose - habitat-sim can't export an assembled scene) and an
input_mosaic.png of every fused frame's GT segmentation.

Read: samples --num-frames novel poses (not used to build the map) and queries each through
.depth_raytracing, saved as backprojected{i}_sem.png plus a backprojected_mosaic.png, showing how
well the map generalizes to unseen views.

Not collected by pytest - run it directly. Requires HABITAT_DATA pointing at real ai2thor-hab data
(see INSTALL.MD).
"""

import argparse
import os
import shutil
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Tuple

import cv2
import habitat_sim
import magnum as mn
import numpy as np
import open3d as o3d
import quaternion
import torch
import trimesh
from detectron2.structures import Boxes, Instances

from habitat.config.default_structured_configs import ObjectDetectorGTSensorConfig
from common.env_utils.object_detector_sensors import ObjectDetectorGTSensor
from common.env_utils.sense import bbs_instances_as_gt, keep_valid_instances
from common.env_utils.vocab_constants import make_colors
from common.utils.plot_utils import make_mosaic, plot_segmentation
from common.utils.semantic_point_cloud import SemanticPointCloud, project_semantic_masks_to_3d

HABITAT_DATA = os.environ.get("HABITAT_DATA")
_RESOLUTION = [640, 640]
HFOV_DEG = 90
MIN_DEPTH = 0.5
MAX_DEPTH = 15.0
_AGENT_HEIGHT = 0.88  # matches common/config/hssd-hab/default.yaml's agent height
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

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    if not sim.pathfinder.is_loaded:  # ProcTHOR-hab doesn't ship a precomputed .navmesh
        nav_settings = habitat_sim.NavMeshSettings()
        nav_settings.set_defaults()
        sim.recompute_navmesh(sim.pathfinder, nav_settings)
    return sim


def _largest_island(pathfinder: habitat_sim.PathFinder) -> int:
    """Recomputed ProcTHOR-hab navmeshes (see _build_sim) can include a disconnected rooftop
    island - open-top houses let recast voxelize the roof as walkable - so unrestricted
    get_random_navigable_point calls can spawn the agent up there. The real floor is by far the
    biggest connected component, so restricting sampling to it keeps poses off the roof."""
    return max(range(pathfinder.num_islands), key=pathfinder.island_area)


def _random_agent_state(sim, rng, island_index: int = -1) -> habitat_sim.AgentState:
    state = habitat_sim.AgentState()
    state.position = sim.pathfinder.get_random_navigable_point(island_index=island_index)
    yaw = rng.uniform(0, 2 * np.pi)
    state.rotation = np.quaternion(np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)
    return state


def _agent_state_near(sim, center: np.ndarray, radius: float, rng, max_tries: int = 100, island_index: int = -1) -> habitat_sim.AgentState:
    """Like _random_agent_state, but within `radius` of `center` - get_random_navigable_point_near
    can return NaN if the navmesh has nothing navigable in range, so retry a few times."""
    for _ in range(max_tries):
        position = sim.pathfinder.get_random_navigable_point_near(
            mn.Vector3(*map(float, center)), radius, island_index=island_index
        )
        if not np.isnan(position).any():
            break
    else:
        raise SystemExit(f"no navigable point found within radius {radius} of {center} after {max_tries} tries")

    state = habitat_sim.AgentState()
    state.position = position
    yaw = rng.uniform(0, 2 * np.pi)
    state.rotation = np.quaternion(np.cos(yaw / 2), 0.0, np.sin(yaw / 2), 0.0)
    return state


def _camera_pose_matrix(sensor_state) -> np.ndarray:
    """4x4 camera-to-world transform, same convention project_instance_segmentation expects
    (Pose.get_T() in common/env_utils/sense.py) - built directly from the depth sensor's own
    world-space state rather than AgentPoseSense's agent-to-camera offset, since habitat
    already gives us the camera pose."""
    T = np.eye(4)
    T[:3, :3] = quaternion.as_rotation_matrix(sensor_state.rotation)
    T[:3, 3] = sensor_state.position
    return T


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
        obj = rom.get_object_by_handle(handle)
        asset_path = obj.creation_attributes.render_asset_fullpath
        if not asset_path or not os.path.exists(asset_path):
            continue
        mesh = trimesh.load(asset_path, force="mesh")
        mesh.apply_scale(np.array(obj.scale))
        mesh.apply_transform(np.array(obj.transformation))
        meshes.append(mesh)

    if meshes:
        trimesh.util.concatenate(meshes).export(out_path)


def _frame_gt_instances(sensor: ObjectDetectorGTSensor, agent_state, obs: dict, classes_keep: set) -> Instances:
    """gt_instances with a synthetic one-hot gt_logits field - both maps fuse per-instance class
    logits, which ground truth doesn't have; one-hot over the true class is the least assuming
    stand-in for a model's confidence scores."""
    instances = sensor.decompose_frame(obs["semantic"], agent_state, obs["depth"])["instances"]
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


def _instances_from_masks(masks: torch.Tensor, classes: torch.Tensor, image_size) -> Instances:
    boxes = []
    for mask in masks.numpy().astype("uint8"):
        x, y, w, h = cv2.boundingRect(mask)
        boxes.append([x, y, x + w, y + h])
    instances = Instances(image_size)
    instances.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32))
    instances.gt_classes = classes.long()
    instances.gt_masks = masks.bool()
    return instances


def save_point_cloud(fused, colors, out_path: str) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fused.resolved_points)
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors)[fused.resolved_infos[:, 0]] / 255.0)
    o3d.io.write_point_cloud(out_path, pcd)


@dataclass
class _Viewer:
    sim: habitat_sim.Simulator
    sensor: ObjectDetectorGTSensor
    colors: list
    out_dir: str
    out_stem: str
    rng: np.random.Generator
    island_index: int
    center: np.ndarray = None


def viewer_arg_parser(description: str, map_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--scene-id", default=_DEFAULT_SCENE_ID, help="ProcTHOR-hab scene id (pick a known single-room one from your data if that matters)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=10, help=f"frames fused into the {map_name}, and separately, novel backprojected views queried from it")
    parser.add_argument("--radius", type=float, default=2.0, help="every frame but the 1st fused one is sampled within this radius of it")
    parser.add_argument("--classes-keep", nargs="*", default=[], help="drop gt_instances whose class isn't in this list; empty means keep every class")
    parser.add_argument("--out", default=None, help="output .ply path (default: tests/testdump/<viewer>/<scene_id>.ply); a matching .glb merged scene mesh is saved alongside it")
    return parser


def run_viewer(args: argparse.Namespace, out_name: str, fuse_frames: Callable) -> None:
    """Builds a map with the viewer's own fuse_frames(viewer, args) loop, saves it, then backprojects it."""
    if not _DATASET_CONFIG or not os.path.exists(_DATASET_CONFIG):
        raise SystemExit(f"no ProcTHOR-hab data at {_DATASET_CONFIG} (check HABITAT_DATA)")
    viewer = _start_viewer(args, out_name)
    try:
        run_st = time.time()
        fused = fuse_frames(viewer, args)
        if len(fused.resolved_points) == 0:
            raise SystemExit(f"none of the {args.num_frames} sampled frames produced anything to resolve")
        print(f"{args.scene_id}: fused {args.num_frames} frames in {time.time() - run_st:.3f}s, {len(fused.resolved_points)} resolved points")
        out_path = args.out or os.path.join(viewer.out_dir, f"{viewer.out_stem}.ply")
        save_point_cloud(fused, viewer.colors, out_path)
        print(f"saved to {out_path}")

        mesh_st = time.time()
        mesh_path = os.path.splitext(out_path)[0] + ".glb"
        _export_scene_mesh(viewer.sim, mesh_path)
        print(f"saved merged scene mesh to {mesh_path} in {time.time() - mesh_st:.3f}s")

        _backproject(viewer, args, fused)
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
        area_thr=250, filter_low_visibility=True, min_visibility_fraction=0.15,
        filter_classes=[],
    ))
    sensor.setup_semantic_labels()
    colors = make_colors(len(sensor.get_classes()), seed=0, ctype=0)
    return _Viewer(sim, sensor, colors, out_dir, out_stem, np.random.default_rng(args.seed), _largest_island(sim.pathfinder))


def _observe(viewer: _Viewer, state: habitat_sim.AgentState):
    """rgb, depth and camera-to-world pose seen from state."""
    agent = viewer.sim.get_agent(0)
    agent.set_state(state)
    obs = viewer.sim.get_sensor_observations()
    pose = _camera_pose_matrix(agent.get_state().sensor_states["depth"])
    return obs, agent.get_state(), pose


def fusion_states(viewer: _Viewer, args: argparse.Namespace) -> Iterator[habitat_sim.AgentState]:
    """args.num_frames poses: a random first one, then every other within args.radius of it."""
    first_state = _random_agent_state(viewer.sim, viewer.rng, viewer.island_index)
    viewer.center = first_state.position
    yield first_state
    for _ in range(args.num_frames - 1):
        yield _agent_state_near(viewer.sim, viewer.center, args.radius, viewer.rng, island_index=viewer.island_index)


def observe_frame(viewer: _Viewer, state: habitat_sim.AgentState, classes_keep: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Instances]:
    """rgb, depth, camera-to-world pose and gt_instances seen from state."""
    obs, agent_state, pose = _observe(viewer, state)
    gt_instances = _frame_gt_instances(viewer.sensor, agent_state, obs, set(classes_keep))
    return obs["rgb"][:, :, :3].copy(), obs["depth"].squeeze(), pose, gt_instances


def save_input_mosaic(viewer: _Viewer, input_tiles: list) -> None:
    make_mosaic(input_tiles, N_cols=4).save(os.path.join(viewer.out_dir, f"{viewer.out_stem}_input_mosaic.png"))


def save_segmentation(viewer: _Viewer, rgb: np.ndarray, instances: Instances, label: str) -> Tuple[str, np.ndarray]:
    im = plot_segmentation(rgb=rgb.copy(), gt_instances=instances, pred_instances=None, classes=viewer.sensor.get_classes(), colors=viewer.colors)
    im.save(os.path.join(viewer.out_dir, f"{viewer.out_stem}_{label}_sem.png"))
    return label, np.array(im)


def _backproject(viewer: _Viewer, args: argparse.Namespace, fused) -> None:
    """Raytraces the map from novel poses - not used to build it - to show how it generalizes to unseen views."""
    backprojected_tiles = []
    backproj_st = time.time()
    for i in range(args.num_frames):
        st = time.time()
        state = _agent_state_near(viewer.sim, viewer.center, args.radius, viewer.rng, island_index=viewer.island_index)
        obs, _, pose = _observe(viewer, state)
        depth = obs["depth"].squeeze()
        resolved_masks, _, resolved_classes, _ = fused.depth_raytracing(depth, pose, hfov=HFOV_DEG, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
        if len(resolved_masks) == 0:
            print(f"backprojected frame {i:02d}: no resolved points in range, skipped ({time.time() - st:.3f}s)")
            continue
        back_instances = _instances_from_masks(resolved_masks, resolved_classes, depth.shape)
        label = f"backprojected{i:02d}"
        backprojected_tiles.append(save_segmentation(viewer, obs["rgb"][:, :, :3].copy(), back_instances, label))
        print(f"backprojected frame {i:02d}: raytraced in {time.time() - st:.3f}s")

    if backprojected_tiles:
        make_mosaic(backprojected_tiles, N_cols=4).save(os.path.join(viewer.out_dir, f"{viewer.out_stem}_backprojected_mosaic.png"))
    print(f"backprojection pass: {time.time() - backproj_st:.3f}s total")


def frame_point_cloud(gt_instances: Instances, depth: np.ndarray, pose: np.ndarray) -> SemanticPointCloud:
    infos = [{"id_object": int(info["object_id"])} for info in gt_instances.infos]
    return project_semantic_masks_to_3d(depth, pose, gt_instances, infos, hfov=HFOV_DEG, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)


def fuse_point_cloud_frames(viewer: _Viewer, args: argparse.Namespace) -> SemanticPointCloud:
    fused, input_tiles = SemanticPointCloud(), []
    for i, state in enumerate(fusion_states(viewer, args)):
        rgb, depth, pose, gt_instances = observe_frame(viewer, state, args.classes_keep)
        input_tiles.append(save_segmentation(viewer, rgb, gt_instances, f"frame{i:02d}"))
        if len(gt_instances):
            st = time.time()
            fused = fused + frame_point_cloud(gt_instances, depth, pose)
            print(f"frame {i:02d}: project+fuse {len(gt_instances)} instances in {time.time() - st:.3f}s")
        if len(fused) == 0:
            continue
        st = time.time()
        fused.resolve()
        print(f"step {i + 1}: resolve in {time.time() - st:.3f}s, {len(fused.resolved_points)} resolved points")
        step_label = "step1" if i == 0 else f"step1:{i + 1}"
        if len(fused.resolved_points):
            save_point_cloud(fused, viewer.colors, os.path.join(viewer.out_dir, f"{viewer.out_stem}_{step_label}.ply"))
    save_input_mosaic(viewer, input_tiles)
    return fused


if __name__ == "__main__":
    args = viewer_arg_parser(__doc__, "SemanticPointCloud").parse_args()
    run_viewer(args, "semantic_point_cloud_viewer", fuse_point_cloud_frames)
