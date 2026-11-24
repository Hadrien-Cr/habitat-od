"""Stage 1: records a train scene raw (common/utils/raw_scene.py) under <raw_root>/<run_name>/<scene>/ for
habitat_embodied_sl/build_dataset.py (stage 2) to build its datasets from, without the simulator (see
habitat_embodied_sl/config/procthor_train20_collection.yaml and habitat_embodied_sl/README.md); test scenes are collect_test.py's.
One scene per call, so one call per scene can run in parallel, each on the GPU its caller's CUDA_VISIBLE_DEVICES names.
common/hybrid_exploration's HybridFrontierAgent
explores in exhaustive mode for up to collection.steps_per_scene steps (or until it gets stuck -- no net translation
over BaseExplorationAgent.is_stuck's window, which stops it, logged as a warning) while a Detector (Detic's proposals scored by
objectness alone, SAM masks) segments every frame; its masks and region embeddings are fused into concept-graphs objects
(common/utils/concept_graph.py's add_processed), while the agent's own ObjectTracker
(common/hybrid_exploration/object_tracker.py) tracks the same detections into the objects it inspects. Every frame and its
detections are recorded as they come; once exploration ends, the fused objects, the agent's map grid, the scene's annotated
objects, with their ground_truth.json (eval_utils.py's single-scene COCO dataset: 3D boxes and top-down footprints on the
agent's map grid, its navigable cells the simulator's navmesh), and every frame's ground truth. The last step's 3-tile
(observation, belief map, ground truth) snapshot is saved to the record's exploration_snapshot.png, and every step's
detections and ground truth, read back from the record, are drawn on its rgb as the record's predictions_mosaic.jpg and
ground_truth_mosaic.jpg. A call whose scene already has a complete record skips it unless --overwrite; otherwise it first
deletes the scene's record. The run's first call saves the config,
its fusion section completed with every ConceptGraphConfig default and its kept classes as classes, as
<raw_root>/<run_name>/config.yaml, and a call with another config or other classes fails (save_run_config): a run's
records share one, and the build classifies among and indexes those classes.

Usage (the config's detector: Detic's Swin-B checkpoint from https://github.com/facebookresearch/Detic's model zoo and
SAM's; Detic classifies against the CLIP text embeddings of the config's vocab_name/filter_classes class names):
  export BASE_DIR=$(pwd) PYTHONNOUSERSITE=1  # HABITAT_DATA already exported, see INSTALL.MD
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python habitat_embodied_sl/collect_train.py <scene> habitat_embodied_sl/config/procthor_train20_collection.yaml
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python habitat_embodied_sl/build_dataset.py <scene> habitat_embodied_sl/config/procthor30_build.yaml variant=V2_drop_add  # stage 2
"""
import argparse
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from habitat.tasks.nav.nav import TopDownMap
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from common.env_utils.env_base import ExplorationEnv, build_agent, build_env
from common.env_utils.object_annotations import kept_classes_from_args, resolve_classes, scene_world_mesh
from common.env_utils.sense import ground_truth_instances
from common.hybrid_exploration.exploration_map import MapGrid
from common.hybrid_exploration.frontier_exp_agent import exploration_snapshot, save_steps_mosaic
from common.hybrid_exploration.hybrid_agent import HybridFrontierAgent
from common.utils.concept_graph import ConceptGraphConfig, ConceptGraphObject, add_processed, predicted_objects, processed_frame
from common.utils.eval_utils import GroundTruthObjects, ground_truth_dataset, save_ground_truth_ply
from common.utils.ground_truth_recovery import GroundTruthView, encoded_views, ground_truth_frame, views_by_frame
from common.utils.image_memory import Frame
from common.utils.plot_utils import PNG_COMPRESS_LEVEL
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import DepthCamera, nearest_resized
from common.utils.raw_scene import (
    EXPLORATION_SNAPSHOT_PNG, RawScene, is_recorded, load_detections, load_raw_scene, raw_scene_dir, save_frame, save_raw_scene, save_run_config,
)
from common.utils.retrieval import SegmentationStyle, segmentation_tile
from common.vision.detector import Detector, build_detector


def observe_frame(obs: dict, env: ExplorationEnv, lift_image_size: int) -> Frame:
    """obs' rgb, depth resized to lift_image_size x lift_image_size and depth-sensor pose."""
    rgb = obs["rgb"][:, :, :3]
    pose = sensor_pose_matrix(env.get_agent_state().sensor_states["depth"])
    return Frame(rgb, nearest_resized(obs["depth"].squeeze(), (lift_image_size, lift_image_size)).numpy(), pose)


@dataclass
class SceneMapper:
    """Records each frame the agent observes and its detections under raw_dir, fuses them into objects and feeds them to the
    agent's tracker, keeping the frame's RLE-encoded ground truth in ground_truth_views."""

    agent: HybridFrontierAgent
    detector: Detector
    camera: DepthCamera
    vocabulary: list
    floor_offset: float
    lift_image_size: int
    raw_dir: Path
    config: ConceptGraphConfig
    objects: List[ConceptGraphObject] = field(default_factory=list)
    ground_truth_views: List[GroundTruthView] = field(default_factory=list)
    num_frames: int = 0

    def observe(self, obs: dict, env: ExplorationEnv) -> None:
        frame = observe_frame(obs, env, self.lift_image_size)
        frame_id, self.num_frames = self.num_frames, self.num_frames + 1
        instances = self.detector.predict(frame.rgb)
        floor_height = env.get_agent_state().position[1] - self.floor_offset
        save_frame(self.raw_dir, frame_id, frame, floor_height, instances)
        detections = processed_frame(predicted_objects(frame_id, frame, instances, self.camera, floor_height), self.config)
        self.objects = add_processed(self.objects, detections, self.config)
        self.agent.update_detections(detections)
        self.ground_truth_views += encoded_views(frame_id, ground_truth_instances(obs, self.vocabulary, self.detector.classes))


def explore(env: ExplorationEnv, mapper: SceneMapper, obs: dict, steps: int, ground_truth) -> bool:
    """Has mapper observe every frame from obs on until its agent stops (also after too many successive collisions),
    gets stuck or steps run out, saving the last step's 3-tile (observation, belief map, ground truth) snapshot to
    mapper's raw_dir/EXPLORATION_SNAPSHOT_PNG; whether the agent got stuck."""
    collided = False
    trajectory_px: list = []
    for _ in tqdm(range(steps), desc="mapping", ncols=80):
        state = env.get_agent_state()
        heading = float(TopDownMap.get_polar_angle(state))
        trajectory_px.append(ground_truth.to_pixel(np.array(state.position)))
        ground_truth.reveal(np.array(state.position), heading)
        mapper.observe(obs, env)
        action, _ = mapper.agent.act(env.get_agent_state(), obs["depth"], collided)
        if action == "stop":
            break
        obs, _, _, _ = env.step(action)
        collided = bool(np.any(obs["agent_collision_sensor"]))
    snapshot = exploration_snapshot(obs, mapper.agent, ground_truth, trajectory_px)
    if snapshot is not None:
        snapshot.save(mapper.raw_dir / EXPLORATION_SNAPSHOT_PNG, compress_level=PNG_COMPRESS_LEVEL)
    return mapper.agent.is_stuck()


def save_step_mosaics(raw_dir: Path, raw: RawScene, style: SegmentationStyle) -> None:
    """raw_dir's predictions_mosaic.jpg and ground_truth_mosaic.jpg (save_steps_mosaic): every recorded step's detections
    (load_detections) and ground truth drawn on its rgb, tile for tile."""
    frame_views, frames = views_by_frame(raw.ground_truth_views), sorted(raw.frames.items())
    prediction_tiles = [segmentation_tile(frame.rgb, load_detections(raw_dir, frame_id), style, f"step{frame_id:04d}") for frame_id, frame in frames]
    ground_truth_tiles = [
        segmentation_tile(frame.rgb, ground_truth_frame(frame_views[frame_id], frame.rgb.shape[:2]), style, f"step{frame_id:04d}") for frame_id, frame in frames
    ]
    save_steps_mosaic(prediction_tiles, raw_dir / "predictions_mosaic.jpg")
    save_steps_mosaic(ground_truth_tiles, raw_dir / "ground_truth_mosaic.jpg")


def map_scene(scene: str, cfg: DictConfig, detector: Detector) -> None:
    """Records the scene's exploration raw under raw_scene_dir, its annotated objects those of detector's classes, with
    the last step's 3-tile snapshot and, read back from the record, its step mosaics, fusing with cfg's fusion settings."""
    env, habitat_cfg = build_env(scene, cfg.seed, cfg.collection.steps_per_scene, cfg.object_params, cfg.input_image_size)
    raw_dir, fusion = raw_scene_dir(cfg, scene), ConceptGraphConfig(**cfg.fusion)
    agent = build_agent(
        "hybrid", np.random.default_rng(cfg.seed), env, habitat_cfg, dump_location=str(raw_dir), exp_name="planner", fusion_config=fusion,
    )
    obs = env.reset(env.episodes[0], rng=agent.rng)
    floor_offset = env.navmesh_floor_offset()
    agent.reset(env.get_agent_state(), env.get_agent_state().position[1] - floor_offset)
    depth_cfg = habitat_cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
    camera = DepthCamera(depth_cfg.hfov, depth_cfg.min_depth, depth_cfg.max_depth)
    vocabulary = resolve_classes(cfg.object_params.env_name, cfg.object_params.vocab_name)
    mapper = SceneMapper(agent, detector, camera, vocabulary, floor_offset, cfg.lift_image_size, raw_dir, fusion)
    gt_meshes = env.get_gt_meshes(detector.classes)
    map_config = agent.config.map_config
    ground_truth = env.gt_map_tracker(map_config.fov_deg, map_config.visibility_dist_m)
    ground_truth.set_object_footprints(gt_meshes)
    stuck = explore(env, mapper, obs, cfg.collection.steps_per_scene, ground_truth)
    if stuck:
        print(f"Warning: agent got stuck in {scene} (no net translation)")
    grid = MapGrid.of(agent.estimated_map, agent.ref_state)
    style = SegmentationStyle.of(detector.classes)
    save_ground_truth_ply(gt_meshes, style.colors, raw_dir / "ground_truth_point_cloud.ply")
    raw = RawScene(
        scene, camera, mapper.objects, fusion, cfg.object_params.min_mesh_visibility_fraction, GroundTruthObjects.from_meshes(gt_meshes),
        mapper.ground_truth_views, grid, ground_truth_dataset(scene, gt_meshes, grid, env.get_pathfinder(), detector.classes),
        scene_world_mesh(env.get_sim()),
    )
    env.close()
    save_raw_scene(raw_dir, raw)
    save_step_mosaics(raw_dir, load_raw_scene(raw_dir), style)


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    cfg.fusion = asdict(ConceptGraphConfig(**cfg.fusion))
    cfg.classes = kept_classes_from_args(cfg.object_params)  # recorded with the run: its ground truth's class ids index them
    raw_dir = raw_scene_dir(cfg, args.scene)
    save_run_config(raw_dir.parent, cfg)
    if is_recorded(raw_dir) and not args.overwrite:
        print(f"{args.scene} already recorded in {raw_dir}, skipped (--overwrite recollects it)")
        return
    shutil.rmtree(raw_dir, ignore_errors=True)
    map_scene(args.scene, cfg, build_detector(cfg, cfg.detector.sam_checkpoint))
    print(f"{args.scene} recorded in {raw_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="Train scene to collect, e.g. ProcTHOR-Train-5.")
    parser.add_argument("config", help="Collection config: run_name, raw_root, image sizes, object_params, detector, budgets.")
    parser.add_argument("--overwrite", action="store_true", help="Recollect the scene even if its raw record is complete.")
    main(parser.parse_args())
