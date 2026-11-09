"""Evaluates habitat_od_search's SearchAgent on IMPRINT's HSSD-rare ObjectNav episodes (--config, see
habitat_od_search/config/hssd_rare_eval.yaml). For each scene in `scenes` (every saved hssd_rare_eps scene when
empty), one ExplorationEnv is built; each of its episodes whose start navmesh island covers at least
min_island_area_m2 (smaller ones, e.g. a balcony cut off from the house, are skipped) and at most max_island_area_m2
(larger ones are usually a navmesh merging areas that shouldn't be connected), and whose episode_id is in
cfg.episode_ids when given (every episode otherwise), resets the env to the episode's
own start pose and runs a fresh SearchAgent, targeting the CLIP text embedding of the episode's main category (the text before the first
comma of its fine-grained object_category), until it stops (also once stuck, BaseExplorationAgent.is_stuck) or
runs out of agent.max_steps -- for at most max_episodes episodes in all (every one when null). Each episode's success, SPL,
distance-to-goal and failure category (habitat_od_search/evaluator.py) is appended to
habitat_od_search/logs/<run_name>/metrics.jsonl as soon as it ends, and the per-rarity-tier summary and failure
breakdown tables are printed and saved to summary.md there. With agent.use_gt_detections, the agent uses the goal
objects' ground-truth detections (target_instances, scored 1.0) instead of Detic's.
Success is a stop within 1 m of a goal object's mesh footprint. With save_visuals, each step's 3-tile (detections,
belief map cropped to the 256 x 256 cells around the agent, ground truth with the episode's goal object footprints) snapshot is saved to images/<run_index>_ep<episode_id>_<scene_id>/ (run_index counting every episode run so far,
across scenes) and assembled into episode.webp, its last one also copied next to that folder as
images/<run_index>_ep<episode_id>_<scene_id>_last_snapshot.png. With save_mesh, each scene's ground-truth mesh (stage
+ every placed object, ceiling stripped) is written once to meshes/<scene_id>.glb.

Usage:
  export HABITAT_DATA=$HOME/habitat_data/data BASE_DIR=$(pwd) PYTHONNOUSERSITE=1
  PYTHONPATH=. python habitat_od_search/main.py --config habitat_od_search/config/hssd_rare_eval.yaml \
      --config-file third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
      --checkpoint third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
"""
import argparse
import itertools
import json
import shutil
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import numpy as np
import trimesh
from habitat.tasks.nav.nav import NavigationEpisode, TopDownMap  # type: ignore
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from common.env_utils.env_base import ExplorationEnv, build_env, sensor_geometry
from common.env_utils.object_annotations import get_all_objects, scene_world_mesh
from common.env_utils.sense import target_instances, visible_object_ids
from common.hybrid_exploration.exploration_map import MapConfig
from common.hybrid_exploration.frontier_exp_agent import DiscretePlannerConfig, exploration_snapshot, obs_dilation_cells
from common.hybrid_exploration.ground_truth_map import GroundTruthMapTracker
from common.utils.plot_utils import save_gif
from common.vision.detic import Vocabulary
from common.vision.detector import Detector
from habitat_od_search.evaluator import EpisodeResult, EpisodeTrace, episode_result, failure_table, summary_table
from habitat_od_search.hssd_rare_dataset import SceneGoal, available_scenes, load_scene_episodes, main_category
from habitat_od_search.search_agent import SearchAgent, SearchAgentConfig

_RUNS_ROOT = Path("habitat_od_search") / "logs"
_GIF_FPS = 10


def search_agent_config(agent_cfg: DictConfig, fov_deg: float, turn_angle_deg: float, agent_radius_m: float) -> SearchAgentConfig:
    return SearchAgentConfig(
        map_config=MapConfig(
            fov_deg, resolution_m=agent_cfg.resolution_m, visibility_dist_m=agent_cfg.visibility_dist_m,
            lift_image_size=agent_cfg.lift_image_size, score_decay=agent_cfg.score_decay,
        ),
        goal_score_thresh=agent_cfg.goal_score_thresh,
        max_steps=agent_cfg.max_steps,
        confidence_thresh=agent_cfg.confidence_thresh,
        planner_config=DiscretePlannerConfig(
            turn_angle_deg,
            obs_dilation_selem_radius=obs_dilation_cells(agent_radius_m, agent_cfg.resolution_m),
            debug=agent_cfg.get("planner_debug", False),
        ),
        reach_m=agent_cfg.reach_m,
        use_gt_detections=agent_cfg.use_gt_detections,
    )


def save_snapshot(obs: dict, agent: SearchAgent, ground_truth: GroundTruthMapTracker, trace: EpisodeTrace, images_dir: Path) -> None:
    """Reveals the agent's current view on ground_truth, then saves the step's 3-tile exploration_snapshot -- the
    ground-truth tile from the agent's world poses, its belief map in the agent's own start frame."""
    position, heading = trace.world_trajectory[-1]
    ground_truth.reveal(position, heading)
    trajectory_px = [ground_truth.to_pixel(visited) for visited, _ in trace.world_trajectory]
    mosaic = exploration_snapshot(obs, agent, ground_truth, trajectory_px)
    if mosaic is not None:
        mosaic.save(images_dir / f"snapshot_gif_{len(trace.world_trajectory) - 1:04d}.png")


def goal_meshes(env: ExplorationEnv, goal_ids: Set[int]) -> List[Tuple[int, int, trimesh.Trimesh]]:
    """env.get_gt_meshes of the goal objects alone, for GroundTruthMapTracker.set_object_footprints and episode_result."""
    annotation = env.get_object_annotation()
    goal_classes = [annotation.classes[annotation.obj_id_to_class_id[object_id]] for object_id in goal_ids]
    return [mesh for mesh in env.get_gt_meshes(goal_classes) if mesh[0] in goal_ids]


def run_episode(
    env: ExplorationEnv, agent: SearchAgent, episode: NavigationEpisode, goals: List[SceneGoal], images_dir: Optional[Path],
    navmesh_bounds: Tuple[np.ndarray, np.ndarray],
) -> EpisodeResult:
    """Runs agent on episode until it stops or gets stuck, saving per-step snapshots to images_dir if given."""
    obs = env.reset(episode=episode)
    start_state = env.get_agent_state()
    category = episode.info["object_category"]
    target = main_category(category)
    agent.reset(
        start_state, start_state.position[1] - env.navmesh_floor_offset(),
        Vocabulary(f"search_{target}", (target,)).clip_embeddings()[:, 0], category, navmesh_bounds,
    )
    goal_handles = {goal.object_handle for goal in goals}
    goal_ids = {obj.object_id for obj in get_all_objects(env.get_sim()) if obj.handle in goal_handles}
    trace = EpisodeTrace()
    meshes = goal_meshes(env, goal_ids)
    ground_truth = env.gt_map_tracker(agent.map_config.fov_deg, agent.map_config.visibility_dist_m)
    if images_dir is not None:
        images_dir.mkdir(parents=True, exist_ok=True)
        ground_truth.set_object_footprints(meshes)
    for _ in range(agent.config.max_steps):
        state = env.get_agent_state()
        trace.world_trajectory.append((np.array(state.position), float(TopDownMap.get_polar_angle(state))))
        action, stop_reason = agent.act(state, obs["rgb"], obs["depth"], gt_instances=target_instances(obs, goal_ids))
        trace.goal_seen |= bool(goal_ids & visible_object_ids(obs))
        if images_dir is not None:
            save_snapshot(obs, agent, ground_truth, trace, images_dir)
        if action == "stop":
            trace.stop_reason = stop_reason.value if stop_reason else None
            break
        obs, _, _, _ = env.step(action)
    trace.stuck = agent.is_stuck()
    if images_dir is not None:
        save_gif(images_dir, "episode.webp", fps=_GIF_FPS)
        shutil.copyfile(max(images_dir.glob("snapshot_gif_*.png")), images_dir.parent / f"{images_dir.name}_last_snapshot.png")
    return episode_result(agent, episode, goals, [mesh for _, _, mesh in meshes], trace)


def episode_rng(cfg_seed: int, scene_id: str, episode_id: str) -> np.random.Generator:
    """Generator seeded from cfg_seed, scene_id and episode_id alone -- so an episode's own randomness (BaseExplorationAgent.spin's
    initial turn direction) is the same whether it runs as part of a full scene or rerun by itself, unlike a single
    rng shared and advanced across a scene's episodes in order."""
    return np.random.default_rng(zlib.crc32(f"{cfg_seed}_{scene_id}_{episode_id}".encode()))


def episode_island_area(pathfinder, episode: NavigationEpisode) -> float:
    return pathfinder.island_area(pathfinder.get_island(np.array(episode.start_position)))


def evaluate_scene(scene_id: str, cfg: DictConfig, detector: Detector, run_dir: Path, run_indices: Iterator[int]) -> Iterator[EpisodeResult]:
    """EpisodeResult of every HSSD-rare episode of scene_id, yielded as each one ends; run_indices numbers them across scenes."""
    scene_episodes = load_scene_episodes(scene_id)
    env, habitat_cfg = build_env(scene_id, cfg.seed, cfg.agent.max_steps, cfg.object_params, cfg.agent.input_image_size)
    if cfg.save_mesh:
        (run_dir / "meshes").mkdir(parents=True, exist_ok=True)
        scene_world_mesh(env.get_sim()).export(str(run_dir / "meshes" / f"{scene_id}.glb"))
    fov_deg, camera = sensor_geometry(habitat_cfg)
    config = search_agent_config(cfg.agent, fov_deg, habitat_cfg.habitat.simulator.turn_angle, habitat_cfg.habitat.simulator.agents.main_agent.radius)
    pathfinder = env.get_pathfinder()
    navmesh_bounds = pathfinder.get_bounds()
    episodes = [
        episode for episode in scene_episodes.episodes
        if cfg.min_island_area_m2 <= episode_island_area(pathfinder, episode) <= cfg.max_island_area_m2
        and episode.episode_id in cfg.get("episode_ids", [episode.episode_id])
    ]
    try:
        for episode in tqdm(episodes, desc=f"search {scene_id}", ncols=80):
            category = episode.info["object_category"]
            run_name = f"{next(run_indices):03d}_ep{episode.episode_id}_{scene_id}"
            rng = episode_rng(cfg.seed, scene_id, episode.episode_id)
            agent = SearchAgent(detector, camera, config, rng, str(run_dir), run_name, main_category(category))
            images_dir = run_dir / "images" / run_name if cfg.save_visuals else None
            yield run_episode(env, agent, episode, scene_episodes.goals_by_category[f"{scene_id}_{category}"], images_dir, navmesh_bounds)
    finally:
        env.close()


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    detector = Detector(
        args.config_file, args.checkpoint, Vocabulary("search_unused", ("object",)), args.sam_checkpoint,
        score_thresh=args.score_thresh, nms_thresh=args.nms_thresh, mask_thresh=args.mask_thresh, device=args.device, input_size=cfg.agent.input_image_size,
    )
    run_dir = _RUNS_ROOT / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_indices = itertools.count()
    episode_results = itertools.islice(
        (result for scene_id in cfg.scenes or available_scenes() for result in evaluate_scene(scene_id, cfg, detector, run_dir, run_indices)),
        cfg.max_episodes,
    )
    results: List[EpisodeResult] = []
    with open(run_dir / "metrics.jsonl", "w") as metrics_file:
        for result in episode_results:
            results.append(result)
            metrics_file.write(json.dumps(asdict(result)) + "\n")
            metrics_file.flush()
    report = f"{summary_table(results)}\n\n{failure_table(results)}\n"
    (run_dir / "summary.md").write_text(report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="habitat_od_search/config/hssd_rare_eval.yaml", help="Evaluation config: scenes, agent, object_params.")
    parser.add_argument("--config-file", default="third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    parser.add_argument("--checkpoint", default="third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
    parser.add_argument("--sam-checkpoint", default=None)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
