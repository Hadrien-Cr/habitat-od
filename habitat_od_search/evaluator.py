"""Runs habitat_od_search's SearchAgent over HSSD-rare episodes, computing success/SPL/failure-category and
writing per-episode logs -- one composed two-panel PNG per step (observation: rgb + single-target Detic
detections over the agent's own confidence threshold; semantic map: the agent's own depth-constructed
obstacle map, white background/gray obstacles/blue target marker, not the ground-truth navmesh), loosely
home-robot-styled (third_party/home-robot/src/home_robot_sim/home_robot_sim/env/habitat_objectnav_env/
visualizer.py) but built with this repo's own tiling helper (rows_mosaic) instead of vendoring its OpenCV
layout code -- plus a metrics.jsonl loadable by common/utils/plot_utils.py::load_metrics/plot_metrics.
See README.md for the full per-step loop, success/SPL/failure-category definitions and output layout.
"""
import argparse
import json
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import List

import cv2
import habitat_sim
import numpy as np
from detectron2.utils.visualizer import ColorMode  # type: ignore
from detectron2.utils.visualizer import Visualizer as DetVisualizer  # type: ignore
from habitat.utils.visualizations import maps  # type: ignore
from omegaconf import OmegaConf
from tabulate import tabulate
from tqdm import tqdm

from common.env_utils.env_base import build_env, gt_navigable_mask
from common.hybrid_exploration.frontiers import reveal_fog_of_war
from common.utils.plot_utils import plot_exploration_map, rows_mosaic, save_gif
from habitat_od_search.agent import SearchAgent, build_search_agent, target_embedding
from habitat_od_search.hssd_rare_dataset import SceneGoal, available_scenes, load_scene_episodes, main_category, rarity_tiers

_SUCCESS_DIST_M = 1.0
_BELIEVED_GOAL_MATCH_M = 1.0
_GT_MAP_RESOLUTION = 512


class FailureCategory(str, Enum):
    """Mirrors IMPRINT's failure-case appendix, narrowed to the 4 categories that matter here (success isn't a
    failure category, and is reported separately): a stop is a True Positive (TP) if select_action's found_goal
    latched on a believed goal within _BELIEVED_GOAL_MATCH_M of a real one, a False Positive (FP) otherwise."""
    SUCCESS = "success"
    EXPLORATION_FAILURE = "exploration_failure"  # ran out of steps/frontiers, no true goal ever came into view
    FN_RECOGNITION_FAILURE = "fn_recognition_failure"  # a true goal was in view at some point, but never confidently detected
    FP_RECOGNITION_FAILURE = "fp_recognition_failure"  # confidently stopped, but the believed goal wasn't a real one
    TP_EARLY_PROXIMITY_STOP = "tp_early_proximity_stop"  # correctly recognized the goal, but stopped outside _SUCCESS_DIST_M


@dataclass
class EpisodeResult:
    scene_id: str
    episode_id: str
    object_category: str
    rarity_tier: str
    success: bool
    spl: float
    steps: int
    geodesic_distance: float
    failure_category: str


class GroundTruthTracker:
    """Ground-truth navmesh topdown view and reveal tracking, built once per episode purely for the logged
    comparison panel -- never consulted by the agent or select_action, which only ever see the agent's own
    depth-built EstimatedMap."""

    def __init__(self, pathfinder: habitat_sim.PathFinder, position: np.ndarray, fov_deg: float, visibility_dist_m: float):
        self.pathfinder = pathfinder
        self.meters_per_pixel = maps.calculate_meters_per_pixel(_GT_MAP_RESOLUTION, pathfinder=pathfinder)
        self.navigable = gt_navigable_mask(pathfinder, position, _GT_MAP_RESOLUTION)
        self.revealed = np.zeros_like(self.navigable)
        self.fov_deg = fov_deg
        self.visibility_px = int(visibility_dist_m / self.meters_per_pixel)

    def to_pixels(self, positions: np.ndarray) -> np.ndarray:
        lower, upper = (np.array(bound)[[2, 0]] for bound in self.pathfinder.get_bounds())
        cell_size = np.abs(upper - lower) / np.array(self.navigable.shape)
        return ((positions[:, [2, 0]] - lower) / cell_size).astype(int)

    def to_pixel(self, position: np.ndarray) -> np.ndarray:
        return self.to_pixels(np.asarray(position)[None])[0]

    def reveal(self, position: np.ndarray, heading: float) -> None:
        in_view = reveal_fog_of_war(self.navigable, np.zeros_like(self.revealed), self.to_pixel(position), heading, self.fov_deg, self.visibility_px)
        self.revealed = self.revealed | in_view


def _goal_positions(goals: List[SceneGoal]) -> np.ndarray:
    return np.array([goal.position for goal in goals])


def _reached_goal(position: np.ndarray, goals: List[SceneGoal]) -> bool:
    return bool(np.any(np.linalg.norm(_goal_positions(goals) - position, axis=1) <= _SUCCESS_DIST_M))


def _goal_ever_in_view(agent: SearchAgent, goals: List[SceneGoal], in_view_history: np.ndarray) -> bool:
    """Whether any true goal's top-down pixel fell inside in_view_history, the union of every step's
    EstimatedMap.in_view -- line-of-sight visibility, not just proximity."""
    pixels = agent.estimated_map.to_pixels(_goal_positions(goals))
    in_bounds = (pixels[:, 0] >= 0) & (pixels[:, 0] < in_view_history.shape[0]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < in_view_history.shape[1])
    pixels = pixels[in_bounds]
    return bool(len(pixels)) and bool(np.any(in_view_history[pixels[:, 0], pixels[:, 1]] > 0))


def _believed_goal_correct(agent: SearchAgent, goals: List[SceneGoal]) -> bool:
    """Whether GoalScoreMap's best_goal_world (what the agent locked onto) is actually near a real goal."""
    if agent.goal_score.best_goal_world is None:
        return False
    return bool(np.any(np.linalg.norm(_goal_positions(goals) - agent.goal_score.best_goal_world, axis=1) <= _BELIEVED_GOAL_MATCH_M))


def classify_failure(success: bool, found_goal: bool, agent: SearchAgent, goals: List[SceneGoal], in_view_history: np.ndarray) -> FailureCategory:
    if success:
        return FailureCategory.SUCCESS
    if not found_goal:
        return (
            FailureCategory.FN_RECOGNITION_FAILURE if _goal_ever_in_view(agent, goals, in_view_history)
            else FailureCategory.EXPLORATION_FAILURE
        )
    return FailureCategory.TP_EARLY_PROXIMITY_STOP if _believed_goal_correct(agent, goals) else FailureCategory.FP_RECOGNITION_FAILURE


def run_episode(env, agent: SearchAgent, episode, goals: List[SceneGoal], vis_dir: Path) -> EpisodeResult:
    obs = env.reset(episode=episode)
    embedding = target_embedding(main_category(episode.info["object_category"]))
    position = np.array(env.get_agent_state().position)
    agent.reset(position, env.navmesh_floor_offset(), embedding)
    ground_truth = GroundTruthTracker(env.get_pathfinder(), position, agent.map_config.fov_deg, agent.map_config.visibility_dist_m)

    path_length = 0.0
    previous_position = position
    in_view_history = np.zeros_like(agent.estimated_map.obstacles)
    target_class = main_category(episode.info["object_category"])
    step = 0
    for step in range(agent.config.max_steps):
        agent_state = env.get_agent_state()
        action = agent.act(agent_state, obs["rgb"][:, :, :3], obs["depth"])
        in_view_history |= agent.estimated_map.in_view
        ground_truth.reveal(np.array(agent_state.position), agent.trajectory[-1][1])
        _save_step_visualization(agent, ground_truth, obs, target_class, goals, step, vis_dir)
        if action == "stop":
            break
        obs, _, _, _ = env.step(action)
        position = np.array(env.get_agent_state().position)
        path_length += float(np.linalg.norm(position - previous_position))
        previous_position = position

    success = _reached_goal(np.array(env.get_agent_state().position), goals)
    geodesic_distance = float(episode.info["geodesic_distance"])
    spl = geodesic_distance / max(path_length, geodesic_distance) if success and path_length > 0 else float(success)
    failure_category = classify_failure(success, agent.found_goal, agent, goals, in_view_history)
    save_gif(vis_dir, "episode.webp")
    return EpisodeResult(
        scene_id=episode.scene_id, episode_id=episode.episode_id, object_category=episode.info["object_category"],
        rarity_tier=rarity_tiers()[main_category(episode.info["object_category"])], success=success, spl=spl,
        steps=step + 1, geodesic_distance=geodesic_distance, failure_category=failure_category.value,
    )


_OBSTACLE_MAP_BG = (255, 255, 255)
_OBSTACLE_MAP_EXPLORED = (225, 225, 225)
_OBSTACLE_MAP_OBSTACLE = (170, 170, 170)
_OBSTACLE_MAP_TARGET = (255, 120, 0)  # BGR -- draws as blue
_OBSTACLE_MAP_VISITED = (0, 0, 255)  # BGR -- draws as red
_OBSTACLE_MAP_NAV_GOAL = (255, 0, 160)  # BGR -- draws as purple


def _draw_target_detections(rgb: np.ndarray, instances, target_class: str, score_thresh: float) -> np.ndarray:
    """rgb with every instance whose score clears score_thresh (the same threshold select_action uses to
    latch onto a goal, so the panel only shows what the agent itself would act on), box/mask and a
    "<target_class> <score%>" label, detectron2's own Visualizer draw style (as plot_segmentation_pred uses)."""
    if instances is None or len(instances) == 0:
        return rgb
    instances = instances[instances.scores >= score_thresh]
    if len(instances) == 0:
        return rgb
    det_visualizer = DetVisualizer(rgb, instance_mode=ColorMode.SEGMENTATION)
    for box, score in zip(instances.pred_boxes.tensor.cpu().numpy(), instances.scores.cpu().numpy()):
        det_visualizer.draw_text(f"{target_class} {round(100 * score)}%", position=(box[0], box[1]), horizontal_alignment="left", color="lime")
    vis_img = det_visualizer.overlay_instances(
        boxes=instances.pred_boxes.tensor.cpu().numpy(),
        masks=instances.pred_masks.cpu().numpy() if instances.has("pred_masks") else None,
        labels=None, assigned_colors=[(0.0, 1.0, 0.0)] * len(instances),
    )
    return vis_img.get_image()


def _obstacle_map(agent: SearchAgent, score_thresh: float, nav_target_px) -> np.ndarray:
    """Agent-constructed top-down view, built purely from the agent's own depth (EstimatedMap.explored/obstacles,
    not the ground-truth navmesh, which stays reserved for the comparison panel): white background, light gray
    explored area, gray obstacles, every visited pose as a red point (connected in visiting order), every cell
    whose accumulated goal_score clears score_thresh (the same bar select_action uses to latch a goal) shaded
    blue, and the navigation_target select_action is actually heading for this step (a frontier or the goal once
    confident) marked in purple."""
    image = np.full(agent.map.obstacles.shape + (3,), _OBSTACLE_MAP_BG, dtype=np.uint8)
    image[agent.map.explored > 0] = _OBSTACLE_MAP_EXPLORED
    image[agent.map.obstacles > 0] = _OBSTACLE_MAP_OBSTACLE
    image[agent.goal_score.goal_score >= score_thresh] = _OBSTACLE_MAP_TARGET
    visited_px = [(int(pixel[1]), int(pixel[0])) for pixel in (agent.map.to_pixel(position) for position, _ in agent.trajectory)]
    if len(visited_px) > 1:
        cv2.polylines(image, [np.array(visited_px)], isClosed=False, color=_OBSTACLE_MAP_VISITED, thickness=1)
    for point in visited_px:
        cv2.circle(image, point, 2, _OBSTACLE_MAP_VISITED, -1)
    if nav_target_px is not None:
        cv2.drawMarker(image, (int(nav_target_px[1]), int(nav_target_px[0])), _OBSTACLE_MAP_NAV_GOAL, cv2.MARKER_DIAMOND, 6, 1)
    return image


def _gt_map(agent: SearchAgent, ground_truth: GroundTruthTracker, goals: List[SceneGoal]) -> np.ndarray:
    """Ground-truth navmesh top-down view (plot_exploration_map, already BGR->RGB internally consistent with its
    other callers): navigable/revealed area, every visited pose, and every true goal position -- a trusted
    reference to check the depth-constructed semantic map (_obstacle_map) against."""
    trajectory_px = [ground_truth.to_pixel(position) for position, _ in agent.trajectory]
    goal_pixels = ground_truth.to_pixels(_goal_positions(goals))
    image = plot_exploration_map(ground_truth.navigable, ground_truth.revealed, trajectory_px, goal_px=goal_pixels[0] if len(goal_pixels) else None)
    for pixel in goal_pixels[1:]:
        cv2.drawMarker(image, (int(pixel[1]), int(pixel[0])), (0, 255, 0), cv2.MARKER_CROSS, 14, 3)
    return image[:, :, ::-1]  # BGR -> RGB, make_mosaic expects RGB


def _save_step_visualization(
    agent: SearchAgent, ground_truth: GroundTruthTracker, obs: dict, target_class: str, goals: List[SceneGoal], step: int, vis_dir: Path,
) -> None:
    score_thresh = agent.config.goal_policy.goal_score_thresh

    first_person = _draw_target_detections(obs["rgb"][:, :, :3].copy(), agent.last_instances, target_class, score_thresh)
    height, width = first_person.shape[:2]

    nav_target_px = agent.map.to_pixel(agent.navigation_target) if agent.navigation_target is not None else None
    top_down = _obstacle_map(agent, score_thresh, nav_target_px)[:, :, ::-1]  # BGR -> RGB, make_mosaic expects RGB
    top_down_resized = cv2.resize(top_down, (width, height), interpolation=cv2.INTER_NEAREST)

    gt_map = cv2.resize(_gt_map(agent, ground_truth, goals), (width, height), interpolation=cv2.INTER_NEAREST)

    mosaic = rows_mosaic([[
        (f"observation (query: {target_class})", first_person),
        ("semantic map", np.ascontiguousarray(top_down_resized)),
        ("ground truth", np.ascontiguousarray(gt_map)),
    ]])
    if mosaic is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        mosaic.save(vis_dir / f"snapshot_{step:03d}.png")


def run_evaluation(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    scenes = cfg.scenes if cfg.scenes else available_scenes()
    run_dir = Path("habitat_od_search") / "logs" / cfg.run_name
    metrics_path = run_dir / "metrics.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)

    results: List[EpisodeResult] = []
    with open(metrics_path, "w") as metrics_file:
        for scene_id in scenes:
            scene_episodes = load_scene_episodes(scene_id)
            env, habitat_cfg = build_env(scene_id, cfg.seed, cfg.agent.max_steps, OmegaConf.to_container(cfg.object_params), cfg.agent.input_image_size)
            agent = build_search_agent(args, cfg.agent, habitat_cfg, np.random.default_rng(cfg.seed))
            for episode in tqdm(scene_episodes.episodes, desc=f"search {scene_id}", ncols=80):
                goals_key = f"{scene_id}_{episode.info['object_category']}"
                vis_dir = run_dir / "images" / f"{scene_id}_{episode.episode_id}"
                result = run_episode(env, agent, episode, scene_episodes.goals_by_category[goals_key], vis_dir)
                results.append(result)
                metrics_file.write(json.dumps(asdict(result)) + "\n")
                metrics_file.flush()

    _print_summary(results)
    _print_failure_breakdown(results)


def _print_summary(results: List[EpisodeResult]) -> None:
    tiers = ["easy", "medium", "hard"]
    rows = [
        (tier, np.mean([r.success for r in tier_results]), np.mean([r.spl for r in tier_results]), len(tier_results))
        for tier in tiers
        for tier_results in [[r for r in results if r.rarity_tier == tier]]
        if tier_results
    ]
    rows.append(("overall", np.mean([r.success for r in results]), np.mean([r.spl for r in results]), len(results)))
    print(tabulate(rows, tablefmt="pipe", floatfmt=".3f", headers=["tier", "success", "spl", "n episodes"], numalign="left"))


def _print_failure_breakdown(results: List[EpisodeResult]) -> None:
    """Share of every episode (success included) per FailureCategory, as IMPRINT's failure-case appendix does."""
    total = len(results)
    rows = [
        (category.value, sum(r.failure_category == category.value for r in results) / total)
        for category in FailureCategory
    ]
    print(tabulate(rows, tablefmt="pipe", floatfmt=".3f", headers=["category", "share"], numalign="left"))
