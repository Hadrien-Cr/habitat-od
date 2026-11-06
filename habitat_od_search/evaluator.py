"""Per-episode success/SPL/distance-to-goal and IMPRINT-style failure categories of habitat_od_search's SearchAgent
over HSSD-rare episodes, and their per-rarity-tier summary tables -- the metrics habitat_od_search/main.py logs.
Distance to goal is the horizontal distance to the nearest goal object's mesh footprint, and success is a stop within
SUCCESS_DIST_M of one -- IMPRINT's "within 1 m of a goal object's outline", without its precomputed view points."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from habitat.tasks.nav.nav import NavigationEpisode  # type: ignore
from scipy.spatial import cKDTree
from tabulate import tabulate

from habitat_od_search.hssd_rare_dataset import SceneGoal, main_category, rarity_tiers
from habitat_od_search.search_agent import SearchAgent, mesh_footprint

SUCCESS_DIST_M = 1.0
_BELIEVED_GOAL_MATCH_M = 1.0
_RARITY_TIERS = ("easy", "medium", "hard")


class FailureCategory(str, Enum):
    """IMPRINT's failure-case appendix, narrowed to the 4 categories that matter here, plus success: a stop is a
    True Positive (TP) if the agent latched onto a believed goal within _BELIEVED_GOAL_MATCH_M of a real one,
    a False Positive (FP) otherwise."""
    SUCCESS = "success"
    EXPLORATION_FAILURE = "exploration_failure"  # ran out of steps/frontiers, no true goal ever visible in a ground-truth frame
    FN_RECOGNITION_FAILURE = "fn_recognition_failure"  # a true goal was visible in some ground-truth frame, but never confidently detected
    FP_RECOGNITION_FAILURE = "fp_recognition_failure"  # confidently latched onto a goal that wasn't a real one
    TP_EARLY_STOP = "tp_early_stop"  # latched onto the real goal, but stopped farther than SUCCESS_DIST_M from its footprint


@dataclass
class EpisodeTrace:
    """What habitat_od_search/main.py accumulates while running an episode, beyond the agent's own state."""
    goal_seen: bool = False
    stuck: bool = False
    stop_reason: Optional[str] = None  # the StopReason agent.act() returned alongside its last "stop" action
    world_trajectory: List[Tuple[np.ndarray, float]] = field(default_factory=list)  # (world position, heading) at every act()


@dataclass
class EpisodeResult:
    scene_id: str
    episode_id: str
    object_category: str
    rarity_tier: str
    success: bool
    spl: float
    steps: int
    stuck: bool
    stop_reason: Optional[str]
    path_length: float
    geodesic_distance: float
    distance_to_goal: float
    failure_category: str


def goal_positions(goals: List[SceneGoal]) -> np.ndarray:
    return np.array([goal.position for goal in goals])


def distance_to_footprints(position: np.ndarray, meshes: List[trimesh.Trimesh]) -> float:
    """Horizontal distance from position to the nearest of meshes' top-down footprints."""
    return min(float(cKDTree(mesh_footprint(mesh)).query(position[[0, 2]])[0]) for mesh in meshes)


def _believed_goal_correct(agent: SearchAgent, goals: List[SceneGoal]) -> bool:
    """Whether any cell the estimated map currently believes is the goal (goal_mask()'s own footprint) is
    actually horizontally near a real one."""
    assert agent.estimated_map is not None, "Call reset() first"
    believed = agent.estimated_map.believed_goal_positions(agent.config.goal_score_thresh)
    if len(believed) == 0:
        return False
    goals_start_frame_xz = agent.start_frame_points(goal_positions(goals))[:, [0, 2]]
    return bool(np.any(np.linalg.norm(goals_start_frame_xz[:, None] - believed[None], axis=2) <= _BELIEVED_GOAL_MATCH_M))


def classify_failure(success: bool, agent: SearchAgent, goals: List[SceneGoal], goal_seen: bool) -> FailureCategory:
    assert agent.estimated_map is not None, "Call reset() first"
    if success:
        return FailureCategory.SUCCESS
    if agent.estimated_map.goal_score.max() < agent.config.goal_score_thresh:
        return FailureCategory.FN_RECOGNITION_FAILURE if goal_seen else FailureCategory.EXPLORATION_FAILURE
    return FailureCategory.TP_EARLY_STOP if _believed_goal_correct(agent, goals) else FailureCategory.FP_RECOGNITION_FAILURE


def episode_result(agent: SearchAgent, episode: NavigationEpisode, goals: List[SceneGoal], meshes: List[trimesh.Trimesh], trace: EpisodeTrace) -> EpisodeResult:
    """EpisodeResult of episode, from the agent's final belief, the trace of its run and the goal objects' meshes."""
    positions = np.array([position for position, _ in trace.world_trajectory])
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    distance_to_goal = distance_to_footprints(positions[-1], meshes)
    success = distance_to_goal <= SUCCESS_DIST_M
    geodesic_distance = float(episode.info["geodesic_distance"])
    category = episode.info["object_category"]
    return EpisodeResult(
        scene_id=episode.scene_id, episode_id=episode.episode_id, object_category=category,
        rarity_tier=rarity_tiers()[main_category(category)], success=success,
        spl=float(success) * geodesic_distance / max(path_length, geodesic_distance), steps=len(positions), stuck=trace.stuck,
        stop_reason=trace.stop_reason, path_length=path_length, geodesic_distance=geodesic_distance,
        distance_to_goal=distance_to_goal,
        failure_category=classify_failure(success, agent, goals, trace.goal_seen).value,
    )


def summary_table(results: List[EpisodeResult]) -> str:
    """Success, SPL, distance-to-goal and episode count per rarity tier, then overall."""
    groups = [(tier, [result for result in results if result.rarity_tier == tier]) for tier in _RARITY_TIERS] + [("overall", results)]
    rows = [
        (name, 100 * np.mean([r.success for r in group]), 100 * np.mean([r.spl for r in group]), np.mean([r.distance_to_goal for r in group]), len(group))
        for name, group in groups
        if group
    ]
    return tabulate(rows, tablefmt="pipe", floatfmt=".2f", headers=["tier", "success", "spl", "dtg (m)", "n episodes"], numalign="left")


def failure_table(results: List[EpisodeResult]) -> str:
    """Share of every episode (success included) per FailureCategory, as IMPRINT's failure-case appendix does."""
    rows = [(category.value, 100 * np.mean([r.failure_category == category.value for r in results])) for category in FailureCategory]
    return tabulate(rows, tablefmt="pipe", floatfmt=".2f", headers=["category", "share"], numalign="left")
