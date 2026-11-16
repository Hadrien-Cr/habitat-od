"""Per-episode success/SPL/distance-to-goal and IMPRINT-style failure categories of habitat_od_search's SearchAgent
over HSSD-rare episodes, and their per-rarity-tier summary tables -- the metrics habitat_od_search/main.py logs.
Distance to goal is the horizontal distance to the nearest goal object's mesh footprint, and success is a stop within
SUCCESS_DIST_M of one -- IMPRINT's "within 1 m of a goal object's outline", without its precomputed view points, widened
to 2 m since some goals stand more than 1 m from any navigable point (e.g. a vase on a deep sideboard) -- and only while
the agent's own believed goal lands on a real one, so a stop that merely lands near a goal by chance doesn't count."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import trimesh
from detectron2.structures import Instances
from habitat.tasks.nav.nav import NavigationEpisode  # type: ignore
from scipy.spatial import cKDTree
from tabulate import tabulate

from habitat_od_search.hssd_rare_dataset import main_category, rarity_tiers
from habitat_od_search.search_agent import SearchAgent, mesh_footprint

SUCCESS_DIST_M = 2.0
_BELIEVED_GOAL_MATCH_M = 1.0
_RARITY_TIERS = ("easy", "medium", "hard")


class FailureCategory(str, Enum):
    """IMPRINT's failure-case appendix, narrowed to the categories that matter here, plus success: a stop is a True
    Positive (TP) if the agent's believed goal lands on a real one (believed_object), a False Positive (FP) otherwise.
    A failed stop that happens to land near a goal is classified by its belief like any other."""
    SUCCESS = "success"
    EXPLORATION_FAILURE = "exploration_failure"  # ran out of steps/frontiers, no true goal ever visible in a ground-truth frame
    FN_DET = "fn_det"  # a true goal was visible in some ground-truth frame, but never confidently detected
    FP_DET = "fp_det"  # confidently latched onto a believed goal that is neither a real goal nor an object synonym
    TP_WRONG_STOP = "tp_wrong_stop"  # latched onto the real goal, but stopped farther than SUCCESS_DIST_M from its footprint
    FP_SYNONYM = "fp_synonym"  # confidently latched onto an object synonym: another object of the goal's main category


@dataclass
class EpisodeTrace:
    """What habitat_od_search/main.py accumulates while running an episode, beyond the agent's own state."""
    first_seen_step: Optional[int] = None  # first step a true goal was visible in a ground-truth frame
    stuck: bool = False
    stop_reason: Optional[str] = None  # the StopReason agent.act() returned alongside its last "stop" action
    world_trajectory: List[Tuple[np.ndarray, float]] = field(default_factory=list)  # (world position, heading) at every act()
    detections: List[Tuple[float, str]] = field(default_factory=list)  # every act()'s best score and its detection_hit


@dataclass(frozen=True)
class EpisodeObjects:
    """An episode's goal and object-synonym ids, every annotated scene object's class, and the top-down footprint
    points (mesh_footprint) of those with a mesh."""
    goal_ids: Set[int]
    synonym_ids: Set[int]
    classes: Dict[int, str]
    footprints: Dict[int, np.ndarray]

    def hit(self, object_id: Optional[int]) -> str:
        """What a detection or belief landed on: "goal", "synonym", "other:<class>", or "background" (no object)."""
        if object_id is None:
            return "background"
        if object_id in self.goal_ids:
            return "goal"
        return "synonym" if object_id in self.synonym_ids else f"other:{self.classes[object_id]}"


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
    n_synonyms: int
    believed_object: str  # EpisodeObjects.hit of believed_object at the stop
    first_seen_step: Optional[int]
    first_confident_step: Optional[int]  # first step whose best detection reached confidence_thresh
    max_goal_score: float  # the highest best-detection score that landed on a goal, 0 when none did
    max_synonym_score: float
    max_other_score: float  # ... on anything else, background included


def object_footprints(meshes: List[Tuple[int, int, trimesh.Trimesh]]) -> Dict[int, np.ndarray]:
    """{object id: mesh_footprint} of env.get_gt_meshes' (object id, class id, mesh) triples."""
    return {object_id: mesh_footprint(mesh) for object_id, _, mesh in meshes}


def detection_hit(objects: EpisodeObjects, instances: Instances, id_image: np.ndarray) -> str:
    """objects.hit of the object (sense.object_id_image) covering most of instances' best detection's mask, "none"
    without any detection."""
    if len(instances) == 0:
        return "none"
    mask = instances.pred_masks[int(instances.scores.argmax())].numpy().astype(bool)
    ids, counts = np.unique(id_image[mask], return_counts=True)
    best = int(ids[counts.argmax()]) if len(ids) else -1
    return objects.hit(None if best < 0 else best)


def distance_to_footprints(position: np.ndarray, footprints: List[np.ndarray]) -> float:
    """Horizontal distance from position to the nearest of footprints' (N, 2) top-down points."""
    return min(float(cKDTree(points).query(position[[0, 2]])[0]) for points in footprints)


def believed_object(agent: SearchAgent, objects: EpisodeObjects) -> Optional[int]:
    """The object goal_mask()'s believed cells land on: of the objects whose footprint comes within
    _BELIEVED_GOAL_MATCH_M of one of them, the nearest goal, else the nearest object synonym, else the nearest one;
    None when none does."""
    assert agent.estimated_map is not None, "Call reset() first"
    believed = agent.estimated_map.believed_goal_positions(agent.config.goal_score_thresh)
    if len(believed) == 0:
        return None
    tree = cKDTree(believed)
    distances = {
        object_id: tree.query(agent.ref_frame_points(np.insert(points, 1, 0.0, axis=1))[:, [0, 2]])[0].min()
        for object_id, points in objects.footprints.items()
    }
    near = {object_id: distance for object_id, distance in distances.items() if distance <= _BELIEVED_GOAL_MATCH_M}
    for group in (objects.goal_ids, objects.synonym_ids, set(near)):
        matches = [object_id for object_id in near if object_id in group]
        if matches:
            return min(matches, key=near.__getitem__)
    return None


def classify_failure(near_goal: bool, believed: str, confident: bool, goal_seen: bool) -> FailureCategory:
    """FailureCategory of a stop near_goal or not, its believed object's EpisodeObjects.hit, and whether the goal map
    ever reached goal_score_thresh (confident) and a goal was ever visible."""
    if near_goal and believed == "goal":
        return FailureCategory.SUCCESS
    if not confident:
        return FailureCategory.FN_DET if goal_seen else FailureCategory.EXPLORATION_FAILURE
    if believed == "goal":
        return FailureCategory.TP_WRONG_STOP
    return FailureCategory.FP_SYNONYM if believed == "synonym" else FailureCategory.FP_DET


def score_maxima(trace: EpisodeTrace) -> Tuple[float, float, float]:
    """The highest best-detection score of trace that landed on a goal, on an object synonym, and on anything else."""
    groups = [(score, hit if hit in ("goal", "synonym") else "other") for score, hit in trace.detections]
    return tuple(max([score for score, group in groups if group == wanted], default=0.0) for wanted in ("goal", "synonym", "other"))


def episode_result(agent: SearchAgent, episode: NavigationEpisode, objects: EpisodeObjects, trace: EpisodeTrace) -> EpisodeResult:
    """EpisodeResult of episode, from the agent's final belief, the trace of its run and the episode's objects."""
    assert agent.estimated_map is not None, "Call reset() first"
    positions = np.array([position for position, _ in trace.world_trajectory])
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    distance_to_goal = distance_to_footprints(positions[-1], [objects.footprints[object_id] for object_id in objects.goal_ids if object_id in objects.footprints])
    believed = objects.hit(believed_object(agent, objects))
    confident = bool(agent.estimated_map.goal_score.max() >= agent.config.goal_score_thresh)
    failure_category = classify_failure(distance_to_goal <= SUCCESS_DIST_M, believed, confident, trace.first_seen_step is not None)
    success = failure_category == FailureCategory.SUCCESS
    geodesic_distance = float(episode.info["geodesic_distance"])
    category = episode.info["object_category"]
    confident_steps = [step for step, (score, _) in enumerate(trace.detections) if score >= agent.config.confidence_thresh]
    max_goal_score, max_synonym_score, max_other_score = score_maxima(trace)
    return EpisodeResult(
        scene_id=episode.scene_id, episode_id=episode.episode_id, object_category=category,
        rarity_tier=rarity_tiers()[main_category(category)], success=success,
        spl=float(success) * geodesic_distance / max(path_length, geodesic_distance), steps=len(positions), stuck=trace.stuck,
        stop_reason=trace.stop_reason, path_length=path_length, geodesic_distance=geodesic_distance,
        distance_to_goal=distance_to_goal, failure_category=failure_category.value, n_synonyms=len(objects.synonym_ids),
        believed_object=believed, first_seen_step=trace.first_seen_step,
        first_confident_step=confident_steps[0] if confident_steps else None,
        max_goal_score=max_goal_score, max_synonym_score=max_synonym_score, max_other_score=max_other_score,
    )


def summary_table(results: List[EpisodeResult]) -> str:
    """Success (with its 95% normal-approximation half-width), SPL, distance-to-goal and episode count per rarity tier,
    on the episodes with and without object synonyms, then overall."""
    groups = [(tier, [result for result in results if result.rarity_tier == tier]) for tier in _RARITY_TIERS] + [
        ("synonyms", [result for result in results if result.n_synonyms > 0]),
        ("no synonyms", [result for result in results if result.n_synonyms == 0]),
        ("overall", results),
    ]
    rows = [
        (name, 100 * np.mean([r.success for r in group]), 196 * np.std([r.success for r in group]) / np.sqrt(len(group)),
         100 * np.mean([r.spl for r in group]), np.mean([r.distance_to_goal for r in group]), len(group))
        for name, group in groups
        if group
    ]
    return tabulate(rows, tablefmt="pipe", floatfmt=".2f", headers=["episodes", "success", "± 95%", "spl", "dtg (m)", "n episodes"], numalign="left")


def failure_table(results: List[EpisodeResult]) -> str:
    """Share of every episode (success included) per FailureCategory, as IMPRINT's failure-case appendix does."""
    rows = [(category.value, 100 * np.mean([r.failure_category == category.value for r in results])) for category in FailureCategory]
    return tabulate(rows, tablefmt="pipe", floatfmt=".2f", headers=["category", "share"], numalign="left")


def stop_reason_table(results: List[EpisodeResult]) -> str:
    """Episode counts per stop reason (rows) and FailureCategory (columns)."""
    frame = pd.DataFrame({"stop reason": [str(r.stop_reason) for r in results], "category": [r.failure_category for r in results]})
    return tabulate(pd.crosstab(frame["stop reason"], frame["category"]), tablefmt="pipe", headers="keys", numalign="left")
