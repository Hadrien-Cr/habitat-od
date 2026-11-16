"""Evaluates habitat_od_search's SearchAgent on IMPRINT's HSSD-rare ObjectNav episodes (--config, see
habitat_od_search/config/hssd_rare_eval.yaml). For each scene in `scenes` (every saved hssd_rare_eps scene when
empty), one ExplorationEnv is built; each of its episodes whose start navmesh island covers at least
min_island_area_m2 (smaller ones, e.g. a balcony cut off from the house, are skipped) and at most max_island_area_m2
(larger ones are usually a navmesh merging areas that shouldn't be connected), and whose episode_id is in
cfg.episode_ids when given (every episode otherwise), the first max_episodes_per_category of each object category, is
numbered across scenes; skipping the first first_episode (0 by default, e.g. to tune thresholds on episodes outside an
evaluation's own), at most max_episodes of them (every one when null) reset the env to the episode's own start pose and
run a fresh SearchAgent, targeting agent.query (habitat_od_search/query_bank.py: the CLIP text embedding of the
episode's main category or whole object_category, a bank of region embeddings of example views of it, or both), until
it stops (also once stuck, BaseExplorationAgent.is_stuck) or runs out of agent.max_steps. Each episode's success, SPL,
distance-to-goal, failure category and detection diagnostics (habitat_od_search/evaluator.py) is appended to
habitat_od_search/logs/<run_name>/metrics.jsonl as soon as it ends, and every step's best detection score and what it
landed on (goal, object synonym, other object or background, by the ground-truth object under its mask) to
detections.jsonl; the summary (per rarity tier and with/without object synonyms), failure breakdown and stop-reason
tables are printed and saved to summary.md there. With agent.use_gt_detections, the agent uses the goal objects'
ground-truth detections (target_instances, scored 1.0) instead of Detic's.
Success is a stop within evaluator.SUCCESS_DIST_M of a goal object's mesh footprint, believing in a real goal. With save_visuals, each step's 3-tile (detections,
belief map cropped to the 256 x 256 cells around the agent, ground truth with the episode's goal object footprints) snapshot is saved to images/<run_index>_ep<episode_id>_<scene_id>/ (run_index counting every episode run so far,
across scenes) and assembled into episode.webp, its last one also copied next to that folder as
images/<run_index>_ep<episode_id>_<scene_id>_last_snapshot.png. With save_frames, each step's rgb and ground-truth
object-id image are saved to frames/<run_index>_ep<episode_id>_<scene_id>/ with an episode.json, for
tune_threshold.py. With save_mesh, each scene's ground-truth mesh (stage + every placed object, ceiling stripped) is
written once to meshes/<scene_id>.glb.

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
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

import cv2
import numpy as np
import trimesh
from habitat.tasks.nav.nav import NavigationEpisode, TopDownMap  # type: ignore
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from common.env_utils.env_base import ExplorationEnv, build_env, sensor_geometry
from common.env_utils.object_annotations import get_all_objects, scene_world_mesh
from common.env_utils.sense import object_id_image, target_instances, visible_object_ids
from common.hybrid_exploration.exploration_map import MapConfig
from common.hybrid_exploration.frontier_exp_agent import DiscretePlannerConfig, exploration_snapshot, obs_dilation_cells
from common.hybrid_exploration.ground_truth_map import GroundTruthMapTracker
from common.utils.plot_utils import save_gif
from common.vision.detic import Vocabulary
from common.vision.detector import Detector
from habitat_od_search.evaluator import (
    EpisodeObjects, EpisodeResult, EpisodeTrace, detection_hit, episode_result, failure_table, object_footprints, stop_reason_table,
    summary_table,
)
from habitat_od_search.hssd_rare_dataset import (
    SceneEpisodes, SceneGoal, available_scenes, load_scene_episodes, main_category, template_main_categories,
)
from habitat_od_search.query_bank import target_query
from habitat_od_search.search_agent import SearchAgent, SearchAgentConfig

RUNS_ROOT = Path("habitat_od_search") / "logs"
_GIF_FPS = 10
_JPEG_QUALITY = 95


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
        query=agent_cfg.get("query", "text_cat"),
    )


@dataclass(frozen=True)
class EpisodeOutputs:
    """run_episode's optional per-episode output directories."""
    images_dir: Optional[Path]  # save_visuals' per-step snapshots and episode.webp
    frames_dir: Optional[Path]  # save_frames' per-step rgb and object-id images, for tune_threshold.py


def save_snapshot(obs: dict, agent: SearchAgent, ground_truth: GroundTruthMapTracker, trace: EpisodeTrace, images_dir: Path) -> None:
    """Reveals the agent's current view on ground_truth, then saves the step's 3-tile exploration_snapshot -- the
    ground-truth tile from the agent's world poses, its belief map in the agent's own start frame."""
    position, heading = trace.world_trajectory[-1]
    ground_truth.reveal(position, heading)
    trajectory_px = [ground_truth.to_pixel(visited) for visited, _ in trace.world_trajectory]
    mosaic = exploration_snapshot(obs, agent, ground_truth, trajectory_px)
    if mosaic is not None:
        mosaic.save(images_dir / f"snapshot_gif_{len(trace.world_trajectory) - 1:04d}.png")


def save_frame(frames_dir: Path, step: int, obs: dict, id_image: np.ndarray) -> None:
    """Saves step's rgb as rgb_<step>.jpg and its object_id_image as ids_<step>.png (16-bit, shifted by 1 so 0 is no object)."""
    assert id_image.max() < np.iinfo(np.uint16).max, "object ids overflow the 16-bit id image"
    cv2.imwrite(str(frames_dir / f"rgb_{step:04d}.jpg"), obs["rgb"][:, :, 2::-1], [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    cv2.imwrite(str(frames_dir / f"ids_{step:04d}.png"), (id_image + 1).astype(np.uint16))


def goal_meshes(env: ExplorationEnv, goal_ids: Set[int]) -> List[Tuple[int, int, trimesh.Trimesh]]:
    """env.get_gt_meshes of the goal objects alone, for GroundTruthMapTracker.set_object_footprints."""
    annotation = env.get_object_annotation()
    goal_classes = [annotation.classes[annotation.obj_id_to_class_id[object_id]] for object_id in goal_ids]
    return [mesh for mesh in env.get_gt_meshes(goal_classes) if mesh[0] in goal_ids]


def synonym_objects(env: ExplorationEnv, category: str, goal_handles: Set[str]) -> list:
    """category's object synonyms: the scene's non-goal objects whose objects.csv main category is category's own."""
    main_categories = template_main_categories()
    return [
        obj for obj in get_all_objects(env.get_sim())
        if obj.handle not in goal_handles and main_categories.get(obj.handle.split("_:")[0]) == main_category(category)
    ]


def episode_objects(env: ExplorationEnv, category: str, goals: List[SceneGoal]) -> EpisodeObjects:
    """The reset env's goal objects (matched by handle), category's object synonyms, and every annotated object's class and footprint."""
    goal_handles = {goal.object_handle for goal in goals}
    annotation = env.get_object_annotation()
    return EpisodeObjects(
        goal_ids={obj.object_id for obj in get_all_objects(env.get_sim()) if obj.handle in goal_handles},
        synonym_ids={obj.object_id for obj in synonym_objects(env, category, goal_handles)},
        classes={object_id: annotation.classes[class_id] for object_id, class_id in annotation.obj_id_to_class_id.items()},
        footprints=object_footprints(env.get_gt_meshes(annotation.classes)),
    )


def start_episode(env: ExplorationEnv, agent: SearchAgent, episode: NavigationEpisode, goals: List[SceneGoal]) -> Tuple[dict, EpisodeObjects]:
    """Resets env to episode and agent to its start pose and query, returning the first observation and episode_objects."""
    obs = env.reset(episode=episode)
    start_state = env.get_agent_state()
    category = episode.info["object_category"]
    agent.reset(
        start_state, start_state.position[1] - env.navmesh_floor_offset(), target_query(agent.config.query, episode.scene_id, category),
        category, env.get_pathfinder().get_bounds(),
    )
    return obs, episode_objects(env, category, goals)


def prepare_outputs(env: ExplorationEnv, episode: NavigationEpisode, objects: EpisodeObjects, outputs: EpisodeOutputs, ground_truth: GroundTruthMapTracker) -> None:
    """Creates outputs' directories, drawing the goal footprints on ground_truth for snapshots and describing the episode
    in frames_dir/episode.json for tune_threshold.py."""
    if outputs.images_dir is not None:
        outputs.images_dir.mkdir(parents=True, exist_ok=True)
        ground_truth.set_object_footprints(goal_meshes(env, objects.goal_ids))
    if outputs.frames_dir is not None:
        outputs.frames_dir.mkdir(parents=True, exist_ok=True)
        (outputs.frames_dir / "episode.json").write_text(json.dumps({
            "scene_id": episode.scene_id, "episode_id": episode.episode_id, "object_category": episode.info["object_category"],
            "goal_ids": sorted(objects.goal_ids), "synonym_ids": sorted(objects.synonym_ids), "classes": objects.classes,
        }))


def run_episode(env: ExplorationEnv, agent: SearchAgent, episode: NavigationEpisode, goals: List[SceneGoal], outputs: EpisodeOutputs) -> Tuple[EpisodeResult, EpisodeTrace]:
    """Runs agent on episode until it stops or gets stuck, recording every step's best detection and what it landed on."""
    obs, objects = start_episode(env, agent, episode, goals)
    ground_truth = env.gt_map_tracker(agent.map_config.fov_deg, agent.map_config.visibility_dist_m)
    prepare_outputs(env, episode, objects, outputs, ground_truth)
    trace = EpisodeTrace()
    for step in range(agent.config.max_steps):
        state = env.get_agent_state()
        trace.world_trajectory.append((np.array(state.position), float(TopDownMap.get_polar_angle(state))))
        action, stop_reason = agent.act(state, obs["rgb"], obs["depth"], gt_instances=target_instances(obs, objects.goal_ids))
        id_image = object_id_image(obs)
        best_score = float(agent.last_instances.scores.max()) if len(agent.last_instances) else 0.0
        trace.detections.append((best_score, detection_hit(objects, agent.last_instances, id_image)))
        if trace.first_seen_step is None and objects.goal_ids & visible_object_ids(obs):
            trace.first_seen_step = step
        if outputs.images_dir is not None:
            save_snapshot(obs, agent, ground_truth, trace, outputs.images_dir)
        if outputs.frames_dir is not None:
            save_frame(outputs.frames_dir, step, obs, id_image)
        if action == "stop":
            trace.stop_reason = stop_reason.value if stop_reason else None
            break
        obs, _, _, _ = env.step(action)
    trace.stuck = agent.is_stuck()
    if outputs.images_dir is not None:
        save_gif(outputs.images_dir, "episode.webp", fps=_GIF_FPS)
        shutil.copyfile(max(outputs.images_dir.glob("snapshot_gif_*.png")), outputs.images_dir.parent / f"{outputs.images_dir.name}_last_snapshot.png")
    return episode_result(agent, episode, objects, trace), trace


def episode_rng(cfg_seed: int, scene_id: str, episode_id: str) -> np.random.Generator:
    """Generator seeded from cfg_seed, scene_id and episode_id alone -- so an episode's own randomness (BaseExplorationAgent.spin's
    initial turn direction) is the same whether it runs as part of a full scene or rerun by itself, unlike a single
    rng shared and advanced across a scene's episodes in order."""
    return np.random.default_rng(zlib.crc32(f"{cfg_seed}_{scene_id}_{episode_id}".encode()))


def episode_island_area(pathfinder, episode: NavigationEpisode) -> float:
    return pathfinder.island_area(pathfinder.get_island(np.array(episode.start_position)))


def capped_per_category(episodes: List[NavigationEpisode], max_per_category: int) -> List[NavigationEpisode]:
    """The first max_per_category of episodes of each object_category, in episodes' order."""
    counts: Counter = Counter()
    kept = []
    for episode in episodes:
        counts[episode.info["object_category"]] += 1
        if counts[episode.info["object_category"]] <= max_per_category:
            kept.append(episode)
    return kept


def selected_episodes(cfg: DictConfig, scene_episodes: SceneEpisodes, pathfinder) -> List[NavigationEpisode]:
    """scene_episodes' episodes on a start island within [min_island_area_m2, max_island_area_m2], in cfg.episode_ids when
    given, capped_per_category."""
    return capped_per_category([
        episode for episode in scene_episodes.episodes
        if cfg.min_island_area_m2 <= episode_island_area(pathfinder, episode) <= cfg.max_island_area_m2
        and episode.episode_id in cfg.get("episode_ids", [episode.episode_id])
    ], cfg.max_episodes_per_category)


def evaluate_scene(scene_id: str, cfg: DictConfig, detector: Detector, run_dir: Path, run_indices: Iterator[int]) -> Iterator[Tuple[EpisodeResult, EpisodeTrace]]:
    """run_episode of every selected HSSD-rare episode of scene_id, yielded as each one ends; run_indices numbers them
    across scenes, and those numbered below cfg.first_episode are skipped."""
    scene_episodes = load_scene_episodes(scene_id)
    env, habitat_cfg = build_env(scene_id, cfg.seed, cfg.agent.max_steps, cfg.object_params, cfg.agent.input_image_size)
    if cfg.save_mesh:
        (run_dir / "meshes").mkdir(parents=True, exist_ok=True)
        scene_world_mesh(env.get_sim()).export(str(run_dir / "meshes" / f"{scene_id}.glb"))
    fov_deg, camera = sensor_geometry(habitat_cfg)
    config = search_agent_config(cfg.agent, fov_deg, habitat_cfg.habitat.simulator.turn_angle, habitat_cfg.habitat.simulator.agents.main_agent.radius)
    try:
        for episode in tqdm(selected_episodes(cfg, scene_episodes, env.get_pathfinder()), desc=f"search {scene_id}", ncols=80):
            run_index = next(run_indices)
            if run_index < cfg.get("first_episode", 0):
                continue
            category = episode.info["object_category"]
            run_name = f"{run_index:03d}_ep{episode.episode_id}_{scene_id}"
            rng = episode_rng(cfg.seed, scene_id, episode.episode_id)
            agent = SearchAgent(detector, camera, config, rng, str(run_dir), run_name, main_category(category))
            outputs = EpisodeOutputs(
                run_dir / "images" / run_name if cfg.save_visuals else None,
                run_dir / "frames" / run_name if cfg.get("save_frames", False) else None,
            )
            yield run_episode(env, agent, episode, scene_episodes.goals_by_category[f"{scene_id}_{category}"], outputs)
    finally:
        env.close()


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    detector = Detector(
        args.config_file, args.checkpoint, Vocabulary("search_unused", ("object",)), args.sam_checkpoint,
        score_thresh=args.score_thresh, nms_thresh=args.nms_thresh, mask_thresh=args.mask_thresh, device=args.device, input_size=cfg.agent.input_image_size,
    )
    run_dir = RUNS_ROOT / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_indices = itertools.count()
    episode_runs = itertools.islice(
        (run for scene_id in cfg.scenes or available_scenes() for run in evaluate_scene(scene_id, cfg, detector, run_dir, run_indices)),
        cfg.max_episodes,
    )
    results: List[EpisodeResult] = []
    with open(run_dir / "metrics.jsonl", "w") as metrics_file, open(run_dir / "detections.jsonl", "w") as detections_file:
        for result, trace in episode_runs:
            results.append(result)
            metrics_file.write(json.dumps(asdict(result)) + "\n")
            metrics_file.flush()
            scores, hits = zip(*trace.detections)
            detections_file.write(json.dumps({"scene_id": result.scene_id, "episode_id": result.episode_id, "scores": scores, "hits": hits}) + "\n")
            detections_file.flush()
    report = f"{summary_table(results)}\n\n{failure_table(results)}\n\n{stop_reason_table(results)}\n"
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
