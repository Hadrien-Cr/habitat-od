"""Enhanced-detector baseline for habitat_od_search: the search target's embedding becomes an ensemble of Detic region
embeddings of example views of the target object, instead of the CLIP text embedding of its category. One run handles
one target object in one scene (--scene; --category, by default the scene's easiest_single_category), writing to
habitat_od_search/logs/enhanced_detector/<scene_id>/<category>/:

- GT views: every object but the target's is removed and the navmesh recomputed over the cleared floor; the camera
  is aimed at the target's center from navigable viewpoints ringed around it (_N_AZIMUTHS bearings, _STANDOFF_M beyond
  its footprint, the agent lifted by each of _CAMERA_LIFTS_M, the camera then turned by each of _YAW_OFFSETS_DEG so the
  target also lands off-center, where the wide field of view stretches it as in exploration frames). Up to _N_VIEWS
  views where it passes the ground-truth area/visibility filters, spread over that order, are kept: a crop each in
  gt_crops/, the Detic region embedding of its ground-truth box (Detector.box_embeddings) in gt_features.npz, and
  a gt_views.png mosaic of the crops.
- Collected boxes: the scene is rebuilt with every object, and _N_POSES_PER_ANCHOR random poses are drawn around each of
  _N_ANCHORS anchors: object_anchors (every object of the target's class, or _N_ANCHORS k-means centroids of them when
  there are more), filled up by covering_anchors (farthest-point picks among random navigable points) when fewer --
  a random navigable point within _POSE_RADIUS_M of the anchor, facing it up to +-_YAW_JITTER_DEG off. Every
  class-agnostic Detector.predict box of every pose's rgb is kept: its crop in collected_crops/pose<pose>_box<index>.png,
  its own region embedding in collected_features.npz.
- Web images: the scraping section pastes IMPRINT's web-image scraper (on Bing); save_scraped_views downloads
  _N_SCRAPED images for the full category string and keeps each one's largest Detic box from single-class prediction
  against the main category's text embedding (Detector.predict_target; the whole image without any): its crop in
  scraped_crops/, its region embedding in scraped_features.npz, and scraped_views.png / scraped_frames.png mosaics of
  the crops and of the boxed images.
- Matching: query_matches.png ranks the collected region embeddings, as their crops, for the gt views' mean and max
  ensemble (normalized mean embedding; first tile, the mean crop; and each collected box's highest similarity to any
  of them; first tile, the pixel-wise max crop), every web image, the web images' own mean and max ensemble, and
  Detic's own text embedding of the full category string (Vocabulary.clip_embeddings; first tile blank), _TOP_K per
  row.
  save_threshold_matches saves one query_matches_thresh<threshold>.png per _SIMILARITY_THRESHOLDS value instead: every
  row lists every collected box at or above that threshold (as Detector.predict_target would flag it), ranked by
  similarity and capped at _MAX_COLS - 1 per row.

Every crop is the max(w, h) square centered on its box (shifted inward at the frame's borders, never padded or
stretched), resized to _CROP_SIZE x _CROP_SIZE; crops are for display only, every embedding comes from the box itself.

Usage:
  export HABITAT_DATA=$HOME/habitat_data/data BASE_DIR=$(pwd) PYTHONNOUSERSITE=1
  PYTHONPATH=. python habitat_od_search/enhanced_detector.py --scene 102344094 [--category "table_lamp, TABLE LAMP-PORTER"]
"""
import argparse
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import habitat_sim
import numpy as np
import torch
import trimesh
from habitat.tasks.nav.nav import NavigationEpisode  # type: ignore
from habitat_sim.utils.common import quat_from_angle_axis
from icrawler.builtin import BingImageCrawler
from omegaconf import DictConfig, OmegaConf
from scipy.cluster.vq import kmeans2
from PIL import Image
from tqdm import tqdm

from common.env_utils.env_base import ExplorationEnv, build_env
from common.env_utils.object_annotations import get_all_objects
from common.env_utils.sense import target_instances
from common.utils.diversity_sampling import k_centroid_greedy
from common.utils.plot_utils import centered_crop, letterboxed, make_mosaic
from common.utils.pose_utils import yaw_to_face
from common.vision.detic import Vocabulary
from common.vision.detector import Detector
from habitat_od_search.hssd_rare_dataset import SceneEpisodes, load_scene_episodes, main_category, rarity_tiers
from habitat_od_search.main import goal_meshes

_CONFIG_PATH = Path("habitat_od_search/config/hssd_rare_eval.yaml")
_DETIC_CONFIG = "third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
_DETIC_WEIGHTS = "third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
_N_POSES_PER_ANCHOR = 50
_N_ANCHORS = 3
_N_COVER_CANDIDATES = 500
_POSE_RADIUS_M = 3.0
_YAW_JITTER_DEG = 45.0
_OUT_DIR = Path("habitat_od_search/logs/enhanced_detector")
_N_AZIMUTHS = 8
_STANDOFF_M = 1.5
_CAMERA_LIFTS_M = (-0.5, 0.0, 0.6)
_YAW_OFFSETS_DEG = (-30, 0, 30)
_N_VIEWS = 8
_CROP_SIZE = 512
_FRAME_SIZE = 1024
_TOP_K = 5
_SIMILARITY_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
_MAX_COLS = 32
_N_SCRAPED = 8
_UP = np.array([0.0, 1.0, 0.0])
_RIGHT = np.array([1.0, 0.0, 0.0])
_BOX_COLOR = (255, 0, 255)


@dataclass(frozen=True)
class Target:
    """One run's target object: its scene, fine-grained category, the first episode searching for it, its goal handles."""
    scene_id: str
    category: str
    episode: NavigationEpisode
    goal_handles: Set[str]


def easiest_single_category(scene_episodes: SceneEpisodes) -> str:
    """The first object_category, in episode order, of the "easy" (i.e. common) rarity tier among scene_episodes'
    single-goal-instance ones."""
    singles = [
        episode.info["object_category"] for episode in scene_episodes.episodes
        if len(scene_episodes.goals_by_category[f"{scene_episodes.scene_id}_{episode.info['object_category']}"]) == 1
        and rarity_tiers()[main_category(episode.info["object_category"])] == "easy"
    ]
    return singles[0]


def load_target(scene_id: str, category: Optional[str]) -> Target:
    """scene_id's Target for category, its easiest_single_category when None."""
    scene_episodes = load_scene_episodes(scene_id)
    category = category or easiest_single_category(scene_episodes)
    episode = next(episode for episode in scene_episodes.episodes if episode.info["object_category"] == category)
    return Target(scene_id, category, episode, {goal.object_handle for goal in scene_episodes.goals_by_category[f"{scene_id}_{category}"]})


def remove_distractors(sim: habitat_sim.Simulator, keep_handles: Set[str]) -> None:
    """Removes every rigid and articulated object of sim whose handle isn't in keep_handles."""
    rigid_manager, articulated_manager = sim.get_rigid_object_manager(), sim.get_articulated_object_manager()
    for obj in [obj for obj in get_all_objects(sim) if obj.handle not in keep_handles]:
        manager = rigid_manager if rigid_manager.get_library_has_handle(obj.handle) else articulated_manager
        manager.remove_object_by_handle(obj.handle)


def recompute_navmesh(sim: habitat_sim.Simulator, agent_radius: float, agent_height: float) -> None:
    """Rebuilds sim's navmesh for an agent_radius x agent_height agent over what remains of the scene."""
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_radius, settings.agent_height = agent_radius, agent_height
    settings.include_static_objects = True
    assert sim.recompute_navmesh(sim.pathfinder, settings), "navmesh recomputation failed"


def facing_rotation(camera: np.ndarray, target: np.ndarray, yaw_offset_deg: float) -> np.quaternion:
    """Agent rotation whose camera at camera is yawed toward target, then by yaw_offset_deg, and pitched up/down to it."""
    offset = target - camera
    pitch = np.arctan2(offset[1], np.linalg.norm(offset[[0, 2]]))
    return quat_from_angle_axis(yaw_to_face(camera, target) + np.deg2rad(yaw_offset_deg), _UP) * quat_from_angle_axis(pitch, _RIGHT)


def camera_poses(pathfinder: habitat_sim.PathFinder, mesh: trimesh.Trimesh, floor_y: float, camera_height: float) -> List[Tuple[str, habitat_sim.AgentState]]:
    """(label, agent state) of every pose facing mesh's bounding-box center from a navigable floor point _N_AZIMUTHS
    bearings around it, _STANDOFF_M beyond its top-down half-diagonal, the agent lifted by each of _CAMERA_LIFTS_M and
    the camera turned by each of _YAW_OFFSETS_DEG -- ordered by bearing, then lift, then yaw offset."""
    center = mesh.bounds.mean(axis=0)
    radius = np.linalg.norm((mesh.bounds[1] - mesh.bounds[0])[[0, 2]]) / 2 + _STANDOFF_M
    grounds = [
        np.array([center[0] + radius * np.cos(azimuth), floor_y, center[2] + radius * np.sin(azimuth)])
        for azimuth in np.linspace(0.0, 2 * np.pi, _N_AZIMUTHS, endpoint=False)
    ]
    return [
        (f"h{lift:+.1f} y{yaw_offset:+d}", habitat_sim.AgentState(position=ground + lift * _UP, rotation=facing_rotation(ground + (lift + camera_height) * _UP, center, yaw_offset)))
        for ground in grounds
        if pathfinder.is_navigable(ground)
        for lift in _CAMERA_LIFTS_M
        for yaw_offset in _YAW_OFFSETS_DEG
    ]


def object_crop(rgb: np.ndarray, box: List[int]) -> np.ndarray:
    """The max(w, h) square of rgb centered on the xyxy box (centered_crop shifts it inward at rgb's borders), resized to
    _CROP_SIZE x _CROP_SIZE."""
    x0, y0, x1, y1 = box
    square = centered_crop(rgb, np.array([(y0 + y1) // 2, (x0 + x1) // 2]), min(max(x1 - x0, y1 - y0, 1), *rgb.shape[:2]))
    return cv2.resize(square, (_CROP_SIZE, _CROP_SIZE))


def goal_views(env: ExplorationEnv, goal_id: int, mesh: trimesh.Trimesh, floor_y: float, camera_height: float) -> List[Tuple[str, np.ndarray, List[int]]]:
    """(label, rgb, goal's xyxy box) of every camera_poses view in which goal_id passes the ground-truth filters."""
    views = []
    for label, state in camera_poses(env.get_pathfinder(), mesh, floor_y, camera_height):
        obs, _, _, _ = env.teleport(state)
        instances = target_instances(obs, {goal_id})
        if len(instances):
            views.append((label, np.ascontiguousarray(obs["rgb"][:, :, :3]), instances.pred_boxes.tensor[0].int().tolist()))
    return views


def save_features(path: Path, named_features: List[Tuple[str, np.ndarray]]) -> None:
    """Saves (name, (D,) embedding) pairs to the .npz path as a names array and an (N, D) features array."""
    np.savez(path, names=np.array([name for name, _ in named_features]), features=np.stack([feature for _, feature in named_features]))


def save_gt_views(cfg: DictConfig, target: Target, detector: Detector, scene_dir: Path) -> None:
    """Removes every object but target's from its scene, then saves each goal object's spread goal_views to scene_dir:
    one object_crop each in gt_crops/, its box's detector region embedding in gt_features.npz, and every goal's
    crops' mosaic as gt_views.png."""
    env, habitat_cfg = build_env(target.scene_id, cfg.seed, cfg.agent.max_steps, cfg.object_params, cfg.agent.input_image_size)
    env.reset(episode=target.episode)
    handles = {obj.object_id: obj.handle for obj in get_all_objects(env.get_sim()) if obj.handle in target.goal_handles}
    meshes = goal_meshes(env, set(handles))
    remove_distractors(env.get_sim(), target.goal_handles)
    agent_cfg = habitat_cfg.habitat.simulator.agents.main_agent
    recompute_navmesh(env.get_sim(), agent_cfg.radius, agent_cfg.height)
    floor_y = env.get_agent_state().position[1]
    camera_height = agent_cfg.sim_sensors.rgb_sensor.position[1]
    crops_dir = scene_dir / "gt_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    named_features, crops = [], []
    for goal_id, _, mesh in meshes:
        views = goal_views(env, goal_id, mesh, floor_y, camera_height)
        assert views, f"{handles[goal_id]} is visible from none of its camera poses"
        spread = [views[index] for index in np.linspace(0, len(views) - 1, min(_N_VIEWS, len(views))).astype(int)]
        for index, (label, rgb, box) in enumerate(spread):
            name = f"{handles[goal_id]}_view{index:02d}"
            crops.append((label, object_crop(rgb, box)))
            Image.fromarray(crops[-1][1]).save(crops_dir / f"{name}.png")
            named_features.append((name, detector.box_embeddings(rgb, np.array([box]))[0].numpy()))
    save_features(scene_dir / "gt_features.npz", named_features)
    make_mosaic(crops).save(scene_dir / "gt_views.png")
    env.close()


def save_box_crops(detector: Detector, rgb: np.ndarray, crops_dir: Path, pose: int) -> List[Tuple[str, np.ndarray]]:
    """Saves an object_crop of every class-agnostic detector box of rgb to crops_dir, returning (crop name, the box's
    region embedding) of each."""
    instances = detector.predict(rgb)
    named_features = []
    for index, (box, feature) in enumerate(zip(instances.pred_boxes.tensor.int().tolist(), instances.region_features.numpy())):
        name = f"pose{pose:04d}_box{index:03d}"
        Image.fromarray(object_crop(rgb, box)).save(crops_dir / f"{name}.png")
        named_features.append((name, feature))
    return named_features


def object_anchors(env: ExplorationEnv, target: Target, rng: np.random.Generator) -> np.ndarray:
    """(N, 3) bounding-box centers of every scene object sharing a goal object's annotated class, or _N_ANCHORS k-means
    centroids of them when there are more."""
    annotation = env.get_object_annotation()
    goal_ids = {obj.object_id for obj in get_all_objects(env.get_sim()) if obj.handle in target.goal_handles}
    classes = sorted({annotation.classes[annotation.obj_id_to_class_id[object_id]] for object_id in goal_ids})
    centers = np.array([mesh.bounds.mean(axis=0) for _, _, mesh in env.get_gt_meshes(classes)])
    return centers if len(centers) <= _N_ANCHORS else kmeans2(centers, _N_ANCHORS, seed=rng, minit="++")[0]


def covering_anchors(pathfinder: habitat_sim.PathFinder, anchors: np.ndarray, start: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """anchors, filled up to _N_ANCHORS with the farthest-point picks (horizontal distance, k_centroid_greedy seeded with
    anchors) among _N_COVER_CANDIDATES random navigable points of start's navmesh island."""
    island = pathfinder.get_island(start)
    candidates = np.array([pathfinder.get_random_navigable_point(island_index=island) for _ in range(_N_COVER_CANDIDATES)])
    points = np.concatenate([anchors, candidates])
    distances = np.linalg.norm(points[:, None, [0, 2]] - points[None, :, [0, 2]], axis=-1)
    return points[k_centroid_greedy(distances, _N_ANCHORS, rng, initial=list(range(len(anchors))))]


def random_poses(pathfinder: habitat_sim.PathFinder, anchors: np.ndarray, rng: np.random.Generator) -> List[habitat_sim.AgentState]:
    """_N_POSES_PER_ANCHOR agent states per anchor, each at a random navigable point within _POSE_RADIUS_M of it, facing
    it up to _YAW_JITTER_DEG off -- fewer when habitat-sim finds no such point for some."""
    positions = [(np.array(pathfinder.get_random_navigable_point_near(anchor, _POSE_RADIUS_M)), anchor) for anchor in anchors for _ in range(_N_POSES_PER_ANCHOR)]
    return [
        habitat_sim.AgentState(position=position, rotation=quat_from_angle_axis(yaw_to_face(position, anchor) + np.deg2rad(rng.uniform(-_YAW_JITTER_DEG, _YAW_JITTER_DEG)), _UP))
        for position, anchor in positions
        if not np.isnan(position).any()
    ]


def save_collected_crops(cfg: DictConfig, target: Target, detector: Detector, scene_dir: Path) -> None:
    """Saves save_box_crops of every random_poses view around target's scene's object_anchors, filled up by
    covering_anchors, to scene_dir/collected_crops/ and their region embeddings to scene_dir/collected_features.npz."""
    crops_dir = scene_dir / "collected_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    env, _ = build_env(target.scene_id, cfg.seed, cfg.agent.max_steps, cfg.object_params, cfg.agent.input_image_size)
    env.reset(episode=target.episode)
    pathfinder = env.get_pathfinder()
    pathfinder.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    anchors = covering_anchors(pathfinder, object_anchors(env, target, rng), np.array(env.get_agent_state().position), rng)
    poses = random_poses(pathfinder, anchors, rng)
    named_features = []
    for pose, state in enumerate(tqdm(poses, desc="collecting boxes")):
        obs, _, _, _ = env.teleport(state)
        named_features += save_box_crops(detector, np.ascontiguousarray(obs["rgb"][:, :, :3]), crops_dir, pose)
    save_features(scene_dir / "collected_features.npz", named_features)
    env.close()


def load_named_features(features_path: Path, crops_dir: Path) -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray]:
    """([(name, its crops_dir crop resized to _CROP_SIZE x _CROP_SIZE)], (N, D) unit-norm features) of features_path's .npz."""
    saved = np.load(features_path)
    crops = [(str(name), cv2.resize(np.array(Image.open(crops_dir / f"{name}.png").convert("RGB")), (_CROP_SIZE, _CROP_SIZE))) for name in saved["names"]]
    return crops, saved["features"] / np.linalg.norm(saved["features"], axis=1, keepdims=True)


def ensemble_rows(name: str, queries: List[Tuple[str, np.ndarray]], query_embeddings: np.ndarray, collected_embeddings: np.ndarray) -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray]:
    """("<name> mean_ensemble" and "<name> max_ensemble" first tiles -- queries' mean and pixel-wise max crop -- and their
    (2, M) similarities to collected_embeddings: the normalized mean query embedding's, and each one's max over queries)."""
    crops = np.stack([crop for _, crop in queries])
    mean_embedding = query_embeddings.mean(axis=0)
    tiles = [(f"{name} mean_ensemble", crops.mean(axis=0).astype(np.uint8)), (f"{name} max_ensemble", crops.max(axis=0))]
    return tiles, np.vstack([mean_embedding / np.linalg.norm(mean_embedding) @ collected_embeddings.T, (query_embeddings @ collected_embeddings.T).max(axis=0)])


def query_rows(scene_dir: Path, category: str) -> Tuple[List[Tuple[str, np.ndarray]], np.ndarray, List[Tuple[str, np.ndarray]]]:
    """Every query (gt views' mean/max ensembles, scraped web crops and their mean/max ensembles, category's text) as
    (name, crop) tiles, their (Q, M) similarities to collected_features.npz's region embeddings, and those embeddings'
    own (name, crop) tiles."""
    gt, gt_embeddings = load_named_features(scene_dir / "gt_features.npz", scene_dir / "gt_crops")
    web_scraped, web_scraped_embeddings = load_named_features(scene_dir / "scraped_features.npz", scene_dir / "scraped_crops")
    collected, collected_embeddings = load_named_features(scene_dir / "collected_features.npz", scene_dir / "collected_crops")

    text_embedding = Vocabulary(f"enhanced_{category}", (category,)).clip_embeddings()[:, 0].numpy()
    gt_tiles, gt_similarities = ensemble_rows("gt", gt, gt_embeddings, collected_embeddings)
    web_tiles, web_similarities = ensemble_rows("web", web_scraped, web_scraped_embeddings, collected_embeddings)
    queries = gt_tiles + [(f"web {name}", crop) for name, crop in web_scraped] + web_tiles
    queries.append((f"text query: {category}", np.zeros((_CROP_SIZE, _CROP_SIZE, 3), dtype=np.uint8)))

    similarities = np.vstack([
        gt_similarities,
        web_scraped_embeddings @ collected_embeddings.T,
        web_similarities,
        text_embedding / np.linalg.norm(text_embedding) @ collected_embeddings.T,
    ])
    return queries, similarities, collected


def save_query_matches(scene_dir: Path, category: str) -> None:
    """Saves scene_dir/query_matches.png: every query tile next to the crops of its _TOP_K most similar collected
    region embeddings."""
    queries, similarities, collected = query_rows(scene_dir, category)
    top = np.argsort(-similarities, axis=1)[:, :_TOP_K]
    tiles = [
        tile
        for (query_name, query_crop), row, indices in zip(queries, similarities, top)
        for tile in [(query_name, query_crop)] + [(f"{collected[index][0]} sim {row[index]:.3f}", collected[index][1]) for index in indices]
    ]
    make_mosaic(tiles, N_cols=1 + _TOP_K).save(scene_dir / "query_matches.png")


def save_threshold_matches(scene_dir: Path, category: str) -> None:
    """For each of _SIMILARITY_THRESHOLDS, saves scene_dir/query_matches_thresh<threshold>.png: every query tile next
    to the crops of every collected region embedding whose similarity to it is >= threshold (as predict_target would
    flag it), ranked by similarity and capped at _MAX_COLS - 1 per row."""
    queries, similarities, collected = query_rows(scene_dir, category)
    row_cap = _MAX_COLS - 1
    for threshold in _SIMILARITY_THRESHOLDS:
        row_sizes, tiles = [], []
        for (query_name, query_crop), row in zip(queries, similarities):
            indices = np.argsort(-row)[:row_cap]
            indices = indices[row[indices] >= threshold]
            row_sizes.append(1 + len(indices))
            tiles.append((query_name, query_crop))
            tiles += [(f"{collected[index][0]} sim {row[index]:.3f}", collected[index][1]) for index in indices]
        make_mosaic(tiles, N_cols=_MAX_COLS, row_sizes=row_sizes).save(scene_dir / f"query_matches_thresh{threshold:.1f}.png")


# Pasted from third_party/IMPRINT/static_baselines/utils/scrape_img.py (save_dir default dropped, icrawler imported above,
# Bing instead of Google: icrawler's Google Images parser no longer parses Google's result pages).
def run_crawler(query, num_images, save_folder, file_idx_offset):
    crawler = BingImageCrawler(storage={'root_dir': save_folder})
    crawler.crawl(keyword=query, max_num=num_images, file_idx_offset=file_idx_offset)

def get_unique_count_and_remove_duplicates(save_folder):
    unique_hashes = {}
    files = [f for f in os.listdir(save_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for f in files:
        path = os.path.join(save_folder, f)
        try:
            with open(path, 'rb') as file:
                file_hash = hashlib.md5(file.read()).hexdigest()
            if file_hash in unique_hashes:
                os.remove(path)
            else:
                unique_hashes[file_hash] = path
        except Exception as e:
            print(f"Error processing file {f}: {e}")
    return len(unique_hashes)

def download_google_images(query, num_images, save_dir, extra_tries = 1):
    current_path = os.getcwd()
    query_dir = os.path.join(current_path, save_dir, query)
    os.makedirs(query_dir, exist_ok=True)

    unique_count = get_unique_count_and_remove_duplicates(query_dir)
    file_idx_offset = unique_count
    pbar = tqdm(total=num_images, desc=f"Downloading {query}", unit="img")
    pbar.update(unique_count)

    n_tries = 0
    while unique_count < num_images:
        remaining = num_images - unique_count
        run_crawler(query, remaining, query_dir, file_idx_offset)
        time.sleep(2)
        new_unique_count = get_unique_count_and_remove_duplicates(query_dir)
        pbar.update(new_unique_count - unique_count)

        if new_unique_count == unique_count:
            n_tries += 1
            if n_tries >= extra_tries:
                print(f"No image could be found! Aborting...")
                return

        unique_count = new_unique_count
        file_idx_offset = unique_count

    pbar.close()
    print(f"Download complete for query: {query}! Saved at: {query_dir}")


def largest_target_box(detector: Detector, rgb: np.ndarray, target_embedding: torch.Tensor) -> Tuple[List[int], np.ndarray]:
    """(xyxy box, region embedding) of the largest of detector.predict_target's boxes on rgb for target_embedding, the
    whole image when it finds none."""
    instances, _ = detector.predict_target(rgb, target_embedding)
    box = instances.pred_boxes.tensor[int(instances.pred_boxes.area().argmax())].int().tolist() if len(instances) else [0, 0, rgb.shape[1], rgb.shape[0]]
    return box, detector.box_embeddings(rgb, np.array([box]))[0].numpy()


def save_scraped_views(scene_dir: Path, detector: Detector, category: str) -> None:
    """Downloads up to _N_SCRAPED Bing Images results for the full category string to scene_dir/scraped_imgs/<category>/
    (download_google_images), then saves each image's largest_target_box for category's main category text embedding as
    an object_crop in scraped_crops/ and its region embedding in scraped_features.npz, and mosaics of every crop
    (scraped_views.png) and of every image with its box drawn, fitted into a _FRAME_SIZE square (scraped_frames.png)."""
    download_google_images(category, _N_SCRAPED, str(scene_dir / "scraped_imgs"))
    coarse = main_category(category)
    target_embedding = Vocabulary(f"search_{coarse}", (coarse,)).clip_embeddings()[:, 0]
    crops_dir = scene_dir / "scraped_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    named_features, crops, frames = [], [], []
    for path in sorted(path for path in (scene_dir / "scraped_imgs" / category).iterdir() if path.suffix.lower() in (".jpg", ".jpeg", ".png")):
        rgb = np.array(Image.open(path).convert("RGB"))
        box, feature = largest_target_box(detector, rgb, target_embedding)
        crops.append((path.stem, object_crop(rgb, box)))
        Image.fromarray(crops[-1][1]).save(crops_dir / f"{path.stem}.png")
        named_features.append((path.stem, feature))
        boxed = cv2.rectangle(rgb.copy(), tuple(box[:2]), tuple(box[2:]), _BOX_COLOR, max(rgb.shape[:2]) // 100)
        frames.append((path.stem, letterboxed(boxed, _FRAME_SIZE, _FRAME_SIZE)))
    save_features(scene_dir / "scraped_features.npz", named_features)
    make_mosaic(crops).save(scene_dir / "scraped_views.png")
    make_mosaic(frames).save(scene_dir / "scraped_frames.png")


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(_CONFIG_PATH)
    target = load_target(args.scene, args.category)
    scene_dir = _OUT_DIR / target.scene_id / target.category
    detector = Detector(_DETIC_CONFIG, _DETIC_WEIGHTS, Vocabulary("enhanced_unused", ("object",)), None, input_size=cfg.agent.input_image_size)
    save_gt_views(cfg, target, detector, scene_dir)
    save_collected_crops(cfg, target, detector, scene_dir)
    save_scraped_views(scene_dir, detector, target.category)
    save_query_matches(scene_dir, target.category)
    save_threshold_matches(scene_dir, target.category)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="HSSD-rare scene id.")
    parser.add_argument("--category", default=None, help="Fine-grained object_category to target; the scene's easiest_single_category by default.")
    main(parser.parse_args())
