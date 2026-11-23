"""Patch-feature object retrieval of one HSSD-rare scene's first episode's object category. A FrontierExplorationAgent
explores the scene from that episode's start. Each frame's SigLIP patch embeddings (common/vision/siglip.py) are placed
on a ground-truth top-down map at _RESOLUTION_M by back-projecting a _DEPTH_SAMPLES x _DEPTH_SAMPLES grid of its depth
pixels. Only points between _MIN_HEIGHT_M and _MAX_HEIGHT_M above the floor are written. Each point carries the
embedding of the patch its image pixel lies in or, with --mask-pooling, of its ground-truth object's segment: the mean
of the patch embeddings the segment overlaps, weighted by overlap (pixels of no object form one segment). A map pixel's
embedding is the mean of its points'. The map is cached as feature_map_<max_steps>steps.npz, and every frame that
placed a point is saved to frames_<max_steps>steps/. Outputs, under
habitat_od_search/logs/static_retrieval/<scene>/{patches,mask_pooling}/: retrieval.png (the ground-truth map, goals in
blue, beside the map pixels' cosine similarity to the category's SigLIP text embedding), feature_map.png (the map
pixels' embeddings' top 3 principal components as RGB) and, next to each frame, similarity_<step>.jpg (the frame
half-blended with its embeddings' similarity to the query, on one color scale across frames).

Usage:
  export BASE_DIR=$(pwd) PYTHONNOUSERSITE=1  # HABITAT_DATA already exported, see INSTALL.MD
  PYTHONPATH=. python habitat_od_search/static_retrieval.py --scene 102344094 [--mask-pooling]
"""
import argparse
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import habitat_sim
import matplotlib
import numpy as np
from habitat.tasks.nav.nav import TopDownMap  # type: ignore
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

from common.env_utils.env_base import ExplorationEnv, build_agent, build_env
from common.env_utils.object_annotations import get_all_objects
from common.env_utils.sense import object_id_image
from common.hybrid_exploration.frontier_exp_agent import FrontierExplorationAgent
from common.hybrid_exploration.ground_truth_map import GroundTruthMapTracker
from common.utils.plot_utils import letterboxed, make_mosaic
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import DepthCamera, object_points
from common.vision.siglip import SiglipEncoder
from habitat_od_search.hssd_rare_dataset import SceneEpisodes, available_scenes, load_scene_episodes
from habitat_od_search.main import goal_meshes

_CONFIG_PATH = Path("habitat_od_search/config/hssd_rare_eval.yaml")
_OUT_DIR = Path("habitat_od_search/logs/static_retrieval")
_RESOLUTION_M = 0.05
_DEPTH_SAMPLES = 128
_MIN_HEIGHT_M = 0.2
_MAX_HEIGHT_M = 2.0
_JPEG_QUALITY = 95
_TILE_PX = 768
_PIXEL_ALPHA = 0.8
_FRAME_OVERLAY_ALPHA = 0.5
_LABEL_WIDTH = 32
_NAVIGABLE_GRAY = 200
_GOAL_COLOR = (0, 120, 255)  # GroundTruthMapTracker's BGR footprint color, as RGB
_ARGMAX_RADIUS_PX = 10


@dataclass
class FeatureMap:
    features: np.ndarray  # (R, D) float16 SigLIP patch or segment embeddings of every recorded frame
    frame_steps: np.ndarray  # (F,) exploration step of each recorded frame
    frame_rows: np.ndarray  # (F, _DEPTH_SAMPLES ** 2) features row of each frame's depth samples
    point_pixels: np.ndarray  # (M,) flat ground-truth map pixel of each back-projected depth point
    point_rows: np.ndarray  # (M,) features row of each point
    ground_truth: np.ndarray  # (H, W, 3) RGB ground-truth map: navmesh, revealed area, trajectory, goals in blue
    navigable: np.ndarray  # (H, W) bool
    goal_footprints: np.ndarray  # (H, W) bool

    def pixel_means(self, row_values: np.ndarray) -> np.ndarray:
        """(H, W, C) mean of the (R, C) row_values over the points on each map pixel, NaN where none fell."""
        size = self.navigable.size
        counts = np.bincount(self.point_pixels, minlength=size)
        point_values = row_values[self.point_rows]
        sums = np.stack([np.bincount(self.point_pixels, weights=channel, minlength=size) for channel in point_values.T], axis=-1)
        with np.errstate(invalid="ignore"):
            return (sums / counts[:, None]).reshape(*self.navigable.shape, -1)


@dataclass
class Recording:
    frames_dir: Path
    n_rows: int = 0
    steps: List[int] = field(default_factory=list)
    features: List[np.ndarray] = field(default_factory=list)
    frame_rows: List[np.ndarray] = field(default_factory=list)
    pixels: List[np.ndarray] = field(default_factory=list)
    point_rows: List[np.ndarray] = field(default_factory=list)

    def add(self, step: int, features: np.ndarray, sample_rows: np.ndarray, points: Tuple[np.ndarray, np.ndarray]) -> None:
        """Appends a frame's (R, D) features, its depth samples' rows into them and its (map pixels, sample indices)
        points, rows offset past earlier frames' features."""
        pixels, samples = points
        rows = sample_rows + self.n_rows
        self.n_rows += len(features)
        self.steps.append(step)
        self.features.append(features.astype(np.float16))
        self.frame_rows.append(rows)
        self.pixels.append(pixels)
        self.point_rows.append(rows[samples])


def query_category(scene_episodes: SceneEpisodes) -> str:
    return scene_episodes.episodes[0].info["object_category"]


def load_map(path: Path) -> FeatureMap:
    saved = np.load(path)
    return FeatureMap(**{name: saved[name] for name in FeatureMap.__dataclass_fields__})


def sampled(image: np.ndarray) -> np.ndarray:
    """image's (_DEPTH_SAMPLES, _DEPTH_SAMPLES) nearest-neighbor samples."""
    return image[::image.shape[0] // _DEPTH_SAMPLES, ::image.shape[1] // _DEPTH_SAMPLES]


def sample_patches(grid: int) -> np.ndarray:
    """(_DEPTH_SAMPLES ** 2,) row-major index, on a grid x grid patch layout, of the patch each depth sample lies in."""
    rows, cols = np.divmod(np.arange(_DEPTH_SAMPLES ** 2), _DEPTH_SAMPLES)
    return (rows * grid // _DEPTH_SAMPLES) * grid + cols * grid // _DEPTH_SAMPLES


def mask_pooled(patch_features: np.ndarray, patches: np.ndarray, segments: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """((K, D) mean of patch_features over each of the K segments' samples' patches, (_DEPTH_SAMPLES ** 2,) segment
    row of each sample), for the samples' patches and segment ids."""
    _, sample_rows = np.unique(segments, return_inverse=True)
    overlap = np.zeros((sample_rows.max() + 1, len(patch_features)))
    np.add.at(overlap, (sample_rows, patches), 1)
    return overlap @ patch_features / overlap.sum(axis=1, keepdims=True), sample_rows


def start_exploration(cfg: DictConfig, scene_episodes: SceneEpisodes, max_steps: int, out_dir: Path) -> Tuple[ExplorationEnv, FrontierExplorationAgent, dict]:
    """(env, fbe agent, first observation), reset to the scene's first episode's start."""
    env, habitat_cfg = build_env(scene_episodes.scene_id, cfg.seed, max_steps, cfg.object_params, cfg.agent.input_image_size)
    agent = build_agent("fbe", np.random.default_rng(cfg.seed), env, habitat_cfg, cfg.agent.visibility_dist_m, str(out_dir), "planner", _RESOLUTION_M)
    obs = env.reset(episode=scene_episodes.episodes[0])
    start = env.get_agent_state()
    agent.reset(start, start.position[1] - env.navmesh_floor_offset(), env.get_pathfinder().get_bounds())
    return env, agent, obs


def ground_truth_map(env: ExplorationEnv, agent: FrontierExplorationAgent, goal_handles: Set[str]) -> GroundTruthMapTracker:
    """env's GroundTruthMapTracker at _RESOLUTION_M per pixel, its footprints the goal objects'."""
    lower, upper = (np.array(bound) for bound in env.get_pathfinder().get_bounds())
    resolution = int(np.abs(upper - lower)[[0, 2]].min() / _RESOLUTION_M)
    ground_truth = env.gt_map_tracker(agent.map_config.fov_deg, agent.map_config.visibility_dist_m, resolution)
    goal_ids = {obj.object_id for obj in get_all_objects(env.get_sim()) if obj.handle in goal_handles}
    ground_truth.set_object_footprints(goal_meshes(env, goal_ids))
    return ground_truth


def frame_points(ground_truth: GroundTruthMapTracker, state: habitat_sim.AgentState, depth: np.ndarray, floor_height: float, camera: DepthCamera) -> Tuple[np.ndarray, np.ndarray]:
    """(flat ground-truth map pixels, flat sample indices) of depth's in-range sampled points between _MIN_HEIGHT_M and
    _MAX_HEIGHT_M above floor_height and inside the map."""
    depth_samples = sampled(depth.squeeze())
    in_range = (depth_samples > camera.min_depth) & (depth_samples < camera.max_depth)
    points = object_points(depth_samples, sensor_pose_matrix(state.sensor_states["depth"]), in_range, camera)
    pixels = ground_truth.to_pixels(points)
    rows, cols = ground_truth.navigable.shape
    heights = points[:, 1] - floor_height
    keep = (heights > _MIN_HEIGHT_M) & (heights < _MAX_HEIGHT_M) & (pixels >= 0).all(axis=1) & (pixels < (rows, cols)).all(axis=1)
    return pixels[keep, 0] * cols + pixels[keep, 1], np.flatnonzero(in_range)[keep]


def record_frame(recording: Recording, encoder: SiglipEncoder, step: int, obs: dict, points: Tuple[np.ndarray, np.ndarray], segments: Optional[np.ndarray]) -> None:
    """Adds obs' patch embeddings, mask_pooled over the sampled segments when given, and its points to recording and
    saves its rgb, unless it placed no point."""
    if not len(points[0]):
        return
    rgb = obs["rgb"][:, :, :3]
    patch_features, patches = encoder.patch_embeddings([rgb])[0].numpy(), sample_patches(encoder.grid)
    features, sample_rows = (patch_features, patches) if segments is None else mask_pooled(patch_features, patches, segments)
    recording.add(step, features, sample_rows, points)
    Image.fromarray(rgb).save(recording.frames_dir / f"rgb_{step:04d}.jpg", quality=_JPEG_QUALITY)


def explore(cfg: DictConfig, scene_episodes: SceneEpisodes, encoder: SiglipEncoder, args: argparse.Namespace, out_dir: Path) -> FeatureMap:
    """FeatureMap of an fbe agent's exploration, until it stops or args.max_steps run out, its frames saved to
    out_dir/frames_<max_steps>steps/."""
    env, agent, obs = start_exploration(cfg, scene_episodes, args.max_steps, out_dir)
    goals = scene_episodes.goals_by_category[f"{scene_episodes.scene_id}_{query_category(scene_episodes)}"]
    ground_truth = ground_truth_map(env, agent, {goal.object_handle for goal in goals})
    floor_height = env.get_agent_state().position[1] - env.navmesh_floor_offset()
    recording, trajectory_px = Recording(out_dir / f"frames_{args.max_steps}steps"), []
    recording.frames_dir.mkdir(exist_ok=True)
    for step in tqdm(range(args.max_steps), desc=f"explore {scene_episodes.scene_id}", ncols=80):
        state = env.get_agent_state()
        ground_truth.reveal(state.position, TopDownMap.get_polar_angle(state))
        trajectory_px.append(ground_truth.to_pixel(state.position))
        action, _ = agent.act(state, obs["depth"])
        points = frame_points(ground_truth, state, obs["depth"], floor_height, agent.camera)
        record_frame(recording, encoder, step, obs, points, sampled(object_id_image(obs)).ravel() if args.mask_pooling else None)
        if action == "stop":
            break
        obs, _, _, _ = env.step(action)
    env.close()
    return FeatureMap(
        np.concatenate(recording.features), np.array(recording.steps), np.stack(recording.frame_rows), np.concatenate(recording.pixels),
        np.concatenate(recording.point_rows), np.ascontiguousarray(ground_truth.image(trajectory_px)[:, :, ::-1]), ground_truth.navigable > 0,
        ground_truth.footprints,
    )


def painted(feature_map: FeatureMap, pixel_colors: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """The navmesh in gray, with each observed pixel's (H, W, 3) RGB color blended over it."""
    image = np.full(feature_map.navigable.shape + (3,), 255, dtype=np.uint8)
    image[feature_map.navigable] = _NAVIGABLE_GRAY
    image[observed] = (_PIXEL_ALPHA * pixel_colors[observed] + (1 - _PIXEL_ALPHA) * image[observed]).astype(np.uint8)
    return image


def turbo_colors(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """uint8 RGB turbo colors of values, scaled from [low, high] to [0, 1]."""
    return (255 * matplotlib.colormaps["turbo"](np.nan_to_num((values - low) / max(high - low, 1e-8)))[..., :3]).astype(np.uint8)


def heatmap_tile(feature_map: FeatureMap, similarity: np.ndarray, observed: np.ndarray) -> Tuple[str, np.ndarray]:
    """similarity's turbo colors, min-max scaled over the observed pixels, the goal footprints outlined in blue and the
    argmax pixel circled."""
    low, high = similarity[observed].min(), similarity[observed].max()
    image = painted(feature_map, turbo_colors(similarity, low, high), observed)
    contours, _ = cv2.findContours(feature_map.goal_footprints.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, _GOAL_COLOR, 2)
    row, col = np.unravel_index(np.nanargmax(similarity), similarity.shape)
    cv2.circle(image, (int(col), int(row)), _ARGMAX_RADIUS_PX, (0, 0, 0), 4)
    cv2.circle(image, (int(col), int(row)), _ARGMAX_RADIUS_PX, (255, 255, 255), 2)
    return f"similarity, max cos {high:.3f}", letterboxed(image, _TILE_PX, _TILE_PX)


def pca_colors(features: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 RGB of features' top 3 principal components, min-max scaled."""
    projected = PCA(n_components=3).fit_transform(features.astype(np.float32))
    low, high = projected.min(axis=0), projected.max(axis=0)
    return (255 * (projected - low) / np.maximum(high - low, 1e-8)).astype(np.uint8)


def save_similarity_snapshots(feature_map: FeatureMap, row_scores: np.ndarray, frames_dir: Path) -> None:
    """Saves similarity_<step>.jpg next to each recorded frame's rgb_<step>.jpg: the frame blended with its depth samples'
    (R,) query row_scores' turbo colors, min-max scaled over every frame's samples."""
    sample_scores = row_scores[feature_map.frame_rows]
    colors = turbo_colors(sample_scores, sample_scores.min(), sample_scores.max()).reshape(-1, _DEPTH_SAMPLES, _DEPTH_SAMPLES, 3)
    for step, frame_colors in zip(feature_map.frame_steps, colors):
        rgb = Image.open(frames_dir / f"rgb_{step:04d}.jpg")
        overlay = Image.fromarray(frame_colors).resize(rgb.size, Image.NEAREST)
        Image.blend(rgb, overlay, _FRAME_OVERLAY_ALPHA).save(frames_dir / f"similarity_{step:04d}.jpg", quality=_JPEG_QUALITY)


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(_CONFIG_PATH)
    scene_episodes = load_scene_episodes(args.scene)
    out_dir = _OUT_DIR / args.scene / ("mask_pooling" if args.mask_pooling else "patches")
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = SiglipEncoder(device=args.device)
    map_path = out_dir / f"feature_map_{args.max_steps}steps.npz"
    if not map_path.exists():
        np.savez_compressed(map_path, **asdict(explore(cfg, scene_episodes, encoder, args, out_dir)))
    feature_map = load_map(map_path)
    observed = np.bincount(feature_map.point_pixels, minlength=feature_map.navigable.size).reshape(feature_map.navigable.shape) > 0
    category = query_category(scene_episodes)
    row_scores = feature_map.features.astype(np.float32) @ encoder.text_embeddings([category]).numpy()[0]
    ground_truth_tile = ("ground truth\n" + "\n".join(textwrap.wrap(category, _LABEL_WIDTH)), letterboxed(feature_map.ground_truth, _TILE_PX, _TILE_PX))
    make_mosaic([ground_truth_tile, heatmap_tile(feature_map, feature_map.pixel_means(row_scores[:, None])[..., 0], observed)], N_cols=2).save(out_dir / "retrieval.png")
    feature_colors = np.nan_to_num(feature_map.pixel_means(pca_colors(feature_map.features).astype(np.float32))).astype(np.uint8)
    make_mosaic([("SigLIP embeddings, PCA to RGB", letterboxed(painted(feature_map, feature_colors, observed), _TILE_PX, _TILE_PX))], N_cols=1).save(out_dir / "feature_map.png")
    save_similarity_snapshots(feature_map, row_scores, out_dir / f"frames_{args.max_steps}steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=available_scenes()[0])
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-pooling", action="store_true", help="Pool patch embeddings over ground-truth object segments.")
    main(parser.parse_args())
