import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import habitat_sim
import numpy as np
from PIL import Image
from pathlib import Path

from common.agents.abstract_agent import StopReason
from common.agents.base_exploration_agent import BaseExplorationAgent
from common.hybrid_exploration.discrete_planner import DiscretePlanner
from common.hybrid_exploration.exploration_map import EstimatedMap, MapConfig
from common.hybrid_exploration.hybrid_agent import HybridFrontierAgent
from common.utils.plot_utils import PNG_COMPRESS_LEVEL, centered_crop, letterboxed, make_mosaic, rows_mosaic, save_tile
from common.utils.projection_utils import DepthCamera


def obs_dilation_cells(agent_radius_m: float, resolution_m: float) -> int:
    """Obstacle dilation radius (cells) covering at least the agent's own collision radius, so a cell marked
    traversible is actually safe for the agent to occupy, not just for its center point."""
    return math.ceil(agent_radius_m / resolution_m)


@dataclass(frozen=True)
class DiscretePlannerConfig:
    turn_angle_deg: float
    collision_threshold: float = 0.05
    step_size_m: float = 0.25  # one forward move (habitat's forward_step_size): the planner's short-term goal distance
    obs_dilation_selem_radius: int = 3
    debug: bool = False


@dataclass(frozen=True)
class FrontierExplorationConfig:
    map_config: MapConfig
    planner_config: DiscretePlannerConfig
    max_successive_collisions: int = 3


class FrontierExplorationAgent(BaseExplorationAgent):
    """Vanilla frontier-based exploration"""
    map_config: MapConfig
    rng: np.random.Generator
    planner: Optional[DiscretePlanner]
    estimated_map: Optional[EstimatedMap]
    trajectory: List[Tuple[np.ndarray, float]]
    _spin_action: Optional[str]
    
    def __init__(self, config: FrontierExplorationConfig, camera: DepthCamera, rng: np.random.Generator, dump_location: str, exp_name: str):
        planner = DiscretePlanner(
            turn_angle=config.planner_config.turn_angle_deg,
            collision_threshold=config.planner_config.collision_threshold,
            step_size=config.planner_config.step_size_m / config.map_config.resolution_m,
            obs_dilation_selem_radius=config.planner_config.obs_dilation_selem_radius,
            map_size_cm=int(config.map_config.map_size_m * 100),
            map_resolution=int(config.map_config.resolution_m * 100),
            dump_location=dump_location,
            exp_name=exp_name,
            debug=config.planner_config.debug,
        )
        super().__init__(config.map_config, rng, config.planner_config.turn_angle_deg, planner)
        self.camera = camera
        self.config = config
        self._successive_collisions = 0

    def reset(
        self, start_state: habitat_sim.AgentState, floor_height: float, navmesh_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        super().reset(start_state, floor_height, navmesh_bounds)
        self._successive_collisions = 0

    def act(self, agent_state: habitat_sim.AgentState, depth: np.ndarray, collided: bool = False) -> Tuple[str, Optional[StopReason]]:
        assert self.estimated_map is not None, "Call reset() first"
        assert self.planner is not None, "Call reset() first"

        self._successive_collisions = self._successive_collisions + 1 if collided else 0
        if self._successive_collisions >= self.config.max_successive_collisions:
            return "stop", StopReason.COLLISION
        agent_state = self.ref_frame_state(agent_state)
        position, heading = self.observe(agent_state, depth, self.camera)
        if self.is_stuck():
            return "stop", StopReason.STUCK
        spin_action = self.initial_spin()
        if spin_action is not None:
            return spin_action, None
        no_goal = np.zeros_like(self.estimated_map.obstacles)
        return self.plan_action(no_goal, len(self.estimated_map.frontier_pixels) > 0, position, heading, len(self.trajectory) - 1)


_SNAPSHOT_TILE_SIZE = 1024
_BELIEF_CROP_M = 12.0
_SAVE_WORKERS = 8


def exploration_snapshot(
    obs: dict, agent, ground_truth, trajectory_px: list,
) -> Optional[Image.Image]:
    """3-tile mosaic, every tile upscaled to _SNAPSHOT_TILE_SIZE: first-person rgb (or, when agent defines
    detections_tile() -- e.g. SearchAgent's Detic predictions -- that same frame with its detections drawn on
    it), the agent's own belief map cropped to _BELIEF_CROP_M around the agent, the ground-truth comparison map."""
    size = (_SNAPSHOT_TILE_SIZE, _SNAPSHOT_TILE_SIZE)
    if hasattr(agent, "detections_tile"):
        observation_label, first_person = agent.detections_tile()
    else:
        observation_label, first_person = "observation", obs["rgb"][:, :, :3]
    first_person = letterboxed(np.ascontiguousarray(first_person), *size)
    agent_pixel = agent.estimated_map.to_pixel(agent.trajectory[-1][0])
    belief_map = cv2.resize(centered_crop(agent.map_image(), agent_pixel, int(round(_BELIEF_CROP_M / agent.estimated_map.meters_per_pixel)))[:, :, ::-1], size, interpolation=cv2.INTER_NEAREST)
    gt_map = letterboxed(ground_truth.image(trajectory_px)[:, :, ::-1], *size)
    return rows_mosaic([[
        (observation_label, np.ascontiguousarray(first_person)),
        ("belief map", np.ascontiguousarray(belief_map)),
        ("ground truth", np.ascontiguousarray(gt_map)),
    ]])

def save_exploration_visuals(agent: HybridFrontierAgent, prediction_tiles: list, out_dir: Path) -> None:
    """predictions_mosaic.png and prediction_frames/step_<step>.png of every step, the frames saved in parallel (PIL
    releases the GIL while encoding)."""
    assert agent.estimated_map is not None, "Call reset() first"

    frame_paths = [out_dir / "prediction_frames" / f"step_{step:04d}.png" for step in range(len(prediction_tiles))]
    with ThreadPoolExecutor(_SAVE_WORKERS) as pool:
        frames_saved = pool.map(save_tile, prediction_tiles, frame_paths)
        make_mosaic(prediction_tiles, N_cols=5).save(out_dir / "predictions_mosaic.png", compress_level=PNG_COMPRESS_LEVEL)
        list(frames_saved)

