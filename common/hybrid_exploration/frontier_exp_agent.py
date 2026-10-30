from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import habitat_sim
import numpy as np
from PIL import Image
from pathlib import Path

from common.agents.base_exploration_agent import BaseExplorationAgent
from common.hybrid_exploration.discrete_planner import DiscretePlanner
from common.hybrid_exploration.exploration_map import EstimatedMap, MapConfig
from common.hybrid_exploration.hybrid_agent import HybridFrontierAgent
from common.hybrid_exploration.plotting import plot_occupancy
from common.utils.plot_utils import letterboxed, make_mosaic, rows_mosaic, save_tile
from common.utils.projection_utils import DepthCamera


@dataclass(frozen=True)
class DiscretePlannerConfig:
    turn_angle_deg: float
    collision_threshold: float = 0.05
    step_size: int = 5
    obs_dilation_selem_radius: int = 3
    goal_dilation_selem_radius: int = 3


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
            step_size=config.planner_config.step_size,
            obs_dilation_selem_radius=config.planner_config.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.planner_config.goal_dilation_selem_radius,
            map_size_cm=int(config.map_config.map_size_m * 100),
            map_resolution=int(config.map_config.resolution_m * 100),
            dump_location=dump_location,
            exp_name=exp_name,
        )
        super().__init__(config.map_config, rng, planner)
        self.camera = camera
        self.config = config
        self._successive_collisions = 0
        self._had_frontier = False

    def reset(self, position: np.ndarray, floor_height: float) -> None:
        super().reset(position, floor_height)
        self._successive_collisions = 0
        self._had_frontier = False

    def act(self, agent_state: habitat_sim.AgentState, depth: np.ndarray, collided: bool = False) -> str:
        assert self.estimated_map is not None, "Call reset() first"
        assert self.planner is not None, "Call reset() first"

        self._successive_collisions = self._successive_collisions + 1 if collided else 0
        if self._successive_collisions >= self.config.max_successive_collisions:
            return "stop"
        position, heading = self.observe(agent_state, depth, self.camera)
        frontier_map = self.estimated_map.frontier_map()
        if not frontier_map.any():
            return "stop" if self._had_frontier else self.spin()
        self._had_frontier = True
        action, self._closest_goal_map, self._short_term_goal, _ = self.planner.plan(
            obstacle_map=self.estimated_map.obstacles,
            goal_map=frontier_map,
            frontier_map=frontier_map,
            sensor_pose=self.sensor_pose(position, heading),
            found_goal=False,
            use_dilation_for_stg=True,
        )
        return self.action_name(action)


_SNAPSHOT_TILE_SIZE = 1024


def exploration_snapshot(
    obs: dict, agent, ground_truth, trajectory_px: list,
) -> Optional[Image.Image]:
    """3-tile mosaic, every tile upscaled to _SNAPSHOT_TILE_SIZE: first-person rgb (or, when agent defines
    detections_tile() -- e.g. SearchAgent's Detic predictions -- that same frame with its detections drawn on
    it), the agent's own belief map, the ground-truth comparison map."""
    size = (_SNAPSHOT_TILE_SIZE, _SNAPSHOT_TILE_SIZE)
    if hasattr(agent, "detections_tile"):
        observation_label, first_person = agent.detections_tile()
    else:
        observation_label, first_person = "observation", obs["rgb"][:, :, :3]
    first_person = letterboxed(np.ascontiguousarray(first_person), *size)
    belief_map = cv2.resize(agent.map_image()[:, :, ::-1], size, interpolation=cv2.INTER_NEAREST)
    gt_map = letterboxed(ground_truth.image(trajectory_px)[:, :, ::-1], *size)
    return rows_mosaic([[
        (observation_label, np.ascontiguousarray(first_person)),
        ("belief map", np.ascontiguousarray(belief_map)),
        ("ground truth", np.ascontiguousarray(gt_map)),
    ]])

def save_exploration_visuals(agent: HybridFrontierAgent, prediction_tiles: list, out_dir: Path) -> None:
    """predictions_mosaic.png and prediction_frames/step_<step>.png of every step, and decision.png and object.png of
    the last step."""
    assert agent.estimated_map is not None, "Call reset() first"
    
    save_tile(agent.map_tile(), out_path = out_dir / "decision.png")
    make_mosaic(prediction_tiles, N_cols=5).save(out_dir / "predictions_mosaic.png")
    frames_dir = out_dir / "prediction_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for step, tile in enumerate(prediction_tiles):
        make_mosaic([tile], N_cols=1).save(frames_dir / f"step_{step:04d}.png")
    occupancy = plot_occupancy(agent.estimated_map.explored, [agent.estimated_map.to_pixels(belief.points) for belief in agent.objects.values()])
    make_mosaic([("objects", occupancy[:, :, ::-1].copy())], N_cols=1).save(out_dir / "object.png")

