"""Detic-driven single-target object-search agent: scores every step's rgb against the episode's target
embedding, fuses the best detection's mask into a GoalScoreMap over the agent's own depth-built EstimatedMap,
and asks goal_policy.select_action for the next action -- home-robot's ObjectNavAgent shape (detect -> update
map -> plan -> act), on this repo's own frontier/path primitives. 
"""
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple

import habitat_sim
import numpy as np
import torch

from common.agents.base_exploration_agent import BaseExplorationAgent, DiscretePlanner, EstimatedMap
from common.hybrid_exploration.exploration_map import MapConfig
from common.hybrid_exploration.frontier_exp_agent import DiscretePlannerConfig
from common.hybrid_exploration.semantic_map import GoalScoreConfig, GoalScoreMap
from common.utils.projection_utils import DepthCamera
from common.utils.retrieval import SegmentationStyle, segmentation_tile
from common.vision.detic import Vocabulary
from common.vision.detector import Detector

_DETECTION_COLOR = (255, 0, 255)  # RGB -- magenta, the single target class predict_target scores against

@dataclass
class SearchAgentConfig:
    map_config: MapConfig
    goal_score_config: GoalScoreConfig
    goal_score_thresh: float
    max_steps: int
    confidence_thresh: float
    planner_config: DiscretePlannerConfig



class SearchAgent(BaseExplorationAgent):
    """Detic-driven single-target object-search agent: scores every step's rgb against the episode's target"""

    map_config: MapConfig
    rng: np.random.Generator
    planner: Optional[DiscretePlanner]
    estimated_map: Optional[EstimatedMap]
    trajectory: List[Tuple[np.ndarray, float]]
    _spin_action: Optional[str]

    detector: Detector
    camera: DepthCamera
    goal_score_map: Optional[GoalScoreMap]

    def __init__(self, detector: Detector, camera: DepthCamera, config: SearchAgentConfig, rng: np.random.Generator, dump_location: str, exp_name: str, target_name: str = "target"):
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
        self.detector = detector
        self.camera = camera
        self.config = config
        self.target_embedding = None
        self.goal_score_map = None
        self.step_count = 0
        self.last_instances = None
        self.last_rgb = None
        self.found_goal = False
        self.navigation_target = None
        self._detection_style = SegmentationStyle(classes=[target_name], colors=[_DETECTION_COLOR])


    def reset(self, position: np.ndarray, floor_height: float, target_embedding: torch.Tensor) -> None:
        super().reset(position, floor_height)
        assert self.estimated_map is not None, "BaseExplorationAgent.reset() must be called first"
        assert self.planner is not None, "Call reset() first"

        self.target_embedding = target_embedding
        self.goal_score_map = GoalScoreMap(self.config.goal_score_config, self.estimated_map)
        self.step_count = 0
        self.last_instances = None
        self.last_rgb = None
        self.found_goal = False
        self.navigation_target = None

    def act(self, agent_state: habitat_sim.AgentState, rgb: np.ndarray, depth: np.ndarray, collided: bool = False) -> str:
        assert self.estimated_map is not None, "Call reset() first"
        assert self.target_embedding is not None, "Call reset() first"
        assert self.goal_score_map is not None, "Call reset() first"
        assert self.planner is not None, "Call reset() first"

        self.step_count += 1
        position, heading = self.observe(agent_state, depth, self.camera)
        self.goal_score_map.decay()
        frontier_map = self.estimated_map.frontier_map()

        # forward the rgb to Detic, get the best detection's mask and score
        instances, best_score = self.detector.predict_target(rgb[:, :, :3], self.target_embedding)
        self.last_instances = instances
        self.last_rgb = rgb[:, :, :3]

        # If the best detection's score exceeds the threshold, update the GoalScoreMap with its mask.
        if best_score >= self.config.goal_score_thresh:
            best_idx = int(instances.scores.argmax())
            self.goal_score_map.update(
                depth, agent_state.sensor_states["depth"], instances.pred_masks[best_idx], best_score, self.camera,
            )

        if self.step_count >= self.config.max_steps:
            self.navigation_target = None
            return "stop"
        

        goal_map = self.goal_score_map.goal_mask(self.config.goal_score_thresh)
        if goal_map.any():
            action, self._closest_goal_map, self._short_term_goal, _ = self.planner.plan(
                obstacle_map=self.estimated_map.obstacles,
                goal_map=goal_map,
                frontier_map=frontier_map,
                sensor_pose=self.sensor_pose(position, heading),
                found_goal=True,
                use_dilation_for_stg=True,
            )

        else:
            action, self._closest_goal_map, self._short_term_goal, _ = self.planner.plan(
                obstacle_map=self.estimated_map.obstacles,
                goal_map=frontier_map,
                frontier_map=frontier_map,
                sensor_pose=self.sensor_pose(position, heading),
                found_goal=False,
                use_dilation_for_stg=True,
            )

        return self.action_name(action)

    def detections_tile(self) -> Tuple[str, np.ndarray]:
        """(label, RGB) tile of the last act() rgb frame with predict_target's detections (boxes/masks/scores)
        drawn on it -- so the goal_score_map updates driving navigation stay visible, not just their effect."""
        label = "observation"
        if self.last_rgb is None or self.last_instances is None or len(self.last_instances) == 0:
            rgb = np.zeros((1, 1, 3), dtype=np.uint8) if self.last_rgb is None else self.last_rgb
            return label, rgb
        return segmentation_tile(self.last_rgb, self.last_instances, self._detection_style, label)

    def goal_score_image(self) -> Optional[np.ndarray]:
        """goal_score_map.goal_score (float32, each cell 0-1), for BaseExplorationAgent.map_image() to blend
        into the belief map as a blue heatmap -- where the detector has ever been confident the target is."""
        return None if self.goal_score_map is None else self.goal_score_map.goal_score
