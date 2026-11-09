"""Detic-driven single-target object-search agent: scores every step's rgb against the episode's target
embedding, fuses the best detection's mask into the agent's own depth-built EstimatedMap's goal_score grid,
and asks goal_policy.select_action for the next action -- home-robot's ObjectNavAgent shape (detect -> update
map -> plan -> act), on this repo's own frontier/path primitives.
"""
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import habitat_sim
import numpy as np
import torch
import trimesh
from detectron2.structures import Instances

from common.agents.abstract_agent import StopReason
from common.agents.base_exploration_agent import BaseExplorationAgent, DiscretePlanner, EstimatedMap
from common.hybrid_exploration.exploration_map import MapConfig
from common.hybrid_exploration.frontier_exp_agent import DiscretePlannerConfig
from common.utils.projection_utils import DepthCamera
from common.utils.retrieval import SegmentationStyle, segmentation_tile
from common.vision.detic import Vocabulary
from common.vision.detector import Detector
from common.hybrid_exploration.plotting import draw_goal_map

_DETECTION_COLOR = (255, 0, 255)  # RGB -- magenta, the single target class predict_target scores against

def mesh_footprint(mesh: trimesh.Trimesh) -> np.ndarray:
    """(1000, 2) top-down (x, z) points sampled on mesh's surface."""
    return trimesh.sample.sample_surface(mesh, 1000, seed=0)[0][:, [0, 2]]


@dataclass
class SearchAgentConfig:
    map_config: MapConfig
    goal_score_thresh: float
    max_steps: int
    confidence_thresh: float
    planner_config: DiscretePlannerConfig
    reach_m: float = 1.0
    use_gt_detections: bool = False  # act()'s gt_instances (all scored 1.0) instead of the detector's



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

    def __init__(self, detector: Detector, camera: DepthCamera, config: SearchAgentConfig, rng: np.random.Generator, dump_location: str, exp_name: str, target_name: str = "target"):
        planner = DiscretePlanner(
            turn_angle=config.planner_config.turn_angle_deg,
            collision_threshold=config.planner_config.collision_threshold,
            step_size=config.planner_config.step_size_m / config.map_config.resolution_m,
            obs_dilation_selem_radius=config.planner_config.obs_dilation_selem_radius,
            map_size_cm=int(config.map_config.map_size_m * 100),
            map_resolution=int(config.map_config.resolution_m * 100),
            min_goal_distance_cm=0.75 * config.reach_m * 100,
            min_obs_dilation_selem_radius=config.planner_config.obs_dilation_selem_radius,
            dump_location=dump_location,
            exp_name=exp_name,
            debug=config.planner_config.debug,
        )

        super().__init__(config.map_config, rng, config.planner_config.turn_angle_deg, planner)
        self.detector = detector
        self.camera = camera
        self.config = config
        self.target_embedding = None
        self.target_label = None
        self._goal_map = None
        self.step_count = 0
        self.last_instances = None
        self.last_rgb = None
        self._detection_style = SegmentationStyle(classes=[target_name], colors=[_DETECTION_COLOR])


    def reset(
        self, start_state: habitat_sim.AgentState, floor_height: float, target_embedding: torch.Tensor, target_label: str,
        navmesh_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        super().reset(start_state, floor_height, navmesh_bounds)
        assert self.estimated_map is not None, "BaseExplorationAgent.reset() must be called first"
        assert self.planner is not None, "Call reset() first"

        self.target_embedding = target_embedding
        self.target_label = target_label
        self._goal_map = None
        self.step_count = 0
        self.last_instances = None
        self.last_rgb = None

    def act(
        self, agent_state: habitat_sim.AgentState, rgb: np.ndarray, depth: np.ndarray,
        gt_instances: Optional[Instances] = None,
    ) -> Tuple[str, Optional[StopReason]]:
        assert self.estimated_map is not None, "Call reset() first"
        assert self.target_embedding is not None, "Call reset() first"
        assert self.planner is not None, "Call reset() first"

        self.step_count += 1
        agent_state = self.ref_frame_state(agent_state)
        position, heading = self.observe(agent_state, depth, self.camera)
        self.estimated_map.decay_goal_score()

        if self.config.use_gt_detections:
            assert gt_instances is not None, "use_gt_detections needs act()'s gt_instances"
            instances, best_score = gt_instances, 1.0 if len(gt_instances) else 0.0
        else:
            instances, best_score = self.detector.predict_target(rgb[:, :, :3], self.target_embedding)

        self.last_instances = instances
        self.last_rgb = rgb[:, :, :3]

        # If the best detection is confident enough, mark the estimated map's goal_score with its mask; goal_score_thresh
        # is the separate threshold on the accumulated map belief, read by goal_mask() and evaluator.py.
        if best_score >= self.config.confidence_thresh:
            best_idx = int(instances.scores.argmax())
            self.estimated_map.mark_goal(
                depth, agent_state.sensor_states["depth"], instances.pred_masks[best_idx], best_score, self.camera, self.floor_height,
            )

        if self.step_count >= self.config.max_steps:
            return "stop", StopReason.MAX_STEPS
        if self.is_stuck():
            return "stop", StopReason.STUCK

        spin_action = self.initial_spin()
        if spin_action is not None:
            return spin_action, None

        goal_map = self.estimated_map.goal_mask(self.config.goal_score_thresh)
        self._goal_map = goal_map  # for visualization/debugging

        frontier_map = self.estimated_map.frontier_map()

        action, self._closest_goal_map, self._short_term_goal, _, no_feasible_plan, no_frontier = self.planner.plan(
            obstacle_map=self.estimated_map.obstacles,
            goal_map=goal_map,
            frontier_map=frontier_map,
            sensor_pose=self.sensor_pose(position, heading),
            found_goal=np.any(goal_map),
            use_dilation_for_stg=False,
            current_timestep=self.step_count - 1,
        )

        action_name = self.action_name(action)
        return action_name, self.plan_stop_reason(action_name, no_feasible_plan, no_frontier)

    def map_image(self) -> np.ndarray:
        image = super().map_image()
        assert self.estimated_map is not None, "Call reset() first"

        if self._goal_map is not None and np.any(self._goal_map):
            draw_goal_map(
                image,
                self._goal_map,
                radius=int(self.config.reach_m / self.estimated_map.meters_per_pixel),
                goal_score=self.estimated_map.goal_score,
            )

        return image


    def detections_tile(self) -> Tuple[str, np.ndarray]:
        """(label, RGB) tile of the last act() rgb frame with predict_target's detections (boxes/masks/scores)
        drawn on it -- so the goal_score updates driving navigation stay visible, not just their effect."""
        label = f"observation\n{self.target_label}"
        if self.last_rgb is None or self.last_instances is None or len(self.last_instances) == 0:
            rgb = np.zeros((1, 1, 3), dtype=np.uint8) if self.last_rgb is None else self.last_rgb
            return label, rgb
        return segmentation_tile(self.last_rgb, self.last_instances, self._detection_style, label)

    def goal_score_image(self) -> Optional[np.ndarray]:
        """estimated_map.goal_score (float32, each cell 0-1), for BaseExplorationAgent.map_image() to blend
        into the belief map as a blue heatmap -- where the detector has ever been confident the target is."""
        return None if self.estimated_map is None else self.estimated_map.goal_score
