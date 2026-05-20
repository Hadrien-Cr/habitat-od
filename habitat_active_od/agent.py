from typing import Any, Optional

import numpy as np
import cv2
import scipy
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.cluster import DBSCAN

from common.exploration.frontier_exploration import FrontierExplorationPolicy
from common.planning.discrete_planner import DiscretePlanner
from habitat_active_od.visualizer import Visualizer
from common.mapping.categorical_2d_semantic_map_module import Categorical2DSemanticMapModule
from common.mapping.categorical_2d_semantic_map_state import Categorical2DSemanticMapState
from common.interfaces import DiscreteNavigationAction, Observations
import common.utils.pose_utils as pu
from common.utils.plot_utils import plot_segmentation_pred, plot_segmentation_gt


class ObsPreprocessor:
    """Preprocess raw observations for consumption by an agent."""

    def __init__(self, config: DictConfig, device: torch.device) -> None:
        self.device = device
        self.frame_height = config.HABITAT_ACTIVE_OD.frame_height
        self.frame_width = config.HABITAT_ACTIVE_OD.frame_width

        self.depth_filtering = config.AGENT.SEMANTIC_PREDICTION.depth_filtering
        self.depth_filter_range_cm = config.AGENT.SEMANTIC_PREDICTION.depth_filter_range_cm

        # init episode variables
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self.last_pose = None
        self.step = 0

    def reset(self) -> None:
        """Reset for a new episode since pre-processing is temporally dependent."""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self.last_pose = np.zeros(3)
        self.step = 0
        

    def preprocess(
        self, obs: Observations
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess observations of a single timestep batched across
        environments.

        Returns:
            obs_preprocessed: frame containing (RGB, depth) of
               shape (3 + 1 + 1, frame_height, frame_width)
            pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
               of shape (num_environments, 3)
            camera_pose: camera extrinsic pose of shape (num_environments, 4, 4)
        """

        pose_delta, self.last_pose = self._preprocess_pose_and_delta(obs)
        pose_delta = (
            torch.tensor(pose_delta)
            .unsqueeze(0)
            .to(device=self.device)
        )
        obs_preprocessed = self._preprocess_frame(obs)
        obs_preprocessed = torch.tensor(obs_preprocessed).unsqueeze(0).to(self.device)

        camera_pose = obs.camera_pose
        camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0).to(self.device)
        self.step += 1
        return obs_preprocessed, pose_delta, camera_pose

    def _preprocess_frame(self, obs: Observations) -> np.ndarray:
        """Preprocess frame information in the observation."""

        def downscale(rgb: np.ndarray, depth:  np.ndarray) -> tuple[ np.ndarray,  np.ndarray]:
            """downscale RGB and depth frames to self.frame_{width,height}"""
            ds = rgb.shape[1] / self.frame_width
            if ds == 1:
                return rgb, depth
            dim = (self.frame_width, self.frame_height)
            rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, dim, interpolation=cv2.INTER_NEAREST)[:, :, None]
            return rgb, depth


        depth = np.expand_dims(obs.depth, axis=2) * 100.0
        rgb, depth = downscale(obs.rgb, depth)
        obs_preprocessed = np.concatenate([rgb, depth, np.zeros_like(depth)], axis=2).transpose(2,0,1)
        assert obs_preprocessed.shape == (5, self.frame_height, self.frame_width)
        return obs_preprocessed

    def _preprocess_pose_and_delta(self, obs: Observations) -> tuple[np.ndarray, np.ndarray]:
        """merge GPS+compass. Compute the delta from the previous timestep."""
        curr_pose = np.array([obs.gps[0], obs.gps[1], obs.compass[0]])
        pose_delta = np.array(pu.get_rel_pose_change(curr_pose, self.last_pose))
        return pose_delta, curr_pose

class ActiveODModule(nn.Module):
    """
    An agent module that maintains a 2D map, explores with FBE, and detects and
    localizes object goals from keypoint correspondences.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=config.HABITAT_ACTIVE_OD.frame_height,
            frame_width=config.HABITAT_ACTIVE_OD.frame_width,
            camera_height=config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[
                1
            ],
            hfov=config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            vision_range=config.AGENT.SEMANTIC_MAP.vision_range,
            explored_radius=config.AGENT.SEMANTIC_MAP.explored_radius,
            been_close_to_radius=config.AGENT.SEMANTIC_MAP.been_close_to_radius,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
            max_depth=config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth,
            min_depth=config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth,
            du_scale=config.AGENT.SEMANTIC_MAP.du_scale,
            cat_pred_threshold=config.AGENT.SEMANTIC_MAP.cat_pred_threshold,
            exp_pred_threshold=config.AGENT.SEMANTIC_MAP.exp_pred_threshold,
            map_pred_threshold=config.AGENT.SEMANTIC_MAP.map_pred_threshold,
            must_explore_close=config.AGENT.SEMANTIC_MAP.must_explore_close,
            min_obs_height_cm=config.AGENT.SEMANTIC_MAP.min_obs_height_cm,
            dilate_obstacles=config.AGENT.SEMANTIC_MAP.dilate_obstacles,
            dilate_size=config.AGENT.SEMANTIC_MAP.dilate_size,
            dilate_iter=config.AGENT.SEMANTIC_MAP.dilate_iter,
        )
        self.exploration_policy = FrontierExplorationPolicy()

    @property
    def goal_update_steps(self) -> int:
        return self.exploration_policy.goal_update_steps
         
    def forward(
        self,
        seq_obs: torch.Tensor,
        seq_pose_delta: torch.Tensor,
        seq_dones: torch.Tensor,
        seq_update_global: torch.Tensor,
        seq_camera_poses: Optional[torch.Tensor],
        seq_found_goal: torch.Tensor,
        seq_goal_map: torch.Tensor,
        init_local_map: torch.Tensor,
        init_global_map: torch.Tensor,
        init_local_pose: torch.Tensor,
        init_global_pose: torch.Tensor,
        init_lmb: torch.Tensor,
        init_origins: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Update maps and poses with a sequence of observations, and predict
        high-level goals from map features.

        Arguments:
            seq_obs: sequence of frames containing (RGB, depth, segmentation)
             of shape (batch_size, sequence_length, 3 + 1 + num_sem_categories,
             frame_height, frame_width)
            seq_pose_delta: sequence of delta in pose since last frame of shape
             (batch_size, sequence_length, 3)
            seq_dones: sequence of (batch_size, sequence_length) done flags that
             indicate episode restarts
            seq_update_global: sequence of (batch_size, sequence_length) binary
             flags that indicate whether to update the global map and pose
            seq_camera_poses: sequence of (batch_size, sequence_length, 4, 4) camera poses
            init_local_map: initial local map before any updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            init_global_map: initial global map before any updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            init_local_pose: initial local pose before any updates of shape
             (batch_size, 3)
            init_global_pose: initial global pose before any updates of shape
             (batch_size, 3)
            init_lmb: initial local map boundaries of shape (batch_size, 4)
            init_origins: initial local map origins of shape (batch_size, 3)

        Returns:
            seq_goal_map: sequence of binary maps encoding goal(s) of shape
             (batch_size, sequence_length, M, M)
            seq_found_goal: binary variables to denote whether we found the object
             goal category of shape (batch_size, sequence_length)
            final_local_map: final local map after all updates of shape
             (batch_size, 4 + num_sem_categories, M, M)
            final_global_map: final global map after all updates of shape
             (batch_size, 4 + num_sem_categories, M * ds, M * ds)
            seq_local_pose: sequence of local poses of shape
             (batch_size, sequence_length, 3)
            seq_global_pose: sequence of global poses of shape
             (batch_size, sequence_length, 3)
            seq_lmb: sequence of local map boundaries of shape
             (batch_size, sequence_length, 4)
            seq_origins: sequence of local map origins of shape
             (batch_size, sequence_length, 3)
        """
        # Reset the last channel of the local map each step when found_goal=False
        # init_local_map: [8, 21, 480, 480]
        init_local_map[:, -1][seq_found_goal[:, 0] == 0] *= 0.0

        # Update map with observations and generate map features
        batch_size, sequence_length = seq_obs.shape[:2]
        (
            seq_map_features,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.semantic_map_module(
            seq_obs,
            seq_pose_delta,
            seq_dones,
            seq_update_global,
            seq_camera_poses,
            init_local_map,
            init_global_map,
            init_local_pose,
            init_global_pose,
            init_lmb,
            init_origins,
        )

        # Predict high-level goals from map features.
        map_features = seq_map_features.flatten(0, 1)

        # the last channel of map_features is cut off -- used for goal det/loc.
        frontier_map = self.exploration_policy(map_features[:, :-1])
        seq_goal_map[seq_found_goal[:, 0] == 0] = frontier_map[
            seq_found_goal[:, 0] == 0
        ]

        seq_goal_map = seq_goal_map.view(
            batch_size, sequence_length, *seq_goal_map.shape[-2:]
        )

        seq_frontier_map = frontier_map.view(
            batch_size, sequence_length, *frontier_map.shape[-2:]
        )

        return (
            seq_goal_map,
            seq_found_goal,
            seq_frontier_map,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) # type: ignore


class ActiveODAgent:

    def __init__(self, config: DictConfig, predictor, classes, device_id: int = 0) -> None:
        self.device = torch.device(f"cuda:{device_id}")
        self.obs_preprocessor = ObsPreprocessor(config, self.device)
        self.max_steps = config.habitat.environment.max_episode_steps
        self.num_environments = 1

        self._module = ActiveODModule(config).to(self.device)
        self.goal_update_steps = self._module.goal_update_steps
        self.verbose = config.AGENT.VERBOSE

        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=config.AGENT.SEMANTIC_MAP.num_sem_categories,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            global_downscaling=config.AGENT.SEMANTIC_MAP.global_downscaling,
        )
        agent_radius_cm = config.habitat.simulator.agents.main_agent.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.AGENT.SEMANTIC_MAP.map_resolution)
        )
        self.use_dilation_for_stg = config.AGENT.PLANNER.use_dilation_for_stg
        self.planner = DiscretePlanner(
            turn_angle=config.habitat.simulator.turn_angle,
            collision_threshold=config.AGENT.PLANNER.collision_threshold,
            step_size=config.AGENT.PLANNER.step_size,
            obs_dilation_selem_radius=config.AGENT.PLANNER.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.AGENT.PLANNER.goal_dilation_selem_radius,
            map_size_cm=config.AGENT.SEMANTIC_MAP.map_size_cm,
            map_resolution=config.AGENT.SEMANTIC_MAP.map_resolution,
            visualize=config.AGENT.PLANNER.visualize,
            print_images=False,
            dump_location=config.AGENT.DUMP_LOCATION,
            exp_name=config.AGENT.EXP_NAME,
            agent_cell_radius=agent_cell_radius,
            min_obs_dilation_selem_radius=config.AGENT.PLANNER.min_obs_dilation_selem_radius,
            map_downsample_factor=config.AGENT.PLANNER.map_downsample_factor,
            map_update_frequency=config.AGENT.PLANNER.map_update_frequency,
            discrete_actions=config.AGENT.PLANNER.discrete_actions,
            min_goal_distance_cm=config.AGENT.PLANNER.min_goal_distance_cm,
        )

        self.goal_filtering = config.AGENT.SEMANTIC_PREDICTION.goal_filtering
        self.timesteps = [0]
        self.timesteps_before_goal_update = None
        self.found_goal = torch.zeros(
            self.num_environments, 1, dtype=bool, device=self.device # type: ignore
        )
        self.goal_map = torch.zeros(
            self.num_environments,
            1,
            *self.semantic_map.local_map.shape[2:],
            dtype=self.semantic_map.local_map.dtype,
            device=self.device,
        )
        self.predictor = predictor
        self.classes = classes

        self.visualizer = None
        if self.verbose:
            self.visualizer = Visualizer(config)

    def reset(self) -> None:
        """Initialize agent state."""
        self.obs_preprocessor.reset()

        if self.visualizer is not None:
            self.visualizer.reset()

        self.timesteps = [0]
        self.timesteps_before_goal_update = [0]
        self.semantic_map.init_map_and_pose()
        self.found_goal[:] = False
        self.goal_map[:] *= 0
        self.planner.reset()


    def act(self, obs: Observations) -> DiscreteNavigationAction:
        """Act end-to-end."""
        # Predict on observation
        outputs = self.predictor(obs.rgb)
        pred_instances = outputs["instances"].to("cpu")
        
        sem_img = plot_segmentation_pred(obs.rgb, pred_instances, self.classes)    
    
        # Else; plan with the module's internal semantic map state and return a discrete action.
        (
            obs_preprocessed,
            pose_delta,
            camera_pose,
        ) = self.obs_preprocessor.preprocess(obs)
        
        planner_inputs, vis_inputs = self._prepare_planner_inputs(
            obs_preprocessed, pose_delta, camera_pose
        )

        closest_goal_map = None

        if self.timesteps[0] < int(360 / self.planner.turn_angle):
            action = DiscreteNavigationAction.TURN_LEFT
        elif self.timesteps[0] >= (self.max_steps - 1):
            action = DiscreteNavigationAction.STOP
        else:
            action,closest_goal_map,short_term_goal,dilated_obstacle_map,could_not_find_path,planner_stop = self.planner.plan(
                **planner_inputs[0],
                use_dilation_for_stg=self.use_dilation_for_stg,
                debug=True,
            )
            print(f"Action: {action}, Short-term goal: {short_term_goal}, Could not find path: {could_not_find_path}, Planner stop: {planner_stop}")

        if self.visualizer is not None:
            collision = obs.task_observations.get("collisions")
            if collision is None:
                collision = {"is_collision": False}
            info = {
                **planner_inputs[0],
                **vis_inputs[0],
                "rgb_frame": obs.rgb.copy(),
                "depth_frame": obs.depth.copy(),
                "semantic_frame": sem_img.copy(),
                "closest_goal_map": closest_goal_map,
                "last_goal_image": obs.task_observations["goal_image"],
                "last_collisions": collision,
                "last_td_map": obs.task_observations.get("top_down_map"),
            }
            self.visualizer.visualize(**info)

        return action # type: ignore

    @torch.no_grad()
    def _prepare_planner_inputs(
        self,
        obs: torch.Tensor,
        pose_delta: torch.Tensor,
        camera_pose: Optional[torch.Tensor] = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Determine a long-term navigation goal in 2D map space for a local policy to
        execute.
        """

        dones = torch.zeros(self.num_environments, dtype=torch.bool)
        update_global = torch.tensor(
            [
                self.timesteps_before_goal_update[e] == 0 # type: ignore
                for e in range(self.num_environments)
            ]
        )
        (
            self.goal_map,
            self.found_goal,
            frontier_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self._module(
            obs.unsqueeze(1),
            pose_delta.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            camera_pose.unsqueeze(1), # type: ignore
            self.found_goal,
            self.goal_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
        )

        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]
        goal_map = self._prep_goal_map_input()
        for e in range(self.num_environments):
            self.semantic_map.update_frontier_map(e, frontier_map[e][0].cpu().numpy())
            if self.found_goal[e].item():
                self.semantic_map.update_global_goal_for_env(e, goal_map[e]) # type: ignore
            elif self.timesteps_before_goal_update[e] == 0: # type: ignore
                self.semantic_map.update_global_goal_for_env(e, goal_map[e]) # type: ignore
                self.timesteps_before_goal_update[e] = self.goal_update_steps # type: ignore

        self.timesteps = [self.timesteps[e] + 1 for e in range(self.num_environments)]
        self.timesteps_before_goal_update = [
            self.timesteps_before_goal_update[e] - 1 # type: ignore
            for e in range(self.num_environments)
        ]

        planner_inputs = [
            {
                "obstacle_map": self.semantic_map.get_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e),
                "frontier_map": self.semantic_map.get_frontier_map(e),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
                "found_goal": self.found_goal[e].item(),
            }
            for e in range(self.num_environments)
        ]
        vis_inputs = [
            {
                "explored_map": self.semantic_map.get_explored_map(e),
                "timestep": self.timesteps[e],
            }
            for e in range(self.num_environments)
        ]
        if self.semantic_map.num_sem_categories > 1:
            for e in range(self.num_environments):
                vis_inputs[e]["semantic_map"] = self.semantic_map.get_semantic_map(e)

        return planner_inputs, vis_inputs

    def _prep_goal_map_input(self) -> None:
        """
        Perform optional clustering of the goal channel to mitigate noisy projection
        splatter.
        """
        goal_map = self.goal_map.squeeze(1).cpu().numpy()

        if not self.goal_filtering:
            return goal_map

        for e in range(goal_map.shape[0]):
            if not self.found_goal[e]:
                continue

            # cluster goal points
            c = DBSCAN(eps=4, min_samples=1)
            data = np.array(goal_map[e].nonzero()).T
            c.fit(data)

            # mask all points not in the largest cluster
            mode = scipy.stats.mode(c.labels_, keepdims=False).mode.item()
            mode_mask = (c.labels_ != mode).nonzero()
            x = data[mode_mask]
            goal_map_ = np.copy(goal_map[e])
            goal_map_[x] = 0.0

            # adopt masked map if non-empty
            if goal_map_.sum() > 0:
                goal_map[e] = goal_map_

        return np.ceil(goal_map)
