# https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/navigation_planner/fmm_planner.py
# https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/navigation_planner/discrete_planner.py
import math
import os
import shutil
import time
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology


from common.utils import pose_utils as pu
from common.utils.pose_utils import normalize_angle
from common.hybrid_exploration.fmm_planner import FMMPlanner
from dataclasses import dataclass
from enum import Enum

# enum DiscreteNavigationAction
class DiscreteNavigationAction(Enum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    MOVE_FORWARD = "move_forward"
    STOP = "stop"

@dataclass
class ContinuousNavigationAction:
    xyt: List[float]  # [x, y, theta] in meters and radians

CM_TO_METERS = 0.01


def add_boundary(mat: np.ndarray, value=1) -> np.ndarray:
    h, w = mat.shape
    new_mat = np.zeros((h + 2, w + 2)) + value
    new_mat[1 : h + 1, 1 : w + 1] = mat
    return new_mat


def remove_boundary(mat: np.ndarray, value=1) -> np.ndarray:
    return mat[value:-value, value:-value]




class DiscretePlanner:
    """
    This class translates planner inputs into a discrete low-level action
    using an FMM planner.

    This is a wrapper used to navigate to a particular object/goal location.
    """

    collision_map: Optional[np.ndarray]
    visited_map: Optional[np.ndarray] 
    collision_width: int
    fmm_distance: Optional[np.ndarray]
    last_action : Optional[DiscreteNavigationAction]

    def __init__(
        self,
        turn_angle: float,
        collision_threshold: float,
        step_size: int,
        obs_dilation_selem_radius: int,
        goal_dilation_selem_radius: int,
        map_size_cm: int,
        map_resolution: int,
        dump_location: str,
        exp_name: str,
        min_goal_distance_cm: float = 50.0,
        min_obs_dilation_selem_radius: int = 1,
        agent_cell_radius: int = 1,
        map_downsample_factor: float = 1.0,
        map_update_frequency: int = 1,
        goal_tolerance: float = 0.01,
        continuous_angle_tolerance: float = 30.0,
    ) -> None:
        """
        Arguments:
            turn_angle (float): agent turn angle (in degrees)
            collision_threshold (float): forward move distance under which we
             consider there's a collision (in meters)
            obs_dilation_selem_radius: radius (in cells) of obstacle dilation
             structuring element
            obs_dilation_selem_radius: radius (in cells) of goal dilation
             structuring element
            map_size_cm: global map size (in centimeters)
            map_resolution: size of map bins (in centimeters)
            visualize: if True, render planner internals for debugging
        """
        self.default_vis_dir = f"{dump_location}/images/{exp_name}"
        os.makedirs(self.default_vis_dir, exist_ok=True)

        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.map_shape = (
            self.map_size_cm // self.map_resolution,
            self.map_size_cm // self.map_resolution,
        )
        self.turn_angle = turn_angle
        self.collision_threshold = collision_threshold
        self.step_size = step_size
        self.start_obs_dilation_selem_radius = obs_dilation_selem_radius
        self.goal_dilation_selem_radius = goal_dilation_selem_radius
        self.min_obs_dilation_selem_radius = min_obs_dilation_selem_radius
        self.agent_cell_radius = agent_cell_radius
        self.goal_tolerance = goal_tolerance
        self.continuous_angle_tolerance = continuous_angle_tolerance

        self.vis_dir = None

        self.collision_map = None
        self.visited_map = None
        self.fmm_distance = None

        self.last_pose = None
        self.curr_pose = None
        self.last_action = None

        self.timestep = 0
        self.collision_width = 1

        self.curr_obs_dilation_selem_radius = None
        self.obs_dilation_selem = None
        self.min_goal_distance_cm = min_goal_distance_cm

        self.map_downsample_factor = map_downsample_factor
        self.map_update_frequency = map_update_frequency

    def reset(self) -> None:
        self.vis_dir = self.default_vis_dir
        self.collision_map = np.zeros(self.map_shape)
        self.visited_map = np.zeros(self.map_shape)
        self.collision_width = 1
        self.last_pose = None
        self.curr_pose = [
            self.map_size_cm / 100.0 / 2.0,
            self.map_size_cm / 100.0 / 2.0,
            0.0,
        ]
        self.last_action = None
        self.timestep = 1
        self.curr_obs_dilation_selem_radius = self.start_obs_dilation_selem_radius
        self.obs_dilation_selem = skimage.morphology.disk(
            self.curr_obs_dilation_selem_radius
        )
        self.goal_dilation_selem = skimage.morphology.disk(
            self.goal_dilation_selem_radius
        )

        self.vis_dir = os.path.join(self.default_vis_dir)
        shutil.rmtree(self.vis_dir, ignore_errors=True)
        os.makedirs(self.vis_dir, exist_ok=True)

    def plan(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        frontier_map: np.ndarray,
        sensor_pose: np.ndarray,
        found_goal: bool,
        debug: bool = True,
        use_dilation_for_stg: bool = False,
    ) -> Tuple[DiscreteNavigationAction, np.ndarray, Tuple[int, int], np.ndarray]:
        """Plan a low-level action.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gx1, gx2, gy1, gy2)
            found_goal: whether we found the object goal category

        Returns:
            action: low-level action
            closest_goal_map: (M, M) binary array denoting closest goal location in the goal map in geodesic distance
            stg: short-term goal position (x, y) in map
            dilated_obstacles: (M, M) binary array denoting dilated obstacles
        """
        assert self.collision_map is not None, "Obstacle map is None"
        assert self.visited_map is not None, "Goal map is None"

        assert obstacle_map.dtype == np.uint8, f"Obstacle map must be of type np.uint8, got {obstacle_map.dtype}"
        assert goal_map.dtype == np.uint8, f"Goal map must be of type np.uint8, got {goal_map.dtype}"
        assert frontier_map.dtype == np.uint8, f"Frontier map must be of type np.uint8, got {frontier_map.dtype}"

        self.last_pose = self.curr_pose

        # set start and planning window 
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = sensor_pose
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        start = [
            int(start_y * 100.0 / self.map_resolution - gx1),
            int(start_x * 100.0 / self.map_resolution - gy1),
        ]
        if start[0] < 0 or start[0] >= obstacle_map.shape[0] or start[1] < 0 or start[1] >= obstacle_map.shape[1]:
            raise ValueError(
                f"Start position {start} is out of bounds for obstacle map of shape {obstacle_map.shape}"
            )
        start = np.array(start)

        self.curr_pose = [start_x, start_y, start_o]
        self.visited_map[gx1:gx2, gy1:gy2][
            start[0] - 0 : start[0] + 1, start[1] - 0 : start[1] + 1
        ] = 1

        # check collisions if we have just moved and are uncertain
        if self.last_action == DiscreteNavigationAction.MOVE_FORWARD or (
            type(self.last_action) == ContinuousNavigationAction
            and np.linalg.norm(self.last_action.xyt[:2]) > 0
        ):
            self._check_collision()

        # Extracts a local waypoint
        (
            short_term_goal,
            closest_goal_map,
            replan,
            stop,
            closest_goal_pt,
            dilated_obstacles,
        ) = self._get_short_term_goal(
            obstacle_map=obstacle_map,
            goal_map=np.copy(goal_map),
            start=start,
            planning_window=planning_window,
            plan_to_dilated_goal=use_dilation_for_stg,
            frontier_map=frontier_map,
        )

        # if returned 'replan', retry with a smaller obstacle dilation radius
        if replan and not stop:
            assert self.curr_obs_dilation_selem_radius is not None, "curr_obs_dilation_selem_radius is None"
            self.collision_map *= 0
            if self.curr_obs_dilation_selem_radius > self.min_obs_dilation_selem_radius:
                self.curr_obs_dilation_selem_radius -= 1
                self.obs_dilation_selem = skimage.morphology.disk(
                    self.curr_obs_dilation_selem_radius
                )
                if debug:
                    print(
                        f"reduced obs dilation to {self.curr_obs_dilation_selem_radius}"
                    )

            if found_goal:
                (
                    short_term_goal,
                    closest_goal_map,
                    replan,
                    stop,
                    closest_goal_pt,
                    dilated_obstacles,
                ) = self._get_short_term_goal(
                    obstacle_map=obstacle_map,
                    goal_map=goal_map,
                    start=start,
                    planning_window=planning_window,
                    frontier_map=frontier_map,
                    plan_to_dilated_goal=True,
                )

                found_goal = False
                
                if replan:
                    print("Nowhere left to explore. Stopping.")
                    # Calling the STOP action here will cause the agent to try grasping
                    #  TODO separate out STOP_SUCCESS and STOP_FAILURE actions
                    return (
                        DiscreteNavigationAction.STOP,
                        closest_goal_map,
                        short_term_goal,
                        dilated_obstacles,
                    )

        # Normalize agent angle
        angle_agent = normalize_angle(start_o)

        # If we found a short term goal worth moving towards...
        stg_x, stg_y = short_term_goal
        relative_stg_x, relative_stg_y = stg_x - start[0], stg_y - start[1]
        angle_st_goal = math.degrees(math.atan2(relative_stg_x, relative_stg_y))
        relative_angle_to_stg = normalize_angle(angle_agent - angle_st_goal)

        # Compute angle to the final goal
        goal_x, goal_y = closest_goal_pt
        angle_goal = math.degrees(math.atan2(goal_x - start[0], goal_y - start[1]))
        relative_angle_to_closest_goal = normalize_angle(angle_agent - angle_goal)

        action = self.get_action(
            relative_stg_x,
            relative_stg_y,
            relative_angle_to_stg,
            relative_angle_to_closest_goal,
            start_o,
            found_goal,
            stop,
            debug,
        )

        self.last_action = action
        return action, closest_goal_map, short_term_goal, dilated_obstacles

    def get_action(
        self,
        relative_stg_x: float,
        relative_stg_y: float,
        relative_angle_to_stg: float,
        relative_angle_to_closest_goal: float,
        start_compass: float,
        found_goal: bool,
        stop: bool,
        debug: bool,
    ) -> DiscreteNavigationAction:
        # Short-term goal -> deterministic local policy
        if not (found_goal and stop):
            if relative_angle_to_stg > self.turn_angle / 2.0:
                action = DiscreteNavigationAction.TURN_RIGHT
            elif relative_angle_to_stg < -self.turn_angle / 2.0:
                action = DiscreteNavigationAction.TURN_LEFT
            else:
                action = DiscreteNavigationAction.MOVE_FORWARD

        else:
            action = DiscreteNavigationAction.STOP

        return action

    def _get_short_term_goal(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        start: np.ndarray,
        planning_window: List[int],
        frontier_map,
        plan_to_dilated_goal: bool = False,
        visualize=False,
    ) -> Tuple[Tuple[int, int], np.ndarray, bool, bool, tuple[int, int], np.ndarray]:
        """Get short-term goal.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            start: start location (x, y)
            planning_window: local map boundaries (gx1, gx2, gy1, gy2)
            plan_to_dilated_goal: for objectnav; plans to dialted goal points instead of explicitly checking reach.

        Returns:
            short_term_goal: short-term goal position (x, y) in map
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
            replan: binary flag to indicate we couldn't find a plan to reach
             the goal
            stop: binary flag to indicate we've reached the goal
            closest_goal_pt: closest goal point (x, y) in map
            dilated_obstacles: (M, M) binary array denoting dilated obstacles
        """
        assert self.collision_map is not None, "Obstacle map is None"
        assert self.visited_map is not None, "Goal map is None"

        assert obstacle_map.dtype == np.uint8, f"Obstacle map must be of type np.uint8, got {obstacle_map.dtype}"
        assert goal_map.dtype == np.uint8, f"Goal map must be of type np.uint8, got {goal_map.dtype}"
        assert frontier_map.dtype == np.uint8, f"Frontier map must be of type np.uint8, got {frontier_map.dtype}"

        gx1, gx2, gy1, gy2 = planning_window
        (x1, y1,) = (0,0)
        x2, y2 = obstacle_map.shape
        obstacles = obstacle_map[x1:x2, y1:y2]

        # Dilate obstacles
        dilated_obstacles = cv2.dilate(obstacles, self.obs_dilation_selem, iterations=1) # type:ignore

        # Create inverse map of obstacles - this is territory we assume is traversible
        # Traversible is now the map
        traversible = 1 - dilated_obstacles
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[self.visited_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 1
        agent_rad = self.agent_cell_radius
        traversible[
            int(start[0] - x1) - agent_rad : int(start[0] - x1) + agent_rad + 1,
            int(start[1] - y1) - agent_rad : int(start[1] - y1) + agent_rad + 1,
        ] = 1
        traversible = add_boundary(traversible)
        goal_map = add_boundary(goal_map, value=0)
        fmm_planner = FMMPlanner(
            traversible,
            step_size=self.step_size,
            vis_dir=self.vis_dir,
            visualize=False,
            print_images=False,
            goal_tolerance=self.goal_tolerance,
        )

        if plan_to_dilated_goal:
            # Compute dilated goal map for use with simulation code - use this to compute closest goal
            dilated_goal_map = cv2.dilate(
                goal_map, self.goal_dilation_selem, iterations=1
            )
            # Set multi goal to the dilated goal map
            # We will now try to find a path to any of these spaces
            self.fmm_distance = fmm_planner.set_multi_goal(
                dilated_goal_map,
                self.timestep,
                self.fmm_distance,
                self.map_downsample_factor,
                self.map_update_frequency,
            )
            
            assert self.fmm_distance is not None
            goal_distance_map, closest_goal_pt = self.get_closest_traversible_goal(
                traversible, goal_map, start, dilated_goal_map=dilated_goal_map
            )
        else:
            raise NotImplementedError

        self.timestep += 1

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        # This is where we create the planner to get the trajectory to this state
        stg_x, stg_y, replan, stop = fmm_planner.get_short_term_goal(
            state, continuous=False
        )
        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
        short_term_goal = int(stg_x), int(stg_y)

        return (
            short_term_goal,
            goal_distance_map,
            replan,
            stop,
            closest_goal_pt,
            dilated_obstacles,
        )

    def get_closest_traversible_goal(
        self, traversible, goal_map, start, dilated_goal_map=None
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        traversible_ = traversible.copy()
        if dilated_goal_map is None:
            traversible_[goal_map == 1] = 1
        else:
            traversible_[dilated_goal_map == 1] = 1

        vis_planner = FMMPlanner(traversible_)
        curr_loc_map = np.zeros_like(goal_map)
        # Update our location for finding the closest goal
        curr_loc_map[start[0], start[1]] = 1
        # curr_loc_map[short_term_goal[0], short_term_goal]1]] = 1
        vis_planner.set_multi_goal(curr_loc_map)
        fmm_dist_ = vis_planner.fmm_dist.copy()
        # find closest point on non-dilated goal map
        goal_map_ = goal_map.copy()
        goal_map_[goal_map_ == 0] = 10000
        fmm_dist_[fmm_dist_ == 0] = 10000
        closest_goal_map = (goal_map_ * fmm_dist_) == (goal_map_ * fmm_dist_).min()
        closest_goal_map = remove_boundary(closest_goal_map)
        closest_goal_pt = np.unravel_index(
            closest_goal_map.argmax(), closest_goal_map.shape
        )
        closest_goal_pt = (int(closest_goal_pt[0]), int(closest_goal_pt[1]))
        return closest_goal_map, closest_goal_pt

    def get_closest_goal(self, goal_map, start) -> Tuple[np.ndarray, Tuple[int, int]]:
        """closest goal, avoiding any obstacles."""
        empty = np.ones_like(goal_map)
        empty_planner = FMMPlanner(empty)
        empty_planner.set_goal(start)
        dist_map = empty_planner.fmm_dist * goal_map
        dist_map[dist_map == 0] = 10000
        closest_goal_map = dist_map == dist_map.min()
        closest_goal_map = remove_boundary(closest_goal_map)
        closest_goal_pt = np.unravel_index(
            closest_goal_map.argmax(), closest_goal_map.shape
        )
        closest_goal_pt = (int(closest_goal_pt[0]), int(closest_goal_pt[1]))
        return closest_goal_map, closest_goal_pt

    def _check_collision(self):
        """Check whether we had a collision and update the collision map."""
        x1, y1, t1 = self.last_pose
        x2, y2, _ = self.curr_pose
        buf = 4
        length = 2

        # You must move at least 5 cm when doing forward actions Otherwise we assume there has been a collision
        if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
            self.collision_width += 2
            if self.collision_width == 7:
                length = 4
                buf = 3
            self.collision_width = min(self.collision_width, 5)
        else:
            self.collision_width = 1

        dist = pu.get_l2_distance(x1, x2, y1, y2)

        # We have a collision
        if dist < self.collision_threshold:
            width = self.collision_width
            for i in range(length):
                for j in range(width):
                    wx = x1 + 0.05 * (
                        (i + buf) * np.cos(np.deg2rad(t1))
                        + (j - width // 2) * np.sin(np.deg2rad(t1))
                    )
                    wy = y1 + 0.05 * (
                        (i + buf) * np.sin(np.deg2rad(t1))
                        - (j - width // 2) * np.cos(np.deg2rad(t1))
                    )
                    r, c = wy, wx
                    r, c = int(r * 100 / self.map_resolution), int(
                        c * 100 / self.map_resolution
                    )
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r, c] = 1


if __name__ == "__main__":
    import numpy as np
    import tempfile

    discrete_planner = DiscretePlanner(
        turn_angle=30,
        collision_threshold=0.05,
        step_size=5,
        obs_dilation_selem_radius=3,
        goal_dilation_selem_radius=3,
        map_size_cm=10 * 100,
        map_resolution=5,
        dump_location="tmp_planner",
        exp_name="single_step",
    )
    discrete_planner.reset()

    # (10 m x 10 m) map
    obstacle_map = np.zeros((200, 200), dtype=np.uint8)
    goal_map = np.zeros_like(obstacle_map, dtype=np.uint8)
    frontier_map = np.zeros_like(obstacle_map, dtype=np.uint8)

    # Two nested square obstacles, with a gap between them.
    obstacle_map[30:170, 30:34] = 1
    obstacle_map[30:170, 166:170] = 1
    obstacle_map[30:34, 30:170] = 1
    obstacle_map[166:170, 30:170] = 1

    obstacle_map[65:135, 65:69] = 1
    obstacle_map[65:135, 131:135] = 1
    obstacle_map[65:69, 65:135] = 1
    obstacle_map[131:135, 65:135] = 1

    # Start in the corridor between the squares; goal inside the inner square.
    start_x, start_y = 45, 100
    goal_map[100, 100] = 1

    sensor_pose = np.array([
        start_x * 0.05, start_y * 0.05, 0.0,
        0, 200, 0, 200
    ])

    action, closest_goal, stg, dilated = discrete_planner.plan(
        obstacle_map, goal_map, frontier_map, sensor_pose,
        found_goal=True, debug=False, use_dilation_for_stg=True
    )

    assert action in DiscreteNavigationAction
    assert closest_goal.shape == obstacle_map.shape
    assert dilated.shape == obstacle_map.shape
    assert obstacle_map[100, 100] == 0
    assert goal_map[100, 100] == 1
    assert obstacle_map[start_y, start_x] == 0

    print("Nested-square test passed:", action, "STG:", stg)

    # save images for visual inspection
    cv2.imwrite(os.path.join(discrete_planner.vis_dir, "obstacle_map.png"), obstacle_map * 255)
    cv2.imwrite(os.path.join(discrete_planner.vis_dir, "goal_map.png"), goal_map * 255)
    cv2.imwrite(os.path.join(discrete_planner.vis_dir, "closest_goal_map.png"), closest_goal * 255)
    cv2.imwrite(os.path.join(discrete_planner.vis_dir, "dilated_obstacles.png"), dilated * 255)

    print("Planner images saved to:", discrete_planner.vis_dir)