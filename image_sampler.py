
import numpy as np

from typing import List, Optional, Tuple

import numpy as np
from habitat_sim import registry as registry
from habitat_sim.agent.agent import AgentConfiguration, AgentState


import attr
import numpy as np
from numpy import bool_, float32, float64, ndarray
import quaternion

import habitat_sim
from habitat_sim import registry as registry
import magnum as mn
import math
import random
from transforms3d.euler import euler2quat, quat2euler
from habitat_sim.utils.common import quat_from_angle_axis



def quaternion_from_rpy(roll, yaw, pitch):
    return quat_from_angle_axis(roll, np.array([1, 0.0, 1])) * quat_from_angle_axis(yaw, np.array([0, 1, 0])) * quat_from_angle_axis(pitch, np.array([0, 0, 1]))

class HabitatGrid:
    topdown_view: habitat_sim.nav.PathFinder
    meters_per_pixel: float
    ref_point: tuple[float, float]
    gridpoints: list[Tuple[int,int]]

    def __init__(self, sim, height, meters_per_pixel):
        bound1, bound2 = sim.pathfinder.get_bounds()
        startw = min(bound1[0], bound2[0])
        starth = min(bound1[2], bound2[2])
        starty = height
        self.ref_point = (startw, starty, starth)
        
        self.topdown_view = sim.pathfinder.get_topdown_view(
            meters_per_pixel, 0.1
        ).astype(np.float64)
        self.meters_per_pixel = meters_per_pixel

        rows, cols = self.topdown_view.shape
        self.gridpoints = []

        for row in range(rows):
            for col in range(cols):
                if self.topdown_view[row][col] == 1.0:
                    self.gridpoints.append((row,col))

    def convert_to_scene_coordinate_system(
        self,
        poses: List[Tuple[Tuple[int, int], quaternion.quaternion]],
    ) -> List[Tuple[Tuple[float, float, float], quaternion.quaternion, str]]:
        # Convert from topdown map coordinate system to that of the scene
        startw, starty, starth = self.ref_point

        out = []
        for i, pose in enumerate(poses):
            pos, quaternion_rot = pose
            row, col = pos
            new_pos = np.array(
                [
                    startw + col * self.meters_per_pixel,
                    starty,
                    starth + row * self.meters_per_pixel,
                ]
            )
            new_pos_t: Tuple[int, int] = tuple(new_pos)
            out.append((new_pos_t, quaternion_rot))

        return out
    

class CustomImageSampler:
    def __init__(
        self,
        sim,
        shuffle: bool = False,
        output: Optional[List[str]] = None,
        meters_per_pixel: float = 0.25,
        yaw_bins: int = 12
    ):
        if output is None:
            output = ["rgba"]

        self.sim = sim

        self.habitat_grid = HabitatGrid(sim, 0.5, meters_per_pixel)
        self.poses = []

        all_grid_poses = []
        
        for row,col in self.habitat_grid.gridpoints:
            all_grid_poses.extend([((row,col), quaternion_from_rpy(0.0, 6.28 * yaw / yaw_bins, 0.0))for yaw in range(yaw_bins)])


        scene_poses = self.habitat_grid.convert_to_scene_coordinate_system(all_grid_poses)
        self.poses.extend(scene_poses)


        if shuffle:
            random.shuffle(self.poses)

        self.out_name_to_sensor_name = {
            "rgba": "rgb",
            "depth": "depth",
            "semantic": "semantic",
        }
        self.output = output

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            if start is None:
                start = 0
            if stop is None:
                stop = len(self.poses)
            if step is None:
                step = 1

            return [
                self.__getitem__(i)
                for i in range(start, stop, step)
                if i < len(self.poses)
            ]

        pos, rot = self.poses[idx]
        new_state = AgentState()
        new_state.position = pos
        new_state.rotation = rot
        self.sim.agents[0].set_state(new_state)

        obs = self.sim.get_sensor_observations()
        sample = {
            out_name: obs[self.out_name_to_sensor_name[out_name]]
            for out_name in self.output
        }

        return sample

    def close(self) -> None:
        r"""Deletes the instance of the simulator. Necessary for instantiating a different ImageExtractor."""
        if self.sim is not None:
            self.sim.close()
            del self.sim
            self.sim = None


