import numpy as np
import habitat_sim
import math
from dataclasses import dataclass
import cv2
from habitat_sim.agent.agent import AgentState

from common.utils.plot_utils import plot_mask
from common.utils.pose_utils import quaternion_from_rpy

from scipy.ndimage import distance_transform_edt

def cells_in_range(occupancy, min_range, max_range):
    dist = distance_transform_edt(~occupancy)
    return (dist >= min_range) & (dist <= max_range)

def object_in_view(
    row,
    col,
    obj_occupancy,
    obstacles,
    yaw,
    min_range,
    max_range,
    fov_deg=30.0,
    n_rays=3,
):
    H, W = obj_occupancy.shape

    half_fov = np.deg2rad(fov_deg / 2.0)

    angles = np.linspace(-half_fov, half_fov, n_rays, endpoint=True) + yaw

    for angle in angles:
        sin_a = np.sin(angle)
        cos_a = np.cos(angle)

        for dist in range(min_range, max_range + 1):
            rr = int(round(row + dist * sin_a))
            cc = int(round(col + dist * cos_a))

            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                continue

            if obstacles[rr, cc]:
                continue

            if obj_occupancy[rr, cc]:
                return True

    return False


@dataclass
class HabitatObjOccupancyGrid:
    ref_point: tuple[float,float,float]
    world_bounds: tuple[tuple[float,float,float],tuple[float,float,float]]
    topdown_view: np.ndarray
    obj_occupancy_td_view: np.ndarray

    def __init__(
        self,
        sim,
        meters_per_grid_pixel: float,
        class_mapping: dict[str, int],
        list_object_info: list[dict],
    ):
        ref_y = sim.agents[0].state.position[1]
        self.turn_angle = 30

        navmesh_verts = sim.pathfinder.build_navmesh_vertices(-1)
        height = min(x[1] for x in navmesh_verts)

        self.world_bounds = sim.pathfinder.get_bounds()
        (b1, b2) = self.world_bounds


        startx = min(b1[0], b2[0])
        startz = min(b1[2], b2[2])

        self.ref_point = (startx, ref_y, startz)
        self.meters_per_grid_pixel = meters_per_grid_pixel

        # Topdown occupancy (H, W)
        self.topdown_view = sim.pathfinder.get_topdown_view(
            meters_per_grid_pixel, height=height
        ).astype(np.uint8)

        H, W = self.topdown_view.shape

        # Collect navigable grid points
        self.gridpoints: list[tuple[int, int]] = []
        for row in range(H):
            for col in range(W):
                if self.topdown_view[row, col] == 1.0:
                    self.gridpoints.append((row, col))

        # Object Occupancy grid: obj_occupancy_td_view[row][col][obj_id] == 1 if object occupies the cell (row,col)
        n = len(list_object_info)
        self.obj_occupancy_td_view = np.zeros((H, W, n), dtype=np.uint8)

        # Add objects
        for obj_info in list_object_info:
            corners_3d = obj_info["corners"]
            corners_2d = [
                (corners_3d[i][0], corners_3d[i][2]) for i in [0,1,6,7]
            ]
            self.add_object(corners_2d, obj_info['object_id'])

        self.obstacles = np.zeros_like(self.topdown_view, dtype=bool)
        self.obstacles = np.logical_or(self.obstacles, self.topdown_view == 0)
        
        for i in range(n):
            self.obstacles = np.logical_and(self.obstacles, self.obj_occupancy_td_view[:, :, i] == 0)
        
        self.obstacles = cv2.erode(self.obstacles.astype(np.uint8), np.ones((3, 3)), iterations=2)

    def world_to_grid(
        self, point: tuple[float, float], do_round: bool
    ) -> tuple[int, int]:
        x, z = point
        startx, _, startz = self.ref_point

        col = (x - startx) / self.meters_per_grid_pixel
        row = (z - startz) / self.meters_per_grid_pixel

        if do_round:
            return round(row), round(col)
        else:
            return math.floor(row), math.floor(col)

    def grid_to_world(self, point: tuple[int, int]) -> tuple[float, float]:
        row, col = point
        startx, _, startz = self.ref_point

        x = startx + col * self.meters_per_grid_pixel
        z = startz + row * self.meters_per_grid_pixel

        return x, z


    def is_navigable(self, point: tuple[int, int]) -> bool:
        row, col = point
        return bool(self.topdown_view[row, col])

    def add_object(
        self,
        obj_corners: list[tuple[float, float]],  # [(x1,z1), (x2,z2), (x3,z3), (x4,z4)]
        obj_id: int,
    ):
        """
        Fills the quadrilateral formed by the 4 world-space corners
        into the object occupancy top-down grid.
        """

        def order_polygon_points(pts):
            center = pts.mean(axis=0)
            angles = np.arctan2(
                pts[:,1] - center[1],   # y - cy
                pts[:,0] - center[0]    # x - cx
            )
            return pts[np.argsort(angles)]
        
        if len(obj_corners) != 4:
            raise ValueError("obj_corners must contain exactly 4 corners")

        H, W = self.topdown_view.shape

        grid_pts = []
        for (x,y) in obj_corners:
            row, col = self.world_to_grid((x,y), do_round=True)
            grid_pts.append([col, row])

        pts = np.array(grid_pts, dtype=np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        pts = order_polygon_points(pts)

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1) # type: ignore

        self.obj_occupancy_td_view[:, :, obj_id][mask == 1] = 1


    def get_all_agent_states(self) -> list[AgentState]:
        agent_states = []

        for (row, col) in self.gridpoints:
            x, z = self.grid_to_world((row, col))

            for k in range(int(360 / self.turn_angle)):
                yaw = 2 * np.pi * (k * self.turn_angle / 360)

                new_state = AgentState()
                new_state.position = np.array([x,self.ref_point[1],z], dtype = np.float32)
                new_state.rotation = quaternion_from_rpy(0, 0,  - yaw - np.pi / 2)
                agent_states.append(new_state)

        return agent_states

    def get_all_viewpoints(self, obj_id: int, visibility_range: tuple[float, float] = (0.5, 2.0), viewpoint_spacing: float = 0.25) -> list[AgentState]:
        agent_states = []

        obj_occupancy = self.obj_occupancy_td_view[:, :, obj_id] == 1

        min_range = visibility_range[0] / self.meters_per_grid_pixel
        max_range = visibility_range[1] / self.meters_per_grid_pixel

        in_range = cells_in_range(obj_occupancy, min_range, max_range)
        in_range_downsampled = np.zeros_like(in_range, dtype=bool)
        
        downsampling_step = max(1,int(viewpoint_spacing/self.meters_per_grid_pixel))
        in_range_downsampled[::downsampling_step, ::downsampling_step] = in_range[::downsampling_step, ::downsampling_step]

        rows, cols = np.where(in_range_downsampled)

        for yaw_idx, yaw in enumerate([2 * np.pi * (yaw_deg / 360) for yaw_deg in range(0, 360, self.turn_angle)]):    
            for row, col in zip(rows, cols):
                if not self.is_navigable((row, col)):
                    continue
                
                if not object_in_view(
                    row=row,
                    col=col,
                    obj_occupancy=obj_occupancy,
                    obstacles=self.obstacles,
                    yaw=yaw,
                    min_range=int(visibility_range[0] / self.meters_per_grid_pixel),
                    max_range=int(visibility_range[1] / self.meters_per_grid_pixel),
                    fov_deg=30.0,
                    n_rays=3,
                ):
                    continue
                
                x, z = self.grid_to_world((row, col))
                new_state = AgentState()
                new_state.position = np.array([x,self.ref_point[1],z], dtype = np.float32)
                new_state.rotation = quaternion_from_rpy(0, 0, - yaw - np.pi / 2)
                agent_states.append(new_state)

        return agent_states