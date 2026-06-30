import numpy as np
import habitat_sim
import math
from dataclasses import dataclass
import cv2
from habitat_sim.agent.agent import AgentState

from common.utils.plot_utils import plot_mask
from common.utils.pose_utils import quaternion_from_rpy, rpy_from_quaternion, get_pose

from scipy.ndimage import distance_transform_edt

def cells_in_range(occupancy, min_range, max_range):
    dist = distance_transform_edt(~occupancy)
    return (dist >= min_range) & (dist <= max_range)

def object_in_view(
    row,
    col,
    obj_occupancy,
    obstacles,
    camera_yaw,
    camera_hfov,
    min_dist,
    max_dist,
    min_corner_relative_angle,
    max_corner_relative_angle,
    n_rays=3,
):
    H, W = obj_occupancy.shape
    rays_hits = 0

    min_camera_angle = camera_yaw - np.deg2rad(camera_hfov / 2.0)
    max_camera_angle = camera_yaw + np.deg2rad(camera_hfov / 2.0)
    if min_camera_angle < 0:
        min_camera_angle += 2 * np.pi
        max_camera_angle += 2 * np.pi

    ray_angles = np.linspace(min_corner_relative_angle, max_corner_relative_angle, n_rays, endpoint=True)

    for ray_angle in ray_angles:
        if ray_angle < 0:
            ray_angle += 2 * np.pi

        if ray_angle < min_camera_angle or ray_angle > max_camera_angle:
            continue  # Skip rays that are outside the camera's field of view

        sin_a = np.sin(ray_angle)
        cos_a = np.cos(ray_angle)

        for dist in range(min_dist, max_dist + 1):
            rr = round(row + dist * sin_a)
            cc = round(col + dist * cos_a)

            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                break

            if obstacles[rr, cc]:
                break

            if obj_occupancy[rr, cc]:
                rays_hits += 1
                break  # Stop checking further along this ray if we hit the object
            
    return rays_hits


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
        list_object_info: list[dict],
        turn_angle: float = 30.0
    ):
        ref_y = sim.agents[0].state.position[1]
        height = sim.pathfinder.get_random_navigable_point()[1]

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
        n = max(obj_info["object_id"] for obj_info in list_object_info) + 1
        self.obj_occupancy_td_view = np.zeros((H, W, n), dtype=np.uint8)

        self.corners_2d = {}
        self.obj_class_ids = {}

        for obj_info in list_object_info:
            corners_3d = obj_info["corners"]
            corners_2d = [
                (corners_3d[i][0], corners_3d[i][2]) for i in [0,1,6,7]
            ]
            self.add_object(corners_2d, obj_info['object_id'])
            self.corners_2d[obj_info['object_id']] = corners_2d
            self.obj_class_ids[obj_info['object_id']] = obj_info['class_name']
    
        self.list_object_info = list_object_info
        self.obstacles = (self.topdown_view == 0).astype(bool)
        
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


    def object_is_visible(self, obj_id, agent_state: AgentState, min_depth: float = 0.0, max_depth: float = 3.0, n_rays=3, min_object_fov=5.0, camera_hfov=90.0) -> bool:
        _,_, o = get_pose(agent_state.position, agent_state.rotation)
        camera_yaw = - np.pi / 2 - o
        row, col = self.world_to_grid((agent_state.position[0], agent_state.position[2]), do_round=True)
        
        obj_corners_2d = self.corners_2d[obj_id]
        obj_corner_relative_angles = [np.arctan2(c[1] - agent_state.position[2], c[0] - agent_state.position[0]) for c in obj_corners_2d]
        min_corner_relative_angle = min(obj_corner_relative_angles)
        max_corner_relative_angle = max(obj_corner_relative_angles)

        rays_hits = object_in_view(
            row,
            col,
            self.obj_occupancy_td_view[:, :, obj_id],
            self.obstacles,
            camera_yaw=camera_yaw,
            camera_hfov=camera_hfov,
            min_corner_relative_angle=min_corner_relative_angle,
            max_corner_relative_angle=max_corner_relative_angle,
            max_dist=int(max_depth / self.meters_per_grid_pixel),
            min_dist=int(min_depth / self.meters_per_grid_pixel),
            n_rays=n_rays,
        )
        return (rays_hits / n_rays) * (max_corner_relative_angle - min_corner_relative_angle) >= np.deg2rad(min_object_fov)