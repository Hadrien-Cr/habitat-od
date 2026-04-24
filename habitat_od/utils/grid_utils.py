import numpy as np
import habitat_sim
import math

import cv2
from habitat_sim.agent.agent import AgentState
from habitat_od.utils.plot_utils import plot_mask
from habitat_od.utils.pose_utils import quaternion_from_rpy

def array_visibility(
    occupancy_array: np.ndarray,
    rpy: tuple[float, float, float],
    fov_deg: float = 90.0,
    min_range: int = 5,
    max_range: int = 10,
) -> np.ndarray:
    """
    Returns (H, W) boolean mask:
    True at (row, col) if a pixel with occupancy==1 is visible
    from that position given yaw/fov/range.
    """
    _, _, yaw = rpy
    half_fov = np.deg2rad(fov_deg / 2.0)

    # Locations of occupied cells
    cls_rows, cls_cols = np.where(occupancy_array)

    if len(cls_rows) == 0:
        return np.zeros_like(occupancy_array, dtype=bool)

    H, W = occupancy_array.shape

    # Observer grid
    obs_rows, obs_cols = np.mgrid[0:H, 0:W]

    # Displacements (broadcasted)
    dcol = cls_cols[None, None, :] - obs_cols[:, :, None]
    drow = cls_rows[None, None, :] - obs_rows[:, :, None]

    dist_sq = dcol**2 + drow**2

    in_range = (min_range**2 <= dist_sq) & (dist_sq <= max_range**2)

    angles = np.arctan2(drow, dcol)
    delta = (angles - yaw - np.pi) % (2 * np.pi) - np.pi
    in_cone = np.abs(delta) <= half_fov

    visible = np.any(in_range & in_cone, axis=-1)

    return visible


class HabitatSemGrid:
    def __init__(
        self,
        sim,
        meters_per_grid_pixel: float,
        class_mapping: dict[str, int],
        list_object_info: list[dict],
    ):
        self.ref_y = sim.agents[0].state.position[1]
        self.turn_angle = 30

        navmesh_verts = sim.pathfinder.build_navmesh_vertices(-1)
        height = min(x[1] for x in navmesh_verts)

        self.world_bounds = sim.pathfinder.get_bounds()
        (b1, b2) = self.world_bounds

        startx = min(b1[0], b2[0])
        startz = min(b1[2], b2[2])

        self.ref_point = (startx, self.ref_y, startz)
        self.meters_per_grid_pixel = meters_per_grid_pixel

        # Topdown occupancy (H, W)
        self.topdown_view = sim.pathfinder.get_topdown_view(
            meters_per_grid_pixel, height=height
        ).astype(np.float64)

        self.class_mapping = class_mapping

        H, W = self.topdown_view.shape

        # Semantic grid: channel 0 = non-navigable, others = classes
        self.sem_td_view = np.zeros((H, W, len(class_mapping) + 1), dtype=np.uint8)

        # Collect navigable grid points
        self.gridpoints: list[tuple[int, int]] = []

        for row in range(H):
            for col in range(W):
                if self.topdown_view[row, col] == 1.0:
                    self.gridpoints.append((row, col))
                else:
                    self.sem_td_view[row, col, 0] = 1

        # Add objects
        for obj_info in list_object_info:
            corners_3d = obj_info["corners"]
            corners_2d = [
                (corners_3d[i][0], corners_3d[i][2]) for i in [0,1,6,7]
            ]
            self.add_object(corners_2d, obj_info["class_name"])

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
        obj_class: str,
    ):
        """
        Fills the quadrilateral formed by the 4 world-space corners
        into the semantic top-down grid.
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
        class_id = self.class_mapping[obj_class]

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

        self.sem_td_view[:, :, 1 + class_id][mask == 1] = 1


    def get_all_agent_states(self) -> list[AgentState]:
        agent_states = []

        for (row, col) in self.gridpoints:
            x, z = self.grid_to_world((row, col))

            for k in range(int(360 / self.turn_angle)):
                yaw = 2 * np.pi * (k * self.turn_angle / 360)

                new_state = AgentState()
                new_state.position = np.array([x,self.ref_y,z], dtype = np.float32)
                new_state.rotation = quaternion_from_rpy(0, 0,  yaw - np.pi / 2)
                agent_states.append(new_state)

        return agent_states

    def get_all_agent_states_viewing_class(self, obj_class: str, visibility_range: tuple[float, float] = (0.5, 2.0)) -> list[AgentState]:
        agent_states = []

        class_id = self.class_mapping[obj_class]
        occupancy = self.sem_td_view[:, :, 1 + class_id] == 1
        
        for yaw_deg in range(0, 360, self.turn_angle):
            yaw = 2 * np.pi * (yaw_deg / 360)
            
            visibility = array_visibility(
                occupancy_array=occupancy,
                rpy=(0, 0, yaw),
                fov_deg=10,
                min_range=int(visibility_range[0] / self.meters_per_grid_pixel),
                max_range=int(visibility_range[1] / self.meters_per_grid_pixel),
            )

            rows, cols = np.where(visibility)

            for row, col in zip(rows, cols):
                if self.is_navigable((row, col)):
                    x, z = self.grid_to_world((row, col))

                    new_state = AgentState()
                    new_state.position = np.array([x,self.ref_y,z], dtype = np.float32)
                    new_state.rotation = quaternion_from_rpy(0, 0,  yaw - np.pi / 2)
                    agent_states.append(new_state)

        return agent_states