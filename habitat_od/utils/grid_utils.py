import numpy as np
import habitat_sim
import math

from common.utils.plot_utils import plot_mask


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

        self.world_bounds = sim.pathfinder.get_bounds()
        (b1, b2) = self.world_bounds

        startx = min(b1[0], b2[0])
        startz = min(b1[2], b2[2])

        self.ref_point = (startx, self.ref_y, startz)
        self.meters_per_grid_pixel = meters_per_grid_pixel

        # Topdown occupancy (H, W)
        self.topdown_view = sim.pathfinder.get_topdown_view(
            meters_per_grid_pixel, height=self.ref_y
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
            corners = obj_info["global_corners"]
            xmin, _, zmin = corners[0]
            xmax, _, zmax = corners[7]

            self.add_object((xmin, zmin, xmax, zmax), obj_info["class_name"])

    # ----------------------------
    # Grid / World conversions
    # ----------------------------

    def world_to_grid(
        self, point: tuple[float, float], round_up: bool
    ) -> tuple[int, int]:
        x, z = point
        startx, _, startz = self.ref_point

        col = (x - startx) / self.meters_per_grid_pixel
        row = (z - startz) / self.meters_per_grid_pixel

        if round_up:
            return math.ceil(row), math.ceil(col)
        else:
            return math.floor(row), math.floor(col)

    def grid_to_world(self, point: tuple[int, int]) -> tuple[float, float]:
        row, col = point
        startx, _, startz = self.ref_point

        x = startx + col * self.meters_per_grid_pixel
        z = startz + row * self.meters_per_grid_pixel

        return x, z

    # ----------------------------
    # Core utilities
    # ----------------------------

    def is_navigable(self, point: tuple[int, int]) -> bool:
        row, col = point
        return bool(self.topdown_view[row, col])

    def add_object(
        self,
        obj_2d_bb: tuple[float, float, float, float],
        obj_class: str,
    ):
        xmin, zmin, xmax, zmax = obj_2d_bb

        row_min, col_min = self.world_to_grid((xmin, zmin), round_up=False)
        row_max, col_max = self.world_to_grid((xmax, zmax), round_up=True)

        H, W = self.topdown_view.shape
        class_id = self.class_mapping[obj_class]

        for row in range(max(0, row_min), min(H, row_max + 1)):
            for col in range(max(0, col_min), min(W, col_max + 1)):
                self.sem_td_view[row, col, 1 + class_id] = 1

    # ----------------------------
    # Pose generation
    # ----------------------------

    def get_all_poses(self) -> list:
        poses = []

        for (row, col) in self.gridpoints[0:10]:
            x, z = self.grid_to_world((row, col))

            for k in range(int(360 / self.turn_angle)):
                yaw = 2 * np.pi * (k * self.turn_angle / 360)
                poses.append(((x, self.ref_y, z), (0.0, 0.0, yaw)))
                break

        return poses

    def get_all_poses_viewing_class(self, obj_class: str) -> list:
        poses = []

        class_id = self.class_mapping[obj_class]
        occupancy = self.sem_td_view[:, :, 1 + class_id] == 1
        

        for yaw_deg in range(0, 360, self.turn_angle):
            yaw = 2 * np.pi * (yaw_deg / 360)
            
            visibility = array_visibility(
                occupancy_array=occupancy,
                rpy=(0, 0, yaw),
                fov_deg=10,
                min_range=int(1.0 / self.meters_per_grid_pixel),
                max_range=int(1.0 / self.meters_per_grid_pixel) + 1,
            )

            rows, cols = np.where(visibility)

            for row, col in zip(rows, cols):
                if self.is_navigable((row, col)):
                    x, z = self.grid_to_world((row, col))
                    poses.append(((x, self.ref_y, z), (0.0, 0.0, yaw - np.pi / 2)))

        return poses