import numpy as np
import math
import cv2


class HabitatObjOccupancyGrid:
    def __init__(
        self,
        sim,
        meters_per_grid_pixel: float,
        list_object_info: list[dict],
    ):
        ref_y = sim.agents[0].state.position[1]
        height = sim.pathfinder.get_random_navigable_point()[1]

        b1, b2 = sim.pathfinder.get_bounds()
        startx = min(b1[0], b2[0])
        startz = min(b1[2], b2[2])

        self.ref_point = (startx, ref_y, startz)
        self.meters_per_grid_pixel = meters_per_grid_pixel

        # Topdown occupancy (H, W)
        self.topdown_view = sim.pathfinder.get_topdown_view(
            meters_per_grid_pixel, height=height
        ).astype(np.uint8)

        H, W = self.topdown_view.shape

        # Object Occupancy grid: obj_occupancy_td_view[row][col][obj_id] == 1 if object occupies the cell (row,col)
        n = max(obj_info["object_id"] for obj_info in list_object_info) + 1
        self.obj_occupancy_td_view = np.zeros((H, W, n), dtype=np.uint8)

        for obj_info in list_object_info:
            corners_3d = obj_info["corners"]
            corners_2d = [
                (corners_3d[i][0], corners_3d[i][2]) for i in [0,1,6,7]
            ]
            self.add_object(corners_2d, obj_info['object_id'])

        self.list_object_info = list_object_info
        self.object_info_by_id = {info["object_id"]: info for info in list_object_info}


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