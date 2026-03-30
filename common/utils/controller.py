import itertools
from pathlib import Path
import ai2thor.platform
from ai2thor.controller import Controller
from .scene_utils import get_scene_type, scene_name_to_scene_spec
import numpy as np


def get_scene_bounds(controller) -> tuple[float, float, float, float]:
    scene_objects = controller.last_event.metadata["objects"]
    min_x, max_x = float("inf"), float("-inf")
    min_z, max_z = float("inf"), float("-inf")

    for obj in scene_objects:
        try:
            o_min_x = min([v[0] for v in obj["axisAlignedBoundingBox"]["cornerPoints"]])
            o_max_x = max([v[0] for v in obj["axisAlignedBoundingBox"]["cornerPoints"]])
            o_min_z = min([v[2] for v in obj["axisAlignedBoundingBox"]["cornerPoints"]])
            o_max_z = max([v[2] for v in obj["axisAlignedBoundingBox"]["cornerPoints"]])
            min_x = min(min_x, o_min_x)
            max_x = max(max_x, o_max_x)
            min_z = min(min_z, o_min_z)
            max_z = max(max_z, o_max_z)
        except:
            pass
    return (min_x, max_x, min_z, max_z)


def get_grid_bounds(controller, grid_size) -> tuple[float, float, float, float]:

    (min_x, max_x, min_z, max_z) = get_scene_bounds(controller)
    all_pos_reachable2d = list(
        itertools.product(
            np.arange(
                int((grid_size + min_x) / grid_size) * grid_size,
                max_x,
                grid_size,
            ),
            np.arange(
                int((grid_size + min_z) / grid_size) * grid_size,
                max_z,
                grid_size,
            ),
        )
    )

    grid_min_x, grid_max_x = min([c[0] for c in all_pos_reachable2d]), max(
        [c[0] for c in all_pos_reachable2d]
    )
    grid_min_z, grid_max_z = min([c[1] for c in all_pos_reachable2d]), max(
        [c[1] for c in all_pos_reachable2d]
    )
    return (grid_min_x, grid_max_x, grid_min_z, grid_max_z)






def setup_controller(
    scene_name: str,
    img_shape: tuple[int, int],
    grid_size: float,
    visibility_distance: float,
    rotate_step_degrees: int,
    render_instance_segmentation: bool,
    render_depth_image: bool,
    snap_to_grid: bool,
    continuous: bool,
    cloud_rendering: bool,
    id: int = 0,
    quality: str = "Ultra",
):
    scene_type = get_scene_type(scene_name)

    if scene_type == "procthor-train" or scene_type == "procthor-test":
        scene = scene_name_to_scene_spec(scene_name)
    else:
        scene = scene_name

    if cloud_rendering:
        controller = Controller(
            platform=ai2thor.platform.CloudRendering,
            scene=scene,
            visibilityDistance=visibility_distance,
            gridSize=grid_size,
            height=img_shape[0],
            width=img_shape[1],
            rotateStepDegrees=rotate_step_degrees,
            renderInstanceSegmentation=render_instance_segmentation,
            renderDepthImage=render_depth_image,
            snapToGrid=snap_to_grid,
            continuous=continuous,
            host=f"127.0.0.{id}",
            quality=quality,
        )

    else:
        controller = Controller(
            scene=scene,
            gridSize=grid_size,
            height=img_shape[0],
            width=img_shape[1],
            visibilityDistance=visibility_distance,
            rotateStepDegrees=rotate_step_degrees,
            renderInstanceSegmentation=render_instance_segmentation,
            renderDepthImage=render_depth_image,
            snapToGrid=snap_to_grid,
            continuous=continuous,
            host=f"127.0.0.{id}",
            quality=quality,
        )

    controller.scene_name = scene_name
    controller.grid_size = grid_size
    controller.yaw_bins = int(360 / rotate_step_degrees)
    controller.pitch_bins = 4
    controller.min_pitch = -30
    controller.max_pitch = 60


    (min_x, max_x, min_z, max_z) = get_scene_bounds(controller)
    (grid_min_x, grid_max_x, grid_min_z, grid_max_z) = get_grid_bounds(
        controller, controller.grid_size
    )
    controller.scene_bounds = (min_x, max_x, min_z, max_z)
    controller.grid_bounds = (grid_min_x, grid_max_x, grid_min_z, grid_max_z)

    all_pos_reachable2d = list(
        itertools.product(
            np.arange(
                int((controller.grid_size + min_x) / controller.grid_size) * controller.grid_size,
                max_x,
                controller.grid_size,
            ),
            np.arange(
                int((controller.grid_size + min_z) / controller.grid_size) * controller.grid_size,
                max_z,
                controller.grid_size,
            ),
        )
    )

    grid_cols = int((grid_max_x - grid_min_x) / controller.grid_size) + 1
    grid_rows = int((grid_max_z - grid_min_z) / controller.grid_size) + 1
    controller.grid_rows = grid_rows
    controller.grid_cols = grid_cols

    out = controller.step(
        action="GetReachablePositions",
        raise_for_failure=True,
    ).metadata["actionReturn"]

    reachable_positions = set(
        (round(p["x"] / controller.grid_size) * controller.grid_size, round(p["z"] / controller.grid_size) * controller.grid_size)
        for p in out
    )

    unreachable_positions = set(all_pos_reachable2d).difference(reachable_positions)

    grid_reachable_positions = [
        (
            round((pos[0] - controller.grid_bounds[0]) / controller.grid_size),
            round((pos[1] - controller.grid_bounds[2]) / controller.grid_size),
        )
        for pos in reachable_positions
    ]
    grid_unreachable_positions = [
        (
            round((pos[0] - controller.grid_bounds[0]) / controller.grid_size),
            round((pos[1] - controller.grid_bounds[2]) / controller.grid_size),
        )
        for pos in unreachable_positions
    ]
    controller.grid_reachable_positions = grid_reachable_positions
    controller.grid_unreachable_positions = grid_unreachable_positions
    return controller
