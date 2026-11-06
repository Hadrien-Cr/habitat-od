"""
Corrupts the ground-truth semantic mask of one procthor_scene.py ProcTHOR-Train-0 object per case into a degenerate one:
the garbage can standing 0.24 m off the outer wall gets the convex hull of its mask (covering the wall seen through and
around it), the counter top and the wine bottle on it get blobs of noise (random pixels of their mask, grown by
dilation, smaller for the bottle) spilling over what lies around them. Each is viewed diagonally, from a horizontal
offset to its mesh's center, snapped onto the navmesh.
The degenerate mask is projected with depth and voxel-downsampled (ConceptGraphConfig's voxel size), its floor and the
walls behind it are flagged with projection_utils' floor_mask and background_wall_mask, and concept-graphs' own
per-detection denoising runs on the rest: main_cluster_mask, called with the same detection_dbscan_eps/
detection_dbscan_min_points/dbscan_eps/detection_max_attached_fraction concept_graph.py's processed_detections uses,
keeps the largest DBSCAN cluster plus any smaller one attached to it. Into tests/testdump/test_denoise/<case>/ go the
view's rgb.png, clusters.ply (floor in brown, walls in gray, noise in black and every other point colored by its
DBSCAN cluster, before denoise drops everything but the kept clusters), degenerate_mask.png, denoised.ply (the points
denoise kept) and raytraced_mask.png, the cleaned mask they raytrace in the view
(ConceptGraphResolvedObjects.visible_objects).
"""

import math
import os
import shutil
from dataclasses import dataclass
from functools import partial
from typing import Callable

import habitat_sim
import numpy as np
import open3d as o3d
import pytest
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import KDTree

from common.env_utils.object_annotations import object_world_mesh
from common.env_utils.vocab_constants import make_colors
from common.utils.concept_graph import ConceptGraphConfig, ConceptGraphResolvedObjects
from common.utils.eval_utils import _box_iou_3d
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import background_wall_mask, dbscan_labels, downsample_points, floor_mask, main_cluster_mask, object_points
from common.utils.retrieval import SegmentationStyle, instances_from_masks, segmentation_tile
from common.utils.spot import fill_convex_hull
from procthor_scene import DATASET_CONFIG, DEPTH_CAMERA, floor_height, procthor_sim

pytestmark = pytest.mark.skipif(not os.path.exists(DATASET_CONFIG), reason="requires HABITAT_DATA pointing at a real ai2thor-hab dataset (see INSTALL.MD)")

_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_denoise")
_FLOOR_COLOR = (140, 90, 40)
_WALL_COLOR = (160, 160, 160)
_NOISE_COLOR = (0, 0, 0)
_MASK_COLOR = (30, 60, 200)
_NUM_BLOBS = 12
_TABLE_BLOB_RADIUS_PX = 8
_BOTTLE_BLOB_RADIUS_PX = 3
_MIN_GT_BOX_IOU = 0.3


def _convex_hull(mask: np.ndarray) -> np.ndarray:
    return fill_convex_hull(mask.astype(np.uint8)).astype(bool)


def _with_blobs(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """mask plus _NUM_BLOBS random pixels of it, each grown by radius_px."""
    seeds = np.zeros_like(mask)
    seeds.flat[np.random.default_rng(0).choice(np.flatnonzero(mask), _NUM_BLOBS, replace=False)] = True
    return mask | ndimage.binary_dilation(seeds, iterations=radius_px)


@dataclass(frozen=True)
class _DenoiseCase:
    name: str
    handle: str
    view_offset: tuple
    degenerate_mask: Callable[[np.ndarray], np.ndarray]


_CASES = [
    _DenoiseCase("garbage_can_hull", "bin_25", (1.6, 1.6), _convex_hull),
    _DenoiseCase("counter_top_blobs", "Countertop_I_8x2", (-1.6, -1.6), partial(_with_blobs, radius_px=_TABLE_BLOB_RADIUS_PX)),
    _DenoiseCase("wine_bottle_blobs", "Wine_Bottle_1", (-1.2, -1.2), partial(_with_blobs, radius_px=_BOTTLE_BLOB_RADIUS_PX)),
]


def _object_view(sim: habitat_sim.Simulator, obj, view_offset: tuple) -> tuple:
    """(observations, depth sensor pose, floor height) from view_offset to obj's mesh center, snapped onto the navmesh,
    facing obj."""
    center_x, _, center_z = object_world_mesh(obj).bounds.mean(axis=0)
    position = np.array(sim.pathfinder.snap_point(np.array([center_x + view_offset[0], 0.0, center_z + view_offset[1]])))
    yaw = math.atan2(position[0] - center_x, position[2] - center_z)
    state = habitat_sim.AgentState()
    state.position = position
    state.rotation = np.quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
    agent = sim.get_agent(0)
    agent.set_state(state)
    return sim.get_sensor_observations(), sensor_pose_matrix(agent.get_state().sensor_states["depth"]), floor_height(sim, position)


def _point_colors(floor: np.ndarray, walls: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """0-1 colors of floor, walls and the DBSCAN labels of the other points."""
    palette = np.array([*make_colors(labels.max() + 1, seed=0, ctype=0), _NOISE_COLOR])
    colors = np.tile(np.array(_FLOOR_COLOR, dtype=float), (len(floor), 1))
    colors[walls] = _WALL_COLOR
    colors[~floor & ~walls] = palette[labels]
    return colors / 255.0


def _save_ply(points: np.ndarray, colors: np.ndarray, out_path: str) -> None:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_path, pcd)


def _aligned_box(points: np.ndarray) -> np.ndarray:
    """(1, 2, 3) (min, max) box of points, eval_utils' own _box_iou_3d input shape."""
    return np.stack([points.min(axis=0), points.max(axis=0)])[None]


def _raytraced_mask(points: np.ndarray, depth: np.ndarray, pose: np.ndarray) -> torch.Tensor:
    """Mask of points raytraced in depth from pose as a single resolved object."""
    resolved = ConceptGraphResolvedObjects()
    resolved.resolved_points, resolved.resolved_infos, resolved.kdtree = points, np.zeros((len(points), 2), dtype=int), KDTree(points)
    masks, _, _ = resolved.visible_objects(depth, pose, DEPTH_CAMERA.hfov, DEPTH_CAMERA.min_depth, DEPTH_CAMERA.max_depth, min_mesh_visibility_fraction=0.0)
    return masks[0]


def _save_mask(rgb: np.ndarray, mask: torch.Tensor, class_name: str, out_path: str) -> None:
    instances = instances_from_masks(torch.as_tensor(mask)[None], torch.zeros(1), mask.shape)
    Image.fromarray(segmentation_tile(rgb, instances, SegmentationStyle([class_name], [_MASK_COLOR]), "")[1]).save(out_path)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_denoise_on_degenerate_mask(case: _DenoiseCase):
    out_dir = os.path.join(_TESTDUMP_DIR, case.name)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    sim, annotation = procthor_sim()
    obj = next(iter(sim.get_rigid_object_manager().get_objects_by_handle_substring(case.handle).values()))
    obs, pose, floor = _object_view(sim, obj, case.view_offset)
    semantic_id, class_name = obj.visual_scene_nodes[0].semantic_id, annotation.classes[annotation.obj_id_to_class_id[obj.object_id]]
    sim.close()
    rgb, object_mask = obs["rgb"][:, :, :3], obs["semantic"] == semantic_id
    assert object_mask.any(), f"the {class_name} is not in view"

    degenerate_mask = case.degenerate_mask(object_mask)
    config = ConceptGraphConfig()
    points = downsample_points(object_points(obs["depth"], pose, degenerate_mask, DEPTH_CAMERA), config.downsample_voxel_size)
    on_floor = floor_mask(points, floor)
    walls = np.zeros(len(points), dtype=bool)
    walls[~on_floor] = background_wall_mask(points[~on_floor], pose[:3, 3])
    foreground = points[~on_floor & ~walls]
    labels = dbscan_labels(foreground, config.detection_dbscan_eps, config.detection_dbscan_min_points)
    kept = main_cluster_mask(
        foreground, config.detection_dbscan_eps, config.detection_dbscan_min_points, config.dbscan_eps, config.detection_max_attached_fraction,
    )

    Image.fromarray(rgb).save(os.path.join(out_dir, "rgb.png"))
    _save_ply(points, _point_colors(on_floor, walls, labels), os.path.join(out_dir, "clusters.ply"))
    _save_mask(rgb, degenerate_mask, class_name, os.path.join(out_dir, "degenerate_mask.png"))
    _save_ply(foreground[kept], np.tile(np.array(_MASK_COLOR) / 255.0, (kept.sum(), 1)), os.path.join(out_dir, "denoised.ply"))
    cleaned_mask = _raytraced_mask(foreground[kept], obs["depth"], pose)
    _save_mask(rgb, cleaned_mask, class_name, os.path.join(out_dir, "raytraced_mask.png"))

    gt_points = object_points(obs["depth"], pose, object_mask, DEPTH_CAMERA)
    box_iou = float(_box_iou_3d(_aligned_box(foreground[kept]), _aligned_box(gt_points))[0, 0])
    assert box_iou >= _MIN_GT_BOX_IOU, f"denoised box barely overlaps the ground-truth {class_name} box (IoU={box_iou:.2f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
