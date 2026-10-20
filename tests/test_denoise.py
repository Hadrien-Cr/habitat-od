"""
Corrupts the ground-truth semantic mask of one test_dummy_scene.py object per case into a degenerate one: the chair
standing 0.1 m off the outer wall gets the convex hull of its mask (covering the wall seen through and around it), the
table and a bottle on it get blobs of noise (random pixels of their mask, grown by dilation, smaller for the bottle)
spilling over what lies around them. Each is viewed diagonally, from a horizontal offset to its floor center.
The degenerate mask is projected with depth and voxel-downsampled (ConceptGraphConfig's voxel size), its floor and the
walls behind it are flagged with projection_utils' floor_mask and background_wall_mask, and concept-graphs' DBSCAN runs
on the rest. Into tests/testdump/test_denoise/<case>/ go the view's rgb.png, clusters.ply (floor in brown, walls in
gray, noise in black and every other point colored by its DBSCAN cluster, before denoise drops all but the largest
one), degenerate_mask.png and raytraced_mask.png, the cleaned mask the largest cluster raytraces in the view
(ConceptGraphResolvedObjects.visible_objects).
"""

import math
import os
import shutil
import tempfile
from collections import Counter
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

from common.env_utils.vocab_constants import make_colors
from common.utils.concept_graph import ConceptGraphConfig, ConceptGraphResolvedObjects
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import background_wall_mask, dbscan_labels, downsample_points, floor_mask, object_points
from common.utils.retrieval import SegmentationStyle, instances_from_masks, segmentation_tile
from common.utils.spot import fill_convex_hull
from test_dummy_scene import _DEPTH_CAMERA, _PROCTHOR_OBJECTS_DIR, _TABLE, _TABLETOP_OBJECTS, _WALL_OBJECT, _dummy_sim

pytestmark = pytest.mark.skipif(not os.path.isdir(_PROCTHOR_OBJECTS_DIR), reason="requires HABITAT_DATA pointing at a real ai2thor-hab dataset (see INSTALL.MD)")

_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_denoise")
_FLOOR_HEIGHT = 0.0
_FLOOR_COLOR = (140, 90, 40)
_WALL_COLOR = (160, 160, 160)
_NOISE_COLOR = (0, 0, 0)
_MASK_COLOR = (30, 60, 200)
_NUM_BLOBS = 12
_TABLE_BLOB_RADIUS_PX = 8
_BOTTLE_BLOB_RADIUS_PX = 3


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
    scene_object: tuple
    view_offset: tuple
    degenerate_mask: Callable[[np.ndarray], np.ndarray]


_CASES = [
    _DenoiseCase("chair_hull", _WALL_OBJECT, (1.6, -1.6), _convex_hull),
    _DenoiseCase("table_blobs", _TABLE, (1.8, 1.8), partial(_with_blobs, radius_px=_TABLE_BLOB_RADIUS_PX)),
    _DenoiseCase("bottle_blobs", _TABLETOP_OBJECTS[0], (-1.2, 1.2), partial(_with_blobs, radius_px=_BOTTLE_BLOB_RADIUS_PX)),
]


def _object_view(sim: habitat_sim.Simulator, case: _DenoiseCase) -> tuple:
    """(observations, depth sensor pose) from case's view offset to its object's floor center, facing the object."""
    (center_x, center_z), (offset_x, offset_z) = case.scene_object[3], case.view_offset
    yaw = math.atan2(offset_x, offset_z)
    state = habitat_sim.AgentState()
    state.position = np.array([center_x + offset_x, 0.0, center_z + offset_z])
    state.rotation = np.quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
    agent = sim.get_agent(0)
    agent.set_state(state)
    return sim.get_sensor_observations(), sensor_pose_matrix(agent.get_state().sensor_states["depth"])


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


def _largest_cluster(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return points[labels == Counter(labels[labels >= 0]).most_common(1)[0][0]]


def _raytraced_mask(points: np.ndarray, depth: np.ndarray, pose: np.ndarray) -> torch.Tensor:
    """Mask of points raytraced in depth from pose as a single resolved object."""
    resolved = ConceptGraphResolvedObjects()
    resolved.resolved_points, resolved.resolved_infos, resolved.kdtree = points, np.zeros((len(points), 2), dtype=int), KDTree(points)
    masks, _, _ = resolved.visible_objects(depth, pose, _DEPTH_CAMERA.hfov, _DEPTH_CAMERA.min_depth, _DEPTH_CAMERA.max_depth, min_visibility_fraction=0.0)
    return masks[0]


def _save_mask(rgb: np.ndarray, mask: torch.Tensor, class_name: str, out_path: str) -> None:
    instances = instances_from_masks(torch.as_tensor(mask)[None], torch.zeros(1), mask.shape)
    Image.fromarray(segmentation_tile(rgb, instances, SegmentationStyle([class_name], [_MASK_COLOR]), "")[1]).save(out_path)


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_denoise_on_degenerate_mask(case: _DenoiseCase):
    out_dir = os.path.join(_TESTDUMP_DIR, case.name)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    with tempfile.TemporaryDirectory() as assets_dir:
        sim = _dummy_sim(assets_dir)
        obs, pose = _object_view(sim, case)
        sim.close()
    semantic_id, class_name = case.scene_object[:2]
    rgb, object_mask = obs["rgb"][:, :, :3], obs["semantic"] == semantic_id
    assert object_mask.any(), f"the {class_name} is not in view"

    degenerate_mask = case.degenerate_mask(object_mask)
    config = ConceptGraphConfig()
    points = downsample_points(object_points(obs["depth"], pose, degenerate_mask, _DEPTH_CAMERA), config.downsample_voxel_size)
    floor = floor_mask(points, _FLOOR_HEIGHT)
    walls = np.zeros(len(points), dtype=bool)
    walls[~floor] = background_wall_mask(points[~floor], pose[:3, 3])
    labels = dbscan_labels(points[~floor & ~walls], config.dbscan_eps, config.dbscan_min_points)

    Image.fromarray(rgb).save(os.path.join(out_dir, "rgb.png"))
    _save_ply(points, _point_colors(floor, walls, labels), os.path.join(out_dir, "clusters.ply"))
    _save_mask(rgb, degenerate_mask, class_name, os.path.join(out_dir, "degenerate_mask.png"))
    cleaned_mask = _raytraced_mask(_largest_cluster(points[~floor & ~walls], labels), obs["depth"], pose)
    _save_mask(rgb, cleaned_mask, class_name, os.path.join(out_dir, "raytraced_mask.png"))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
