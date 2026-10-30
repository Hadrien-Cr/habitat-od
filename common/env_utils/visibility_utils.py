import cv2
import numpy as np
import quaternion


def focal_length_px(width: int, camera_hfov: float) -> float:
    return (width / 2.0) / np.tan(np.deg2rad(camera_hfov / 2.0))


def camera_basis(camera_rot: quaternion.quaternion, width: int, camera_hfov: float):
    """(forward, right, up, focal_px) for the given sensor."""
    axes = (np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    forward, right, up = (
        quaternion.rotate_vectors(camera_rot, axis) for axis in axes
    )
    forward, right, up = (v / np.linalg.norm(v) for v in (forward, right, up))
    return forward, right, up, focal_length_px(width, camera_hfov)

def compute_dimensions_from_aabb(aabb) -> np.ndarray:
    min_v = aabb.min
    max_v = aabb.max
    return np.array([max_v.x - min_v.x, max_v.y - min_v.y, max_v.z - min_v.z])


def compute_obj_dimensions(obj) -> np.ndarray:
    return compute_dimensions_from_aabb(obj.collision_shape_aabb) # type: ignore


def compute_obj_render_dimensions(obj) -> np.ndarray:
    """Tight render-mesh extents; collision_shape_aabb is inflated by Bullet's 0.04 m collision margin per side."""
    return compute_dimensions_from_aabb(obj.root_scene_node.cumulative_bb)


def compute_dimensions_from_obb(obb) -> np.ndarray:
    """SemanticObject.aabb is unreliable for MP3D/Gibson-Semantic (confirmed against real
    data: a couch's aabb can be off by ~7x on one axis vs. its true size) - .obb.sizes is the
    tight, correctly oriented fit test_gibson.py/test_mp3d.py already trust for their own
    wireframes, so use that instead for these two envs."""
    return np.array([obb.sizes.x, obb.sizes.y, obb.sizes.z])


SIZE_MEASURES = ("volume", "shortest_edge", "largest_edge")


def box_size_measures(dimensions: np.ndarray) -> dict[str, float]:
    """SIZE_MEASURES of a box of dimensions."""
    return dict(zip(SIZE_MEASURES, (float(np.prod(dimensions)), float(np.min(dimensions)), float(np.max(dimensions)))))


def within_size_ranges(dimensions: np.ndarray, size_ranges: dict[str, tuple[float, float]]) -> bool:
    """Whether each box_size_measures of a box of dimensions lies in its (low, high) size_ranges entry, if it has one."""
    measures = box_size_measures(dimensions)
    return all(low <= measures[name] <= high for name, (low, high) in size_ranges.items())


def components_bbox_area(mask: np.ndarray) -> int:
    """Sum of each connected component's own bounding-box area in mask."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return int(np.sum(stats[1:, cv2.CC_STAT_WIDTH] * stats[1:, cv2.CC_STAT_HEIGHT]))


def mesh_visibility_fraction(
    obj_dimensions: np.ndarray,
    object_mask: np.ndarray,
    depth_obs: np.ndarray,
    camera_hfov: float = 90.0,
) -> float:
    obj_depth = depth_obs[object_mask].mean()
    if obj_depth <= 0:
        # 0 is the depth sensor's no-return sentinel (e.g. every masked pixel lies beyond
        # max_depth) - there's no reliable depth to size the object's unoccluded silhouette
        # against, so treat it as unverifiable/fully occluded rather than dividing by zero.
        return 0.0

    focal = focal_length_px(depth_obs.shape[1], camera_hfov)

    # Compute the pixels occupied by the object if not occluded

    face_area = max([obj_dimensions[0] * obj_dimensions[1], obj_dimensions[0] * obj_dimensions[2], obj_dimensions[1] * obj_dimensions[2]])
    face_pixels = face_area * (focal / obj_depth) ** 2

    # Compute the fraction of pixels that are visible
    visible_pixels = components_bbox_area(object_mask)
    visibility_fraction = visible_pixels / (face_pixels + 1e-8)

    return visibility_fraction