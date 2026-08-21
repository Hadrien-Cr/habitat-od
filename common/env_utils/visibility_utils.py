import cv2
import numpy as np
import quaternion
from habitat_sim.agent.agent import AgentState


def camera_basis(camera_rot: quaternion.quaternion, width: int, camera_hfov: float):
    """(forward, right, up, focal_px) for the given sensor."""
    axes = (np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    forward, right, up = (
        quaternion.rotate_vectors(camera_rot, axis) for axis in axes
    )
    forward, right, up = (v / np.linalg.norm(v) for v in (forward, right, up))
    focal = (width / 2.0) / np.tan(np.deg2rad(camera_hfov / 2.0))
    return forward, right, up, focal

def compute_obj_dimensions(obj) -> np.ndarray:

    aabb = obj.collision_shape_aabb # type: ignore
    min_v = aabb.min
    max_v = aabb.max

    return np.array([max_v.x - min_v.x, max_v.y - min_v.y, max_v.z - min_v.z])


def mesh_visibility_fraction(
    obj_dimensions: np.ndarray,
    object_mask: np.ndarray,
    agent_state: AgentState,
    depth_obs: np.ndarray,
) -> float:
    obj_depth = depth_obs[object_mask].mean()

    sensor_state = agent_state.sensor_states["rgb"]
    camera_rot = sensor_state.rotation

    forward, right, up, focal = camera_basis(
        camera_rot, depth_obs.shape[1], 90.0
    )

    # Compute the pixels occupied by the object if not occluded

    face_area = max([obj_dimensions[0] * obj_dimensions[1], obj_dimensions[0] * obj_dimensions[2], obj_dimensions[1] * obj_dimensions[2]])
    face_pixels = face_area * (focal / obj_depth) ** 2

    # Compute the fraction of pixels that are visible
    opened_mask = object_mask.copy()
    opened_mask = cv2.morphologyEx(opened_mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    x,y,w,h = cv2.boundingRect(opened_mask.astype(np.uint8))
    visible_pixels = w * h


    visibility_fraction = visible_pixels / (face_pixels + 1e-8)

    return visibility_fraction