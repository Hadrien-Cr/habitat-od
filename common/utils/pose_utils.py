import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis

def quaternion_from_rpy(roll, pitch, yaw):
    """Inputs in radians"""
    return quat_from_angle_axis(roll, np.array([1, 0, 0])) * quat_from_angle_axis(yaw, np.array([0, 1, 0])) * quat_from_angle_axis(pitch, np.array([0, 0, 1]))


def rpy_from_quaternion(q):
    w, x, y, z = q.w, q.x, q.y, q.z
    
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x+z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x+y*y)]
    ])

    yaw = np.arcsin(R[0,2])

    c = np.cos(yaw)

    if abs(c) > 1e-8:
        pitch = np.arctan2(-R[0,1], R[0,0])
        roll  = np.arctan2(-R[1,2], R[2,2])

    else:
        # Gimbal lock
        pitch = 0
        roll = np.arctan2(R[2,1], R[1,1])

    return roll, pitch, yaw