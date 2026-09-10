"""Discretized shortest-path actions (rotations + forward movements only, no teleporting) from
the agent's current pose towards a goal position -- turns to face each of env.find_shortest_path's
waypoints in env.get_turn_angle()-degree increments, then walks to it in
env.get_forward_step_size() hops. Used by common/agents/random_agent.py to walk to a randomly
sampled destination instead of teleporting to it."""
import numpy as np

from common.utils.pose_utils import get_yaw, yaw_to_face

TURN_LEFT, TURN_RIGHT, MOVE_FORWARD = "turn_left", "turn_right", "move_forward"
_ACTIONS = {TURN_LEFT, TURN_RIGHT, MOVE_FORWARD}


def waypoints_to_actions(waypoints: list, start_yaw: float, turn_angle: float, step_size: float) -> list:
    """Pure geometry, no env/simulator needed: turns turn_angle-degree increments to face each
    waypoint (yaw_to_face's convention: positive delta = turn_left), then walks to it in
    step_size hops. start_yaw in radians."""
    actions: list = []
    yaw = start_yaw
    turn_step = np.radians(turn_angle)

    for prev, nxt in zip(waypoints[:-1], waypoints[1:]):
        target_yaw = yaw_to_face(prev, nxt)
        delta = (target_yaw - yaw + np.pi) % (2 * np.pi) - np.pi
        n_turns = int(round(abs(delta) / turn_step))
        actions.extend([TURN_LEFT if delta > 0 else TURN_RIGHT] * n_turns)
        yaw += np.sign(delta) * n_turns * turn_step

        dist = np.hypot(nxt[0] - prev[0], nxt[2] - prev[2])
        actions.extend([MOVE_FORWARD] * max(1, round(dist / step_size)))

    return actions


def forward_shortest_path(env, goal: np.ndarray, max_steps: int) -> list:
    assert _ACTIONS <= env.action_space.spaces.keys(), f"env.action_space is missing one of {_ACTIONS}"

    state = env.get_agent_state()
    waypoints = env.find_shortest_path(np.array(state.position), goal)
    if not waypoints:
        return []

    yaw = get_yaw(state.rotation.w, state.rotation.x, state.rotation.y, state.rotation.z)
    actions = waypoints_to_actions(waypoints, yaw, env.get_turn_angle(), env.get_forward_step_size())
    return actions[:max_steps]
