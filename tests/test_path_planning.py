"""Path-planning test for common/env_utils/env_base.py::ExplorationEnv.find_shortest_path (the
discrete-action shortest-path trajectory, via habitat_sim's GreedyGeodesicFollower -- the real
navmesh geodesic shortest path, greedily fit to move_forward/turn_left/turn_right motion
primitives) and common/agents/random_agent.py (which walks it).

Behavior checks: a reachable goal produces a trajectory of valid env.action_space actions ending
in exactly one "stop" that actually walks the agent to the goal when executed via env.step;
repeated calls from a fixed pose are deterministic; an unreachable goal returns [] instead of
raising.

Visual check: drives RandomAgent for one horizon (RandomAgent.HORIZON actions), stepping the env
with each returned action (rotations/move_forward only, never teleporting) and dumping a rgb |
top-down map side-by-side GIF to tests/testdump/test_path_planning/ for visual inspection of the
walk towards the agent's randomly sampled destination (red circle on the map; blue dot+line is
the agent's position/heading; yellow trail is the path walked so far).

Requires HABITAT_DATA pointing at a real hssd-hab dataset (see INSTALL.MD) -- skipped otherwise.
"""
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.skipif(
    not os.environ.get("HABITAT_DATA"),
    reason="requires HABITAT_DATA pointing at a real hssd-hab dataset (see INSTALL.MD)",
)

import habitat  # type: ignore
from habitat.config import read_write  # type: ignore
from habitat.utils.visualizations import maps  # type: ignore

from common.agents.random_agent import HORIZON, RandomAgent
from common.env_utils.dataset import ExplorationNavDataset  # noqa: F401 - registers "ExplorationSynthetic"
from common.env_utils.env_base import ExplorationEnv
from common.env_utils.object_detector_sensors import ObjectDetectorGTSensorConfig  # registers "bbsgt", required by ExplorationEnv.reset()
import common.env_utils.sensors  # noqa: F401 - registers agent_collision_sensor/position_sensor
from common.utils.pose_utils import get_yaw

_SCENE = "102344022"  # same smoke-test scene as test_sensor_filters.py
_GOAL_TOLERANCE_M = 1.0  # generous vs. the follower's own goal_radius (0.75 * forward_step_size)
_MAP_SIZE = 480
_FRAME_DURATION_MS = 200
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_path_planning")
os.system(f"rm -rf {_TESTDUMP_DIR}")


@pytest.fixture
def env():
    habitat_config = habitat.get_config(config_path="common/config/hssd-hab/default.yaml")
    with read_write(habitat_config):
        habitat_config.habitat.dataset.content_scenes = [_SCENE]
        habitat_config.habitat.task.lab_sensors = {
            "object_detector_gt": ObjectDetectorGTSensorConfig(env_name="HSSD-HAB", vocab_name="COCO80"),
            **habitat_config.habitat.task.lab_sensors,
        }
    e = ExplorationEnv(config=habitat_config)
    e.reset()
    if not e._env.sim.pathfinder.is_loaded:
        e.close()
        pytest.skip(f"no navmesh loaded for scene {_SCENE}")
    yield e
    e.close()


def _walk(env, trajectory: list) -> np.ndarray:
    for action in trajectory:
        env.step(action)
    return np.array(env.get_agent_state().position)


def _planar_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(a[0] - b[0], a[2] - b[2]))


def test_trajectory_actions_are_valid_and_end_in_one_stop(env):
    goal = env.get_random_point(np.random.default_rng(0))
    trajectory = env.find_shortest_path(goal)

    assert trajectory, "expected a non-empty trajectory to a navmesh-sampled goal"
    assert set(trajectory) <= env.action_space.spaces.keys(), (
        f"trajectory contains actions not in env.action_space: {set(trajectory) - env.action_space.spaces.keys()}"
    )
    assert trajectory[-1] == "stop"
    assert trajectory.count("stop") == 1, "stop should only ever be the final action"


def test_walking_the_trajectory_reaches_the_goal(env):
    goal = env.get_random_point(np.random.default_rng(1))
    trajectory = env.find_shortest_path(goal)
    assert trajectory

    final_position = _walk(env, trajectory)
    final_dist = _planar_dist(final_position, goal)
    assert final_dist < _GOAL_TOLERANCE_M, f"walked trajectory ended {final_dist:.2f}m from the goal"


def test_repeated_calls_from_a_fixed_pose_are_deterministic(env):
    """env.find_shortest_path always plans from the agent's *current* live pose (no separate
    `start` argument, unlike the skeleton-based find_shortest_path_waypoints) -- calling it
    twice in a row without moving should return the same trajectory."""
    goal = env.get_random_point(np.random.default_rng(2))
    assert env.find_shortest_path(goal) == env.find_shortest_path(goal)


def test_unreachable_goal_returns_empty_list(env):
    _, upper_bound = env.get_map_bounds()
    far_outside_navmesh = np.array(upper_bound) + np.array([100.0, 0.0, 100.0])
    assert env.find_shortest_path(far_outside_navmesh) == []


def _forward_vector(yaw: float) -> np.ndarray:
    """Unit world-space heading for `yaw`, in pose_utils.get_yaw/yaw_to_face's own convention."""
    return np.array([-np.sin(yaw), 0.0, -np.cos(yaw)])


def _topdown_map(env) -> np.ndarray:
    pathfinder = env._env.sim.pathfinder
    lower_bound, _ = pathfinder.get_bounds()
    return pathfinder.get_topdown_view(
        meters_per_pixel=env._GROUND_FLOOR_MPP, height=lower_bound[1]
    ).astype(np.uint8)


def _render_frame(env, tdmap: np.ndarray, rgb: np.ndarray, trail: list, goal, step: int, action: str) -> Image.Image:
    pathfinder = env._env.sim.pathfinder
    topdown = np.full((*tdmap.shape, 3), 40, dtype=np.uint8)
    topdown[tdmap == 1] = (210, 210, 210)
    map_img = Image.fromarray(topdown).resize((_MAP_SIZE, _MAP_SIZE), Image.NEAREST)
    sx, sy = _MAP_SIZE / tdmap.shape[1], _MAP_SIZE / tdmap.shape[0]
    draw = ImageDraw.Draw(map_img)

    def px(pos):
        row, col = maps.to_grid(pos[2], pos[0], tdmap.shape, pathfinder=pathfinder)
        return col * sx, row * sy

    dist_to_goal = None
    if goal is not None:
        gx, gy = px(goal)
        draw.ellipse([gx - 6, gy - 6, gx + 6, gy + 6], outline=(220, 30, 30), width=3)
        dist_to_goal = np.hypot(trail[-1][0][0] - goal[0], trail[-1][0][2] - goal[2])

    if len(trail) > 1:
        draw.line([px(pos) for pos, _ in trail], fill=(255, 190, 0), width=3)

    pos, yaw = trail[-1]
    ax, ay = px(pos)
    # Direction computed analytically (world-space forward vector scaled by sx/sy), not by
    # differencing two separately-projected points -- pos and pos + forward*d each round to a
    # grid cell via maps.to_grid's int() truncation, and since pos moves every move_forward
    # step, their quantization error doesn't cancel, making the drawn angle jitter even when
    # the real yaw hasn't changed at all (see sensors.py's ColoredTDMapSensor, same bug).
    # Fixed on-screen length (not a fixed world-space length through px()) since sx != sy for
    # non-square rooms would otherwise stretch/shrink the line with heading too.
    dx, dy = -_forward_vector(yaw)[0] * sx, -_forward_vector(yaw)[2] * sy
    norm = np.hypot(dx, dy) or 1.0
    tx, ty = ax + dx / norm * 20, ay + dy / norm * 20
    draw.line([ax, ay, tx, ty], fill=(30, 120, 220), width=3)
    draw.ellipse([ax - 5, ay - 5, ax + 5, ay + 5], fill=(30, 120, 220))

    caption = f"step {step}/{HORIZON}  action={action}"
    if dist_to_goal is not None:
        caption += f"  dist_to_goal={dist_to_goal:.2f}m"
    draw.rectangle([0, 0, _MAP_SIZE, 16], fill=(0, 0, 0))
    draw.text((4, 2), caption, fill=(255, 255, 255))

    rgb_img = Image.fromarray(rgb[:, :, :3]).resize((_MAP_SIZE, _MAP_SIZE))
    return Image.fromarray(np.concatenate([np.array(rgb_img), np.array(map_img.convert("RGB"))], axis=1))


def _agent_pose(env) -> tuple:
    state = env.get_agent_state()
    yaw = get_yaw(state.rotation.w, state.rotation.x, state.rotation.y, state.rotation.z)
    return np.array(state.position), yaw


def test_random_agent_walks_one_horizon_to_gif(env):
    rng = np.random.default_rng(0)
    agent = RandomAgent(rng)
    tdmap = _topdown_map(env)

    trail = [_agent_pose(env)]
    frames = []
    dists_to_goal = []

    for step in range(1, HORIZON + 1):
        action = agent.act(env)
        obs, _, _, _ = env.step(action)
        trail.append(_agent_pose(env))
        frames.append(_render_frame(env, tdmap, obs["rgb"], trail, agent.goal, step, action))
        if agent.goal is not None:
            dists_to_goal.append(np.hypot(trail[-1][0][0] - agent.goal[0], trail[-1][0][2] - agent.goal[2]))

    assert len(frames) == HORIZON
    assert agent.goal is not None, "RandomAgent never found a reachable destination -- check the navmesh"
    assert dists_to_goal[-1] < dists_to_goal[0], (
        f"walked away from the goal over the horizon: {dists_to_goal[0]:.2f}m -> {dists_to_goal[-1]:.2f}m"
    )

    os.makedirs(_TESTDUMP_DIR, exist_ok=True)
    out_path = os.path.join(_TESTDUMP_DIR, f"{_SCENE}_horizon{HORIZON}.gif")
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=_FRAME_DURATION_MS, loop=0)
    print(f"wrote {len(frames)}-frame gif to {out_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
