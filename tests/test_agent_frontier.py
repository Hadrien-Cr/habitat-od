"""Tests for FrontierExplorationAgent (common/hybrid_exploration/frontier_exp_agent.py).
Agent should satisfy stop conditions."""
import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import habitat_sim
import numpy as np
import pytest
from tqdm import tqdm

from common.agents.base_exploration_agent import ProfileTimer
from common.hybrid_exploration.frontier_exp_agent import DiscretePlannerConfig, FrontierExplorationAgent, FrontierExplorationConfig, exploration_snapshot
from common.hybrid_exploration.ground_truth_map import GroundTruthMapTracker
from common.utils.plot_utils import save_gif
from common.utils.pose_utils import quaternion_from_rpy
from habitat.tasks.nav.nav import TopDownMap


_TESTDUMP_DIR = Path(os.path.dirname(__file__)) / "testdump" / "test_agent_frontier"
_EXPLORATION_STEPS = 1000
_GIF_FPS = 10
_COLLISION_DIST_M = 0.15  # matches common/env_utils/sensors.py::AgentCollisionSensor
_STUCK_WINDOW_STEPS = 25
_STUCK_RADIUS_M = 0.1


def _is_stuck(trajectory: list, window: int, radius_m: float) -> bool:
    """True once the agent's last window positions all lie within radius_m of each other -- no net
    translation, even while its heading keeps oscillating between turns."""
    if len(trajectory) < window:
        return False
    recent = np.stack([position for position, _ in trajectory[-window:]])
    return bool(np.all(np.linalg.norm(recent - recent[0], axis=1) < radius_m))


def test_exploration_gif() -> None:
    """Runs FrontierExplorationAgent for _EXPLORATION_STEPS in dummy_scene.py's synthetic flat, saving a
    3-tile (observation, belief map, ground truth) snapshot per step and assembling them into exploration.webp."""
    from dummy_scene import _DEPTH_CAMERA, _MAP_CONFIG, _START, _TURN_ANGLE_DEG, _dummy_sim, _ground_truth_meshes, _navmesh_floor_offset

    shutil.rmtree(_TESTDUMP_DIR, ignore_errors=True)
    images_dir = _TESTDUMP_DIR / "images" / "test_exploration_gif"
    images_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as assets_dir:
        sim = _dummy_sim(assets_dir)

    profile_timer = ProfileTimer()
    try:
        rng = np.random.default_rng(0)
        start = habitat_sim.AgentState()
        start.position = np.array(_START)
        start.rotation = quaternion_from_rpy(0, 0, rng.integers(0, 8) * np.pi / 4)  # random start heading
        sim.get_agent(0).set_state(start)

        config = FrontierExplorationConfig(
            map_config=_MAP_CONFIG,
            planner_config=DiscretePlannerConfig(turn_angle_deg=_TURN_ANGLE_DEG, debug=True),
        )
        agent = FrontierExplorationAgent(config, _DEPTH_CAMERA, rng, str(_TESTDUMP_DIR), "test_exploration_gif")
        floor_height = _START[1] - _navmesh_floor_offset(sim, np.array(_START))
        agent.reset(np.array(_START), floor_height)

        ground_truth = GroundTruthMapTracker(sim.pathfinder, np.array(_START), fov_deg=_MAP_CONFIG.fov_deg, visibility_dist_m=_MAP_CONFIG.visibility_dist_m)
        ground_truth.set_object_footprints(_ground_truth_meshes(sim))
        trajectory_px: list = []

        obs = sim.get_sensor_observations()
        collided = False
        stuck = False

        assert agent.estimated_map is not None, "Call reset() first"

        for step in tqdm(range(_EXPLORATION_STEPS)):
            state = sim.get_agent(0).get_state()
            heading = float(TopDownMap.get_polar_angle(state))
            trajectory_px.append(ground_truth.to_pixel(np.array(state.position)))
            ground_truth.reveal(np.array(state.position), heading)

            act_start = time.perf_counter()
            action = agent.act(state, obs["depth"], collided)
            profile_timer.update("act", time.perf_counter() - act_start)

            mosaic = exploration_snapshot(obs, agent, ground_truth, trajectory_px)
            if mosaic is not None:
                mosaic.save(images_dir / f"snapshot_gif_{step:03d}.png")

            stuck = _is_stuck(agent.trajectory, _STUCK_WINDOW_STEPS, _STUCK_RADIUS_M)
            if action == "stop" or stuck:
                break
            obs = sim.step(action)

            new_state = sim.get_agent(0).get_state()
            collided = bool(np.linalg.norm(new_state.position - state.position) < _COLLISION_DIST_M and np.allclose(new_state.rotation, state.rotation))

        assert not stuck, f"agent got stuck (no net translation over the last {_STUCK_WINDOW_STEPS} steps)"
        assert agent.estimated_map.coverage() > 0.0, "agent wrong marked any cell as explored"
    
    finally:
        sim.close()

    print(profile_timer.display())
    save_gif(images_dir, "exploration.webp", fps=_GIF_FPS)
    assert (images_dir / "exploration.webp").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])