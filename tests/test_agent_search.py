"""Tests for SearchAgent (habitat_od_search/search_agent.py)."""
import os
import shutil
import tempfile
import time
from pathlib import Path

import habitat_sim
import numpy as np
import pytest
from tqdm import tqdm

from common.agents.base_exploration_agent import ProfileTimer
from common.hybrid_exploration.frontier_exp_agent import exploration_snapshot
from common.hybrid_exploration.ground_truth_map import GroundTruthMapTracker
from common.utils.plot_utils import save_gif
from common.vision.detic import Vocabulary, get_clip_embeddings, build_text_encoder # type: ignore
from common.vision.detector import Detector
from habitat_od_search.agent import SearchAgent, SearchAgentConfig, GoalScoreConfig, DiscretePlannerConfig
from habitat.tasks.nav.nav import TopDownMap


_TESTDUMP_DIR = Path(os.path.dirname(__file__)) / "testdump" / "test_search_agent"
_EXPLORATION_STEPS = 1000
_GIF_FPS = 6
_COLLISION_DIST_M = 0.15  # matches common/env_utils/sensors.py::AgentCollisionSensor
_STUCK_WINDOW_STEPS = 15
_STUCK_RADIUS_M = 0.3
_TARGET_OBJECT = "pillow"


def _is_stuck(trajectory: list, window: int, radius_m: float) -> bool:
    """True once the agent's last window positions all lie within radius_m of each other -- no net
    translation, even while its heading keeps oscillating between turns."""
    if len(trajectory) < window:
        return False
    recent = np.stack([position for position, _ in trajectory[-window:]])
    return bool(np.all(np.linalg.norm(recent - recent[0], axis=1) < radius_m))


def test_exploration_gif() -> None:
    """Runs SearchAgent for _EXPLORATION_STEPS in dummy_scene.py's synthetic flat, saving a
    3-tile (observation, belief map, ground truth) snapshot per step and assembling them into exploration.webp."""
    from dummy_scene import _DEPTH_CAMERA, _MAP_CONFIG, _START, _TURN_ANGLE_DEG, _dummy_sim, _ground_truth_meshes, _navmesh_floor_offset

    images_dir = _TESTDUMP_DIR / "images" / "test_exploration_gif"
    shutil.rmtree(images_dir, ignore_errors=True)
    images_dir.mkdir(parents=True)
    plan_log = (_TESTDUMP_DIR / "plan_debug.txt").open("w")

    with tempfile.TemporaryDirectory() as assets_dir:
        sim = _dummy_sim(assets_dir)

    profile_timer = ProfileTimer()
    try:
        start = habitat_sim.AgentState()
        start.position = np.array(_START)
        sim.get_agent(0).set_state(start)
        config = SearchAgentConfig(
            map_config=_MAP_CONFIG,
            planner_config=DiscretePlannerConfig(turn_angle_deg=_TURN_ANGLE_DEG),
            goal_score_config=GoalScoreConfig(score_decay=0.9, lift_image_size=512),
            confidence_thresh=0.5,
            goal_score_thresh=0.5,
            max_steps=_EXPLORATION_STEPS,
        )
        detector = Detector(
            vocabulary=Vocabulary("search_unused", ("object",)),
            config_file="third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml",
            weights_path="third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth",
            device="cuda",
            sam_checkpoint=None,
        )

        agent = SearchAgent(detector=detector,
            config=config, camera=_DEPTH_CAMERA, rng=np.random.default_rng(0), dump_location=str(_TESTDUMP_DIR), exp_name="test_search_agent",
            target_name=_TARGET_OBJECT)

        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        target_embedding = get_clip_embeddings([_TARGET_OBJECT], text_encoder=text_encoder)[:, 0]

        floor_height = _START[1] - _navmesh_floor_offset(sim, np.array(_START))
        agent.reset(np.array(_START), floor_height=floor_height, target_embedding=target_embedding)

        ground_truth = GroundTruthMapTracker(sim.pathfinder, np.array(_START), fov_deg=_MAP_CONFIG.fov_deg, visibility_dist_m=_MAP_CONFIG.visibility_dist_m)
        ground_truth.set_object_footprints(_ground_truth_meshes(sim))
        trajectory_px: list = []

        obs = sim.get_sensor_observations()
        collided = False
        found_target = False
        stuck = False

        assert agent.estimated_map is not None, "Call reset() first"

        for step in tqdm(range(_EXPLORATION_STEPS)):
            state = sim.get_agent(0).get_state()
            heading = float(TopDownMap.get_polar_angle(state))
            trajectory_px.append(ground_truth.to_pixel(np.array(state.position)))
            ground_truth.reveal(np.array(state.position), heading)
            mosaic = exploration_snapshot(obs, agent, ground_truth, trajectory_px)

            if mosaic is not None:
                mosaic.save(images_dir / f"snapshot_{step:03d}.png")

            act_start = time.perf_counter()
            action = agent.act(state, obs["rgb"], obs["depth"], collided)
            profile_timer.update("act", time.perf_counter() - act_start)
            plan_log.write(agent.plan_debug_line(step, action) + "\n")
            if action == "stop":
                found_target = True
                break
            stuck = _is_stuck(agent.trajectory, _STUCK_WINDOW_STEPS, _STUCK_RADIUS_M)
            if stuck:
                break
            obs = sim.step(action)

            new_state = sim.get_agent(0).get_state()
            collided = bool(np.linalg.norm(new_state.position - state.position) < _COLLISION_DIST_M and np.allclose(new_state.rotation, state.rotation))

        assert not stuck, f"agent got stuck (no net progress over the last {_STUCK_WINDOW_STEPS} steps) before stopping at {_TARGET_OBJECT}"
        assert found_target, f"agent reached {_EXPLORATION_STEPS} steps without stopping at {_TARGET_OBJECT}"
        assert agent.estimated_map.coverage() > 0.0, "agent wrong marked any cell as explored"
    finally:
        sim.close()
        plan_log.close()

    print(profile_timer.display())
    save_gif(images_dir, "exploration.webp", fps=_GIF_FPS)
    assert (images_dir / "exploration.webp").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])