"""Tests for SearchAgent (habitat_od_search/search_agent.py).
Agent should satisfy stop conditions."""
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
from common.utils.pose_utils import quaternion_from_rpy
from common.vision.detic import Vocabulary, get_clip_embeddings, build_text_encoder # type: ignore
from common.vision.detector import Detector
from habitat_od_search.search_agent import SearchAgent, SearchAgentConfig, DiscretePlannerConfig
from habitat.tasks.nav.nav import TopDownMap


_TESTDUMP_DIR = Path(os.path.dirname(__file__)) / "testdump" / "test_agent_search"
_EXPLORATION_STEPS = 1000
_GIF_FPS = 10
_TARGET_OBJECT = "pillow"



def test_exploration_gif() -> None:
    """Runs SearchAgent for _EXPLORATION_STEPS in dummy_scene.py's synthetic flat, saving a
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
        config = SearchAgentConfig(
            map_config=_MAP_CONFIG,
            planner_config=DiscretePlannerConfig(turn_angle_deg=_TURN_ANGLE_DEG, debug=True),
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
            config=config, camera=_DEPTH_CAMERA, rng=rng, dump_location=str(_TESTDUMP_DIR), exp_name="test_exploration_gif",
            target_name=_TARGET_OBJECT)

        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        target_embedding = get_clip_embeddings([_TARGET_OBJECT], text_encoder=text_encoder)[:, 0]

        floor_height = _START[1] - _navmesh_floor_offset(sim, np.array(_START))
        agent.reset(sim.get_agent(0).get_state(), floor_height=floor_height, target_embedding=target_embedding, target_label=_TARGET_OBJECT)

        ground_truth = GroundTruthMapTracker(sim.pathfinder, np.array(_START), fov_deg=_MAP_CONFIG.fov_deg, visibility_dist_m=_MAP_CONFIG.visibility_dist_m)
        ground_truth.set_object_footprints(_ground_truth_meshes(sim))
        trajectory_px: list = []

        obs = sim.get_sensor_observations()
        found_target = False
        stuck = False

        assert agent.estimated_map is not None, "Call reset() first"

        for step in tqdm(range(_EXPLORATION_STEPS)):
            state = sim.get_agent(0).get_state()
            heading = float(TopDownMap.get_polar_angle(state))
            trajectory_px.append(ground_truth.to_pixel(np.array(state.position)))
            ground_truth.reveal(np.array(state.position), heading)

            act_start = time.perf_counter()
            action, _ = agent.act(state, obs["rgb"], obs["depth"])
            profile_timer.update("act", time.perf_counter() - act_start)

            mosaic = exploration_snapshot(obs, agent, ground_truth, trajectory_px)
            if mosaic is not None:
                mosaic.save(images_dir / f"snapshot_gif_{step:03d}.png")

            if action == "stop":
                stuck = agent.is_stuck()
                found_target = not stuck
                break
            obs = sim.step(action)

        assert not stuck, f"agent got stuck (no net progress) before stopping at {_TARGET_OBJECT}"
        assert found_target, f"agent reached {_EXPLORATION_STEPS} steps without stopping at {_TARGET_OBJECT}"
        assert agent.estimated_map.coverage() > 0.0, "agent wrong marked any cell as explored"
    finally:
        sim.close()

    print(profile_timer.display())
    save_gif(images_dir, "exploration.webp", fps=_GIF_FPS)
    assert (images_dir / "exploration.webp").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])