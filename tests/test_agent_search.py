"""Tests for SearchAgent (habitat_od_search/search_agent.py).
Agent should satisfy stop conditions."""
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from tqdm import tqdm

from common.agents.base_exploration_agent import ProfileTimer
from common.env_utils.object_annotations import annotated_object_meshes
from common.hybrid_exploration.frontier_exp_agent import exploration_snapshot
from common.hybrid_exploration.ground_truth_map import GroundTruthMapTracker
from common.utils.plot_utils import save_gif
from common.vision.detic import Vocabulary, get_clip_embeddings, build_text_encoder # type: ignore
from common.vision.detector import Detector
from habitat_od_search.search_agent import SearchAgent, SearchAgentConfig, DiscretePlannerConfig
from habitat.tasks.nav.nav import TopDownMap
from procthor_scene import CLASSES, DEPTH_CAMERA, MAP_CONFIG, TURN_ANGLE_DEG, floor_height, procthor_sim, start_state


_TESTDUMP_DIR = Path(os.path.dirname(__file__)) / "testdump" / "test_agent_search"
_EXPLORATION_STEPS = 1000
_GIF_FPS = 10
_TARGET_OBJECT = "fridge"



def test_exploration_gif() -> None:
    """Runs SearchAgent for _EXPLORATION_STEPS in procthor_scene.py's ProcTHOR-Train-0 from its start_state, saving a
    3-tile (observation, belief map, ground truth) snapshot per step and assembling them into exploration.webp."""

    shutil.rmtree(_TESTDUMP_DIR, ignore_errors=True)
    images_dir = _TESTDUMP_DIR / "images" / "test_exploration_gif"
    images_dir.mkdir(parents=True)

    sim, annotation = procthor_sim()

    profile_timer = ProfileTimer()
    try:
        rng = np.random.default_rng(0)
        start = start_state()
        sim.get_agent(0).set_state(start)
        config = SearchAgentConfig(
            map_config=MAP_CONFIG,
            planner_config=DiscretePlannerConfig(turn_angle_deg=TURN_ANGLE_DEG, debug=True),
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
            config=config, camera=DEPTH_CAMERA, rng=rng, dump_location=str(_TESTDUMP_DIR), exp_name="test_exploration_gif",
            target_name=_TARGET_OBJECT)

        text_encoder = build_text_encoder(pretrain=True)
        text_encoder.eval()
        target_embedding = get_clip_embeddings([_TARGET_OBJECT], text_encoder=text_encoder)[:, 0]

        agent.reset(sim.get_agent(0).get_state(), floor_height=floor_height(sim, start.position), target_embedding=target_embedding, target_label=_TARGET_OBJECT)

        ground_truth = GroundTruthMapTracker(sim.pathfinder, start.position, fov_deg=MAP_CONFIG.fov_deg, visibility_dist_m=MAP_CONFIG.visibility_dist_m)
        ground_truth.set_object_footprints(annotated_object_meshes(sim, annotation, CLASSES))
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