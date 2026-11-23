"""Tests for HybridFrontierAgent (common/hybrid_exploration/hybrid_agent.py), standalone: no map is resolved and nothing is
saved. It explores procthor_scene.py's ProcTHOR-Train-0 from its start_state in exhaustive mode for up to
_EXPLORATION_STEPS steps, its ObjectTracker fed every step's Detector predictions lifted by concept_graph.py's
predicted_objects, as collect_train.py feeds it. Agent should satisfy stop conditions: never stuck (no net translation,
BaseExplorationAgent.is_stuck), and some of the scene explored.

Needs ProcTHOR-hab under HABITAT_DATA/scene_datasets/ai2thor-hab, a GPU, Detic's Swin-B checkpoint under
third_party/Detic/models and SAM's ViT-B checkpoint under $HABITAT_DATA/sam.
"""
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from tqdm import tqdm

from common.hybrid_exploration.frontier_exp_agent import DiscretePlannerConfig
from common.hybrid_exploration.hybrid_agent import HybridConfig, HybridFrontierAgent
from common.utils.concept_graph import ConceptGraphConfig, predicted_objects, processed_frame
from common.utils.image_memory import Frame
from common.utils.pose_utils import sensor_pose_matrix
from common.vision.detic import Vocabulary
from common.vision.detector import Detector
from procthor_scene import CLASSES, DATASET_CONFIG, DEPTH_CAMERA, MAP_CONFIG, TURN_ANGLE_DEG, VOCAB_NAME, floor_height, procthor_sim, start_state

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DETIC_CONFIG = _REPO_ROOT / "third_party" / "Detic" / "configs" / "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
_DETIC_CHECKPOINT = _REPO_ROOT / "third_party" / "Detic" / "models" / "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
_SAM_CHECKPOINT = Path(os.environ.get("HABITAT_DATA", "")) / "sam" / "sam_vit_b_01ec64.pth"
_EXPLORATION_STEPS = 1000
_COLLISION_DIST_M = 0.15  # matches common/env_utils/sensors.py::AgentCollisionSensor
_SEED = 0

pytestmark = [
    pytest.mark.skipif(not os.path.exists(DATASET_CONFIG), reason="requires HABITAT_DATA pointing at a real ai2thor-hab dataset (see INSTALL.MD)"),
    pytest.mark.skipif(not _DETIC_CHECKPOINT.exists() or not torch.cuda.is_available(), reason="requires a GPU and Detic's Swin-B checkpoint"),
    pytest.mark.skipif(not _SAM_CHECKPOINT.exists(), reason="requires SAM's ViT-B checkpoint"),
]


def test_exhaustive_exploration(tmp_path: Path) -> None:
    detector = Detector(
        str(_DETIC_CONFIG), str(_DETIC_CHECKPOINT), Vocabulary(f"{VOCAB_NAME}_kept", tuple(CLASSES)), str(_SAM_CHECKPOINT),
        score_thresh=0.4, nms_thresh=0.5, mask_thresh=0.5, input_size=1024,
    )
    sim, _ = procthor_sim()
    try:
        start = start_state()
        sim.get_agent(0).set_state(start)
        config = HybridConfig(MAP_CONFIG, TURN_ANGLE_DEG, planner_config=DiscretePlannerConfig(TURN_ANGLE_DEG))
        agent = HybridFrontierAgent(config, DEPTH_CAMERA, np.random.default_rng(_SEED), str(tmp_path), "test_agent_hybrid")
        floor = floor_height(sim, start.position)
        agent.reset(sim.get_agent(0).get_state(), floor)
        obs, collided, mapping_config = sim.get_sensor_observations(), False, ConceptGraphConfig()
        for step in tqdm(range(_EXPLORATION_STEPS), desc="exploration"):
            state = sim.get_agent(0).get_state()
            frame = Frame(obs["rgb"][:, :, :3], obs["depth"], sensor_pose_matrix(state.sensor_states["depth"]))
            agent.update_detections(processed_frame(predicted_objects(step, frame, detector.predict(frame.rgb), DEPTH_CAMERA, floor), mapping_config))
            action, _ = agent.act(state, obs["depth"], collided)
            if action == "stop":
                break
            obs = sim.step(action)
            new_state = sim.get_agent(0).get_state()
            collided = bool(np.linalg.norm(new_state.position - state.position) < _COLLISION_DIST_M and np.allclose(new_state.rotation, state.rotation))
        assert not (action == "stop" and agent.is_stuck()), "agent got stuck (no net translation)"
        assert agent.estimated_map.coverage() > 0.0, "agent marked no cell as explored"
    finally:
        sim.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
