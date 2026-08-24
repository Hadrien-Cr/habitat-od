"""
Integration test: instantiates a real HSSD scene and generates k frames, then runs
ObjectDetectorGTSensor.decompose_frame on each of them under increasingly strict
min_visibility_fraction settings - reusing the same captured (semantic, depth) pair across
settings so the comparison is apples-to-apples.

Requires real HSSD data (HABITAT_DATA/BASE_DIR set, per CLAUDE.md) - skipped otherwise.
"""

import math
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.skipif(
    not os.environ.get("HABITAT_DATA"),
    reason="requires HABITAT_DATA pointing at real HSSD scene/dataset files",
)

import habitat # type: ignore
from habitat.config import read_write # type: ignore
import habitat_sim
from habitat_sim.utils.common import colorize_ids

from common.env_utils.object_detector_sensors import ObjectDetectorGTSensorConfig
from common.env_utils.env_base import ExplorationEnv
from common.env_utils.dataset import ExplorationNavDataset  # noqa: F401 - registers "ExplorationSynthetic"
import common.env_utils.sensors  # noqa: F401 - registers agent_collision_sensor/position_sensor

_SCENE = "102344022"
_VOCAB = "NYU40"
_K_FRAMES = 5
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_sensor_filters")
os.system(f"rm -rf {_TESTDUMP_DIR}")

SETTINGS = {
    "no_filter": dict(filter_low_visibility=False, min_visibility_fraction=0.0),
    "lenient": dict(filter_low_visibility=True, min_visibility_fraction=0.1),
    "strict": dict(filter_low_visibility=True, min_visibility_fraction=0.5),
}


@pytest.fixture(scope="module")
def env():
    habitat_config = habitat.get_config(config_path="common/config/hssd-hab/default.yaml")
    with read_write(habitat_config):
        habitat_config.habitat.dataset.content_scenes = [_SCENE]
        habitat_config.habitat.task.lab_sensors = {
            "object_detector_gt": ObjectDetectorGTSensorConfig(
                area_thr=0.0,
                env_name="HSSD-HAB",
                vocab_name=_VOCAB,
            ),
            **habitat_config.habitat.task.lab_sensors,
        }
    e = ExplorationEnv(config=habitat_config)
    e.reset()
    yield e
    e.close()


def _sensor(env):
    return env._env.task.sensor_suite.get("bbsgt")


def _random_agent_state(sim, rng) -> habitat_sim.AgentState:
    state = habitat_sim.AgentState()
    state.position = sim.pathfinder.get_random_navigable_point()
    yaw = rng.uniform(0, 2 * math.pi)
    state.rotation = np.quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
    return state


@pytest.fixture(scope="module")
def frames(env):
    """k (rgb, semantic, depth, agent_state) tuples captured once, reused across all 4
    settings. agent_state is captured alongside each frame (rather than re-read from the
    simulator when the frame is later decomposed) since by the time all frames have been
    captured the agent has moved on to the last one."""
    rng = np.random.default_rng(0)
    sim = env._env.sim
    captured = []
    for _ in range(_K_FRAMES):
        state = _random_agent_state(sim, rng)
        obs, _, _, _ = env.teleport(state)
        captured.append((obs["rgb"], obs["semantic"], obs["depth"], sim.get_agent_state()))
    return captured


def _apply_setting(sensor, name: str):
    for attr, value in SETTINGS[name].items():
        setattr(sensor, attr, value)


def _dump_frame(rgb, semantic_obs, instances, name):
    """Saves the semantic frame and the RGB frame (with kept detections' boxes) side by side."""
    os.makedirs(_TESTDUMP_DIR, exist_ok=True)
    semantic_img = colorize_ids(semantic_obs)

    rgb_img = Image.fromarray(rgb[:, :, :3].copy())
    draw = ImageDraw.Draw(rgb_img)
    for x0, y0, x1, y1 in instances.pred_boxes.tensor.tolist():
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)

    combined = np.concatenate([semantic_img, np.array(rgb_img)], axis=1)
    Image.fromarray(combined).save(os.path.join(_TESTDUMP_DIR, f"{name}.png"))


def test_filters_only_ever_remove_detections(env, frames):
    sensor = _sensor(env)
    counts = {name: [] for name in SETTINGS}

    for i, (rgb, semantic_obs, depth_obs, agent_state) in enumerate(frames):
        for name in SETTINGS:
            print(f"Image {i}, filter setting: {name}")
            _apply_setting(sensor, name)
            result = sensor.decompose_frame(semantic_obs.copy(), agent_state=agent_state, depth_obs=depth_obs.copy())
            instances = result["instances"]
            counts[name].append(len(instances))
            _dump_frame(rgb, semantic_obs, instances, f"frame{i}_{name}_n={len(instances)}")

    for i in range(len(frames)):
        assert counts["no_filter"][i] >= counts["lenient"][i]
        assert counts["lenient"][i] >= counts["strict"][i]


if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
