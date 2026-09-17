"""Smoke tests for the grid-based exploration agents (common/agents/frontier_agent.py,
sweep_agent.py) -- same env fixture/scene as test_path_planning.py.

FrontierAgent: drives it for _N_STEPS actions and checks its unexplored-cell count only ever
shrinks (never re-grows) and is strictly smaller at the end than at the start (real coverage
progress, not stuck replanning the same cell forever).

SweepAgent: drives it for _N_STEPS actions and checks its visited-cell count only ever grows
except for a full reset back to empty once every reachable cell has been swept.

Requires HABITAT_DATA pointing at a real hssd-hab dataset (see INSTALL.MD) -- skipped otherwise.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("HABITAT_DATA"),
    reason="requires HABITAT_DATA pointing at a real hssd-hab dataset (see INSTALL.MD)",
)

import habitat  # type: ignore
from habitat.config import read_write  # type: ignore

from common.agents.frontier_agent import FrontierAgent
from common.agents.sweep_agent import SweepAgent
from common.env_utils.dataset import ExplorationNavDataset  # noqa: F401 - registers "ExplorationSynthetic"
from common.env_utils.env_base import ExplorationEnv
from common.env_utils.object_detector_sensors import ObjectDetectorGTSensorConfig  # registers "bbsgt", required by ExplorationEnv.reset()
import common.env_utils.sensors  # noqa: F401 - registers agent_collision_sensor/position_sensor

_SCENE = "102344022"  # same smoke-test scene as test_path_planning.py/test_sensor_filters.py
_N_STEPS = 150


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


def test_frontier_agent_explores(env):
    rng = np.random.default_rng(0)
    agent = FrontierAgent(rng)

    unexplored_counts = []
    for _ in range(_N_STEPS):
        action = agent.act(env)
        assert action in env.action_space.spaces.keys()
        env.step(action)
        unexplored_counts.append(len(agent._unexplored))

    assert agent._reachable, "no reachable cells found on the navmesh"
    assert all(b <= a for a, b in zip(unexplored_counts, unexplored_counts[1:])), (
        "unexplored cell count should never increase"
    )
    assert unexplored_counts[-1] < unexplored_counts[0], (
        f"unexplored cell count didn't shrink over {_N_STEPS} steps: "
        f"{unexplored_counts[0]} -> {unexplored_counts[-1]}"
    )


def test_sweep_agent_covers_cells(env):
    rng = np.random.default_rng(0)
    agent = SweepAgent(rng)

    visited_counts = []
    for _ in range(_N_STEPS):
        action = agent.act(env)
        assert action in env.action_space.spaces.keys()
        env.step(action)
        visited_counts.append(len(agent._visited))

    assert agent._reachable, "no reachable cells found on the navmesh"
    assert visited_counts[-1] > 0

    for prev, curr in zip(visited_counts, visited_counts[1:]):
        if curr < prev:
            assert prev == len(agent._reachable), (
                f"visited count dropped from {prev} to {curr} without having swept all "
                f"{len(agent._reachable)} reachable cells first"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
