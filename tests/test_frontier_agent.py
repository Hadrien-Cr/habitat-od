"""
Standalone test for common/agents/frontier_agent.py::FrontierAgent.

Drives the agent on a raw habitat_sim.Simulator (no habitat-lab Env/task) in
habitat-test-scenes' apartment_1 with vanilla move_forward/turn_left/turn_right actions, and checks
that it has revealed most of the navigable map once it stops. The final top-down map (navigable map,
revealed area, trajectory, circled closest frontier) and a mosaic of that map at every step are saved
to tests/testdump/test_frontier_agent/ for visual inspection.

Requires data/scene_datasets/habitat-test-scenes/apartment_1.glb - skipped otherwise.
"""

import os

import cv2
import habitat_sim
import numpy as np
import pytest
import shutil

from common.agents.frontier_agent import FrontierAgent, FrontierAgentConfig
from common.utils.plot_utils import make_mosaic

_SCENE = "data/scene_datasets/habitat-test-scenes/apartment_1.glb"
_FOV_DEG = 90.0
_FORWARD_STEP_M = 0.25
_TURN_ANGLE_DEG = 30.0
_MAX_STEPS = 1000
_MIN_COVERAGE = 0.9
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_frontier_agent")

pytestmark = pytest.mark.skipif(not os.path.exists(_SCENE), reason=f"requires {_SCENE}")


def _build_sim() -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = _SCENE
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=_FORWARD_STEP_M)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=_TURN_ANGLE_DEG)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=_TURN_ANGLE_DEG)),
    }
    return habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))


@pytest.fixture(scope="module")
def exploration_run():
    """(agent, actions, positions, revealed pixel counts, per-step map tiles) of one FrontierAgent run from a seeded random start."""
    sim = _build_sim()
    sim.pathfinder.seed(0)
    sim_agent = sim.get_agent(0)
    start = habitat_sim.AgentState()
    start.position = sim.pathfinder.get_random_navigable_point()
    sim_agent.set_state(start)

    agent = FrontierAgent(sim.pathfinder, FrontierAgentConfig(_FOV_DEG, _TURN_ANGLE_DEG), np.random.default_rng(0))
    actions, positions, revealed_counts, map_tiles = [], [np.array(start.position)], [], []
    for _ in range(_MAX_STEPS):
        actions.append(agent.act(sim_agent.get_state()))
        revealed_counts.append(int(agent.fog_of_war_mask.sum()))
        map_tiles.append(agent.map_tile())
        if actions[-1] == "stop":
            break
        sim.step(actions[-1])
        positions.append(np.array(sim_agent.get_state().position))
    yield agent, actions, np.array(positions), revealed_counts, map_tiles
    sim.close()


def _save_snapshot(agent: FrontierAgent, map_tiles: list) -> None:
    shutil.rmtree(_TESTDUMP_DIR, ignore_errors=True)
    os.makedirs(_TESTDUMP_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(_TESTDUMP_DIR, "map.png"), agent.map_image())
    make_mosaic(map_tiles, N_cols=10).save(os.path.join(_TESTDUMP_DIR, "map_mosaic.png"))


def test_exploration(exploration_run):
    agent, _, positions, _, map_tiles = exploration_run
    _save_snapshot(agent, map_tiles)
    navigable = agent.top_down_map > 0
    coverage = ((agent.fog_of_war_mask > 0) & navigable).sum() / navigable.sum()
    assert coverage >= _MIN_COVERAGE, f"revealed {coverage:.0%} of the navigable map in {len(positions) - 1} steps"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
