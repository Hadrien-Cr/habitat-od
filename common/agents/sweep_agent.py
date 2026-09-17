"""Sweep exploration: jumps to a random not-yet-visited navmesh cell and does a full rotation
at 3 vertical tilts there, repeating (resetting the visited set once every reachable cell has
been swept). Ported from
habitat_embodied_al/third_party/embodied-active-learning-od's agent_sweep.py::SweepJumpAgent.

The reference precomputes its *entire episode's* action sequence once, in an
`initialize(num_timesteps)` hook -- this repo's Agent.act(env) has no such hook and no
total-step-count argument (habitat_embodied_al/reproduce/main.py reuses one agent instance
across many collect_round calls of unknown total length, not one fixed-length episode), so this
ports as incremental per-cycle generation instead: one walk+sweep cycle is queued whenever the
action queue empties, rather than every cycle for the whole episode up front.

The reference's sweep is 2 full 45deg-step rotations plus a partial third, interleaved with
LookDown (3 vertical tilts). Ported using habitat's own look_up/look_down actions (registered in
common/config/hssd-hab/default.yaml, tilt_angle=30 to match this env's turn_angle): a full
rotation, look_down, another full rotation, look_down, a final rotation one turn short, then 2
look_up to restore a level camera before the next walk leg (habitat doesn't clamp tilt the way
AI2Thor's fixed vertical levels do)."""
from array import array
from typing import Optional

import numpy as np

from common.agents.abstract_agent import Agent
from common.agents.grid import reachable_cells
from common.env_utils.env_base import ExplorationEnv

_MAX_GOAL_RETRIES = 10


class SweepAgent(Agent):
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._reachable: Optional[set] = None
        self._visited: set = set()
        self._queue: list = []

    def act(self, env: ExplorationEnv) -> str:

        if self._reachable is None:
            self._reachable = reachable_cells(env.get_navmesh_grid())

        if not self._queue:
            self._queue = self._plan_next_cycle(env)

        self._visited.add(env.world_to_cell(env.get_agent_state().position))
        return self._queue.pop(0)

    def _plan_next_cycle(self, env: ExplorationEnv) -> list:
        tried: set = set()

        for _ in range(_MAX_GOAL_RETRIES):
            candidates = list(self._reachable - self._visited - tried)
            cell = candidates[self.rng.integers(len(candidates))]
            tried.add(cell)

            trajectory = env.find_shortest_path(env.cell_to_world(*cell))

            # sweep at pitch 0, -30, -60, then restore pitch to 0
            if trajectory:
                num_yaw = 360 // env.get_turn_angle()
                sweep = (
                    ["turn_right"] * num_yaw + ["look_down"]
                    + ["turn_right"] * num_yaw + ["look_down"]
                    + ["turn_right"] * (num_yaw - 1)
                    + ["look_up"] * 2
                )
                return trajectory + sweep

        raise RuntimeError(f"SweepAgent failed to find a reachable goal after {_MAX_GOAL_RETRIES} retries")
