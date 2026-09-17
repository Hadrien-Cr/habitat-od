"""Frontier-based exploration: walks to the nearest not-yet-seen navmesh cell, replanning once
it arrives or once that cell gets marked seen first. Ported from
habitat_embodied_al/third_party/embodied-active-learning-od's agent_fbe.py::FrontierAgent,
adapted from AI2Thor's discretized 8-way transition-graph pathing + integer vision-cone grid
onto habitat-sim's real navmesh: "seen" cells are marked via a continuous distance/angle check
around the agent's live world pose (no 8-way lattice needed, unlike the reference's
YAW_TO_DX_DZ-indexed cone), and cell-to-cell travel reuses ExplorationEnv.find_shortest_path's
real navmesh geodesic path (GreedyGeodesicFollower) instead of a custom A* over a hand-rolled
transition graph -- so no fake terminal yaw/height is needed for a goal either, unlike the
reference (find_shortest_path takes a bare world position)."""
from typing import Optional

import numpy as np

from common.agents.abstract_agent import Agent
from common.agents.grid import reachable_cells
from common.env_utils.env_base import ExplorationEnv
from common.utils.pose_utils import forward_vector, get_yaw

_MAX_GOAL_RETRIES = 10


class FrontierAgent(Agent):
    def __init__(self, rng: np.random.Generator, cone_depth_m: float = 1.0, cone_half_angle_deg: float = 45.0):
        self.rng = rng
        self.cone_depth_m = cone_depth_m
        self.cone_half_angle_rad = np.deg2rad(cone_half_angle_deg)
        self._reachable: Optional[set] = None
        self._unexplored: Optional[set] = None
        self._goal_cell: Optional[tuple] = None
        self._trajectory: list = []

    def act(self, env: ExplorationEnv) -> str:
        if self._reachable is None:
            self._reachable = reachable_cells(env.get_navmesh_grid())
            self._unexplored = set(self._reachable)

        state = env.get_agent_state()
        agent_cell = env.world_to_cell(state.position)
        self._update_explored(env, state, agent_cell)

        if not self._trajectory or self._goal_cell is None:
            self._trajectory = self._plan_to_goal(env, agent_cell)

        return self._trajectory.pop(0)

    def _update_explored(self, env: ExplorationEnv, state, agent_cell: tuple) -> None:
        if not self._unexplored:
            return

        yaw = get_yaw(state.rotation.w, state.rotation.x, state.rotation.y, state.rotation.z)
        heading = forward_vector(yaw)
        cos_half_angle = np.cos(self.cone_half_angle_rad)

        radius_cells = int(np.ceil(self.cone_depth_m / env._GROUND_FLOOR_MPP))
        for row in range(agent_cell[0] - radius_cells, agent_cell[0] + radius_cells + 1):
            for col in range(agent_cell[1] - radius_cells, agent_cell[1] + radius_cells + 1):
                cell = (row, col)
                if cell not in self._unexplored:
                    continue

                delta = env.cell_to_world(row, col) - state.position
                dist = float(np.hypot(delta[0], delta[2]))
                if dist > self.cone_depth_m:
                    continue

                # Cells within ~one grid cell of the agent's true (unquantized) position are
                # always "seen" regardless of facing -- at this range, cell-center snapping
                # error puts `delta` in a near-random direction, which would otherwise fail
                # the forward-angle check on pure quantization noise (observed: the agent's
                # own cell failing this check, getting re-picked as the nearest goal, and
                # deadlocking on a zero-length path forever).
                in_view = dist <= env._GROUND_FLOOR_MPP or (heading[0] * delta[0] + heading[2] * delta[2]) / dist >= cos_half_angle
                if in_view:
                    self._unexplored.discard(cell)
                    if cell == self._goal_cell:
                        self._goal_cell = None  # currently-pursued goal got seen first -- replan

    def _pick_goal(self, agent_cell: tuple, exclude: set) -> tuple:
        candidates = self._unexplored - exclude - {agent_cell}
        if candidates:
            return min(candidates, key=lambda c: abs(c[0] - agent_cell[0]) + abs(c[1] - agent_cell[1]))

        fallback = list(self._reachable - exclude - {agent_cell})  # exploration done -- degrade to a random walk
        return fallback[self.rng.integers(len(fallback))]

    def _plan_to_goal(self, env: ExplorationEnv, agent_cell: tuple) -> list:
        tried: set = set()
        for _ in range(_MAX_GOAL_RETRIES):
            goal_cell = self._pick_goal(agent_cell, tried)
            tried.add(goal_cell)

            trajectory = env.find_shortest_path(env.cell_to_world(*goal_cell))
            if "move_forward" in trajectory:
                self._goal_cell = goal_cell
                return trajectory
            # empty (navmesh mask doesn't encode connectivity) or a trivial already-there
            # ["stop"] plan (goal within the geodesic follower's own goal radius, closer than
            # this grid's resolution can usefully separate from agent_cell) -- neither is a
            # real destination; mark it seen (we're already standing right there) and retry.
            self._unexplored.discard(goal_cell)

        raise RuntimeError(f"FrontierAgent failed to find a reachable goal after {_MAX_GOAL_RETRIES} retries")
