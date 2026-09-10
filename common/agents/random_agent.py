"""Broad random-exploration policy -- the counterpart of collection.py::collect_random's inline
loop, restructured behind the Agent interface so it can be swapped for a smarter policy in
habitat_embodied_al/reproduce/main.py's active-learning loop without changing that loop. Picks a
random navmesh destination and asks env for the discrete-action trajectory to it
(env.find_shortest_path -- habitat_sim's GreedyGeodesicFollower, the real navmesh geodesic
shortest path), one action per act() call; replans to a fresh destination every `horizon`
actions (whether or not the previous one was reached)."""
import numpy as np

from common.agents.abstract_agent import Agent
from common.env_utils.env_base import ExplorationEnv

_MAX_GOAL_RETRIES = 10


class RandomAgent(Agent):
    # Random Goal Agent
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.goal: np.ndarray = None  # current destination, kept for callers that visualize it
        self._trajectory: list = []

    def act(self, env: ExplorationEnv) -> str:
        if not self._trajectory:
            self._trajectory = self._plan_to_random_goal(env)

        return self._trajectory.pop(0)

    def _plan_to_random_goal(self, env: ExplorationEnv) -> list:
        for _ in range(_MAX_GOAL_RETRIES):
            goal = env.get_random_point(self.rng, min_distance=1.0, max_retries=100)
            trajectory = env.find_shortest_path(goal)
  
            if trajectory:
                self.goal = goal
                return trajectory
        
        raise RuntimeError(f"Failed to find a random goal after {_MAX_GOAL_RETRIES} retries")