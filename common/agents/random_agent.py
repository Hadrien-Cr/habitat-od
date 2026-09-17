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
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions available in the environment")
        return self.rng.choice(legal_actions)