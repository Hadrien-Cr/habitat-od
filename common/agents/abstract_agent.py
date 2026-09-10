"""Navigation-policy interface for active-learning collection (see
habitat_embodied_al/reproduce/main.py) -- an Agent picks the next discrete action (one of
env.action_space's keys, e.g. "move_forward"/"turn_left"/"turn_right") to step the simulator
agent with, each step, over an ExplorationEnv. Mirrors third_party/embodied-active-learning-od's
data_collection/baselines/agents/ (ada/fbe/sweep/dqn/random) -- only random_agent.py is ported
so far; smarter agents will need more from ExplorationEnv than this interface currently
requires (e.g. a discretized transition graph, occupancy/frontier access)."""
from abc import ABC, abstractmethod

from common.env_utils.env_base import ExplorationEnv


class Agent(ABC):
    @abstractmethod
    def act(self, env: ExplorationEnv) -> str:
        """Returns the next action (a key of env.action_space) to step the agent with."""
