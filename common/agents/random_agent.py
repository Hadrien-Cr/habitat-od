"""Uniformly random policy over a fixed action set."""
from typing import Optional, Tuple

import habitat_sim
import numpy as np

from common.agents.abstract_agent import Agent, StopReason


class RandomAgent(Agent):
    def __init__(self, rng: np.random.Generator, actions: list[str]):
        self.rng = rng
        self.actions = actions

    def act(self, agent_state: habitat_sim.AgentState) -> Tuple[str, Optional[StopReason]]:
        return self.rng.choice(self.actions), None
