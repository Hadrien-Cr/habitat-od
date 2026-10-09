"""Uniformly random policy over a fixed action set."""
import habitat_sim
import numpy as np

from common.agents.abstract_agent import Agent


class RandomAgent(Agent):
    def __init__(self, rng: np.random.Generator, actions: list[str]):
        self.rng = rng
        self.actions = actions

    def act(self, agent_state: habitat_sim.AgentState) -> str:
        return self.rng.choice(self.actions)
