"""Navigation-policy interface: picks the next vanilla habitat action from the agent's current pose."""
from abc import ABC, abstractmethod

import habitat_sim


class Agent(ABC):
    @abstractmethod
    def act(self, agent_state: habitat_sim.AgentState) -> str:
        """Returns the next action, e.g. "move_forward", "turn_left", "turn_right" or "stop"."""
