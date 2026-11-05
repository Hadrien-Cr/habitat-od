"""Navigation-policy interface: picks the next vanilla habitat action from the agent's current pose."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple

import habitat_sim


class StopReason(str, Enum):
    """Why an agent's act() returned "stop"."""
    STUCK = "stuck"  # BaseExplorationAgent.is_stuck: no net translation over its window
    COLLISION = "collision"  # the last max_successive_collisions steps all collided
    EXPLORATION_EXHAUSTED = "exploration_exhausted"  # no goal (if any) and no frontier left to go to
    GOAL_REACHED = "goal_reached"  # DiscretePlanner's own stop, within goal_tolerance of a found goal
    NO_FEASIBLE_PLAN = "no_feasible_plan"  # DiscretePlanner's own stop: no heading gets closer to the goal, short of it
    MAX_STEPS = "max_steps"  # SearchAgent's own step budget ran out
    ORACLE_STOP = "oracle_stop"  # SearchAgent's use_oracle_path ablation: the oracle itself stopped


class Agent(ABC):
    @abstractmethod
    def act(self, agent_state: habitat_sim.AgentState) -> Tuple[str, Optional[StopReason]]:
        """Returns (the next action, e.g. "move_forward", "turn_left", "turn_right" or "stop"; the StopReason
        when the action is "stop", else None)."""
