from dataclasses import dataclass
from typing import Any
import numpy as np
from detectron2.structures import Instances

@dataclass
class Candidate:
    rbg: np.ndarray
    bbsgt: dict
    colored_tdmap: np.ndarray
    pred_instances: Instances
    agent_state: Any