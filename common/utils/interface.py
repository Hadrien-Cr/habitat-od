from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
from detectron2.structures import Instances

@dataclass
class Candidate:
    rbg: np.ndarray
    bbsgt: dict
    colored_tdmap: np.ndarray
    pred_instances: Instances
    agent_state: Any
    action: Optional[str] = None
    disc_pose: Any = None  # common.samplers.discrepancy.DiscretizedPose, kept loosely typed to avoid a sampler-package import here
    round_idx: Optional[int] = None  # collection round disc_pose's odometry is local to -- neighbor lookups must stay within it
    filename: Optional[str] = None  # rgb file this candidate was first saved as, set once at collection time