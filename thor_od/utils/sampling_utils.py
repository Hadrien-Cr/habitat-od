from collections import Counter
from common.utils.pose_utils import DiscretizedAgentPose
import numpy as np

def balanced_subsampling(samples: list[tuple[DiscretizedAgentPose, np.ndarray, list[dict]]], num_samples: int):

