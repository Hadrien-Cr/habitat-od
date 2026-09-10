"""Uniform-random sample selection -- the simplest baseline (see abstract_sampler.py)."""
import numpy as np

from common.samplers.abstract_sampler import Sampler
from common.utils.dataset_utils import instance_is_empty
from common.utils.interface import Candidate


class RandomSampler(Sampler):
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def select(self, candidates: list[Candidate], budget: int) -> list:
        non_empty_candidates = [c for c in candidates if not instance_is_empty(c.bbsgt["instances"])]

        idx = self.rng.choice(len(non_empty_candidates), size=budget, replace=False)

        assert len(idx) == budget and len(idx) < len(non_empty_candidates)

        return [non_empty_candidates[i] for i in idx]
