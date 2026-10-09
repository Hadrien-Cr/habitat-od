"""Score-and-take-top-k sample selection -- see abstract_sampler.py. Ported from
embodied-active-learning-od's baselines/samplers/greedy_sampler.py."""
from typing import Optional

from common.samplers.abstract_sampler import Sampler
from common.samplers.scoring import score_count, score_entropy, score_oracle
from common.utils.interface import Candidate

METHODS = ["count", "entropy-mean", "entropy-total", "oracle"]


class GreedySampler(Sampler):
    def __init__(self, method: str, classes: Optional[list] = None):
        assert method in METHODS, f"method {method!r} not in {METHODS}"
        if method == "oracle":
            assert classes is not None, "oracle scoring needs `classes` to name each GT box's class"
        self.method = method
        self.classes = classes

    def score(self, candidate: Candidate, classwise_ap: Optional[dict]) -> float:
        if self.method == "count":
            return score_count(candidate)
        if self.method == "entropy-mean":
            return score_entropy(candidate, mode="mean")
        if self.method == "entropy-total":
            return score_entropy(candidate, mode="total")
        return score_oracle(candidate, self.classes, classwise_ap)

    def select(self, candidates: list[Candidate], budget: int, *, annotated: Optional[list] = None, classwise_ap: Optional[dict] = None) -> list:
        assert len(candidates) >= budget

        ranked = sorted(candidates, key=lambda c: self.score(c, classwise_ap), reverse=True)
        return ranked[:budget]
