"""First narrows the pool by score (GreedySampler), then diversifies within that narrowed
pool (DiversitySampler) -- see abstract_sampler.py. Ported from
embodied-active-learning-od's baselines/samplers/two_stage_sampler.py."""
from typing import Optional

from common.samplers.abstract_sampler import Sampler
from common.samplers.diversity_sampler import DiversitySampler
from common.samplers.greedy_sampler import GreedySampler
from common.utils.interface import Candidate


class TwoStageSampler(Sampler):
    def __init__(self, first: GreedySampler, second: DiversitySampler, budget_expanding_ratio: float = 2.0):
        self.first = first
        self.second = second
        self.budget_expanding_ratio = budget_expanding_ratio

    def top_n(self, candidates: list[Candidate], budget: int, *, annotated: Optional[list] = None, classwise_ap: Optional[dict] = None) -> list[Candidate]:
        """The first stage's own ranked output (descending by score) -- exposed separately so
        callers (e.g. habitat_embodied_al/collection.py's write_selection_mosaic) can visualize
        the pool select() narrows from, without re-running DiversitySampler's rng-consuming
        second stage a second time."""
        first_stage_budget = min(len(candidates), int(budget * self.budget_expanding_ratio))
        return self.first.select(candidates, first_stage_budget, annotated=annotated, classwise_ap=classwise_ap)

    def select(self, candidates: list[Candidate], budget: int, *, annotated: Optional[list] = None, classwise_ap: Optional[dict] = None) -> list:
        first_stage = self.top_n(candidates, budget, annotated=annotated, classwise_ap=classwise_ap)
        return self.second.select(first_stage, budget, annotated=annotated)
