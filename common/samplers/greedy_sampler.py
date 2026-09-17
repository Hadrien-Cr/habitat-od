"""Score-and-take-top-k sample selection -- see abstract_sampler.py. Ported from
embodied-active-learning-od's baselines/samplers/greedy_sampler.py, including its `env`/graph
plumbing for discrepancy scoring (get_state_value there looks up transition_graph neighbors;
here score_count_discrepancy looks up pose-adjacent neighbors in the pool passed to select(),
see common/samplers/scoring.py)."""
from typing import Optional

from common.samplers.abstract_sampler import Sampler
from common.samplers.scoring import score_count, score_count_discrepancy, score_entropy, score_oracle
from common.utils.interface import Candidate

METHODS = ["count", "discrepancy-total-count", "entropy-mean", "entropy-total", "oracle"]


class GreedySampler(Sampler):
    def __init__(
        self, method: str, classes: Optional[list] = None,
        width: Optional[int] = None, height: Optional[int] = None,
        grid_size: Optional[float] = None, num_yaw: Optional[int] = None, min_area: Optional[float] = None,
    ):
        assert method in METHODS, f"method {method!r} not in {METHODS}"
        if method == "oracle":
            assert classes is not None, "oracle scoring needs `classes` to name each GT box's class"
        if method == "discrepancy-total-count":
            assert None not in (width, height, grid_size, num_yaw, min_area), \
                "discrepancy-total-count scoring needs width/height/grid_size/num_yaw/min_area"
        self.method = method
        self.classes = classes
        self.width, self.height, self.grid_size, self.num_yaw, self.min_area = width, height, grid_size, num_yaw, min_area

    def score(self, candidate: Candidate, pool: list[Candidate], classwise_ap: Optional[dict]) -> float:
        if self.method == "count":
            return score_count(candidate)
        if self.method == "discrepancy-total-count":
            assert self.width is not None and self.height is not None and self.grid_size is not None \
                and self.num_yaw is not None and self.min_area is not None
            return score_count_discrepancy(candidate, pool, self.width, self.height, self.grid_size, self.num_yaw, self.min_area)
        if self.method == "entropy-mean":
            return score_entropy(candidate, mode="mean")
        if self.method == "entropy-total":
            return score_entropy(candidate, mode="total")
        return score_oracle(candidate, self.classes, classwise_ap)

    def select(self, candidates: list[Candidate], budget: int, *, annotated: Optional[list] = None, classwise_ap: Optional[dict] = None) -> list:
        assert len(candidates) >= budget

        ranked = sorted(candidates, key=lambda c: self.score(c, candidates, classwise_ap), reverse=True)
        return ranked[:budget]
