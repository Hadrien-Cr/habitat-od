"""Sample-selection interface for active-learning collection (see
habitat_embodied_al/reproduce/main.py) -- a Sampler picks which of one round's collected
candidate frames get sent to the oracle for labeling (i.e. kept for training), under a fixed
per-round budget. Mirrors third_party/embodied-active-learning-od's data_collection/baselines/
samplers/ (greedy/diversity/two-stage/random -- common/samplers/{greedy,diversity,two_stage,
random}_sampler.py)."""
from abc import ABC, abstractmethod
from typing import Optional


class Sampler(ABC):
    @abstractmethod
    def select(self, candidates: list, budget: int, *, annotated: Optional[list] = None, classwise_ap: Optional[dict] = None) -> list:
        """Returns up to `budget` items from `candidates` to send to the oracle for labeling.
        `annotated`: every candidate already selected in earlier rounds (the growing labeled
        pool) -- only DiversitySampler/TwoStageSampler use this, to avoid re-picking
        near-duplicates of what's already been trained on. `classwise_ap`: per-class AP from
        the previous round's eval -- only the "oracle" scoring rule (common/samplers/
        scoring.py) uses this."""
