"""Sample-selection interface for active-learning collection (see
habitat_embodied_al/reproduce/main.py) -- a Sampler picks which of one round's collected
candidate frames get sent to the oracle for labeling (i.e. kept for training), under a fixed
per-round budget. Mirrors third_party/embodied-active-learning-od's data_collection/baselines/
samplers/ (greedy/diversity/two-stage/random) -- only random_sampler.py is ported so far;
scoring-based samplers will need each candidate to carry its detector predictions (already
collected, see main.py's Candidate), which this interface already passes through untouched."""
from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def select(self, candidates: list, budget: int) -> list:
        """Returns up to `budget` items from `candidates` to send to the oracle for labeling."""
