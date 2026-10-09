"""Per-candidate scoring rules for GreedySampler/TwoStageSampler (common/samplers/
greedy_sampler.py) -- ported from habitat_embodied_al/third_party/embodied-active-learning-od's
uutils/ranking_utils.py::get_state_value (the "count"/"mean-entropy"/"total-entropy"/"oracle"
branches)."""
from typing import Optional

import numpy as np

from common.env_utils.sense import keep_valid_instances
from common.utils.interface import Candidate


def score_count(candidate: Candidate) -> float:
    return float(len(candidate.pred_instances))


def score_entropy(candidate: Candidate, mode: str = "mean") -> float:
    probs = candidate.pred_instances.class_probs if candidate.pred_instances.has("class_probs") else None
    if probs is None or len(probs) == 0:
        return -1.0 if mode == "mean" else 0.0

    probs = probs.numpy()
    entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
    return float(entropies.mean()) if mode == "mean" else float(entropies.sum())


def score_oracle(candidate: Candidate, classes: list, classwise_ap: Optional[dict]) -> float:
    if not classwise_ap:
        return -1.0

    gt = keep_valid_instances(candidate.bbsgt["instances"])
    if len(gt) == 0:
        return -1.0

    difficulty = 0.0
    any_scored = False
    for class_id in gt.pred_classes.tolist():
        class_name = classes[class_id]
        ap = classwise_ap.get(class_name)
        if ap is None or np.isnan(ap):
            continue
        any_scored = True
        difficulty += 1.0 - ap

    return difficulty if any_scored else -1.0
