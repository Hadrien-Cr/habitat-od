"""Pairwise frame-similarity for diversity_sampler.py's distance matrix. Ported from
habitat_embodied_al/third_party/embodied-active-learning-od's uutils/similarity_utils.py:
`feature`/`other_feature` are detector.py's `box_features`, a frozen ImageNet-ResNet18
embedding of each detection's own cropped patch (see Detector._crop_features) -- same
source as the original's `bbx_features`, just computed inline instead of via a separate
CLIP-ish extractor."""
import numpy as np
from detectron2.structures import Instances  # type: ignore


def _similarity_object_to_frame(class_id: int, feature: np.ndarray, instances: Instances) -> float:
    best_score = 0.0
    classes = instances.pred_classes.numpy() if len(instances) else np.zeros((0,), dtype=int)
    features = instances.box_features.numpy() if len(instances) else np.zeros((0, feature.shape[0]))

    for other_class_id, other_feature in zip(classes, features):
        if other_class_id != class_id:
            continue
        score = (1 + np.dot(feature, other_feature) / (np.linalg.norm(feature) * np.linalg.norm(other_feature) + 1e-10)) / 2
        best_score = max(best_score, score)
    return best_score


def similarity_score_asymmetric(instances1: Instances, instances2: Instances) -> float:
    if len(instances1) == 0 and len(instances2) == 0:
        return 1.0 - 1e-5
    if len(instances1) == 0 or len(instances2) == 0:
        return 1e-5

    classes = instances1.pred_classes.numpy()
    features = instances1.box_features.numpy()
    scores = instances1.scores.numpy()

    total_score = 0.0
    total_confidence = 0.0
    for class_id, feature, confidence in zip(classes, features, scores):
        total_score += confidence * _similarity_object_to_frame(class_id, feature, instances2)
        total_confidence += confidence

    score = total_score / total_confidence if total_confidence > 0 else 0.0
    return max(0.0, min(score, 1.0))


def similarity_score(instances1: Instances, instances2: Instances) -> float:
    """Symmetric [0, 1] similarity between two frames' predictions: same-class detections
    are matched by cosine similarity of their box_features, confidence-weighted and averaged
    in both directions."""
    score = (similarity_score_asymmetric(instances1, instances2) + similarity_score_asymmetric(instances2, instances1)) / 2
    return max(0.0, min(score, 1.0))
