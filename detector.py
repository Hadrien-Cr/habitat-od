"""Live single-frame detectron2 inference, for scoring/agents during an active-learning
collection rollout (see habitat_embodied_al/reproduce/main.py) -- unlike pretrain.py/eval.py,
which only ever score a checkpoint against an already-built static COCO dataset offline, this
wraps a trained checkpoint to run it against one simulator frame at a time as it's collected.
"""
from typing import Optional

import numpy as np
from detectron2.config import get_cfg  # type: ignore
from detectron2.engine import DefaultPredictor  # type: ignore
from detectron2.structures import Instances  # type: ignore


class Detector:
    """Wraps a trained detectron2 checkpoint for single-frame inference. `classes` fixes
    MODEL.ROI_HEADS.NUM_CLASSES (can't be read from a registered dataset here, since there
    isn't one -- frames are scored one at a time, not through a COCOEvaluator/test loader) and
    gives predicted class_ids a name via `class_name`."""

    def __init__(self, config_file: str, weights_path: str, classes: list, score_thresh: Optional[float] = 0.5):
        self.classes = classes
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
        cfg.MODEL.WEIGHTS = weights_path
        if score_thresh is not None:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        self._predictor = DefaultPredictor(cfg)

    def predict(self, rgb: np.ndarray) -> Instances:
        """rgb: HxWx3 uint8 array, RGB order (as returned by habitat_sim's rgb sensor).
        DefaultPredictor always expects BGR (see its own docstring), so this flips channels
        before calling it. Returns detectron2 Instances (pred_boxes/pred_classes/scores) on
        CPU, image-sized."""
        bgr = rgb[:, :, ::-1]
        return self._predictor(bgr)["instances"].to("cpu")

    def class_name(self, class_id: int) -> str:
        return self.classes[class_id]
