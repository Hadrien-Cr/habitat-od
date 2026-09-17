"""Live single-frame detectron2 inference, for scoring/agents during an active-learning
collection rollout (see habitat_embodied_al/reproduce/main.py) -- unlike pretrain.py/eval.py,
which only ever score a checkpoint against an already-built static COCO dataset offline, this
wraps a trained checkpoint to run it against one simulator frame at a time as it's collected.
"""
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision  # type: ignore
from torchvision.models.feature_extraction import create_feature_extractor  # type: ignore
from detectron2.config import get_cfg  # type: ignore
from detectron2.engine import DefaultPredictor  # type: ignore
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN  # type: ignore
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference  # type: ignore
from detectron2.structures import Instances  # type: ignore

_CROP_SIZE = 128


class Detector:
    """Wraps a trained detectron2 checkpoint for single-frame inference. `classes` fixes
    MODEL.ROI_HEADS.NUM_CLASSES (can't be read from a registered dataset here, since there
    isn't one -- frames are scored one at a time, not through a COCOEvaluator/test loader) and
    gives predicted class_ids a name via `class_name`."""

    def __init__(
        self, config_file: str, weights_path: str, classes: list,
        score_thresh: Optional[float] = 0.5, agnostic_nms_thresh: float = 0.5,
    ):
        self.classes = classes
        self.agnostic_nms_thresh = agnostic_nms_thresh
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
        cfg.MODEL.WEIGHTS = weights_path
        if score_thresh is not None:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        self._predictor = DefaultPredictor(cfg)
        self._device = torch.device(cfg.MODEL.DEVICE)
        resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self._feature_extractor = create_feature_extractor(resnet, return_nodes={"avgpool": "features"})
        self._feature_extractor.to(self._device).eval()

    def _crop_features(self, rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Frozen-ImageNet-ResNet18 embedding of each detected box's own cropped+resized
        patch -- independent of this Detector's own (actively retraining) weights, so
        similarity.py's diversity distance stays comparable across active-learning rounds
        instead of drifting with every retrain. Ported from embodied-active-learning-od's
        AbstractDetector._build_prediction."""
        height, width = rgb.shape[:2]
        crops = []
        for x0, y0, x1, y1 in boxes:
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(width, int(x1)), min(height, int(y1))
            crop = (
                cv2.resize(rgb[y0:y1, x0:x1], (_CROP_SIZE, _CROP_SIZE))
                if x1 > x0 and y1 > y0 else np.zeros((_CROP_SIZE, _CROP_SIZE, 3), dtype=np.uint8)
            )
            crops.append(crop)
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)

        batch = torch.as_tensor(np.stack(crops)).permute(0, 3, 1, 2).float().to(self._device) / 255.0
        with torch.no_grad():
            features = self._feature_extractor(batch)["features"].squeeze(-1).squeeze(-1).cpu().numpy()
        features = features.astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True) + 1e-10
        return features

    def predict(self, rgb: np.ndarray) -> Instances:
        """rgb: HxWx3 uint8 array, RGB order (as returned by habitat_sim's rgb sensor).
        DefaultPredictor always expects BGR (see its own docstring), so this flips channels
        before calling it. Manually replays DefaultPredictor/StandardROIHeads/
        FastRCNNOutputLayers's own inference path instead of just calling self._predictor(bgr)
        so the per-detection full class-probability vector -- normally discarded after NMS --
        survives as an extra Instances field for scoring (common/samplers/scoring.py's entropy
        rule). box_features (common/samplers/similarity.py's diversity distance) come from a
        separate frozen crop-feature extractor (see _crop_features), not this detector's own
        RoI head. Returns detectron2 Instances (pred_boxes/pred_classes/scores/class_probs/
        box_features) on CPU, image-sized.

        fast_rcnn_inference's own NMS is per-class (detectron2 has no config flag for
        class-agnostic NMS -- it always groups by predicted class, see
        fast_rcnn_inference_single_image's batched_nms call), so the same object can survive
        as two overlapping boxes under different classes. Unless self.agnostic_nms_thresh is 0
        (disabled), a second torchvision.ops.nms pass (no class grouping) runs after it to
        collapse those."""
        bgr = rgb[:, :, ::-1]
        height, width = bgr.shape[:2]
        model = self._predictor.model

        with torch.no_grad():
            image = self._predictor.aug.get_transform(bgr).apply_image(bgr)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}

            images = model.preprocess_image([inputs])
            features = model.backbone(images.tensor)
            proposals, _ = model.proposal_generator(images, features, None)

            roi_heads = model.roi_heads
            pooled = roi_heads.box_pooler([features[f] for f in roi_heads.box_in_features], [p.proposal_boxes for p in proposals])
            box_features = roi_heads.box_head(pooled)
            predictions = roi_heads.box_predictor(box_features)

            boxes = roi_heads.box_predictor.predict_boxes(predictions, proposals)
            probs = roi_heads.box_predictor.predict_probs(predictions, proposals)
            instances, kept = fast_rcnn_inference(
                boxes, probs, [p.image_size for p in proposals],
                roi_heads.box_predictor.test_score_thresh,
                roi_heads.box_predictor.test_nms_thresh,
                roi_heads.box_predictor.test_topk_per_image,
            )
            instances[0].class_probs = probs[0][kept[0]]

            result = GeneralizedRCNN._postprocess(instances, [inputs], images.image_sizes)[0]["instances"]

        result = result.to("cpu")
        if self.agnostic_nms_thresh > 0.0 and len(result) > 0:
            keep = torchvision.ops.nms(result.pred_boxes.tensor, result.scores, self.agnostic_nms_thresh)
            result = result[keep]
        result.box_features = torch.from_numpy(self._crop_features(rgb, result.pred_boxes.tensor.numpy()))
        return result

    def class_name(self, class_id: int) -> str:
        return self.classes[class_id]
