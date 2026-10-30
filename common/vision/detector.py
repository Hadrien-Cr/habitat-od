"""Live single-frame Detic inference with class-agnostic detection: Detic's CenterNet2 proposals, refined by its
class-agnostic cascade box regression, scored by their objectness alone and kept through one class-agnostic NMS -- no detection vocabulary.
Every detection is of class 0 and carries its Detic region embedding; masks come from SAM prompted with the boxes, or from
Detic's class-agnostic mask head without a SAM checkpoint. Objectness alone also fires on parts (a table's legs, a chair's
back) that a detection vocabulary scores low; without_parts drops detections whose mask lies inside a larger one's,
leaving the whole (its call in predict is commented out). classify_objects, run on fused objects' views, and predict_target,
run live by habitat_od_search's SearchAgent, are the two callers that use a target vocabulary instead.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.data.transforms import Transform  # type: ignore
from detectron2.engine import DefaultPredictor  # type: ignore
from detectron2.modeling.postprocessing import detector_postprocess  # type: ignore
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference  # type: ignore
from detectron2.structures import Boxes, ImageList, Instances  # type: ignore

from common.utils.sam import SamMasker, mask_box
from common.vision.detic import Vocabulary, detic_cfg
from detic.modeling.utils import reset_cls_test  # type: ignore

ObjectViews = Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]



# def without_parts(instances: Instances, min_containment: float) -> Instances:
#     """instances without those whose mask lies at least min_containment inside a larger one's."""
#     if len(instances) < 2:
#         return instances
#     masks = instances.pred_masks.flatten(1).float()
#     areas = masks.sum(dim=1)
#     contained = (masks @ masks.T) / areas.clamp(min=1)[:, None] >= min_containment
#     larger = areas[None, :] > areas[:, None]
#     return instances[~(contained & larger).any(dim=1)]


class Detector:
    """Wraps a Detic checkpoint for class-agnostic single-frame inference; its zero-shot classifier is left random and
    unused, `vocabulary`'s classes are the ones classify_objects classifies among. predict_target instead resets the
    classifier to a single live target class, for Detic's own background-competed scoring (see predict_target)."""

    def __init__(
        self, config_file: str, weights_path: str, vocabulary: Vocabulary, sam_checkpoint: Optional[str], score_thresh: float = 0.4,
        nms_thresh: float = 0.5, mask_thresh: float = 0.5, device: str = "cuda", min_part_containment: float = 0.8,
        input_size: int = 800
    ):
        self.classes = list(vocabulary.classes)
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.min_part_containment = min_part_containment
        self.mask_thresh = mask_thresh
        self._target_embedding = None

        cfg = detic_cfg(config_file)
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.MASK_ON = sam_checkpoint is None
        cfg.MODEL.DEVICE = device
        cfg.INPUT.MIN_SIZE_TEST = input_size
        self._predictor = DefaultPredictor(cfg)
        self._device = torch.device(cfg.MODEL.DEVICE)
        self._class_embeddings = F.normalize(vocabulary.clip_embeddings(), dim=0).to(self._device)
        self.masker = None if sam_checkpoint is None else SamMasker(sam_checkpoint, mask_thresh, device=device)

    def _box_predictions(self, features: dict, proposals: list) -> Tuple[list, List[torch.Tensor]]:
        """DeticCascadeROIHeads._forward_box's boxes, refined through every stage, and (N, 2) [objectness, background] scores
        of their proposals."""
        roi_heads = self._predictor.model.roi_heads
        objectness = [p.scores if p.has("scores") else p.objectness_logits for p in proposals]
        box_features = [features[f] for f in roi_heads.box_in_features]
        image_sizes = [p.image_size for p in proposals]
        for stage in range(roi_heads.num_cascade_stages):
            if stage > 0:
                proposals = roi_heads._create_proposals_from_boxes(boxes, image_sizes, logits=[p.objectness_logits for p in proposals])
            predictions = roi_heads._run_stage(box_features, proposals, stage)
            boxes = roi_heads.box_predictor[stage].predict_boxes((predictions[0], predictions[1]), proposals)
        return boxes, [torch.stack([scores, torch.zeros_like(scores)], dim=1) for scores in objectness]

    def _backbone(self, rgb: np.ndarray) -> Tuple[dict, ImageList, Transform]:
        """Backbone features of rgb (RGB order, flipped to the BGR DefaultPredictor expects), its preprocessed images and resize transform."""
        bgr = rgb[:, :, ::-1]
        transform = self._predictor.aug.get_transform(bgr)
        image = torch.as_tensor(transform.apply_image(bgr).astype("float32").transpose(2, 0, 1))
        images = self._predictor.model.preprocess_image([{"image": image}])
        return self._predictor.model.backbone(images.tensor), images, transform

    def _region_embeddings(self, features: dict, boxes: Boxes) -> torch.Tensor:
        """(N, D) normalized projections of boxes' last-stage box-head features, the ones Detic contrasts with text embeddings."""
        roi_heads = self._predictor.model.roi_heads
        pooled = roi_heads.box_pooler([features[f] for f in roi_heads.box_in_features], [boxes])
        head_features = torch.flatten(roi_heads.box_head[-1](pooled), start_dim=1)
        return F.normalize(roi_heads.box_predictor[-1].cls_score.linear(head_features), dim=-1)

    @torch.no_grad()
    def region_class_softmax(self, embeddings: torch.Tensor) -> torch.Tensor:
        """(N, len(classes)) Detic's temperature-scaled cosine similarities of region embeddings to the class text embeddings,
        softmaxed across classes."""
        cls_score = self._predictor.model.roi_heads.box_predictor[-1].cls_score
        return (cls_score.norm_temperature * embeddings.to(self._device) @ self._class_embeddings).softmax(dim=1).cpu()

    def classify_objects(self, views: ObjectViews) -> Dict[int, Dict[int, torch.Tensor]]:
        """{object id: {frame id: softmax class probabilities of that view's mask-box region embedding}}."""
        return {
            object_id: dict(zip(frame_embeddings, self.region_class_softmax(torch.stack(list(frame_embeddings.values())))))
            for object_id, frame_embeddings in self._view_embeddings(views).items()
        }

    @torch.no_grad()
    def _view_embeddings(self, views: ObjectViews) -> Dict[int, Dict[int, torch.Tensor]]:
        """{object id: {frame id: region embedding of that view's mask box}}, one backbone pass per frame."""
        frame_views = defaultdict(list)
        for object_id, object_views in views.items():
            for frame_id, (rgb, mask) in object_views.items():
                frame_views[frame_id].append((object_id, rgb, mask_box(mask)))
        view_embeddings = defaultdict(dict)
        for frame_id, object_boxes in frame_views.items():
            features, _, transform = self._backbone(object_boxes[0][1])
            boxes = Boxes(torch.as_tensor(transform.apply_box(np.array([box for _, _, box in object_boxes])), dtype=torch.float32, device=self._device))
            for (object_id, _, _), embedding in zip(object_boxes, self._region_embeddings(features, boxes).cpu()):
                view_embeddings[object_id][frame_id] = embedding
        return view_embeddings

    def predict(self, rgb: np.ndarray) -> Instances:
        """rgb: HxWx3 uint8 array, RGB order. Returns detectron2 Instances (pred_boxes/pred_classes, all 0/scores, the
        objectness/region_features/pred_masks) on CPU, image-sized."""
        height, width = rgb.shape[:2]
        model = self._predictor.model

        with torch.no_grad():
            features, images, _ = self._backbone(rgb)
            proposals, _ = model.proposal_generator(images, features, None)
            boxes, scores = self._box_predictions(features, proposals)
            instances, _ = fast_rcnn_inference(
                boxes, scores, [p.image_size for p in proposals], self.score_thresh, self.nms_thresh,
                model.roi_heads.box_predictor[-1].test_topk_per_image,
            )
            instances[0].region_features = self._region_embeddings(features, instances[0].pred_boxes)
            if self.masker is None:
                instances = model.roi_heads.forward_with_given_boxes(features, instances)
            result = detector_postprocess(instances[0], height, width, mask_threshold=self.mask_thresh).to("cpu")

        if self.masker is not None:
            result.pred_masks = self.masker.masks(rgb, result.pred_boxes.tensor)
        # return without_parts(result, self.min_part_containment)
        return result

    def predict_target(self, rgb: np.ndarray, embedding: torch.Tensor) -> Tuple[Instances, float]:
        """Detic's own single-class zero-shot inference against the caller-supplied (D,) embedding (e.g. a class's
        CLIP text embedding): the classifier is reset to [embedding, implicit all-zero background] (reset_cls_test,
        skipped when embedding is unchanged from the last call), so a region scores high only if it beats every
        other region AND the background class on softmax probability -- unlike predict's class-agnostic objectness,
        this naturally yields 0 or very few detections instead of dozens of generic object proposals. Returns
        (Instances with pred_boxes/scores/pred_masks, image-sized, on CPU; the best score, or 0.0 without any
        instance). The caller owns the embedding (and its vocabulary, if any)."""
        if self._target_embedding is None or not torch.equal(self._target_embedding, embedding):
            reset_cls_test(self._predictor.model, embedding[:, None].to(self._device), 1)
            self._target_embedding = embedding
        with torch.no_grad():
            result = self._predictor(rgb[:, :, ::-1])["instances"].to("cpu")
        if self.masker is not None and len(result) > 0:
            result.pred_masks = self.masker.masks(rgb, result.pred_boxes.tensor)
        return result, result.scores.max().item() if len(result) > 0 else 0.0
