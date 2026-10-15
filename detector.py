"""Live single-frame detectron2 inference, for scoring/agents during an active-learning
collection rollout (see habitat_embodied_al/reproduce/main.py) -- unlike pretrain.py/eval.py,
which only ever score a checkpoint against an already-built static COCO dataset offline, this
wraps a trained checkpoint to run it against one simulator frame at a time as it's collected.
Runs Detic checkpoints (third_party/Detic), whose zero-shot classifier is replaced by the CLIP text
embeddings of `classes`; masks come from SAM prompted with Detic's boxes, not Detic's own mask head.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision  # type: ignore
from torchvision.models.feature_extraction import create_feature_extractor  # type: ignore
from detectron2.engine import DefaultPredictor  # type: ignore
from detectron2.modeling.postprocessing import detector_postprocess  # type: ignore
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference  # type: ignore
from detectron2.data.transforms import Transform  # type: ignore
from detectron2.structures import BitMasks, Boxes, ImageList, Instances  # type: ignore

from common.utils.image_memory import ImageMemory
from common.utils.plot_utils import draw_ranked_classes
from common.utils.projection_utils import DepthCamera, ResolvedObjects
from common.utils.sam import SamMasker
from common.vision.detic import detic_cfg
from detic.modeling.roi_heads.detic_fast_rcnn import DeticFastRCNNOutputLayers  # type: ignore
from detic.modeling.utils import reset_cls_test  # type: ignore
from detic.predictor import get_clip_embeddings  # type: ignore

_CROP_SIZE = 128
_TOP_CLASSES = 5
_OBJECT_PROB_PLACEHOLDER = 1.0

ObjectViews = Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]]


def mean_view_class_probs(object_probs: Dict[int, Tuple[float, Dict[int, torch.Tensor]]]) -> Dict[int, torch.Tensor]:
    """{object id: mean of its views' class probabilities} of a Detector.classify_objects result."""
    return {object_id: torch.stack(list(frame_probs.values())).mean(dim=0) for object_id, (_, frame_probs) in object_probs.items()}


def _mask_box(mask: np.ndarray) -> List[int]:
    return BitMasks(torch.as_tensor(mask)[None]).get_bounding_boxes().tensor[0].int().tolist()


class Detector:
    """Wraps a trained detectron2 checkpoint for single-frame inference. `classes` fixes
    MODEL.ROI_HEADS.NUM_CLASSES (can't be read from a registered dataset here, since there
    isn't one -- frames are scored one at a time, not through a COCOEvaluator/test loader) and
    gives predicted class_ids a name via `class_name`."""

    def __init__(
        self, config_file: str, weights_path: str, classes: list, sam_checkpoint: str,
        score_thresh: Optional[float] = 0.5, agnostic_nms_thresh: float = 0.5, device: str = "cuda",
    ):
        self.classes = classes
        self.agnostic_nms_thresh = agnostic_nms_thresh
        cfg = detic_cfg(config_file)
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.MASK_ON = False
        cfg.MODEL.DEVICE = device

        if score_thresh is not None:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        self._predictor = DefaultPredictor(cfg)
        reset_cls_test(self._predictor.model, get_clip_embeddings(classes), len(classes))
        self._device = torch.device(cfg.MODEL.DEVICE)
        resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self._feature_extractor = create_feature_extractor(resnet, return_nodes={"avgpool": "features"})
        self._feature_extractor.to(self._device).eval()
        self._masker = SamMasker(sam_checkpoint, device=device)

    def _box_predictions(self, features: dict, proposals: list) -> Tuple[tuple, tuple, DeticFastRCNNOutputLayers]:
        """DeticCascadeROIHeads._forward_box's inference: stage-averaged class probabilities, times the proposal score, on the last stage's boxes."""
        roi_heads = self._predictor.model.roi_heads
        proposal_scores = [p.scores if p.has("scores") else p.objectness_logits for p in proposals]
        box_features = [features[f] for f in roi_heads.box_in_features]
        image_sizes = [p.image_size for p in proposals]
        stage_probs = []

        for stage in range(roi_heads.num_cascade_stages):
            if stage > 0:
                proposals = roi_heads._create_proposals_from_boxes(boxes, image_sizes, logits=[p.objectness_logits for p in proposals])
            predictions = roi_heads._run_stage(box_features, proposals, stage)
            boxes = roi_heads.box_predictor[stage].predict_boxes((predictions[0], predictions[1]), proposals)
            stage_probs.append(roi_heads.box_predictor[stage].predict_probs(predictions, proposals))
        probs = [sum(cls_probs) / roi_heads.num_cascade_stages for cls_probs in zip(*stage_probs)]
        
        if roi_heads.mult_proposal_score:
            probs = [(cls_probs * box_scores[:, None]) ** 0.5 for cls_probs, box_scores in zip(probs, proposal_scores)]
        
        return boxes, probs, roi_heads.box_predictor[-1]

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

    def _backbone(self, rgb: np.ndarray) -> Tuple[dict, ImageList, Transform]:
        """Backbone features of rgb (RGB order, flipped to the BGR DefaultPredictor expects), its preprocessed images and resize transform."""
        bgr = rgb[:, :, ::-1]
        transform = self._predictor.aug.get_transform(bgr)
        image = torch.as_tensor(transform.apply_image(bgr).astype("float32").transpose(2, 0, 1))
        images = self._predictor.model.preprocess_image([{"image": image}])
        return self._predictor.model.backbone(images.tensor), images, transform

    def _region_embeddings(self, features: dict, boxes: Boxes) -> torch.Tensor:
        """(N, D) normalized last-stage embeddings of boxes, the ones Detic contrasts with the class text embeddings."""
        roi_heads = self._predictor.model.roi_heads
        pooled = roi_heads.box_pooler([features[f] for f in roi_heads.box_in_features], [boxes])
        head_features = torch.flatten(roi_heads.box_head[-1](pooled), start_dim=1)
        return F.normalize(roi_heads.box_predictor[-1].cls_score.linear(head_features), dim=-1)

    @torch.no_grad()
    def encode_regions(self, rgb: np.ndarray, boxes: np.ndarray) -> torch.Tensor:
        """(N, D) region embeddings of the xyxy boxes of rgb."""
        features, _, transform = self._backbone(rgb)
        return self._region_embeddings(features, Boxes(torch.as_tensor(transform.apply_box(boxes), dtype=torch.float32, device=self._device))).cpu()

    @torch.no_grad()
    def region_class_probs(self, embeddings: torch.Tensor) -> torch.Tensor:
        """(N, len(classes)) Detic's sigmoid class probabilities of region embeddings."""
        cls_score = self._predictor.model.roi_heads.box_predictor[-1].cls_score
        return (cls_score.norm_temperature * embeddings.to(self._device) @ cls_score.zs_weight[:, :-1]).sigmoid().cpu()

    def classify_objects(self, views: ObjectViews) -> Dict[int, Tuple[float, Dict[int, torch.Tensor]]]:
        """{object id: (p(object), {frame id: class probabilities of that view's mask-box region embedding})}."""
        embeddings = self._view_embeddings(views)
        view_probs = {
            object_id: dict(zip(frame_embeddings, self.region_class_probs(torch.stack(list(frame_embeddings.values())))))
            for object_id, frame_embeddings in embeddings.items()
        }
        return {object_id: (_OBJECT_PROB_PLACEHOLDER, frame_probs) for object_id, frame_probs in view_probs.items()}

    def classify_resolved(
        self, resolved: ResolvedObjects, memory: ImageMemory, camera: DepthCamera, min_visibility_fraction: float,
    ) -> ResolvedObjects:
        """resolved, each object taking the mean class probabilities of its views raytraced in memory (classify_objects)."""
        views = memory.object_views(resolved, camera, min_visibility_fraction)
        return resolved.classified(mean_view_class_probs(self.classify_objects(views)))

    def _view_embeddings(self, views: ObjectViews) -> Dict[int, Dict[int, torch.Tensor]]:
        """{object id: {frame id: region embedding of that view's mask box}}, one backbone pass per frame."""
        frame_views = defaultdict(list)
        for object_id, object_views in views.items():
            for frame_id, (rgb, mask) in object_views.items():
                frame_views[frame_id].append((object_id, rgb, _mask_box(mask)))
        embeddings = defaultdict(dict)
        for frame_id, object_boxes in frame_views.items():
            rgb = object_boxes[0][1]
            for (object_id, _, _), embedding in zip(object_boxes, self.encode_regions(rgb, np.array([box for _, _, box in object_boxes]))):
                embeddings[object_id][frame_id] = embedding
        return dict(embeddings)

    def draw_top_classes(self, class_probs: torch.Tensor, image: np.ndarray) -> np.ndarray:
        """image with the top classes of class_probs and their probabilities drawn on it."""
        top_probs, top_classes = class_probs.topk(min(_TOP_CLASSES, len(class_probs)))
        return draw_ranked_classes(image, [self.class_name(idx) for idx in top_classes.tolist()], top_probs.tolist())

    def predict(self, rgb: np.ndarray) -> Instances:
        """rgb: HxWx3 uint8 array, RGB order (as returned by habitat_sim's rgb sensor).
        DefaultPredictor always expects BGR (see its own docstring), so this flips channels
        before calling it. Manually replays DefaultPredictor/DeticCascadeROIHeads's own
        inference path (see _box_predictions) instead of just calling self._predictor(bgr)
        so the per-detection full class-probability vector -- normally discarded after NMS --
        survives as an extra Instances field for scoring (common/samplers/scoring.py's entropy
        rule). box_features (common/samplers/similarity.py's diversity distance) come from a
        separate frozen crop-feature extractor (see _crop_features), not this detector's own
        RoI head. Returns detectron2 Instances (pred_boxes/pred_classes/scores/class_probs/
        region_features (see _region_embeddings)/box_features/pred_masks (SAM, see SamMasker)) on CPU, image-sized.

        fast_rcnn_inference's own NMS is per-class (detectron2 has no config flag for
        class-agnostic NMS -- it always groups by predicted class, see
        fast_rcnn_inference_single_image's batched_nms call), so the same object can survive
        as two overlapping boxes under different classes. Unless self.agnostic_nms_thresh is 0
        (disabled), a second torchvision.ops.nms pass (no class grouping) runs after it to
        collapse those."""
        height, width = rgb.shape[:2]
        model = self._predictor.model

        with torch.no_grad():
            features, images, _ = self._backbone(rgb)
            proposals, _ = model.proposal_generator(images, features, None)

            boxes, probs, box_predictor = self._box_predictions(features, proposals)
            instances, kept = fast_rcnn_inference(
                boxes, probs, [p.image_size for p in proposals],
                box_predictor.test_score_thresh,
                box_predictor.test_nms_thresh,
                box_predictor.test_topk_per_image,
            )
            instances[0].class_probs = probs[0][kept[0]]
            instances[0].region_features = self._region_embeddings(features, instances[0].pred_boxes)
            result = detector_postprocess(instances[0], height, width)

        result = result.to("cpu")
        if self.agnostic_nms_thresh > 0.0 and len(result) > 0:
            keep = torchvision.ops.nms(result.pred_boxes.tensor, result.scores, self.agnostic_nms_thresh)
            result = result[keep]
        result.box_features = torch.from_numpy(self._crop_features(rgb, result.pred_boxes.tensor.numpy()))
        result.pred_masks = self._masker.masks(rgb, result.pred_boxes.tensor)
        return result

    def class_name(self, class_id: int) -> str:
        return self.classes[class_id]
