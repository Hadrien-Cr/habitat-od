"""Box- (and point-) prompted instance masks with Segment Anything (ViT-B), ported from concept-graphs'
conceptgraph/utils/model_utils.py::get_sam_segmentation_from_xyxy_batched."""
from typing import List, Optional

import numpy as np
import torch
from detectron2.structures import BitMasks
from segment_anything import SamPredictor, sam_model_registry  # type: ignore

_MODEL_TYPE = "vit_b"


def mask_box(mask: np.ndarray) -> List[int]:
    return BitMasks(torch.as_tensor(mask)[None]).get_bounding_boxes().tensor[0].int().tolist()


class SamMasker:
    """One SAM mask per xyxy box prompt, the image encoded once per frame."""

    def __init__(self, checkpoint: str, mask_thresh: float = 0.5, device: str = "cuda"):
        self.predictor = SamPredictor(sam_model_registry[_MODEL_TYPE](checkpoint=checkpoint).to(device))
        self.predictor.model.mask_threshold = float(torch.logit(torch.tensor(mask_thresh)))

    def masks(self, rgb: np.ndarray, boxes: torch.Tensor) -> torch.Tensor:
        """(N, H, W) bool masks on CPU of the (N, 4) xyxy boxes of the uint8 RGB image rgb."""
        return self.prompted_masks(rgb, boxes, None, None)

    @torch.no_grad()
    def prompted_masks(
        self, rgb: np.ndarray, boxes: torch.Tensor, point_coords: Optional[torch.Tensor], point_labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """masks of the boxes, each also prompted with its (N, P, 2) xy points labelled 1 (in), 0 (out) or -1 (padding)."""
        if len(boxes) == 0:
            return torch.zeros((0, *rgb.shape[:2]), dtype=torch.bool)
        self.predictor.set_image(rgb)
        prompts = self.predictor.transform.apply_boxes_torch(boxes.to(self.predictor.device), rgb.shape[:2])
        if point_coords is not None:
            point_coords = self.predictor.transform.apply_coords_torch(point_coords.float().to(self.predictor.device), rgb.shape[:2])
            point_labels = point_labels.to(self.predictor.device)
        masks, _, _ = self.predictor.predict_torch(point_coords=point_coords, point_labels=point_labels, boxes=prompts, multimask_output=False)
        return masks[:, 0].cpu()
