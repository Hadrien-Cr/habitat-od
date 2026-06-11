import os
from typing import List, Tuple

import numpy as np
import pycocotools.mask as mask_util
import torch
import torchvision
from detectron2 import model_zoo
from detectron2.config import configurable, get_cfg
from detectron2.layers import (
    Conv2d,
    ConvTranspose2d,
    ShapeSpec,
    batched_nms,
    cat,
    get_norm,
)
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.structures.boxes import BoxMode
from detectron2.structures.masks import BitMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from torch.nn import functional as F


def fast_rcnn_inference(
    boxes,
    scores,
    image_shapes,
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    logits=None,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
            logits_per_image,
        )
        for scores_per_image, boxes_per_image, image_shape, logits_per_image in zip(
            scores, boxes, image_shapes, logits
        )
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    logits: None,
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # save the scores for later
    scores_all = scores

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = torchvision.ops.nms(boxes, scores, nms_thresh)
    # keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)

    if topk_per_image >= 0:
        keep = keep[:topk_per_image]

    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    filter_mask = filter_inds[:, 0]
    scores_all = scores_all[filter_mask]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    # result.scores = scores # original code
    if logits is not None:
        result.gt_logits = logits[filter_mask]

    result.scores = scores
    result.pred_classes = scores_all.max(1).indices  # filter_inds[:, 1]
    return result, filter_inds[:, 0]


def get_coco_item_dict(labels):
    instances = []

    for index, y in enumerate(labels):
        class_labels = y.gt_classes

        annotations = [
            {
                'bbox': y[id_instance].gt_boxes.tensor[0].tolist(),
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': class_labels[id_instance],
                'segmentation': mask_util.encode(
                    np.asfortranarray(y.gt_masks[id_instance])
                ),
                # TODO introduce 'uncertainties': y[id_instance].gt_uncertainty_masks[0],
                'iscrowd': 0,
                'infos': y[id_instance].infos[0],
                'gt_logits': y[id_instance].gt_logits[0],
            }
            for id_instance in range(len(y))
        ]

        instances.append(annotations)

    return instances
