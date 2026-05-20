import copy
import os

import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.evaluation import (
    LVISEvaluator,
)
from detectron2.data import detection_utils
from omegaconf import OmegaConf
import habitat_od.od_dataset_registry



class InMemoryMapper:
    def __init__(self):
        self.augmentations = T.AugmentationList([
            T.ResizeTransform(h=480, w=640, new_h=800, new_w=1067, interp=2)
        ])

    def __call__(self, dataset_dict):
        """Uses same processing as DatasetMapper"""
        dataset_dict = copy.deepcopy(dataset_dict)
        image = dataset_dict["image"]   # HWC uint8 numpy
        detection_utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        
        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        annos = [
            detection_utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        instances = detection_utils.annotations_to_instances(
            annos, image_shape, mask_format="polygon"
        )

        filtered_instances = detection_utils.filter_empty_instances(instances)
        dataset_dict["instances"] = filtered_instances
        return dataset_dict


if __name__ == "__main__":
    metadata = MetadataCatalog.get("hssd_od_openvoc_full")
    classes = metadata.thing_classes
    
    from common.vision.detic import build_detic_predictor
    detic_config = OmegaConf.load("config/detic_config.yaml")
    detic_predictor = build_detic_predictor(detic_config, classes) # type: ignore

    model = detic_predictor.model

    cfg = get_cfg()
    raise ValueError(cfg)