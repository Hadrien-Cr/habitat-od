
import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import random

import habitat # type: ignore
from habitat_sim.agent.agent import AgentState
from detectron2.data import (
    build_detection_test_loader, build_detection_train_loader, 
    MetadataCatalog, DatasetCatalog, DatasetMapper,
)
from detectron2.evaluation import inference_on_dataset, LVISEvaluator
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.data.samplers import InferenceSampler
from detectron2.data import detection_utils
from detectron2.data.build  import print_instances_class_histogram
from detectron2.data import transforms as T

from common.hssd_od_open_voc.hssd_env import HSSD_OpenVoc_Env
from common.vision.detic import build_detic_predictor
from common.utils.plot_utils import make_mosaic, plot_segmentation_gt, plot_segmentation_pred
from habitat_transfer_od.utils import do_train, InMemoryMapper
import habitat_od.od_dataset_registry

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")
logger.setLevel(logging.INFO)


def evaluate(predictor, dataset_name: str) -> None:
    test_dataset = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    logger = logging.getLogger("detectron2")
    print_instances_class_histogram(test_dataset, metadata.thing_classes)

    i = 0
    import random
    random.shuffle(test_dataset)

    mosaic = []
    for i, d in enumerate(test_dataset):
        img = detection_utils.read_image(d["file_name"])
        o = predictor(img)
    
        sem_im_pred = plot_segmentation_pred(img, o["instances"], metadata)
        sem_im_pred.save(f"vis_pred_{i}.jpg", format="JPEG")
        mosaic.append((f"test_{dataset_name}/vis{i}_pred", sem_im_pred))

        sem_im_gt = plot_segmentation_gt(img, d, metadata)
        sem_im_gt.save(f"vis_gt_{i}.jpg", format="JPEG")
        mosaic.append((f"test_{dataset_name}/vis{i}_gt", sem_im_gt))
        i += 1

        if i >= 20:
            break

    make_mosaic(mosaic, N_cols=2).save(f"mosaic_detic_{dataset_name}.png")

    test_loader = build_detection_test_loader(
        dataset=test_dataset,
        mapper=DatasetMapper(
            is_train=True,
            augmentations=[T.ResizeShortestEdge(640, 640, "choice")],
            image_format="BGR",
        ),
        sampler=InferenceSampler(len(test_dataset)),
        batch_size=4,
        num_workers=1,
    )
    evaluator = LVISEvaluator(
        dataset_name, output_dir="json_results"
    )

    results = inference_on_dataset(detic_predictor.model, test_loader, evaluator)


if __name__ == "__main__":
    for dataset_name in ["hssd_od_mpcat40_small"]:
        test_dataset = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        classes = metadata.thing_classes

        detic_config = OmegaConf.load("config/detic_config.yaml")
        detic_predictor = build_detic_predictor(detic_config, classes, dataset_name) # type: ignore
        detic_predictor.model.eval()

        evaluate(detic_predictor, dataset_name)