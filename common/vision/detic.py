from omegaconf import OmegaConf, DictConfig

import sys
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path

from detectron2.data import transforms as T
from detectron2.config import CfgNode, get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor

sys.path.insert(0, "third_party/Detic/third_party/CenterNet2")
from centernet.config import add_centernet_config # type: ignore

sys.path.insert(0, "third_party/Detic/third_party/Deformable-DETR")
from third_party.Detic.detic.config import add_detic_config

from common.vision.clip import get_clip_embeddings, reset_cls_test
from common.utils.plot_utils import make_mosaic

DETIC_ROOT = "third_party/Detic"

def setup_cfg(detic_config: DictConfig) -> CfgNode:
    config_file = str(Path(DETIC_ROOT).resolve().parent / detic_config.config_file)
    weights = str(Path(DETIC_ROOT).resolve().parent / detic_config.weights)

    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(config_file)

    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = detic_config.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detic_config.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        detic_config.confidence_threshold
    )
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = str(
        Path(DETIC_ROOT)
        / cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
    )
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.freeze()
    return cfg


def build_detic_predictor(detic_config: CfgNode, vocab: list[str]):
    detic_config = OmegaConf.load("config/detic_config.yaml") # type: ignore
    cfg = setup_cfg(detic_config) # type: ignore

    predictor = DefaultPredictor(cfg)
    checkpointer = DetectionCheckpointer(predictor.model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    classifier = get_clip_embeddings(vocab)
    reset_cls_test(predictor.model, classifier, len(vocab))

    return predictor

# def process_detic_output(
#     output
# )
    
#     ds = DatasetCatalog.get(ds_name)
# import time
# for class_name in metadata.thing_classes:
#     list_fname_images = []
#     class_samples_ids = [
#         i for i in range(len(ds)) if ds[i]["file_name"].split("/")[-1].startswith("cls_" + class_name)
#     ]

#     if len(class_samples_ids) < 16:
#         continue

#     class_id = metadata.thing_classes.index(class_name)
    
#     for sample_id in class_samples_ids[0:16]:
#         d = ds[sample_id]
#         img = cv2.imread(d["file_name"])
    
#         t = time.time()
#         outputs = predictor(img)
#         print(predictor.model.device)
#         print(time.time() - t)

#         v = Visualizer(
#             img[:, :, ::-1],   # BGR -> RGB
#             metadata=metadata,
#             scale=0.5
#         )
#         instances = outputs["instances"].to("cpu")
#         keep = instances.pred_classes == class_id
#         # instances_filtered = instances[keep]
#         vis = v.draw_instance_predictions(instances)
#         result = vis.get_image()
#         list_fname_images.append((class_name, result ))

#     im = make_mosaic(list_fname_images)
#     im.save(f"mosaic_rare/{class_name}.png")
