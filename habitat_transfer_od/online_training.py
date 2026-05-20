
import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import random
import torch
import torch.nn as nn
from detectron2.utils.events import (
    EventStorage,
)
from common.utils.dataset_utils import make_dataset_dict
import habitat # type: ignore
from detectron2.data import (
    build_detection_test_loader, build_detection_train_loader, 
    MetadataCatalog, DatasetCatalog, DatasetMapper,
)
from detectron2.data.build  import print_instances_class_histogram
from detectron2.data import transforms as T

from common.hssd_od_open_voc.hssd_env import HSSD_OpenVoc_Env
from common.vision.detic import build_detic_model
from common.utils.plot_utils import make_mosaic, plot_segmentation_gt, plot_segmentation_pred
from habitat_transfer_od.utils import InMemoryMapper
import habitat_od.od_dataset_registry
from detectron2.structures import Instances, Boxes
from detectron2.data.samplers import TrainingSampler, InferenceSampler
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")
logger.setLevel(logging.INFO)

from detectron2.data import detection_utils


def online_training(model: nn.Module, dataset_name: str, config: DictConfig, vocab_name: str) -> None:
    test_dataset = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    classes = metadata.thing_classes
    print_instances_class_histogram(test_dataset, metadata.thing_classes)

    rng_gen = np.random.default_rng(0)

    habitat_env = HSSD_OpenVoc_Env(config=config, vocab_name=vocab_name)

    class_mapping = habitat_env.get_class_mapping()

    online_buffer = []

    DatasetCatalog.register("online_agent_data", lambda: online_buffer)
    MetadataCatalog.get("online_agent_data").set(thing_classes=classes)

    os.makedirs(config.ONLINE_TRAINING.collection_dir, exist_ok=True)
    os.system(f"rm -rf {config.ONLINE_TRAINING.collection_dir}/*")
    os.makedirs(Path(config.ONLINE_TRAINING.collection_dir) / 'collected', exist_ok=True)
    os.makedirs(Path(config.ONLINE_TRAINING.collection_dir) / 'annotated', exist_ok=True)

    def acquisition_fn(obs, labels):
        if len(labels.instances) >= 5:
            return True
        return False

    pbar = tqdm(range(10), desc="Online Training Episodes")
    
    in_memory_mapper = InMemoryMapper()

    with EventStorage(0) as storage:
            
        for episode in pbar:
            pbar.set_postfix({"collected": len(online_buffer)})
            habitat_env.reset()

            objid_to_class = habitat_env.get_objid_to_class()
            object_id = int(habitat_env._current_episode.episode_id.split("_obj_id_")[-1])

            class_name = objid_to_class[object_id]

            viewpoints = habitat_env.get_episode_viewpoints()
            rng_gen.shuffle(viewpoints) # type: ignore

            # Collect data
            model.eval()
            for step in range(config.ONLINE_TRAINING.steps_per_episode):
                if step >= len(viewpoints):
                    break

                vp = viewpoints[step]
                obs, labels = habitat_env.get_obs_gt(vp)
                dataset_dict = make_dataset_dict(obs.rgb, labels, classes)

                if acquisition_fn(obs, labels):
                    sem_img = plot_segmentation_gt(obs.rgb, dataset_dict=dataset_dict, metadata=metadata)
                    sem_img.save(Path(config.ONLINE_TRAINING.collection_dir) / 'annotated' / f"ep{episode}_step{step}.jpg")
                    instances = model([in_memory_mapper(dataset_dict)])[0]
                    sem_img_pred = plot_segmentation_pred(obs.rgb, instances=instances["instances"], metadata=metadata)
                    sem_img_pred.save(Path(config.ONLINE_TRAINING.collection_dir) / 'annotated' / f"ep{episode}_step{step}_pred.jpg")
                    online_buffer.append(dataset_dict)

                else:
                    sem_img = plot_segmentation_gt(obs.rgb, dataset_dict=dataset_dict, metadata=metadata)
                    sem_img.save(Path(config.ONLINE_TRAINING.collection_dir) / 'collected' / f"ep{episode}_step{step}.jpg")

            model.zero_grad()
            # Dummy training loop
            model.train()

            online_ds = DatasetCatalog.get("online_agent_data")
            online_dataloader = build_detection_train_loader(
                dataset=online_ds,
                total_batch_size=config.ONLINE_TRAINING.batch_size,
                mapper=in_memory_mapper,
                # sampler=TrainingSampler(len(online_buffer), shuffle=True, seed=0),
                sampler=InferenceSampler(len(online_buffer)),
                num_workers=1,
            )

            for batch in online_dataloader:
                losses = model(batch)
                print("--------")
                print(sum(losses.values()))
                
            break


if __name__ == "__main__":
    dataset_name = "hssd_od_scannet200_small"
    vocab_name = "SCANNET200"

    test_dataset = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)
    classes = metadata.thing_classes

    detic_config = OmegaConf.load("config/detic_config.yaml")
    detic_model = build_detic_model(detic_config, classes, vocab_name=vocab_name) # type: ignore
    logger.info(detic_model)
    config = habitat.get_config(config_path="config/habitat_transfer_od.yaml")
    online_training(detic_model, dataset_name, config, vocab_name=vocab_name)