
import os

import habitat # type: ignore[import]
from habitat_baselines.common.baseline_registry import baseline_registry # type: ignore[import]

from common.env_utils.object_detector_sensors import *
from common.env_utils.sensors import *
from common.env_utils.env_base import *
from common.env_utils.dataset import *
from common.baselines.agents import *
from habitat.config.default_structured_configs import (
    ObjectDetectorGTSensorConfig,
) # type: ignore[import]


def collection(ds_name, habitat_config, scenes, steps_per_episode, cfg) -> None:
    dataset_root = f"data/{ds_name}"

    if os.path.exists(dataset_root):
        overwrite = input(f"Dataset {dataset_root} already exists. Do you want to overwrite it? [y/n] ")
        if overwrite != "y":
            print("Exiting without overwriting.")
            return
        
        os.system(f"rm -rf {dataset_root}")
    
    os.makedirs(dataset_root, exist_ok=True)

    with read_write(habitat_config):
        habitat_config.habitat.dataset.content_scenes = scenes
        habitat_config.habitat.environment.max_episode_steps = steps_per_episode
        habitat_config.habitat.task.lab_sensors = {
            "object_detector_gt": ObjectDetectorGTSensorConfig(**habitat_config.object_params),
            **habitat_config.habitat.task.lab_sensors
        }

    habitat_trainer_init = baseline_registry.get_trainer(habitat_config.habitat_baselines.trainer_name)
    habitat_trainer = habitat_trainer_init(habitat_config)
    habitat_trainer.collect(dataset_root, steps_per_episode=steps_per_episode)
    
    
if __name__ == "__main__":
    cfg = OmegaConf.load("config/maskrcnn_train.yaml")

    habitat_config = habitat.get_config(config_path="config/habitat/default.yaml")
    collection(cfg.collected_set, habitat_config, scenes = cfg.train_scenes, steps_per_episode=cfg.steps_per_episode, cfg=cfg)
    collection(cfg.test_set, habitat_config, scenes = cfg.val_scenes, steps_per_episode=cfg.steps_per_episode, cfg=cfg)