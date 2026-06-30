from copy import copy, deepcopy
import os
import hydra
import pytorch_lightning as pl
from detectron2.utils.events import EventStorage # type: ignore
from habitat.config import read_write
from habitat_baselines.common.baseline_registry import baseline_registry # type: ignore[import]
from habitat_learn_od.utils.train_helpers import get_training_params
from habitat_learn_od.utils.data_modules import HabitatDataModule, GTDataModule
from habitat_learn_od.utils.teacher_student_modules import TeacherStudent
from common.env_utils.object_detector_sensors import * 
from common.env_utils.object_detector_sensors import *
from common.env_utils.sensors import *
from common.env_utils.env_base import *
from common.env_utils.dataset import *
from common.baselines.agents import *
from habitat.config.default_structured_configs import (
    ObjectDetectorGTSensorConfig,
) # type: ignore[import]


import matplotlib
matplotlib.use("Agg")

BASE_DIR = os.environ["BASE_DIR"]


def collection(ds_name, habitat_config, scenes, steps_per_episode, cfg) -> str:
    habitat_cfg = deepcopy(habitat_config)
    dataset_path = f"object_detection_dataset/{ds_name}"

    if os.path.exists(dataset_path):
        overwrite = input(f"Dataset {dataset_path} already exists. Do you want to overwrite it? [y/n] ")
        if overwrite != "y":
            print("Exiting without overwriting.")
            return ''
        
        os.system(f"rm -rf {dataset_path}")
    
    os.makedirs(dataset_path, exist_ok=True)

    with read_write(habitat_cfg):
        habitat_cfg.habitat.dataset.content_scenes = scenes
        habitat_cfg.habitat.environment.max_episode_steps = steps_per_episode
        habitat_cfg.habitat.task.lab_sensors = {
            "object_detector_gt": ObjectDetectorGTSensorConfig(**cfg.object_params),
            **habitat_cfg.habitat.task.lab_sensors
        }
    habitat_trainer_init = baseline_registry.get_trainer(habitat_cfg.habitat_baselines.trainer_name)
    habitat_trainer = habitat_trainer_init(habitat_cfg)
    habitat_trainer.collect(dataset_path, steps_per_episode)

    return dataset_path

@hydra.main(config_path="../config", config_name="train_hssd80_100x30.yaml")
def main(cfg):
    data_path = os.path.join(os.getcwd(), "data")
    if not (os.path.exists(data_path)):
        os.symlink(os.path.join(BASE_DIR, "data"), data_path)

    config_path = os.path.join(os.getcwd(), "config")
    if not (os.path.exists(config_path)):
        os.symlink(os.path.join(BASE_DIR, "config"), config_path)

    tp_path = os.path.join(os.getcwd(), "third_party")
    if not (os.path.exists(tp_path)):
        os.symlink(os.path.join(BASE_DIR, "third_party"), tp_path)

    habitat_od_data_path = os.path.join(os.getcwd(), "habitat_od_data")
    if not (os.path.exists(habitat_od_data_path)):
        os.symlink(os.path.join(BASE_DIR, "habitat_od_data"), habitat_od_data_path)
    
    from common.vision.detic import DETIC_ROOT
    os.symlink(os.path.join(DETIC_ROOT, "datasets"), os.path.join(os.getcwd(), "datasets"))
    os.symlink(os.path.join(DETIC_ROOT, "models"), os.path.join(os.getcwd(), "models"))
    os.symlink(os.path.join(DETIC_ROOT, "configs"), os.path.join(os.getcwd(), "configs"))

    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    habitat_config = habitat.get_config(config_path="config/habitat/default.yaml")

    if "collected_exp_name" not in cfg:
        collected_dataset_path = collection(cfg.collected_set, habitat_config, scenes = cfg.train_scenes, steps_per_episode=cfg.steps_per_episode, cfg=cfg)
        test_dataset_path = collection(cfg.test_set, habitat_config, scenes = cfg.val_scenes, steps_per_episode=cfg.steps_per_episode, cfg=cfg)    

        os.makedirs(os.path.join(BASE_DIR, 'habitat_od_data', cfg.exp_name, 'object_detection_dataset'), exist_ok=True)
        os.system(f"cp -r {collected_dataset_path} {os.path.join(BASE_DIR, 'habitat_od_data', cfg.exp_name, 'object_detection_dataset')}")
        os.system(f"cp -r {test_dataset_path} {os.path.join(BASE_DIR, 'habitat_od_data', cfg.exp_name, 'object_detection_dataset')}")

    else:
        collected_dataset_path = os.path.join(BASE_DIR, 'habitat_od_data', cfg.collected_exp_name, 'object_detection_dataset', cfg.collected_set)
        test_dataset_path = os.path.join(BASE_DIR, 'habitat_od_data', cfg.collected_exp_name, 'object_detection_dataset', cfg.test_set)

    teacher_student = TeacherStudent(**cfg,**cfg.training, **cfg.detic_args, device="cuda:0")
    trainer_config = get_training_params(cfg)
    dataset_path = os.path.join(os.getcwd(), collected_dataset_path)
    trainer = pl.Trainer(**trainer_config)    

    checkpoint_path = None

    with EventStorage(start_iter=0) as storage:
        for id_iteration in range(cfg.training.n_iterations):
            if 'use_gt' in cfg.training and cfg.training['use_gt']:
                dm = GTDataModule(
                    pseudo_labeler=teacher_student.pseudo_labeler,
                    collection_policy=None,
                    collected_dataset_path=collected_dataset_path,
                    test_dataset_path=test_dataset_path,
                    **cfg, # type: ignore
                    **cfg.training
                )
            else:
                dm = HabitatDataModule(
                    pseudo_labeler=teacher_student.pseudo_labeler,
                    collection_policy=None,
                    collected_dataset_path=dataset_path,
                    test_dataset_path=test_dataset_path,
                    **cfg, # type: ignore
                    **cfg.training
                )

            if checkpoint_path is not None:
                teacher_student.load_from_checkpoint(checkpoint_path)
            
            if id_iteration == 0:
                trainer.validate(model=teacher_student, datamodule=dm)

            trainer.fit(model=teacher_student, datamodule=dm)

            checkpoint_path = f"iteration-{id_iteration}.ckpt"
            trainer.save_checkpoint(checkpoint_path)

            trainer_config['max_epochs'] += cfg.training.epochs_per_iteration
            if "update_target" in cfg.training and cfg.training['update_target']:
                if not "ema" in cfg.training or not cfg.training.ema:
                    teacher_student.pseudo_labeler.reinit(teacher_student.online_network)


if __name__ == "__main__":
    main()