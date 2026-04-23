import numpy as np
from pathlib import Path

import habitat
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig

from habitat_od.data_gen import random_teleport_collection, get_sample
from hssd_od_open_voc.hssd_open_voc_env import HSSD_OpenVoc_Env, ClassMappingHSSD

from habitat.tasks.nav.object_nav_task import ObjectGoal, ObjectViewLocation

from habitat_od.data_gen import random_teleport_collection, query_collection
from habitat_od.utils.dataset_utils import save_dataset, pose2fname
from habitat_od.utils.grid_utils import HabitatSemGrid
from common.utils.plot_utils import plot_object_detection, plot_segmentation, plot_semantic_2d_map, make_mosaic

from omegaconf import OmegaConf
from PIL import Image

def edit(config):
    with read_write(config):
        config.habitat.dataset.split = "val"
        config.habitat.simulator.habitat_sim_v0.enable_physics = True # needed to interact with rigid object manager
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.update(
            {"semantic_sensor": HabitatSimSemanticSensorConfig(
                height=480, 
                width=640, 
                position=[0.0,0.88,0.0],
                hfov=79
            ),
            }
        )


if __name__ == "__main__": 
    config = habitat.get_config(config_path="config/dataset.yaml")
    edit(config)

    print(OmegaConf.to_yaml(config))

    habitat_env = HSSD_OpenVoc_Env(config=config)

    habitat_env.reset()   
    habitat_env.update_scene()

    objects = habitat_env.get_objects()
    
    ep_goal = habitat_env.get_episode_goal()

    data = []

    list_fname_images = []

    for class_name in sorted(set(habitat_env.get_scene_annotations().values())):
        print("query=",  class_name)
        poses = query_collection(habitat_env, config=config, query_class=class_name)
        sample = get_sample(habitat_env, poses[0])
        masks = habitat_env.get_mask(sample["semantic"])

        fname = pose2fname(class_name + habitat_env.get_scene_name(), poses[0])
        data.append((fname, sample["rgb"][:,:,0:3], masks))

        subset_of_mask = [mask for mask in masks if mask[0] == class_name]
        list_fname_images.append(
            (class_name, plot_segmentation(sample["rgb"][:,:,0:3], [], subset_of_mask, habitat_env.get_colors()))
        )

    splits = {config.DATASET.splits.test: data}
    save_dataset(config, splits, habitat_env.get_class_mapping())
    mosaic = make_mosaic(list_fname_images[0:16])

    img = Image.fromarray(mosaic)
    img.save("mosaic.png")
    