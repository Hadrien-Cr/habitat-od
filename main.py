import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

import habitat
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig

from hssd_od_open_voc.hssd_open_voc_env import HSSD_OpenVoc_Env
from habitat_od.data_gen import query_collection
from habitat_od.utils.grid_utils import HabitatSemGrid
from habitat_od.utils.data_utils import agent_state2fname
from habitat_od.utils.dataset_utils import save_dataset
from habitat_od.utils.plot_utils import plot_object_detection, plot_segmentation, plot_semantic_2d_map, make_mosaic
from habitat_od.utils.sampling_utils import area_bin_sampling


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

    os.system("rm -rf data_od")
    print(OmegaConf.to_yaml(config))
    rng_gen = np.random.default_rng(0)

    habitat_env = HSSD_OpenVoc_Env(config=config)

    scene_names = habitat_env.get_scenes_names()
    per_class_candidate_samples = defaultdict(list)

    for scene in scene_names:
        print("-----------------")
        print("Scene = ", scene)
        habitat_env.change_scene(scene)
        habitat_env.update_scene()

        habitat_grid = HabitatSemGrid(
            habitat_env.sim, 
            config.DATASET.meters_per_grid_pixel, 
            habitat_env.get_class_mapping(),
            habitat_env.get_objects()
        )
        # sem_grid = habitat_env.get_oracle_semantic_grid(meters_per_grid_pixel=0.1).sem_td_view

        # img = plot_semantic_2d_map(sem_grid, habitat_env.get_int2color(), habitat_env.get_class_mapping())
        # img.save(f"semantic_{scene}.png")

        for class_name in sorted(set(habitat_env.get_scene_annotations().values())):
            if class_name != "bed":
                continue
        
            candidate_agent_states = query_collection(
                habitat_grid=habitat_grid, 
                target_class=class_name,
                num_samples=200,
                rng_gen=rng_gen
            )
            rng_gen.shuffle(candidate_agent_states)
            
            candidate_samples = []
            
            for agent_state in tqdm(candidate_agent_states, desc=class_name):
                habitat_env.sim.agents[0].set_state(agent_state)
                obs = habitat_env.sim.get_sensor_observations()
                obs = {
                    key: obs[key]
                    for key in ["rgb", "depth", "semantic"]
                }
                masks = habitat_env.decompose_frame(obs["semantic"])
                subset_of_mask = [mask for mask in masks if mask["class_name"] == class_name]

                if not subset_of_mask:
                    continue

                candidate_samples.append((agent_state, obs, subset_of_mask))


            per_class_candidate_samples[class_name].extend(candidate_samples)
            
        break

    splits = {}

    for class_name in per_class_candidate_samples:
        candidate_samples = per_class_candidate_samples[class_name]
        
        selected_indices = area_bin_sampling(
            candidate_samples,
            rng_gen,
            num_samples=config.DATASET.num_samples,
            min_area=config.DATASET.min_pixel_area
        ) 
        rng_gen.shuffle(selected_indices)
        
        selected_samples = [candidate_samples[i] for i in selected_indices]
        
        list_fname_images_masks = [
            (agent_state2fname(scene, agent_state), obs["rgb"][:,:,0:3], masks) for (agent_state, obs, masks) in selected_samples
        ]

        if list_fname_images_masks:
            splits[class_name] = list_fname_images_masks


    save_dataset(config, splits, habitat_env.get_class_mapping())

    list_fname_images = [
        (
            str(fname), 
            plot_segmentation(
                image, 
                [], 
                masks, 
                habitat_env.get_name2color()
            ) 
        ) for class_name, list_fname_images_masks  in splits.items() for (fname, image, masks) in list_fname_images_masks[0:4] 
    ]
    mosaic = make_mosaic(list_fname_images[0:4*len(splits)], target_height = 200 * len(splits) )

    img = Image.fromarray(mosaic)
    img.save("mosaic.png")
    