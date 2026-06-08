import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

import habitat # type: ignore
from habitat.config import read_write # type: ignore
from habitat.config.default import get_agent_config # type: ignore
from habitat.core.env import Env # type: ignore
from habitat_baselines.common.baseline_registry import baseline_registry # type: ignore
from typing import Any

from common.env_utils.object_detector_sensors import *
from common.env_utils.sensors import *
from common.env_utils.env_base import *
from common.env_utils.dataset import *
from common.baselines.agents import *
from common.utils.data_utils import save_obs, load_data, _remove_data, SenseInfo
from common.utils.plot_utils import plot_segmentation_gt, plot_segmentation_pred
from common.utils.sampling_utils import area_bin_sampling
from common.utils.grid_utils import HabitatObjOccupancyGrid



def od_dataset_gen(config, habitat_config, split_name, scenes_to_include: list[str]) -> None:
    dataset_root = Path(config.DATA_GEN.dataset.data_root) / config.DATA_GEN.dataset.dataset_name

    if os.path.exists(dataset_root / split_name):
        overwrite = input(f"Dataset {dataset_root / split_name} already exists. Do you want to overwrite it? [y/n] ")
        if overwrite != "y":
            print("Exiting without overwriting.")
            return
        
        os.system(f"rm -rf {dataset_root / split_name}")

    os.makedirs(dataset_root / split_name, exist_ok=True)

    rng_gen = np.random.default_rng(0)

    with read_write(habitat_config):
        habitat_config.habitat.dataset.content_scenes = scenes_to_include
        habitat_config.habitat.task.lab_sensors = {
            "object_detector_gt": ObjectDetectorGTSensorConfig(
                area_thr=habitat_config.object_params.area_thr, 
                env_name=f"HSSD-HAB/{config.DATA_GEN.vocab}"
            ),
            **habitat_config.habitat.task.lab_sensors
        }

    rl_env = ExplorationEnv(config=habitat_config)
    
    from detectron2.data import MetadataCatalog
    env_name = rl_env.get_env_name()
    classes = MetadataCatalog.get(env_name).thing_classes
    colors = MetadataCatalog.get(env_name).thing_colors
    class2int = {c: i for i, c in enumerate(classes)}

    per_scene_class_object_occurences = {class_name: 0 for class_name in classes}
    per_class_candidates_samples = {class_name: [] for class_name in classes}

    episode_id = 0
    for scene_idx, scene in enumerate(scenes_to_include):
        rl_env.change_scene(scene)

        objects_info_list = rl_env.get_objects_info()
        habitat_obj_occupancy_grid = HabitatObjOccupancyGrid(
            sim=rl_env.habitat_env.sim,
            meters_per_grid_pixel=config.DATA_GEN.meters_per_grid_pixel,
            list_object_info=objects_info_list,
        )

        print("-----------------")
        print("Collection in Scene = ", scene, f"({scene_idx + 1}/{len(scenes_to_include)})")

        pbar = tqdm(objects_info_list, desc="Processing objects")
    
        for obj_info in pbar:
            object_id = obj_info["object_id"]
            class_name = obj_info["class_name"]

            if class_name == "unknown":
                continue

            if per_scene_class_object_occurences[class_name] >= 5:
                continue
            
            pbar.set_description(f"Processing objects - {class_name:20s}")
            candidate_agent_states = habitat_obj_occupancy_grid.get_all_viewpoints(object_id, viewpoint_spacing=config.DATA_GEN.viewpoint_spacing)
            rng_gen.shuffle(candidate_agent_states) # type: ignore
            candidate_agent_states = candidate_agent_states[0:config.DATA_GEN.per_class_num_samples // 4]

            if not candidate_agent_states:
                continue

            per_scene_class_object_occurences[class_name] += 1
            episode_id += 1

            for step, agent_state in enumerate(candidate_agent_states):
                rl_env.habitat_env.sim.agents[0].set_state(agent_state)

                sim_obs = rl_env.habitat_env.sim.get_sensor_observations()
                task_obs = rl_env.habitat_env.task.sensor_suite.get_observations(
                    observations=sim_obs, episode=rl_env.habitat_env.current_episode
                )
                
                obs = {**sim_obs, **task_obs}
                gt_instances = obs["bbsgt"]["instances"]

                if not len(gt_instances):
                    continue
                
                paths = save_obs(
                    dataset_path = dataset_root / split_name,
                    episode_id = episode_id,
                    observations = [obs],
                    timestamp = step,
                    modalities = ["rgb", "bbsgt"],
                )
                sense_info = SenseInfo(base_path=dataset_root / split_name, mod="rgb", step=step, episode=episode_id, camera_id=0)
                assert os.path.exists(sense_info.get_path()), f"Saved path {sense_info.get_path()} does not exist."

                per_class_candidates_samples[class_name].append((sense_info, gt_instances))


    # Post processing: for each class, performs a downsampling to reach "per_class_num_samples"
    pbar = tqdm(per_class_candidates_samples.items(), desc="Post-Processing classes")

    for class_name, candidate_samples in pbar:
        pbar.set_description(f"Post-Processing classes - {class_name:20s}")

        if not candidate_samples:
            continue

        rng_gen.shuffle(candidate_samples)
        selected_indices = area_bin_sampling(
            [gt_instances for _, gt_instances in candidate_samples],
            rng_gen,
            mask_filtering_fn=lambda i, instances: (instances.pred_classes[i] == class2int[class_name]),
            area_calculation_fn=lambda i, instances: 
                ((instances.pred_boxes.tensor[i,2].item() - instances.pred_boxes.tensor[i,0].item())*(instances.pred_boxes.tensor[i,3].item() - instances.pred_boxes.tensor[i,1].item())),
            num_samples=config.DATA_GEN.per_class_num_samples,
        ) 
        assert len(selected_indices) <= config.DATA_GEN.per_class_num_samples

        selected_samples = [candidate_samples[i] for i in selected_indices]
        rejected_samples = [candidate_samples[i] for i in range(len(candidate_samples)) if i not in selected_indices]
        
        for (sense_info, gt_instances) in rejected_samples:
            _remove_data(
                dataset_path = dataset_root / split_name,
                episode_id = sense_info.episode,
                camera_id = sense_info.camera_id,
                timestamp = sense_info.step,
                modalities = ["rgb", "bbsgt"],
            )
        for (sense_info, gt_instances) in selected_samples:
            rgb = load_data(sense_info.get_path()).data
            sem_im = plot_segmentation_pred(rgb, gt_instances, classes, colors)
            sem_im.save(sense_info.get_path().replace("rgb", "segmentation_gt").replace(".npy", ".png"), format="JPEG")


if __name__ == "__main__":
    for config_path in [
        "config/datagen/data_gen_mpcat40.yaml", 
        "config/datagen/data_gen_nyu40.yaml", 
        "config/datagen/data_gen_scannet200.yaml", 
        "config/datagen/data_gen_coco80.yaml",
        "config/datagen/data_gen_hssd40.yaml",
        "config/datagen/data_gen_hssd500.yaml", 
    ]:
        habitat_config = habitat.get_config(config_path="config/habitat/default.yaml")
        config = OmegaConf.load(config_path)

        for i in range(len(config.DATA_GEN.dataset.splits)):
            split_name = config.DATA_GEN.dataset.splits[i].split_name
            od_dataset_gen(
                habitat_config=habitat_config,
                config=config, 
                split_name=split_name, 
                scenes_to_include=list(config.DATA_GEN.dataset.splits[i].scenes_to_include)
            )