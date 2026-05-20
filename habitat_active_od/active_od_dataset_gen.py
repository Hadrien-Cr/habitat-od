import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import json
import gzip

import habitat # type: ignore
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1, NavigationEpisode, NavigationGoal # type: ignore

from common.hssd_od_open_voc.hssd_env import HSSD_OpenVoc_Env
from common.utils.plot_utils import plot_semantic_2d_map, make_mosaic


def collect_episodes_all_scenes(config) -> list:
    episodes = []
    rng_gen = np.random.default_rng(0)

    habitat_env = HSSD_OpenVoc_Env(config=config, vocab_name=config.DATA_GEN.vocab)
    scene_names = habitat_env.get_scenes_names()

    for scene_idx, scene in enumerate(scene_names[0:config.DATA_GEN.num_scenes]):
        habitat_env.change_scene(scene)

        print("-----------------")
        print("Collection in Scene = ", scene, f"({scene_idx}/{len(scene_names)})")

        habitat_obj_occupancy_grid = habitat_env.get_oracle_object_occupancy_grid(config.DATA_GEN.meters_per_grid_pixel)
        objid_to_class = habitat_env.get_objid_to_class()

        for object_id, class_name in tqdm(objid_to_class.items()):
            if class_name == "unknown":
                continue
            
            start = habitat_obj_occupancy_grid.get_all_agent_states()[0]
            candidate_agent_states = habitat_obj_occupancy_grid.get_all_viewpoints(object_id, viewpoint_spacing=config.DATA_GEN.viewpoint_spacing)
            viewpoints = []

            for agent_state in candidate_agent_states:
                obs, labels = habitat_env.get_obs_gt(agent_state)
                if not sum([inst["mask_area"] for inst in labels.instances if (inst["object_id"] == object_id)]) >= config.DATA_GEN.min_pixel_area:
                    continue
                viewpoints.append(agent_state)

            if len(viewpoints) < 3: 
                continue
            
            goals = [
                NavigationGoal(position=viewpoints[-1].position, radius=0)
            ]
            episode = NavigationEpisode(
                goals=goals,
                episode_id=f"{scene_idx}_obj_id_{object_id}",
                scene_dataset_config=config.habitat.simulator.scene_dataset,
                scene_id=scene,
                start_position=start.position,
                start_rotation=start.rotation,
            )
            
            episode.info = {}
            episode.info["viewpoints"] = [
                {
                    "position": v.position,
                    "rotation": v.rotation
                }
                for v in viewpoints
            ]
            episodes.append(episode)

    return episodes


if __name__ == "__main__":
    config = habitat.get_config(config_path="habitat_active_od/config/data_gen_config.yaml")
    episodes = collect_episodes_all_scenes(config)
    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = episodes
    out_file = config.DATA_GEN.dataset.data_path
    split = config.DATA_GEN.dataset.split
    out_file = out_file.replace('{split}', 'val')

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())
