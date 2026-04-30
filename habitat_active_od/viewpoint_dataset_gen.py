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

from common.hssd_od_open_voc.hssd_open_voc_env import HSSD_OpenVoc_Env
from common.utils.plot_utils import plot_semantic_2d_map, make_mosaic


def collect_episodes_all_scenes(config) -> list:
    episodes = []
    rng_gen = np.random.default_rng(0)

    habitat_env = HSSD_OpenVoc_Env(config=config)
    scene_names = habitat_env.get_scenes_names()

    for scene_idx, scene in enumerate(scene_names[0:config.HABITAT_ACTIVE_OD.num_scenes]):
        habitat_env.change_scene(scene)

        print("-----------------")
        print("Collection in Scene = ", scene, f"({scene_idx}/{len(scene_names)})")

        habitat_obj_occupancy_grid = habitat_env.get_oracle_object_occupancy_grid(config.HABITAT_ACTIVE_OD.meters_per_grid_pixel)
        object2class = habitat_env.get_scene_annotations()

        for object_id, class_name in tqdm(object2class.items()):
            if class_name == "unknown":
                continue

            candidate_agent_states = habitat_obj_occupancy_grid.get_all_viewpoints(object_id)

            viewpoints = []

            for agent_state in candidate_agent_states:
                obs, instances = habitat_env.get_obs_gt(agent_state)

                if not [inst for inst in instances if 
                        (inst["object_id"] == object_id and inst["mask_area"] >= config.HABITAT_ACTIVE_OD.min_pixel_area)]:
                    continue
                
                viewpoints.append(agent_state)

            if not viewpoints: 
                continue
            
            viewpoint = viewpoints[0]
            goals = [
                NavigationGoal(position=viewpoint.position, radius=0)
            ]
            episode = NavigationEpisode(
                goals=goals,
                episode_id=f"{scene_idx}_obj_id_{object_id}",
                scene_id=scene,
                start_position=viewpoint.position,
                start_rotation=viewpoint.rotation,
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
    config = habitat.get_config(config_path="config/habitat_active_od_config.yaml")
    episodes = collect_episodes_all_scenes(config)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = episodes
    out_file = config.HABITAT_ACTIVE_OD.viewpoint_dataset.dataset.data_path
    split = config.HABITAT_ACTIVE_OD.viewpoint_dataset.dataset.split
    out_file = out_file.replace('{split}', 'val')

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    print(out_file)

    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())
