import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

import habitat # type: ignore
from habitat_sim.agent.agent import AgentState
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1, NavigationEpisode, NavigationGoal # type: ignore
from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader, MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

from common.hssd_od_open_voc.hssd_open_voc_env import HSSD_OpenVoc_Env
from common.utils.plot_utils import plot_semantic_2d_map, make_mosaic
from common.utils.data_utils import save_img
from common.vision.detic import build_detic_predictor

import habitat_od.od_dataset_registry

def overlap(mask1: np.ndarray,mask2:np.ndarray):
    return np.logical_and(mask1, mask2).any()

if __name__ == "__main__":
    config = habitat.get_config(config_path="config/habitat_active_od_config.yaml")
    
    rng_gen = np.random.default_rng(0)

    habitat_env = HSSD_OpenVoc_Env(config=config)
    viewpoint_dataset = PointNavDatasetV1(config.HABITAT_ACTIVE_OD.viewpoint_dataset.dataset)

    scene_names = habitat_env.get_scenes_names()
    metadata = MetadataCatalog.get("hssd_od_openvoc_test")

    detic_config = OmegaConf.load("config/detic_config.yaml")
    detic_predictor = build_detic_predictor(detic_config, habitat_env.get_classes()) # type: ignore

    per_class_object_occurences = defaultdict(int)
    class_mapping = habitat_env.get_class_mapping()

    scene_id = None

    for episode in tqdm(viewpoint_dataset.episodes):
        ep_scene_id = episode.scene_id.split('/')[-1]

        if len(episode.info["viewpoints"])<8:
            continue
    
        if ep_scene_id != scene_id:
            scene_id = ep_scene_id
            habitat_env.change_scene(scene_id)
            object2class = habitat_env.get_scene_annotations()

        object_id = int(episode.episode_id.split("_obj_id_")[-1])
        class_name = object2class[object_id]
        
        list_fnames_images = [] 

        
        agent_state = AgentState(
            position=episode.start_position,
            rotation=episode.start_rotation
        )

        obs, instances = habitat_env.get_obs_gt(agent_state)

  

        vis = Visualizer(
            obs,
            metadata=metadata,
            scale=0.5,
            instance_mode=ColorMode.SEGMENTATION
        )

        target_masks = [inst["mask"] for inst in instances if inst["object_id"] == object_id]
        
        for target_mask in target_masks:
            vis.draw_binary_mask(target_mask)

        gt_image = vis.get_output().get_image()

        for v in episode.info["viewpoints"][0:8]:
            agent_state = AgentState(
                position=v["position"],
                rotation=v["rotation"]
            )

            obs, instances = habitat_env.get_obs_gt(agent_state)
            target_masks = [inst["mask"] for inst in instances if inst["object_id"] == object_id]

            if not target_masks:
                continue

            outputs = detic_predictor(obs)
            class_id = metadata.thing_classes.index(class_name)
            vis = Visualizer(
                obs,
                metadata=metadata,
                scale=0.5,
                instance_mode=ColorMode.SEGMENTATION
            )
        
            pred_instances = outputs["instances"].to("cpu")
            keep = np.zeros(len(pred_instances))

            for i, pred_mask in enumerate(pred_instances.pred_masks):
                pred_mask_np = pred_mask.numpy()
                if any([overlap(pred_mask_np,target_mask) for target_mask in target_masks]):
                    keep[i] = 1

            pred_instances = pred_instances[keep == 1]

            vis.draw_instance_predictions(pred_instances)
            result = vis.get_output().get_image()

            list_fnames_images.append((class_name, result))

        list_fnames_images.append(("gt" + class_name, gt_image))

        make_mosaic(list_fnames_images).save(f"datadump/cls_{class_name}_{object_id}.png")