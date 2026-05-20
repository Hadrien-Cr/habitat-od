import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import random

import habitat # type: ignore
from habitat_sim.agent.agent import AgentState
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1, NavigationEpisode, NavigationGoal # type: ignore
from detectron2.data import build_detection_test_loader, MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

from common.hssd_od_open_voc.hssd_env import HSSD_OpenVoc_Env
from common.utils.plot_utils import plot_semantic_2d_map, make_mosaic
from common.utils.data_utils import save_img
from common.vision.detic import build_detic_predictor

import habitat_od.od_dataset_registry
from habitat_active_od.agent import ActiveODAgent, DiscreteNavigationAction



if __name__ == "__main__":
    config = habitat.get_config(config_path="config/habitat_active_od_config.yaml")
    print(OmegaConf.to_yaml(config))

    rng_gen = np.random.default_rng(0)

    habitat_env = HSSD_OpenVoc_Env(config=config, vocab_name=config.DATA_GEN.vocab)
    classes = habitat_env.get_classes()

    detic_config = OmegaConf.load("config/detic_config.yaml")
    detic_predictor = build_detic_predictor(detic_config, classes) # type: ignore

    agent = ActiveODAgent(config=config, predictor=detic_predictor, classes = classes)

    ACTIONS = {
        DiscreteNavigationAction.MOVE_FORWARD: "move_forward",
        DiscreteNavigationAction.TURN_LEFT: "turn_left",
        DiscreteNavigationAction.TURN_RIGHT: "turn_right",
        DiscreteNavigationAction.STOP: "stop",
    }

    class_mapping = habitat_env.get_class_mapping()

    for ep in range(10):
        habitat_env.reset()
        obs, labels = habitat_env.get_obs_gt(habitat_env.get_agent_state())
        agent.reset()

        objid_to_class = habitat_env.get_objid_to_class()
        object_id = int(habitat_env._current_episode.episode_id.split("_obj_id_")[-1])
        class_name = objid_to_class[object_id]

        for t in range(10):
            agent_state = habitat_env.get_agent_state()
            obs, labels = habitat_env.get_obs_gt(agent_state)
            action = agent.act(obs)
            habitat_env.step(ACTIONS[action])