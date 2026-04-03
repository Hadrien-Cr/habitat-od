
import numpy as np
import matplotlib.pyplot as plt

import habitat_sim, habitat
from habitat.config import read_write
from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig
from habitat_sim import registry as registry
from habitat.config.default import get_agent_config
from image_extractor import CustomImageExtractor
import pprint

def cfg_to_dict(cfg):
    result = {}
    for k, v in cfg.items():
        if hasattr(v, 'items'):  # it's a nested config node
            result[k] = cfg_to_dict(v)
        else:
            result[k] = v
    return result


# backend_cfg = habitat_sim.SimulatorConfiguration()
# backend_cfg.scene_id = "data/scene_datasets/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
# backend_cfg.scene_dataset_config_file = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"

# # backend_cfg = habitat_sim.SimulatorConfiguration()
# # backend_cfg.scene_dataset_config_file = "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"

# # Common sensor settings
# resolution = [480, 640]
# position = [0.0, 1.5, 0.0]

# # RGB sensor
# rgb_cfg = habitat_sim.CameraSensorSpec()
# rgb_cfg.uuid = "rgb"
# rgb_cfg.sensor_type = habitat_sim.SensorType.COLOR
# rgb_cfg.resolution = resolution
# rgb_cfg.position = position

# # Depth sensor
# depth_cfg = habitat_sim.CameraSensorSpec()
# depth_cfg.uuid = "depth"
# depth_cfg.sensor_type = habitat_sim.SensorType.DEPTH
# depth_cfg.resolution = resolution
# depth_cfg.position = position

# # Semantic sensor (your original one, completed)
# sem_cfg = habitat_sim.CameraSensorSpec()
# sem_cfg.uuid = "semantic"
# sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
# sem_cfg.resolution = resolution
# sem_cfg.position = position

# # Agent config
# agent_cfg = habitat_sim.agent.AgentConfiguration()
# agent_cfg.sensor_specifications = [rgb_cfg, depth_cfg, sem_cfg]

# # Simulator config
# sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
# sim = habitat_sim.Simulator(sim_cfg)


config = habitat.get_config(config_path="benchmark/nav/objectnav/objectnav_hssd-hab.yaml")
# config = habitat.get_config(config_path="benchmark/nav/pointnav/pointnav_mp3d.yaml")

with read_write(config):
    config.habitat.dataset.split = "val"
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
    config.habitat.simulator.turn_angle = 30

pprint.pprint(cfg_to_dict(config))
env = habitat.Env(config=config)
sim = env.sim

##########################################
extractor = CustomImageExtractor(
    sim=sim,
    img_size=(480, 640),
    output=["rgba", "depth", "semantic"],
)
extractor.set_mode('train')


def display_sample(sample):
    img = sample["rgba"]
    depth = sample["depth"]
    semantic = sample["semantic"]

    arr = [img, depth, semantic]
    titles = ["rgba", "depth", "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()

print(len(extractor))
for i in range(10):
    sample = extractor[i]
    display_sample(sample)