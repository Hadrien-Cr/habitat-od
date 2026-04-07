
import numpy as np
import matplotlib.pyplot as plt
import json

from habitat_sim.scene import SemanticScene
from habitat.sims.habitat_simulator.sim_utilities import get_all_object_ids

import habitat_sim, habitat
from habitat.config import read_write
from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig
from habitat_sim import registry as registry
from habitat.config.default import get_agent_config
from image_sampler import CustomImageSampler
import pprint
import matplotlib.cm as cm


def cfg_to_dict(cfg):
    result = {}
    for k, v in cfg.items():
        if hasattr(v, 'items'):  # it's a nested config node
            result[k] = cfg_to_dict(v)
        else:
            result[k] = v
    return result
 
# config = habitat.get_config(config_path="benchmark/nav/objectnav/objectnav_procthor-hab.yaml")
# with open('data/scene_datasets/ai2thor-hab/ai2thor-hab/configs/object_semantic_id_mapping.json') as f:
#     class2int = json.load(f)

config = habitat.get_config(config_path="benchmark/nav/objectnav/objectnav_hssd-hab.yaml")
with open('data/scene_datasets/hssd-hab/semantics/hssd-hab_semantic_lexicon.json') as f:
    f = json.load(f)
    class2int = {x["name"]: x["id"] for x in f["classes"]}

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

colors = cm.get_cmap('tab20')

def get_class_name(sem_id):
    for c in class2int:
        if class2int[c] == sem_id:
            return c
    return None

int2color = {class2int[class_name]: colors(class2int[class_name]%20) for class_name in class2int}
int2color[0] = (0,0,0,1)

##########################################
extractor = CustomImageSampler(
    sim=sim,
    output=["rgba", "depth", "semantic"],
)

def display_sample(sample):
    img = sample["rgba"]
    depth = sample["depth"]
    semantic = sample["semantic"]

    plt.figure(figsize=(12, 8))

    # rgb
    ax = plt.subplot(1, 3, 1)
    ax.axis("off")
    ax.set_title("rgba")
    plt.imshow(img)

    # depth
    ax = plt.subplot(1, 3, 2)
    ax.axis("off")
    ax.set_title("depth")
    plt.imshow(depth)
    
    # semantic
    w,h = semantic.shape
    colored_semantic = np.zeros((w,h,3), dtype = np.uint8)

    for x in range(w):
        for y in range(h):
            r,g,b,a = int2color[semantic[x,y]]
            colored_semantic[x,y] = np.array([int(255*r),int(255*g),int(255*b)])

    ax = plt.subplot(1, 3, 3)
    ax.axis("off")
    ax.set_title("semantic")
    plt.imshow(colored_semantic)
    plt.show()

for i in range(len(extractor)):
    sample = extractor[i]
    display_sample(sample)

extractor.close()