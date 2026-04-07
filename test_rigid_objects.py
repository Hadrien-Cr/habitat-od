
import numpy as np
import matplotlib.pyplot as plt
import json

from habitat.sims.habitat_simulator.sim_utilities import get_all_object_ids

import habitat_sim, habitat
from habitat.config import read_write
from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig
from habitat_sim import registry as registry
from habitat.config.default import get_agent_config
import pprint


from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm


# config = habitat.get_config(config_path="benchmark/nav/objectnav/objectnav_procthor-hab.yaml")
# with open('data/scene_datasets/ai2thor-hab/ai2thor-hab/configs/object_semantic_id_mapping.json') as f:
#     class2int = json.load(f)

config = habitat.get_config(config_path="benchmark/nav/objectnav/objectnav_hssd-hab.yaml")
with open('data/scene_datasets/hssd-hab/semantics/hssd-hab_semantic_lexicon.json') as f:
    f = json.load(f)
    class2int = {x["name"]: x["id"] for x in f["classes"]}


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

env = habitat.Env(config=config)
sim = env.sim


##########################################
rom = sim.get_rigid_object_manager()

data = []
for obj_id in get_all_object_ids(sim):
    data.append((rom.get_object_by_id(obj_id).translation, rom.get_object_by_id(obj_id).rotation,rom.get_object_by_id(obj_id).aabb, rom.get_object_by_id(obj_id).semantic_id))


colors = cm.get_cmap('tab20')

def get_class_name(sem_id):
    for c in class2int:
        if class2int[c] == sem_id:
            return c
    return None

int2color = {class2int[class_name]: colors(class2int[class_name]%20) for class_name in class2int}
int2color[0] = (0,0,0,1)


def draw_bbox(ax, translation, rot, bb_min, bb_max, color, class_name, alpha=0.3):
    
    bb_min = np.array(bb_min)
    bb_max = np.array(bb_max)
    t = np.array(translation)

    # 8 corners relative to origin
    corners = np.array([
        [bb_min[0], bb_min[1], bb_min[2]],
        [bb_max[0], bb_min[1], bb_min[2]],
        [bb_max[0], bb_max[1], bb_min[2]],
        [bb_min[0], bb_max[1], bb_min[2]],
        [bb_min[0], bb_min[1], bb_max[2]],
        [bb_max[0], bb_min[1], bb_max[2]],
        [bb_max[0], bb_max[1], bb_max[2]],
        [bb_min[0], bb_max[1], bb_max[2]],
    ])

    # Rotate corners using Magnum quaternion
    corners_rot = np.array([np.array(rot.transform_vector(corner)) for corner in corners])

    # Translate corners
    corners_rot += t

    # 6 faces
    faces = [
        [corners_rot[0], corners_rot[1], corners_rot[2], corners_rot[3]],  # bottom
        [corners_rot[4], corners_rot[5], corners_rot[6], corners_rot[7]],  # top
        [corners_rot[0], corners_rot[1], corners_rot[5], corners_rot[4]],  # front
        [corners_rot[2], corners_rot[3], corners_rot[7], corners_rot[6]],  # back
        [corners_rot[0], corners_rot[3], corners_rot[7], corners_rot[4]],  # left
        [corners_rot[1], corners_rot[2], corners_rot[6], corners_rot[5]],  # right
    ]

    poly = Poly3DCollection(faces, alpha=alpha, linewidths=0.5, edgecolors='k')
    poly.set_facecolor(color)
    ax.add_collection3d(poly)

    # Label above rotated box (top center)
    top_center = (bb_min + bb_max) / 2
    top_center_rot = np.array(rot.transform_vector(top_center)) + t
    ax.text(top_center_rot[0], top_center_rot[1], top_center_rot[2], class_name,
            color='k', fontsize=10, ha='center', va='bottom')
        
# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for translation, rot, bb, sem_id in data:
    if sem_id == 0:
        continue # skip undefined
    
    color = int2color[sem_id]
    class_name = get_class_name(sem_id)
    draw_bbox(ax, translation, rot, bb.min, bb.max, color=color, class_name=class_name, alpha=0.4)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
z_span = ax.get_zlim()[1] - ax.get_zlim()[0]
ax.set_box_aspect([x_span, y_span, z_span])

plt.show()