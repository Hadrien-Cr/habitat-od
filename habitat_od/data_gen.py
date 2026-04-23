import numpy as np
from omegaconf import DictConfig

# from habitat_od.utils.dataset_utils import DatasetGenerationConfig, save_dataset, load_dataset
# from habitat_od.utils.sampling_utils import balanced_supsampling, coverage_subsampling, covisibility_subsampling

from habitat.core.env import Env
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim import registry as registry
from habitat_sim.agent.agent import AgentState

from habitat_od.utils.grid_utils import HabitatSemGrid

def quaternion_from_rpy(roll, pitch, yaw):
    """Inputs in radians"""
    return quat_from_angle_axis(roll, np.array([1, 0, 0])) * quat_from_angle_axis(yaw, np.array([0, 1, 0])) * quat_from_angle_axis(pitch, np.array([0, 0, 1]))


def random_teleport_collection(
    habitat_env: Env,
    config: DictConfig
):
    habitat_grid = HabitatSemGrid(
        habitat_env.sim, 
        config.DATASET.meters_per_grid_pixel, 
        habitat_env.get_class_mapping(),
        habitat_env.get_objects()
    )

    return habitat_grid.get_all_poses()


def query_collection(
    habitat_env: Env,
    config: DictConfig,
    query_class
):
    habitat_grid = HabitatSemGrid(
        habitat_env.sim, 
        config.DATASET.meters_per_grid_pixel, 
        habitat_env.get_class_mapping(),
        habitat_env.get_objects()
    )
    return habitat_grid.get_all_poses_viewing_class(query_class)

def get_sample(habitat_env, pose):
    (x,y,z), (roll, pitch, yaw) = pose

    new_state = AgentState()

    new_state.position = np.array([x,y,z], dtype = np.float32)
    new_state.rotation = quaternion_from_rpy(roll, pitch, yaw)
    habitat_env.sim.agents[0].set_state(new_state)
    obs = habitat_env.sim.get_sensor_observations()
    sample = {
        key: obs[key]
        for key in ["rgb", "depth", "semantic"]
    }

    return sample

