import numpy as np
from omegaconf import DictConfig

# from habitat_od.utils.dataset_utils import DatasetGenerationConfig, save_dataset, load_dataset
# from habitat_od.utils.sampling_utils import balanced_supsampling, coverage_subsampling, covisibility_subsampling

from habitat.core.env import Env
from habitat_sim import registry as registry
from habitat_od.utils.grid_utils import HabitatSemGrid
from habitat_sim.agent.agent import AgentState


def query_collection(
    habitat_env: Env,
    habitat_grid: HabitatSemGrid,
    config: DictConfig,
    target_class: str,
) -> list[AgentState]:

    return habitat_grid.get_all_agent_states_viewing_class(target_class)
