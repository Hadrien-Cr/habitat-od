import numpy as np
from omegaconf import DictConfig

# from habitat_od.utils.dataset_utils import DatasetGenerationConfig, save_dataset, load_dataset
# from habitat_od.utils.sampling_utils import balanced_supsampling, coverage_subsampling, covisibility_subsampling

from habitat.core.env import Env
from habitat_sim import registry as registry
from habitat_od.utils.grid_utils import HabitatSemGrid
from habitat_od.utils.sampling_utils import kmeans
from habitat_sim.agent.agent import AgentState


def query_collection(
    habitat_grid: HabitatSemGrid,
    num_samples: int,
    target_class: str,
    rng_gen
) -> list[AgentState]:
    candidate_states =  habitat_grid.get_all_agent_states_viewing_class(target_class)
    
    rng_gen.shuffle(candidate_states)
    candidate_states = candidate_states[0:(5*num_samples)]
    
    x = [np.array([agent_state.position[0], agent_state.position[1]]) for agent_state in candidate_states]

    centers_indices, partitionned_indices = kmeans(
        x,
        k = num_samples,
        rng_gen=rng_gen
    )

    return [candidate_states[i] for i in centers_indices]