"""Per-env_name static config needed to collect data for any of
object_annotations.py's 4 supported env_names from one shared Hydra config
(see habitat_embodied_al/collection.py::collect_raw and
common/config/hssd-hab/default.yaml, which despite its path is now
env-agnostic). The only thing that genuinely differs per env at the
habitat-lab config level is which scene_dataset_config.json scenes/episodes
resolve against -- sensor setup, agent height, task type etc. are shared.
"""

ENV_SCENE_DATASET_CONFIG: dict[str, str] = {
    "HSSD-HAB": "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json",
    "MP3D": "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json",
    "Gibson-Semantic": "data/scene_datasets/gibson_semantic/gibson_semantic.scene_dataset_config.json",
    "ProcTHOR-hab": "data/scene_datasets/ai2thor-hab/ai2thor-hab/ai2thor-hab.scene_dataset_config.json",
}


def resolve_env(env_name: str) -> str:
    """Returns the scene_dataset_config.json path for env_name, to set on
    both habitat.dataset.scene_dataset_config (read by
    ExplorationNavDataset when building episodes -- see
    common/env_utils/dataset.py) and habitat.simulator.scene_dataset
    (belt-and-suspenders: Env.__init__ overwrites the latter from the first
    episode's own scene_dataset_config anyway, see habitat/core/env.py)."""
    if env_name not in ENV_SCENE_DATASET_CONFIG:
        raise NotImplementedError(
            f"No scene_dataset_config registered for env_name={env_name!r}; "
            f"must be one of {sorted(ENV_SCENE_DATASET_CONFIG)}"
        )
    return ENV_SCENE_DATASET_CONFIG[env_name]
