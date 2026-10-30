#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from omegaconf import DictConfig
from habitat.core.dataset import Dataset # type: ignore
from habitat.core.registry import registry # type: ignore
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal # type: ignore

# Sentinel start_position/start_rotation recognized by ExplorationEnv.reset() as "pick a random
# navigable point/rotation instead" -- NavigationEpisode's own validators reject None.
NO_FIXED_START_POSITION = [0.0, 0.0, 0.0]
NO_FIXED_START_ROTATION = [0.0, 0.0, 0.0, 1.0]


@registry.register_dataset(name="ExplorationSynthetic")
class ExplorationNavDataset(Dataset):
    r"""Episode dataset for ExplorationTask (common/env_utils/env_base.py) that needs no real
    per-scene episode files on disk, unlike habitat-lab's usual ObjectNav/PointNav dataset
    types. Its episodes carry NO_FIXED_START as their start_position/start_rotation (habitat-lab's
    own NavigationEpisode requires non-None values, so a real sentinel is used instead of None):
    ExplorationEnv.reset() recognizes it and falls back to a random navigable point/rotation,
    exactly as agent.act()-driven exploration needs.

    One trivial episode is synthesized per scene in config.content_scenes,
    carrying config.scene_dataset_config (see
    common/env_utils/env_registry.py::resolve_env, set by
    common/env_utils/env_base.py::configure_scene from object_params'
    env_name) so Env.__init__ resolves the right scene dataset regardless
    of which of the 4 env_names is being collected -- this is what lets one
    shared Hydra config (common/config/hssd-hab/default.yaml, despite its
    path) work across all of them, instead of needing a real ObjectNav-style
    episode dataset per env. goals gets one placeholder NavigationGoal
    (unused by ExplorationTask, but pointgoal_with_gps_compass_sensor --
    wired in by that same shared config -- indexes goals[0] unconditionally
    and errors on an empty list).
    """

    episodes: List[NavigationEpisode]

    def __init__(self, config: Optional[DictConfig] = None) -> None:
        self.episodes = []
        if config is None:
            return

        for episode_id, scene in enumerate(config.content_scenes):
            self.episodes.append(
                NavigationEpisode(
                    episode_id=str(episode_id),
                    scene_id=scene,
                    scene_dataset_config=config.scene_dataset_config,
                    start_position=NO_FIXED_START_POSITION,
                    start_rotation=NO_FIXED_START_ROTATION,
                    goals=[NavigationGoal(position=[10.0, 0.0, 10.0])],
                )
            )
