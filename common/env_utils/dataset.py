#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
from typing import List, Optional
import copy

import tqdm

import attr
from omegaconf import DictConfig, OmegaConf
import magnum as mn
from habitat.core.dataset import ALL_SCENES_MASK, Dataset # type: ignore
from habitat.core.registry import registry # type: ignore
from habitat.tasks.nav.nav import ( # type: ignore
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)

import numpy as np

CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


def resolve_dataset_split(
    dataset_config: DictConfig, scenes: List[str], candidate_splits=("train", "val")
) -> str:
    r"""HSSD-HAB's objectnav dataset partitions scenes disjointly across each
    split's `content/` dir, so a caller that overrides `content_scenes` with
    an explicit scene list (e.g. `collect_raw`) can't just leave
    `dataset_config.split` at its config default -- it must match whichever
    split dir actually holds those scenes' `.json.gz` files, or loading fails
    with a raw FileNotFoundError. Scenes must all come from the same split.
    """
    matches = []
    for split in candidate_splits:
        dataset_dir = os.path.dirname(dataset_config.data_path.format(split=split))
        content_dir = os.path.join(dataset_dir, "content")
        if all(os.path.exists(os.path.join(content_dir, f"{scene}.json.gz")) for scene in scenes):
            matches.append(split)

    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely resolve the HSSD dataset split for scenes {scenes} "
            f"among candidates {candidate_splits} (matched: {matches}). Scenes passed "
            "together to collect_raw must all belong to the same split's content/ dir."
        )
    return matches[0]


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None
    object_id: Optional[str] = None
    floor_id: Optional[str] = None

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@registry.register_dataset(name="ExplorationNav")
class ExplorationNavDataset(Dataset):
    r"""Class inherited from Dataset that loads Point Navigation dataset."""

    episodes: List[NavigationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: DictConfig) -> bool:
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_dir)

    @classmethod
    def get_scenes_to_load(cls, config: DictConfig) -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """

        assert cls.check_config_paths_exist(config)
        dataset_dir = os.path.dirname(config.data_path.format(split=config.split))

        cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        dataset = cls(cfg) # type: ignore
        
        has_individual_scene_files = os.path.exists(
            dataset.content_scenes_path.split("{scene}")[0].format(
                data_path=dataset_dir
            )
        )
        if has_individual_scene_files:
            return cls._get_scenes_from_folder(
                content_scenes_path=dataset.content_scenes_path,
                dataset_dir=dataset_dir,
            )
        else:
            # Load the full dataset, things are not split into separate files
            cfg.content_scenes = [ALL_SCENES_MASK]
            dataset = cls(cfg) # type: ignore
            return list(map(cls.scene_from_scene_path, dataset.scene_ids))

    @staticmethod
    def _get_scenes_from_folder(
        content_scenes_path: str, dataset_dir: str
    ) -> List[str]:
        scenes: List[str] = []

        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)

        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def __init__(self, config: DictConfig) -> None:
        self.episodes = []
        self.scene_dataset_config = "data/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
        datasetfile_path = config.data_path.format(split=config.split)

        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.scenes_dir)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        all_scenes = os.listdir(os.path.join(dataset_dir, "content"))
        has_individual_scene_files = os.path.exists(
            self.content_scenes_path.split("{scene}")[0].format(data_path=dataset_dir)
        )

        repeat_factor = config.repeat if "REPEAT" in config else 1

        if has_individual_scene_files:
            scenes = config.content_scenes
            if ALL_SCENES_MASK in scenes:
                scenes = self._get_scenes_from_folder(
                    content_scenes_path=self.content_scenes_path,
                    dataset_dir=dataset_dir,
                )

            for scene in tqdm.tqdm(scenes):
                scene_filename = self.content_scenes_path.format(
                    data_path=dataset_dir, scene=scene
                )
                with gzip.open(scene_filename, "rt") as f:
                    self.from_json(
                        f.read(), scenes_dir=config.scenes_dir, repeat=repeat_factor
                    )

        else:
            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )

        ids = {item.split(".")[0]: 0 for i, item in enumerate(all_scenes)}
        assert len(self.episodes) > 0, "No episodes found for the specified config content scenes"

        episodes_unique = []
        episode_id = 0

        for scene in sorted(all_scenes):
            for episode in self.episodes:
                episode_scene = os.path.split(episode.scene_id)[-1].split(".")[0]
                if episode_scene == scene.split(".")[0]:
                    if ids[episode_scene] >= repeat_factor:
                        continue
                    episode.episode_id = episode_id
                    episode_id += 1
                    episodes_unique.append(episode)
                    ids[episode_scene] += 1

        self.episodes = episodes_unique
        assert len(self.episodes) > 0

    def from_json(
        self,
        json_str: str,
        scenes_dir: Optional[str] = None,
        object_category=None,
        repeat=1,
    ) -> None:
        deserialized = json.loads(json_str)

        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for episode_cfg in deserialized["episodes"]:
            for _ in range(repeat):              
                episode = NavigationEpisode(
                    episode_id=episode_cfg["episode_id"],
                    scene_id=episode_cfg["scene_id"],
                    start_position=episode_cfg["start_position"],
                    start_rotation=episode_cfg["start_rotation"],
                    goals=[NavigationGoal(
                        position=episode_cfg["start_position"] + np.array([10,0,10]),
                    )],
                    scene_dataset_config=self.scene_dataset_config
                )
                self.episodes.append(episode)
                break
            break