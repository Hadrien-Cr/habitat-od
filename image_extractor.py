
import numpy as np

from typing import Callable, List, Optional, Union

import numpy as np
from habitat_sim import registry as registry
from habitat_sim.agent.agent import AgentConfiguration, AgentState
from habitat_sim.utils.data.data_structures import ExtractorLRUCache
from habitat_sim.utils.data.pose_extractor import PoseExtractor, TopdownView, ClosestPointExtractor


class CustomImageExtractor:
    def __init__(
        self,
        sim,
        labels: Optional[List[float]] = None,
        img_size: tuple = (512, 512),
        output: Optional[List[str]] = None,
        shuffle: bool = True,
        split: tuple = (70, 30),
        use_caching: bool = True,
        meters_per_pixel: float = 0.1,
    ):
        if labels is None:
            labels = [0.0]
        if output is None:
            output = ["rgba"]
        if sum(split) != 100:
            raise Exception("Train/test split must sum to 100.")


        self.labels = set(labels)
        self.img_size = img_size
        self.sim = sim
        self.meters_per_pixel = meters_per_pixel

        ref_point = self._get_pathfinder_reference_point(self.sim.pathfinder)
        self.tdv_fp_ref_triples = [
            (
                TopdownView(self.sim, ref_point[1], meters_per_pixel),
                self.sim.config.sim_cfg.scene_id,
                ref_point,
            )
        ]

        self.pose_extractor = ClosestPointExtractor(
            self.tdv_fp_ref_triples,
            self.meters_per_pixel
        )

        self.poses = self.pose_extractor.extract_all_poses()

        if shuffle:
            np.random.shuffle(self.poses)

        self.train, self.test = self._handle_split(split, self.poses)
        self.mode = "full"
        self.mode_to_data = {
            "full": self.poses,
            "train": self.train,
            "test": self.test,
            None: self.poses,
        }
        self.instance_id_to_name = self._generate_label_map(self.sim.semantic_scene)
        self.out_name_to_sensor_name = {
            "rgba": "rgb",
            "depth": "depth",
            "semantic": "semantic",
        }
        self.output = output
        self.use_caching = use_caching
        if self.use_caching:
            self.cache = ExtractorLRUCache()

    def __len__(self):
        return len(self.mode_to_data[self.mode])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            if start is None:
                start = 0
            if stop is None:
                stop = len(self.mode_to_data[self.mode])
            if step is None:
                step = 1

            return [
                self.__getitem__(i)
                for i in range(start, stop, step)
                if i < len(self.mode_to_data[self.mode])
            ]

        mymode = self.mode.lower()
        if self.use_caching:
            cache_entry = (idx, mymode)
            if cache_entry in self.cache:
                return self.cache[cache_entry]

        poses = self.mode_to_data[mymode]
        pos, rot, fp = poses[idx]

        new_state = AgentState()
        new_state.position = pos
        new_state.rotation = rot
        self.sim.agents[0].set_state(new_state)
        obs = self.sim.get_sensor_observations()
        sample = {
            out_name: obs[self.out_name_to_sensor_name[out_name]]
            for out_name in self.output
        }

        if self.use_caching:
            self.cache.add(cache_entry, sample)

        return sample

    def close(self) -> None:
        r"""Deletes the instance of the simulator. Necessary for instantiating a different ImageExtractor."""
        if self.sim is not None:
            self.sim.close()
            del self.sim
            self.sim = None

    def set_mode(self, mode: str) -> None:
        r"""Sets the mode of the simulator. This controls which poses to use; train, test, or all (full)"""
        mymode = mode.lower()
        if mymode not in ["full", "train", "test"]:
            raise Exception(
                f'Mode {mode} is not a valid mode for ImageExtractor. Please enter "full, train, or test"'
            )

        self.mode = mymode

    def get_semantic_class_names(self) -> List[str]:
        r"""Returns a list of english class names in the scene(s). E.g. ['wall', 'ceiling', 'chair']"""
        class_names = list(set(self.instance_id_to_name.values()))
        return class_names

    def _handle_split(self, split, poses):
        train, test = split
        num_poses = len(self.poses)
        last_train_idx = int((train / 100) * num_poses)
        train_poses = poses[:last_train_idx]
        test_poses = poses[last_train_idx:]
        return train_poses, test_poses

    def _get_pathfinder_reference_point(self, pf):
        bound1, bound2 = pf.get_bounds()
        startw = min(bound1[0], bound2[0])
        starth = min(bound1[2], bound2[2])
        starty = pf.get_random_navigable_point()[
            1
        ]
        return (startw, starty, starth)  # width, y, height

    def _generate_label_map(self, scene, verbose=False):
        if verbose:
            print(
                f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
            )
            print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

        instance_id_to_name = {}
        for obj in scene.objects:
            if obj and obj.category:
                obj_id = int(obj.id.split("_")[-1])
                instance_id_to_name[obj_id] = obj.category.name()

        return instance_id_to_name
