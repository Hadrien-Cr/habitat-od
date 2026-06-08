import copy
import math
import os
import random
from itertools import compress
from typing import List

import numpy as np
from omegaconf import DictConfig
from habitat.config import read_write # type: ignore
import torch
from habitat import RLEnv, ThreadedVectorEnv, VectorEnv, make_dataset, Env # type: ignore


def make_env_fn(env_class, config, kwargs) -> RLEnv:
    env = env_class(config=config, **kwargs)
    return env


def get_unique_scene_envs_generator(config, env_class, **kwargs) -> RLEnv:
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)

    scenes = config.DATASET.CONTENT_SCENES
    if "*" in config.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(config.DATASET)

    for i, scene_name in enumerate(scenes):
        task_config = config.clone()
        task_config.defrost()

        task_config.SEED = config.SEED + i
        task_config.DATASET.CONTENT_SCENES = [scene_name]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = (
            task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID
        )
        task_config.freeze()
        yield env_class(config=task_config, **kwargs)


def construct_envs(
    config: DictConfig,
    env_class: RLEnv,
    workers_ignore_signals: bool = False,
    mode="train",
    **kwargs
) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    :param config: configs that contain num_environments as well as information
    :param necessary to create individual environments.
    :param env_class: class type of the envs to be created.
    :param workers_ignore_signals: Passed to :ref:`habitat.VectorEnv`'s constructor
    :return: VectorEnv object created according to specification.
    """
    print("Constructing envs...")
    scenes = list(config.habitat.dataset.content_scenes)
    
    (
        sim_gpu_id,
        num_processes,
        num_processes_on_first_gpu,
        num_processes_per_gpu,
    ) = get_multi_gpu_config(len(scenes))
    
    num_processes = min(num_processes, len(scenes))
    configs = []

    env_classes = [env_class for _ in range(num_processes)]

    kwargs_per_env = [kwargs for _ in range(num_processes)]

    random.shuffle(scenes)

    scene_splits: List[List[str]] = [[] for _ in range(num_processes)]

    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)
    for i in range(num_processes):
        task_config = copy.deepcopy(config)
        
        with read_write(task_config):
            task_config.habitat.seed = config.habitat.seed + i
            if len(scenes) > 0:
                task_config.habitat.dataset.content_scenes = scene_splits[i]

            if i < num_processes_on_first_gpu:
                gpu_id = 0
            else:
                gpu_id = (i - num_processes_on_first_gpu) % (
                    torch.cuda.device_count() - 1
                ) + sim_gpu_id

            task_config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id
        
        configs.append(task_config)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(env_classes, configs, kwargs_per_env)),
        workers_ignore_signals=workers_ignore_signals,
    )

    return envs


def get_multi_gpu_config(num_scenes=25, x=10):
    # Automatically configure number of training threads based on
    # number of GPUs available and GPU memory size
    total_num_scenes = num_scenes
    gpu_memory = 100
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        gpu_memory = min(
            gpu_memory,
            torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024,
        )
        if i == 0:
            assert (
                torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024
                > 10.0
            ), "Insufficient GPU memory"

    num_processes_per_gpu = int(gpu_memory / 1.4)

    num_processes_on_first_gpu = int((gpu_memory - x) / 1.4)

    sim_gpu_id = 0

    if num_gpus == 1:
        num_processes_on_first_gpu = num_processes_on_first_gpu
        num_processes_per_gpu = 0
        num_processes = num_processes_on_first_gpu
    else:
        total_threads = (
            num_processes_per_gpu * (num_gpus - 1) + num_processes_on_first_gpu
        )

        num_scenes_per_thread = math.ceil(total_num_scenes / total_threads)
        num_threads = math.ceil(total_num_scenes / num_scenes_per_thread)
        num_processes_per_gpu = min(
            num_processes_per_gpu, math.ceil(num_threads // (num_gpus - 1))
        )

        num_processes_on_first_gpu = max(
            0, num_threads - num_processes_per_gpu * (num_gpus - 1)
        )

        num_processes = num_processes_on_first_gpu + num_processes_per_gpu * (
            num_gpus - 1
        )  # num_threads

        sim_gpu_id = 1

    return sim_gpu_id, num_processes, num_processes_on_first_gpu, num_processes_per_gpu
