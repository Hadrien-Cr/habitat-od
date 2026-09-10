"""Two ways to collect raw rgb + ground-truth sense files for dataset.py's
build_dataset() to convert into a COCO split: collect_random teleports to
random navmesh points (broad exploration, for pretraining data);
collect_validation walks straight to each target object in turn instead (the
counterpart of embodied-active-learning-od's validation_data_create.py). Both
collect exactly one scene per call -- write DATA_ROOT/run_name/{split_name}/
{scene_name}/raw/ -- so callers loop over their own scene lists (see
collect_dataset.py/collect_validation_dataset.py); each call is a single
ExplorationEnv driven directly in a plain loop -- no habitat-baselines
trainer/VectorEnv involved."""
import os
from pathlib import Path
from typing import Any
from tqdm import tqdm  # type: ignore

import numpy as np
from habitat.config import read_write  # type: ignore
from habitat.config.default_structured_configs import ObjectDetectorGTSensorConfig  # type: ignore

from habitat_embodied_al import constants
from common.env_utils.object_annotations import resolve_classes
from common.env_utils.object_detector_sensors import *  # noqa: F401,F403
from common.env_utils.sensors import *  # noqa: F401,F403
from common.env_utils.env_base import *  # noqa: F401,F403
from common.env_utils.env_base import ExplorationEnv
from common.env_utils.dataset import *  # noqa: F401,F403 registers "ExplorationSynthetic"
from common.env_utils.env_registry import resolve_env
from common.env_utils.sense import keep_valid_instances
from common.env_utils.vocab_constants import make_colors
from common.utils.data_utils import save_obs
from common.utils.dataset_utils import SampleLoader, instance_is_empty
from common.utils.plot_utils import make_mosaic, plot_segmentation_gt
# Imported last, deliberately: common.env_utils.sensors's own `import *` above brings in
# habitat.core.simulator.AgentState (a different, incompatible class -- requires `position`
# positionally, unlike habitat_sim.AgentState's all-optional fields), which would otherwise
# silently shadow the one collect_random/collect_validation actually construct below.
from habitat_sim import AgentState  # type: ignore
from common.utils.pose_utils import quaternion_from_rpy, yaw_to_face


def _reset_raw_dir(out_dir: Path) -> None:
    if out_dir.exists():
        input(f"WARNING: {out_dir} already exists, will delete and overwrite. Press Enter to continue...")
        os.system(f"rm -rf {out_dir}")
    os.makedirs(out_dir, exist_ok=True)


def configure_scene(habitat_cfg: Any, scenes: list, steps_per_episode: int, object_params: dict) -> None:
    """Points habitat_cfg at `scenes` (content_scenes -- see ExplorationNavDataset) and wires up
    the ObjectDetectorGTSensor from object_params, in place, before an ExplorationEnv is built
    from it. Public since habitat_embodied_al/reproduce/main.py's AL loop needs the same setup
    collect_random/collect_validation do below."""
    scene_dataset_config = resolve_env(object_params["env_name"])
    with read_write(habitat_cfg):
        habitat_cfg.habitat.dataset.content_scenes = scenes
        habitat_cfg.habitat.dataset.scene_dataset_config = scene_dataset_config
        habitat_cfg.habitat.simulator.scene_dataset = scene_dataset_config
        habitat_cfg.habitat.task.lab_sensors = {
            "object_detector_gt": ObjectDetectorGTSensorConfig(**object_params),
            **habitat_cfg.habitat.task.lab_sensors,
        }


def _write_mosaic(out_dir: Path, object_params: dict, n_samples: int = 32) -> None:
    mosaic_path = out_dir.parent / "mosaic.png"
    visualize_mosaic(
        out_dir, object_params["env_name"], object_params["vocab_name"], mosaic_path,
        n_samples=n_samples, shuffle=True, non_empty=True,
    )


def collect_random(habitat_cfg: Any, ds_cfg: Any, split_name: str, scene_name: str) -> None:
    """Broad random-exploration collection for one scene: every step, teleports to a random
    navmesh point + random yaw. Writes DATA_ROOT/run_name/{split_name}/{scene_name}/raw/ -- call
    once per scene (see collect_dataset.py's loop over ds_cfg.train_scenes/val_scenes)."""
    out_dir = constants.DATA_ROOT / ds_cfg.run_name / split_name / scene_name / "raw"
    _reset_raw_dir(out_dir)
    configure_scene(habitat_cfg, [scene_name], ds_cfg.steps_per_episode, ds_cfg.object_params)

    env = ExplorationEnv(config=habitat_cfg)
    rng = np.random.default_rng(habitat_cfg.habitat.seed)
    env.reset(env.episodes[0])

    for step in tqdm(range(ds_cfg.steps_per_episode), desc=f"{split_name}/{scene_name}"):
        agent_state = AgentState()
        agent_state.position = env.get_random_point()
        agent_state.rotation = env.get_random_rotation(rng)
        obs, _, _, _ = env.teleport(agent_state)
        save_obs(str(out_dir), 0, [obs], step, modalities=["rgb", "bbsgt"])

    env.close()
    _write_mosaic(out_dir, ds_cfg.object_params)


def collect_validation(habitat_cfg: Any, ds_cfg: Any, split_name: str, scene_name: str, view_radius: float = 1.5) -> None:
    """Counterpart of embodied-active-learning-od's validation_data_create.py, for one scene:
    walks straight to each target object in turn (get_target_object_positions ->
    get_point_near -> find_shortest_path), teleporting through the navmesh path one waypoint per frame
    and facing the object on arrival -- a single ExplorationEnv driven directly, no
    trainer/VectorEnv. Objects are revisited (reshuffled) until ds_cfg.steps_per_episode frames
    are collected. Writes DATA_ROOT/run_name/{split_name}/{scene_name}/raw/ -- call once per
    scene (see collect_validation_dataset.py's loop over ds_cfg.val_scenes)."""
    object_params = ds_cfg.object_params
    steps_per_episode = ds_cfg.steps_per_episode

    out_dir = constants.DATA_ROOT / ds_cfg.run_name / split_name / scene_name / "raw"
    _reset_raw_dir(out_dir)
    configure_scene(habitat_cfg, [scene_name], steps_per_episode, object_params)

    env = ExplorationEnv(config=habitat_cfg)
    rng = np.random.default_rng(habitat_cfg.habitat.seed)
    env.reset(env.episodes[0])

    objects = env.get_target_object_positions()
    rng.shuffle(objects)

    pbar = tqdm(total=steps_per_episode, desc=f"{split_name}/{scene_name}")
    step = 0

    while step < steps_per_episode and objects:
        visited_any = False
        for _, _, obj_position in list(objects):
            if step >= steps_per_episode:
                break

            # reset()'s start position is a meaningless placeholder (ExplorationNavDataset
            # synthesizes it just to let Env.reset() succeed), and each object's approach
            # starts over from a fresh point rather than continuing from wherever the
            # previous object's path left off.
            agent_state = AgentState()
            agent_state.position = env.get_random_point()
            agent_state.rotation = env.get_random_rotation(rng)
            obs, _, _, _ = env.teleport(agent_state)
            save_obs(str(out_dir), 0, [obs], step, modalities=["rgb", "bbsgt", "colored_tdmap"])
            step += 1

            if step >= steps_per_episode:
                break

            agent_position = obs["position"]["position"]
            viewpoint = env.get_point_near(obj_position, radius=view_radius)

            if viewpoint is None:
                continue
            path_points = env.find_shortest_path_waypoints(agent_position, viewpoint)
            if not path_points:
                continue

            waypoints = [(p, path_points[i + 2]) for i, p in enumerate(path_points[1:-1])]
            waypoints.append((path_points[-1], obj_position))

            for position, look_at in waypoints:
                if step >= steps_per_episode:
                    break
                agent_state = AgentState()
                agent_state.position = position
                agent_state.rotation = quaternion_from_rpy(0, 0, yaw_to_face(position, look_at))
                obs, _, _, _ = env.teleport(agent_state)
                save_obs(str(out_dir), 0, [obs], step, modalities=["rgb", "bbsgt", "colored_tdmap"])
                step += 1

                visited_any = True
                pbar.update(1)

        if not visited_any:
            break  # every object was unreachable this lap
        rng.shuffle(objects)

    env.close()
    _write_mosaic(out_dir, object_params)


def visualize_mosaic(raw_dir: Path, env_name: str, vocab_name: str, out_path: Path, n_samples: int = 32, shuffle: bool = False, non_empty: bool = True) -> Path:
    """Builds a GT-overlay mosaic PNG from a raw collected sense dir (as
    produced by `collect_random`/`collect_validation`), for quick visual
    sanity-checking of a collection run."""
    classes = resolve_classes(env_name, vocab_name)
    colors = make_colors(len(classes), seed=0, ctype=0)

    sampler = SampleLoader(str(raw_dir))
    episodes, steps = sampler.get_episode_and_steps_dense_list()
    n_samples = min(n_samples, len(episodes))

    def get_non_empty_samples(samples: list[int], n_samples: int) -> list[int]:
        plot_samples = []

        for i in samples:
            episode, step = int(episodes[i]), int(steps[i])
            gt_instances = sampler.get_sample(episode, 0, "bbsgt", step).get_bbs_as_gt()
            if not instance_is_empty(gt_instances):
                plot_samples.append(i)
            if len(plot_samples) >= n_samples:
                break
        return plot_samples

    tiles = []


    if shuffle:
        samples = list(np.random.permutation(len(episodes)))
    else:
        samples = list(np.arange(len(episodes)))

    if non_empty:
        samples = get_non_empty_samples(samples, n_samples)
    else:
        samples = samples[:n_samples]

    for i in samples:
        episode, step = int(episodes[i]), int(steps[i])
        rgb = sampler.get_sample(episode, 0, "rgb", step).data[:, :, :3]
        gt_instances = keep_valid_instances(sampler.get_sample(episode, 0, "bbsgt", step).get_bbs_as_gt())
        im = plot_segmentation_gt(rgb, gt_instances, classes, colors)
        tiles.append((f"ep{episode}_s{step}", np.array(im)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_mosaic(tiles, N_cols=4).save(out_path)
    print(f"Wrote visualization mosaic to {out_path}")
    return out_path
