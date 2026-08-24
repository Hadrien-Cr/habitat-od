"""Drives a habitat-baselines trainer to collect raw rgb + ground-truth
sense files over a set of scenes, for `collect_dataset.py`'s pretraining
data collection (over many scenes)."""
import os
from pathlib import Path
from typing import Any, List

import numpy as np
from detectron2.data import MetadataCatalog  # type: ignore
from habitat.config import read_write  # type: ignore
from habitat.config.default_structured_configs import ObjectDetectorGTSensorConfig  # type: ignore
from habitat_baselines.common.baseline_registry import baseline_registry  # type: ignore

from common.env_utils.object_detector_sensors import *  # noqa: F401,F403
from common.env_utils.sensors import *  # noqa: F401,F403
from common.env_utils.env_base import *  # noqa: F401,F403
from common.env_utils.dataset import *  # noqa: F401,F403
from common.baselines.agents import *  # noqa: F401,F403 registers trainers with baseline_registry
from common.utils.dataset_utils import SampleLoader
from common.utils.plot_utils import make_mosaic, plot_segmentation_gt


def collect_raw(
    habitat_config: Any,
    scenes: List[str],
    steps_per_episode: int,
    trainer_name: str,
    object_params: dict,
    out_dir: Path,
    create_mosaic: bool = False,
    mosaic_samples: int = 32,
) -> Path:
    """Collects rgb+bbsgt sense files for `scenes` into `out_dir`, driven by
    the habitat-baselines trainer registered as `trainer_name`. Returns
    `out_dir`. If `create_mosaic`, also writes a GT-overlay mosaic PNG
    (`<out_dir>_mosaic.png`) of up to `mosaic_samples` collected frames, for
    quickly eyeballing what got collected."""
    if out_dir.exists():
        input(f"WARNING: {out_dir} already exists, will delete and overwrite. Press Enter to continue...")
        os.system(f"rm -rf {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = resolve_dataset_split(habitat_config.habitat.dataset, scenes)
        habitat_config.habitat.dataset.content_scenes = scenes
        habitat_config.habitat.environment.max_episode_steps = steps_per_episode
        habitat_config.habitat.task.lab_sensors = {
            "object_detector_gt": ObjectDetectorGTSensorConfig(**object_params),
            **habitat_config.habitat.task.lab_sensors,
        }
        habitat_config.habitat_baselines.trainer_name = trainer_name

    trainer_cls = baseline_registry.get_trainer(habitat_config.habitat_baselines.trainer_name)
    trainer = trainer_cls(habitat_config)
    trainer.collect(str(out_dir), steps_per_episode=steps_per_episode)

    if create_mosaic:
        vocab_name = object_params["vocab_name"]
        mosaic_path = out_dir.parent / f"{out_dir.name}_mosaic.png"
        visualize_mosaic(out_dir, vocab_name, mosaic_path, n_samples=mosaic_samples, shuffle=True, non_empty=True)

    return out_dir


def visualize_mosaic(raw_dir: Path, vocab_name: str, out_path: Path, n_samples: int = 32, shuffle: bool = False, non_empty: bool = True) -> Path:
    """Builds a GT-overlay mosaic PNG from a raw collected sense dir (as
    produced by `collect_raw`), for quick visual sanity-checking of a
    collection run."""
    classes = MetadataCatalog.get(vocab_name).thing_classes
    colors = MetadataCatalog.get(vocab_name).thing_colors

    sampler = SampleLoader(str(raw_dir))
    episodes, steps = sampler.get_episode_and_steps_dense_list()
    n_samples = min(n_samples, len(episodes))

    def get_non_empty_samples(samples: list[int], n_samples: int) -> list[int]:
        plot_samples = []

        for i in samples:
            episode, step = int(episodes[i]), int(steps[i])
            gt_instances = sampler.get_sample(episode, 0, "bbsgt", step).get_bbs_as_gt()
            if len(gt_instances) > 0:
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
        gt_instances = sampler.get_sample(episode, 0, "bbsgt", step).get_bbs_as_gt()
        im = plot_segmentation_gt(rgb, gt_instances, classes, colors)
        tiles.append((f"ep{episode}_s{step}", np.array(im)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_mosaic(tiles, N_cols=4).save(out_path)
    print(f"Wrote visualization mosaic to {out_path}")
    return out_path
