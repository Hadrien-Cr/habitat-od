"""Test set frames of an evaluation scene: a FrontierExplorationAgent explores it until it stops or frontier_agent.steps run
out, and every step's frame comes with its valid ground-truth instances; collect_test.py keeps every frame_stride-th one
showing some, so a scene contributes in proportion to its exploration and every view is as likely to be kept."""
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from detectron2.structures import Instances
from omegaconf import DictConfig
from tqdm import tqdm

from common.env_utils.env_base import build_agent, build_env
from common.env_utils.object_annotations import resolve_classes
from common.env_utils.sense import ground_truth_instances
from common.utils.plot_utils import save_tile

DECISION_FRAME_PNG = "decision_frame.png"


def evaluation_frames(scene: str, cfg: DictConfig, classes: list, out_dir: Path) -> Iterator[Tuple[int, np.ndarray, Instances]]:
    """(step, rgb, valid ground truth of classes) of every step of a FrontierExplorationAgent's exploration of the scene;
    once they are all consumed, its final map and trajectory are saved as out_dir/DECISION_FRAME_PNG."""
    object_params, frontier_agent = cfg.object_params, cfg.frontier_agent
    env, habitat_cfg = build_env(scene, cfg.seed, frontier_agent.steps, object_params, cfg.input_image_size)
    rng = np.random.default_rng(cfg.seed)
    agent = build_agent("fbe", rng, env, habitat_cfg, frontier_agent.visibility_dist_m, str(out_dir), "planner")
    obs = env.reset(env.episodes[0], rng=rng)
    agent.reset(env.get_agent_state(), env.get_agent_state().position[1] - env.navmesh_floor_offset())
    vocabulary = resolve_classes(object_params.env_name, object_params.vocab_name)
    for step in tqdm(range(frontier_agent.steps), desc=f"test {scene}", ncols=80):
        yield step, obs["rgb"][:, :, :3], ground_truth_instances(obs, vocabulary, classes)
        action, _ = agent.act(env.get_agent_state(), obs["depth"])
        if action == "stop":
            break
        obs, _, _, _ = env.step(action)
    save_tile(agent.map_tile(), out_dir / DECISION_FRAME_PNG)
    env.close()
