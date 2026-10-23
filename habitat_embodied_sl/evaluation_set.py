"""Test set of an evaluation scene: a vanilla FrontierAgent explores it until downsampling x frames_per_scene of its frames
show some valid ground truth (or max_steps_per_scene run out), then the frames_per_scene most diverse of those, by
farthest-point sampling of their CLIP image embeddings, are kept with all their valid ground-truth instances."""
from typing import List, Tuple

import clip
import numpy as np
import torch
from detectron2.structures import Instances
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from common.env_utils.object_annotations import resolve_classes
from common.env_utils.sense import ground_truth_instances
from common.samplers.diversity_sampler import k_centroid_greedy
from common.utils.plot_utils import save_decision_map
from habitat_embodied_al import constants
from habitat_embodied_al.reproduce.main import build_agent, build_env

_CLIP_MODEL = "ViT-B/32"
_CLIP_BATCH_SIZE = 64


class ClipImageEncoder:
    def __init__(self, device: str):
        self.device = device
        self.model, self.preprocess = clip.load(_CLIP_MODEL, device=device)

    @torch.no_grad()
    def embed(self, images: List[np.ndarray]) -> np.ndarray:
        """(N, D) unit-norm CLIP embeddings of the rgb images."""
        batches = [images[start:start + _CLIP_BATCH_SIZE] for start in range(0, len(images), _CLIP_BATCH_SIZE)]
        features = torch.cat([
            self.model.encode_image(torch.stack([self.preprocess(Image.fromarray(image)) for image in batch]).to(self.device)).float()
            for batch in batches
        ])
        return torch.nn.functional.normalize(features, dim=-1).cpu().numpy()


def evaluation_frames(scene: str, cfg: DictConfig, classes: list, image_size: int, encoder: ClipImageEncoder) -> List[Tuple[int, np.ndarray, Instances]]:
    """The cfg.evaluation.frames_per_scene most diverse of explored_frames, by step."""
    frames = explored_frames(scene, cfg, classes, image_size)
    if not frames:
        return []
    embeddings = encoder.embed([rgb for _, rgb, _ in frames])
    kept = k_centroid_greedy(1 - embeddings @ embeddings.T, min(cfg.evaluation.frames_per_scene, len(frames)), np.random.default_rng(cfg.seed))
    return [frames[index] for index in sorted(kept)]


def explored_frames(scene: str, cfg: DictConfig, classes: list, image_size: int) -> List[Tuple[int, np.ndarray, Instances]]:
    """(step, rgb, valid ground truth of classes) of each frame of a FrontierAgent's exploration of the scene showing some,
    until cfg.evaluation.downsampling x frames_per_scene of them are met, the agent stops or max_steps_per_scene run out;
    its final map and trajectory saved as datasets/<run_name>/test/decision_frames/<scene>.png."""
    evaluation, object_params = cfg.evaluation, cfg.object_params
    env, habitat_cfg = build_env(scene, cfg.seed, evaluation.max_steps_per_scene, object_params, image_size)
    rng = np.random.default_rng(cfg.seed)
    agent = build_agent("fbe", rng, env, habitat_cfg)
    obs = env.reset(env.episodes[0], rng=rng)
    vocabulary = resolve_classes(object_params.env_name, object_params.vocab_name)
    frames = []
    for step in tqdm(range(evaluation.max_steps_per_scene), desc=f"test {scene}", ncols=80):
        ground_truth = ground_truth_instances(obs, vocabulary, classes)
        if len(ground_truth):
            frames.append((step, obs["rgb"][:, :, :3], ground_truth))
        action = agent.act(env.get_agent_state())
        if len(frames) >= evaluation.downsampling * evaluation.frames_per_scene or action == "stop":
            break
        obs, _, _, _ = env.step(action)
    save_decision_map(agent, constants.DATASET_ROOT / cfg.run_name / "test" / "decision_frames" / f"{scene}.png")
    env.close()
    return frames
