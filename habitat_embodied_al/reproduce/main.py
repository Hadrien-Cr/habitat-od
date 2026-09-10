"""Runs one active-learning experiment, starting from an already-pretrained checkpoint --
the counterpart of third_party/embodied-active-learning-od's main.py, ported onto this repo's
habitat-sim/detectron2 stack (see CLAUDE.md's Architecture section for the rest of the port).

Round "init" scores --config's `init_checkpoint` as-is (no fine-tuning) to give history.json a
pre-AL baseline. Each numbered round after that: an Agent explores --config's `scene_name` for
`timesteps` steps while a Detector -- running the previous round's own fine-tuned checkpoint
(round 0's Detector runs `init_checkpoint` itself) -- scores every frame; the full collected
round (every candidate, ground truth included, from the same ObjectDetectorGTSensor collect_
random/collect_validation use) is written to raw_collected/ for inspection/reuse; a Sampler
picks `samples` of them to send to the oracle, written separately to raw/; the growing labeled
pool (every round's raw/ selection so far) is rebuilt into one COCO dataset; the detector is
re-fine-tuned from `init_checkpoint` itself every round
(not chained round-to-round, to avoid compounding drift/forgetting across rounds) on that whole
pool (`--config`'s optional `MODEL` section overlays the detectron2 cfg first, e.g. to set
SOLVER.MAX_ITER); the resulting checkpoint becomes next round's Detector. Evaluated each round
against `scene_name`'s own distinct validation set (datasets/procthor_reproduce_valdistinct_
<scene_name>/, built separately by collect_validation_dataset.py against
ds_procthor_reproduce_valdistinct.yaml -- see ../README.md's step 2) -- collect and test scene
are deliberately the same scene here, not held-out from each other. A per-round summary
(candidate/selected counts, checkpoint path, eval results) is appended to a history.json.

Only RandomAgent (common/agents/) + RandomSampler (common/samplers/) are wired in so far --
this reproduces the loop's mechanics (collect -> select -> retrain, repeated for R rounds), not
yet the paper's actual navigation/selection strategies (ada/fbe/discrepancy-ranked sampling),
which need more from ExplorationEnv than get_random_point/get_agent_state (a discretized
transition graph, occupancy/frontier access) and aren't ported yet.

A round's selection pool is its own freshly collected candidates plus every earlier round's
rejects (candidates offered to the Sampler but not picked) -- so round i's Sampler chooses from
candidate(round i) | rejected(round j<i), not just its own round's candidates.

Single scene per experiment (--config's `scene_name`), matching the original's --scene-name.
Always single-process (no --num-gpus) -- each round's fine-tune is short.

Usage:
  PYTHONPATH=. python habitat_embodied_al/reproduce/main.py \
      --config-file habitat_embodied_al/reproduce/config/mask_rcnn_R_50_FPN_coco_procthor_reproduce.yaml \
      --config habitat_embodied_al/reproduce/config/al_procthor_reproduce.yaml
"""
import argparse
import json
import logging
from pathlib import Path
import shutil
import shutil
from typing import Any, Optional

import habitat  # type: ignore
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer  # type: ignore
from detectron2.config import get_cfg  # type: ignore
from detectron2.structures import Instances  # type: ignore
from omegaconf import OmegaConf

from common.agents.random_agent import RandomAgent
from common.env_utils.sense import BBSense
from common.utils.interface import Candidate
from common.env_utils.env_base import ExplorationEnv
from common.env_utils.object_annotations import resolve_classes
from common.samplers.random_sampler import RandomSampler
from common.utils.data_utils import save_obs
from detector import Detector
from eval import Trainer, register_dataset
from habitat_embodied_al import constants
from habitat_embodied_al.coco_writer import build_coco_dataset
from habitat_embodied_al.collection import configure_scene, _write_mosaic
from tqdm import tqdm

# ds_procthor_reproduce_valdistinct.yaml's own run_name -- fixed, since its per-scene val sets
# (see ../README.md's step 2) are what every al_*.yaml here scores against, regardless of the
# AL run's own run_name.
_VALDISTINCT_RUN_NAME = "procthor_reproduce_valdistinct"



def _save_candidate(candidate: Candidate, out_dir: Path, step: int) -> None:
    save_obs(
        str(out_dir), 0,
        [{ "rgb": candidate.rbg, "bbsgt": candidate.bbsgt, "colored_tdmap": candidate.colored_tdmap }],
        step, modalities=["rgb", "bbsgt", "colored_tdmap"],
    )
    
def collect_round(env: ExplorationEnv, agent, detector: Detector, timesteps: int, out_dir: Path) -> list:
    """Explores for `timesteps` steps"""
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for step in tqdm(range(timesteps), desc="collecting round", ncols=80, leave=False):
        action = agent.act(env)
        obs, _, _, _ = env.step(action)
        pred_instances = detector.predict(obs["rgb"][:, :, :3])
        candidate = Candidate(
            obs["rgb"][:, :, :3], obs["bbsgt"], obs["colored_tdmap"], pred_instances, env.get_agent_state(),
        )
        candidates.append(candidate)
        _save_candidate(candidate, out_dir, step)
    return candidates


def save_candidates(candidates: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for step, candidate in enumerate(candidates):
        _save_candidate(candidate, out_dir, step)


def flatten_model_args(model: dict, prefix: str = "") -> list:
    flat = []
    for key, value in model.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.extend(flatten_model_args(value, path))
        else:
            flat.extend([path, value])
    return flat


def train_round(
    config_file: str, train_name: str, val_name: Optional[str], num_classes: int,
    init_weights: str, model_overrides: Optional[dict], output_dir: Path,
) -> tuple[str, dict]:
    """ Train and evaluate a detector on the given train_name/val_name datasets, returning the final checkpoint"""
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    
    if model_overrides:
        cfg.merge_from_list(flatten_model_args(model_overrides))

    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,) if val_name else ()
    cfg.MODEL.WEIGHTS = init_weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.freeze()

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    return str(output_dir / "model_final.pth"), dict(trainer._last_eval_results)


def eval_checkpoint(
    config_file: str, val_name: Optional[str], num_classes: int, weights: str, output_dir: Path,
) -> dict:
    """Scores an already-trained checkpoint against val_name with no further fine-tuning """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.DATASETS.TEST = (val_name,) if val_name else ()
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.freeze()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    return dict(Trainer.test(cfg, model))


def main(config_file: str, al_cfg: Any) -> None:
    # logging.getLogger("detectron2.engine.defaults").setLevel(logging.WARNING)

    object_params = al_cfg.object_params
    filter_classes = object_params.get("filter_classes")
    full_classes = resolve_classes(object_params["env_name"], object_params["vocab_name"])
    kept_classes = [c for c in full_classes if c != "unknown" and (not filter_classes or c in filter_classes)]
    model_overrides = OmegaConf.to_container(al_cfg.MODEL, resolve=True) if "MODEL" in al_cfg else None
    init_checkpoint = al_cfg.init_checkpoint
    checkpoint = init_checkpoint

    # set up the validation dataset
    val_dataset_dir = constants.DATASET_ROOT / f"{_VALDISTINCT_RUN_NAME}_{al_cfg.scene_name}"
    val_name = None
    if (val_dataset_dir / "val.json").exists():
        val_name = register_dataset(val_dataset_dir, "val", name=f"{al_cfg.run_name}_val")
    else:
        print(f"no {val_dataset_dir} -- run collect_validation_dataset.py against "
              "ds_procthor_reproduce_valdistinct.yaml first to get per-round eval numbers")


    base_log_dir = Path(config_file).resolve().parent.parent / "logs" / f"al_{al_cfg.run_name}"
    raw_root = constants.DATA_ROOT / al_cfg.run_name / "al_pool"
    history = []
    
    # empty al_pool and rm dataset
    shutil.rmtree(constants.DATA_ROOT / al_cfg.run_name / "al_pool", ignore_errors=True)
    shutil.rmtree(constants.DATASET_ROOT / al_cfg.run_name, ignore_errors=True)

    # initial eval of init_checkpoint
    init_eval = eval_checkpoint(config_file, val_name, len(kept_classes), checkpoint, base_log_dir / "round_init")
    init_summary = {"round": "init", "n_candidates": 0, "n_selected": 0, "checkpoint": checkpoint, "eval": init_eval}
    history.append(init_summary)
    print(f"round init: {init_summary}")

    # set up detector, scene, env, agent, sampler
    detector = Detector(config_file, checkpoint, kept_classes)
    habitat_cfg = habitat.get_config(config_path="common/config/hssd-hab/default.yaml")
    configure_scene(habitat_cfg, [al_cfg.scene_name], al_cfg.timesteps, object_params)
    env = ExplorationEnv(config=habitat_cfg)
    env.reset(env.episodes[0])

    full_classes = resolve_classes(object_params["env_name"], object_params["vocab_name"])
    kept_classes = [c for c in full_classes if c != "unknown" and (not filter_classes or c in filter_classes)]

    rng = np.random.default_rng(habitat_cfg.habitat.seed)

    if al_cfg.agent == "random" and al_cfg.sampler == "random":
        agent = RandomAgent(rng)
        sampler = RandomSampler(rng)
    else: 
        raise ValueError

    # al loop: collect -> select -> retrain, repeated for R rounds
    rejected: list = []  # candidates offered to the Sampler in earlier rounds but not picked
    for round_idx in range(1, al_cfg.rounds + 1):
        # collection -- progressively written to raw_collected/raw as each candidate is captured
        candidates = collect_round(env, agent, detector, al_cfg.timesteps, raw_root / f"round_{round_idx:03d}_collected" / "raw")
        _write_mosaic(raw_root / f"round_{round_idx:03d}_collected" / "raw", object_params)

        # sample selection -- this round's candidates plus every earlier round's rejects
        pool = candidates + rejected
        selected = sampler.select(pool, al_cfg.samples)
        save_candidates(selected, raw_root / f"round_{round_idx:03d}_selected" / "raw")
        _write_mosaic(raw_root / f"round_{round_idx:03d}_selected" / "raw", object_params)
        selected_ids = {id(c) for c in selected}
        rejected = [c for c in pool if id(c) not in selected_ids]

        # ds creation
        raw_dirs = sorted(raw_root.glob("round_*_selected/raw"))  # every round's selection so far
        dataset_root = constants.DATASET_ROOT / al_cfg.run_name /f"round_{round_idx:03d}"
        build_coco_dataset(
            [str(d) for d in raw_dirs], str(dataset_root), "train",
            object_params["env_name"], object_params["vocab_name"], filter_classes,
            filter_empty=True,
        )
        train_name = register_dataset(dataset_root, "train", name=f"{al_cfg.run_name}_round{round_idx:03d}_train")

        # retraining + eval detector
        checkpoint, eval_results = train_round(
            config_file, train_name, val_name, len(kept_classes),
            init_checkpoint, model_overrides, base_log_dir / f"round_{round_idx:03d}",
        )

        # update collection detector
        detector = Detector(config_file, checkpoint, kept_classes)

        round_summary = {
            "round": round_idx, "n_candidates": len(candidates), "n_pool": len(pool), "n_selected": len(selected),
            "checkpoint": checkpoint, "eval": eval_results,
        }
        history.append(round_summary)
        print(f"round {round_idx}: {round_summary}")

    env.close()
    base_log_dir.mkdir(parents=True, exist_ok=True)
    with open(base_log_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"Active-learning run '{al_cfg.run_name}' logged to {base_log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", default="habitat_embodied_al/reproduce/config/mask_rcnn_R_50_FPN_coco_procthor_reproduce.yaml")
    parser.add_argument(
        "--config",
        default="habitat_embodied_al/reproduce/config/al_procthor_reproduce_ProcTHOR-Val-0.yaml",
    )
    args = parser.parse_args()
    main(args.config_file, OmegaConf.load(args.config))