"""Runs one active-learning experiment, starting from an already-pretrained checkpoint --
the counterpart of third_party/embodied-active-learning-od's main.py, ported onto this repo's
habitat-sim/detectron2 stack (see CLAUDE.md's Architecture section for the rest of the port).

Round "init" scores --init-checkpoint as-is (no fine-tuning) to give history.json a
pre-AL baseline. Each numbered round after that: an Agent explores --scene-name for
--timesteps steps while a Detector -- running the previous round's own fine-tuned checkpoint
(round 0's Detector runs --init-checkpoint itself) -- scores every frame; the full collected
round (every candidate, ground truth included, from the same ObjectDetectorGTSensor collect_
random/collect_validation use) is written to raw_collected/ for inspection/reuse; a Sampler
picks --samples of them to send to the oracle, written separately to raw/; the growing labeled
pool (every round's raw/ selection so far) is rebuilt into one COCO dataset; the detector is
re-fine-tuned from --init-checkpoint itself every round
(not chained round-to-round, to avoid compounding drift/forgetting across rounds) on that whole
pool (trailing `opts` overlay the detectron2 cfg first, e.g. to set SOLVER.MAX_ITER); the
resulting checkpoint becomes next round's Detector. Evaluated each round against --scene-name's
own distinct validation set (datasets/procthor_reproduce_valdistinct_<scene_name>/, built
separately by collect_validation_dataset.py against ds_procthor_reproduce_valdistinct.yaml --
see ../README.md's step 2) -- collect and test scene are deliberately the same scene here, not
held-out from each other. A per-round summary (candidate/selected counts, checkpoint path, eval
results) is appended to a history.json.

Only RandomAgent/FrontierAgent/SweepAgent (common/agents/) + RandomSampler/GreedySampler/
DiversitySampler/TwoStageSampler (common/samplers/) are wired in -- `ada` (the reference's
learned/Bayesian-optimization navigation policy) isn't ported, see ../README.md.

A round's selection pool is its own freshly collected candidates plus every earlier round's
rejects (candidates offered to the Sampler but not picked) -- so round i's Sampler chooses from
candidate(round i) | rejected(round j<i), not just its own round's candidates.

Single scene per experiment (--scene-name), matching the original's --scene-name.
Always single-process (no --num-gpus) -- each round's fine-tune is short.

Usage:
  PYTHONPATH=. python habitat_embodied_al/reproduce/main.py \
      --config-file habitat_embodied_al/reproduce/config/mask_rcnn_R_50_FPN_coco_procthor_reproduce.yaml \
      --scene-name ProcTHOR-Val-0 --rounds 5 --timesteps 400 --samples 20 \
      --agent fbe --sampler two-stage --scoring count \
      --init-checkpoint habitat_embodied_al/reproduce/logs/mask_rcnn_R_50_FPN_coco_procthor_reproduce/model_final.pth \
      SOLVER.IMS_PER_BATCH 32 SOLVER.BASE_LR 0.001 SOLVER.MAX_ITER 100
"""
import argparse
import json
from pathlib import Path
import shutil
from typing import Optional

from detectron2.data.catalog import MetadataCatalog
import habitat  # type: ignore
from habitat.config import read_write  # type: ignore
import numpy as np
from detectron2.checkpoint import DetectionCheckpointer  # type: ignore
from detectron2.config import get_cfg  # type: ignore

from common.agents.abstract_agent import Agent
from common.agents.frontier_agent import FrontierAgent
from common.agents.random_agent import RandomAgent
from common.agents.sweep_agent import SweepAgent
from common.utils.interface import Candidate
from common.env_utils.env_base import ExplorationEnv
from common.env_utils.object_annotations import resolve_classes
from common.samplers.abstract_sampler import Sampler
from common.samplers.discrepancy import DiscretizedPose, advance_pose
from common.samplers.diversity_sampler import DiversitySampler
from common.samplers.greedy_sampler import METHODS, GreedySampler
from common.samplers.random_sampler import RandomSampler
from common.samplers.two_stage_sampler import TwoStageSampler
from common.utils.eval_utils import compute_confusion_matrix

from common.utils.data_utils import save_obs
from common.utils.plot_utils import plot_al_history
from detector import Detector
from eval import Trainer, register_dataset
from habitat_embodied_al import constants
from habitat_embodied_al.coco_writer import build_coco_dataset
from habitat_embodied_al.collection import configure_scene, _write_mosaic, write_selection_mosaic
from habitat_embodied_al.reproduce.classes import PROCTHOR_SHORT_CLASSES
from tqdm import tqdm

# ds_procthor_reproduce_valdistinct.yaml's own run_name -- fixed, since its per-scene val sets
# (see ../README.md's step 2) are what every AL run here scores against, regardless of the
# run's own --run-name.
_VALDISTINCT_RUN_NAME = "procthor_reproduce_valdistinct"



def _save_candidate(candidate: Candidate, out_dir: Path, step: int) -> list[str]:
    return save_obs(
        str(out_dir), 0,
        [{ "rgb": candidate.rbg, "bbsgt": candidate.bbsgt, "colored_tdmap": candidate.colored_tdmap }],
        step, modalities=["rgb", "bbsgt", "colored_tdmap"],
    )

def collect_round(
    env: ExplorationEnv, agent, detector: Detector, timesteps: int, out_dir: Path,
    num_yaw: int, round_idx: int,
) -> list:
    """Explores for `timesteps` steps. Tracks a DiscretizedPose (common/samplers/discrepancy.py)
    odometry-style across the round -- reset every round, so pose-adjacency (and the
    discrepancy scoring built on it, common/samplers/scoring.py::score_count_discrepancy) only
    ever compares candidates sharing the same round_idx."""
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    pose = DiscretizedPose(0.0, 0.0, 0)
    prev_agent_state = env.get_agent_state()

    for step in tqdm(range(timesteps), desc="collecting round", ncols=80, leave=False):
        action = agent.act(env)
        obs, _, _, _ = env.step(action)
        curr_agent_state = env.get_agent_state()
        moved = not np.allclose(prev_agent_state.position, curr_agent_state.position)
        pose = advance_pose(pose, action, moved, num_yaw)

        rgb = obs["rgb"][:, :, :3]
        pred_instances = detector.predict(rgb)
        candidate = Candidate(
            rgb, obs["bbsgt"], obs["colored_tdmap"], pred_instances, curr_agent_state,
            action=action, disc_pose=pose, round_idx=round_idx,
        )
        candidates.append(candidate)
        paths = _save_candidate(candidate, out_dir, step)
        candidate.filename = Path(next(p for p in paths if "modality_rgb" in p)).name
        prev_agent_state = curr_agent_state
    return candidates


def build_sampler(
    sampler_name: str, scoring: Optional[str], budget_expanding_ratio: float, rng: np.random.Generator, classes: list,
    width: int, height: int, grid_size: float, num_yaw: int, min_area: float,
) -> Sampler:
    if sampler_name == "random":
        return RandomSampler(rng)
    if sampler_name == "greedy":
        return GreedySampler(scoring, classes=classes, width=width, height=height, grid_size=grid_size, num_yaw=num_yaw, min_area=min_area)
    if sampler_name == "diversity":
        return DiversitySampler(rng)
    if sampler_name == "two-stage":
        first = GreedySampler(scoring, classes=classes, width=width, height=height, grid_size=grid_size, num_yaw=num_yaw, min_area=min_area)
        return TwoStageSampler(first, DiversitySampler(rng), budget_expanding_ratio=budget_expanding_ratio)
    raise ValueError(f"Unknown sampler {sampler_name!r}")


def build_agent(agent_name: str, rng: np.random.Generator) -> Agent:
    if agent_name == "random":
        return RandomAgent(rng)
    if agent_name == "fbe":
        return FrontierAgent(rng)
    if agent_name == "sweep":
        return SweepAgent(rng)
    raise ValueError(f"Unknown agent {agent_name!r}")


def classwise_ap_from_eval(eval_results: dict) -> dict:
    bbox = eval_results.get("bbox", {})
    return {k[len("AP-"):]: v / 100.0 for k, v in bbox.items() if k.startswith("AP-") and not np.isnan(v)}


def save_candidates(candidates: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for step, candidate in enumerate(candidates):
        _save_candidate(candidate, out_dir, step)


def train_round(
    config_file: str, train_name: str, val_name: str, num_classes: int,
    init_weights: str, opts: list, output_dir: Path,
) -> tuple[str, dict]:
    """ Train and evaluate a detector on the given train_name/val_name datasets, returning the final checkpoint"""
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(opts)

    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,) if val_name else ()
    cfg.MODEL.WEIGHTS = init_weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.freeze()

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    predictions_json = output_dir / "eval" / val_name / "coco_instances_results.json"
    if predictions_json.exists():
        gt_json = Path(MetadataCatalog.get(val_name).json_file)
        confusion_matrix_png = output_dir / "eval" / val_name / "confusion_matrix.png"
        compute_confusion_matrix(predictions_json, gt_json, confusion_matrix_png)
        print(f"{val_name}: wrote confusion matrix to {confusion_matrix_png}")

    return str(output_dir / "model_final.pth"), dict(trainer._last_eval_results)


def eval_checkpoint(
    config_file: str, val_name: str, num_classes: int, weights: str, opts: list, output_dir: Path,
) -> dict:
    """Scores an already-trained checkpoint against val_name with no further fine-tuning """
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(opts)
    cfg.DATASETS.TEST = (val_name,) if val_name else ()
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.freeze()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    eval_results = dict(Trainer.test(cfg, model))

    predictions_json = output_dir / "eval" / val_name / "coco_instances_results.json"
    if predictions_json.exists():
        gt_json = Path(MetadataCatalog.get(val_name).json_file)
        confusion_matrix_png = output_dir / "eval" / val_name / "confusion_matrix.png"
        compute_confusion_matrix(predictions_json, gt_json, confusion_matrix_png)
        print(f"{val_name}: wrote confusion matrix to {confusion_matrix_png}")

    return eval_results


def main(args: argparse.Namespace) -> None:
    object_params = {
        "env_name": args.env_name, "vocab_name": args.vocab_name, "area_thr": args.area_thr,
        "filter_low_visibility": args.filter_low_visibility, "min_visibility_fraction": args.min_visibility_fraction,
        "filter_classes": args.filter_classes,
    }
    full_classes = resolve_classes(args.env_name, args.vocab_name)
    kept_classes = [c for c in full_classes if c != "unknown" and (not args.filter_classes or c in args.filter_classes)]
    checkpoint = args.init_checkpoint

    # set up the validation dataset
    val_dataset_dir = constants.DATASET_ROOT / f"{_VALDISTINCT_RUN_NAME}_{args.scene_name}"
    val_name = None
    if (val_dataset_dir / "val.json").exists():
        val_name = register_dataset(val_dataset_dir, "val", name=f"{args.run_name}_val")
    else:
        print(f"no {val_dataset_dir} -- run collect_validation_dataset.py against "
              "ds_procthor_reproduce_valdistinct.yaml first to get per-round eval numbers")


    base_log_dir = Path(args.config_file).resolve().parent.parent / "logs" / f"al_{args.run_name}"
    raw_root = constants.DATA_ROOT / args.run_name / "al_pool"
    history = []

    # empty al_pool, dataset, logs
    shutil.rmtree(constants.DATA_ROOT / args.run_name / "al_pool", ignore_errors=True)
    shutil.rmtree(constants.DATASET_ROOT / args.run_name, ignore_errors=True)
    shutil.rmtree(base_log_dir, ignore_errors=True)

    # initial eval of init_checkpoint
    init_eval_results = eval_checkpoint(args.config_file, val_name, len(kept_classes), checkpoint, args.opts, base_log_dir / "round_init")
    init_summary = {"round": "init", "n_candidates": 0, "n_selected": 0, "checkpoint": checkpoint, "eval": init_eval_results}
    history.append(init_summary)
    print(f"round init: {init_summary}")

    # set up detector, scene, env, agent, sampler
    detector = Detector(args.config_file, checkpoint, kept_classes, agnostic_nms_thresh=args.agnostic_nms_thresh)
    habitat_cfg = habitat.get_config(config_path="common/config/hssd-hab/default.yaml")
    with read_write(habitat_cfg):
        habitat_cfg.habitat.seed = args.seed


    configure_scene(habitat_cfg, [args.scene_name], args.timesteps, object_params)
    env = ExplorationEnv(config=habitat_cfg)
    rng = np.random.default_rng(args.seed)
    reset_obs = env.reset(env.episodes[0], rng=rng)

    grid_size = habitat_cfg.habitat.simulator.forward_step_size
    num_yaw = 360 // habitat_cfg.habitat.simulator.turn_angle
    min_area = args.area_thr
    height, width = reset_obs["rgb"].shape[:2]

    agent = build_agent(args.agent, rng)
    sampler = build_sampler(args.sampler, args.scoring, args.budget_expanding_ratio, rng, full_classes, width, height, grid_size, num_yaw, min_area)

    # al loop: collect -> select -> retrain, repeated for R rounds
    rejected: list = []  # candidates offered to the Sampler in earlier rounds but not picked
    selected_all: list = []  # every candidate selected in earlier rounds -- the growing labeled pool (DiversitySampler's "annotated")
    classwise_ap = classwise_ap_from_eval(init_eval_results)  # oracle scoring's per-class difficulty, refreshed each round from the latest eval
    for round_idx in range(1, args.rounds + 1):
        # collection -- progressively written to raw_collected/raw as each candidate is captured
        candidates = collect_round(
            env, agent, detector, args.timesteps, raw_root / f"round_{round_idx:03d}_collected" / "raw",
            num_yaw, round_idx,
        )
        _write_mosaic(raw_root / f"round_{round_idx:03d}_collected" / "raw", object_params)

        # sample selection -- this round's candidates plus every earlier round's rejects
        pool = candidates + rejected
        selected = sampler.select(pool, args.samples, annotated=selected_all, classwise_ap=classwise_ap)
        save_candidates(selected, raw_root / f"round_{round_idx:03d}_selected" / "raw")
        _write_mosaic(raw_root / f"round_{round_idx:03d}_selected" / "raw", object_params)

        if isinstance(sampler, TwoStageSampler):
            top_n = sampler.top_n(pool, args.samples, annotated=selected_all, classwise_ap=classwise_ap)
            write_selection_mosaic(
                top_n, selected, full_classes, kept_classes, sampler.first, classwise_ap,
                raw_root / f"round_{round_idx:03d}_selected" / "topN_mosaic.png", pool,
            )

        selected_ids = {id(c) for c in selected}
        rejected = [c for c in pool if id(c) not in selected_ids]
        selected_all = selected_all + selected

        # ds creation
        raw_dirs = sorted(raw_root.glob("round_*_selected/raw"))  # every round's selection so far
        dataset_root = constants.DATASET_ROOT / args.run_name /f"round_{round_idx:03d}"
        build_coco_dataset(
            [str(d) for d in raw_dirs], str(dataset_root), "train",
            args.env_name, args.vocab_name, args.filter_classes,
            filter_empty=True,
        )
        train_name = register_dataset(dataset_root, "train", name=f"{args.run_name}_round{round_idx:03d}_train")

        # retraining + eval detector
        checkpoint, eval_results = train_round(
            args.config_file, train_name, val_name, len(kept_classes),
            args.init_checkpoint, args.opts, base_log_dir / f"round_{round_idx:03d}",
        )

        # update collection detector
        detector = Detector(args.config_file, checkpoint, kept_classes, agnostic_nms_thresh=args.agnostic_nms_thresh)
        classwise_ap = classwise_ap_from_eval(eval_results)

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
    plot_al_history(history, base_log_dir / "map_by_round.png")
    print(f"Active-learning run '{args.run_name}' logged to {base_log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", default="habitat_embodied_al/reproduce/config/mask_rcnn_R_50_FPN_coco_procthor_reproduce.yaml")
    parser.add_argument("--run-name", help="Defaults to a name derived from --scene-name/--agent/--sampler/--scoring/--rounds/--samples/--timesteps/--seed.")
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--timesteps", type=int, required=True, help="Navigation timesteps per round.")
    parser.add_argument("--samples", type=int, required=True, help="Annotated samples selected per round.")
    parser.add_argument("--agent", choices=["random", "fbe", "sweep"], default="random")
    parser.add_argument("--sampler", choices=["random", "greedy", "diversity", "two-stage"], default="random")
    parser.add_argument("--scoring", choices=METHODS, help="Required for --sampler greedy/two-stage.")
    parser.add_argument("--budget-expanding-ratio", type=float, default=2.0, help="Only used by --sampler two-stage.")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--agnostic-nms-thresh", type=float, default=0.5, help="Collapses same-box detections across classes -- detectron2's own NMS is per-class only. Pass 0.0 to disable.")

    parser.add_argument("--env-name", default="ProcTHOR-hab")
    parser.add_argument("--vocab-name", default="ProcTHOR-native")
    parser.add_argument("--area-thr", type=float, default=400.0)
    parser.add_argument("--filter-low-visibility", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-visibility-fraction", type=float, default=0.15)
    parser.add_argument("--filter-classes", nargs="*", default=PROCTHOR_SHORT_CLASSES, help="Allow-list of classes to keep; pass an empty list to keep every class.")

    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
        help="Modify the detectron2 config by adding 'KEY VALUE' pairs at the end of the command, "
             "e.g. SOLVER.MAX_ITER 100 SOLVER.BASE_LR 0.001.",
    )
    args = parser.parse_args()

    if args.sampler in ("greedy", "two-stage") and not args.scoring:
        parser.error("--scoring is required when --sampler is 'greedy' or 'two-stage'")

    args.run_name = args.run_name or "_".join([
        args.scene_name, args.agent, args.sampler, args.scoring or "na",
        str(args.rounds), str(args.samples), str(args.timesteps), f"seed{args.seed}",
    ])

    main(args)
