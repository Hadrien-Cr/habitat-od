"""Hyperparameter search over an already-collected round's dataset (e.g. main.py's own
round_010/) -- reuses main.py's train_round for one detectron2 fine-tune + eval per trial, no
simulator/exploration involved. Optuna (TPE by default) searches the LR schedule.

Usage:
  PYTHONPATH=. python habitat_embodied_al/reproduce/sweep.py \
      --config-file habitat_embodied_al/reproduce/config/mask_rcnn_R_50_FPN_coco_procthor_reproduce_default500.yaml \
      --train-dataset datasets/procthor_hpsweep_s100_baseline/round_010 \
      --val-dataset datasets/procthor_reproduce_valdistinct_ProcTHOR-Val-0 \
      --init-checkpoint habitat_embodied_al/reproduce/logs/mask_rcnn_R_50_FPN_coco_procthor_reproduce/model_final.pth \
      --n-trials 30
"""
import argparse
from pathlib import Path

import numpy as np
import optuna
from detectron2.data import MetadataCatalog  # type: ignore

from habitat_embodied_al.dataset import register_dataset
from habitat_embodied_al.reproduce.main import train_round


def objective(trial: optuna.Trial, args: argparse.Namespace, train_name: str, val_name: str, num_classes: int, out_root: Path) -> float:
    max_iter = trial.suggest_int("max_iter", 100, 2000, step=100)
    base_lr = trial.suggest_float("base_lr", 1e-4, 1e-1, log=True)
    warmup_iters = trial.suggest_int("warmup_iters", 0, max_iter // 4)
    step1, step2 = int(0.7 * max_iter), int(0.9 * max_iter)

    opts = [
        "SOLVER.IMS_PER_BATCH", "32",
        "SOLVER.BASE_LR", str(base_lr),
        "SOLVER.MAX_ITER", str(max_iter),
        "SOLVER.STEPS", f"({step1}, {step2})",
        "SOLVER.WARMUP_ITERS", str(warmup_iters),
    ]

    output_dir = out_root / f"trial_{trial.number:03d}"
    _, eval_results = train_round(
        args.config_file, train_name, val_name, num_classes,
        args.init_checkpoint, opts, output_dir,
    )
    for ckpt in output_dir.glob("*.pth"):
        ckpt.unlink()
    ap = eval_results.get("bbox", {}).get("AP")
    if ap is None or np.isnan(ap):
        raise optuna.TrialPruned("no AP (empty val predictions)")
    return ap


def main(args: argparse.Namespace) -> None:
    train_name = register_dataset(Path(args.train_dataset), "train", name="sweep_train")
    val_name = register_dataset(Path(args.val_dataset), "val", name="sweep_val")
    num_classes = len(MetadataCatalog.get(train_name).thing_classes)

    out_root = Path(args.config_file).resolve().parent.parent / "logs" / f"sweep_{args.study_name}"
    out_root.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name, direction="maximize",
        storage=f"sqlite:///{out_root / 'study.db'}", load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, args, train_name, val_name, num_classes, out_root), n_trials=args.n_trials)

    print(f"best trial: {study.best_trial.number} AP={study.best_value:.3f} params={study.best_params}")
    study.trials_dataframe().to_csv(out_root / "trials.csv", index=False)
    print(f"wrote {out_root / 'trials.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", default="habitat_embodied_al/reproduce/config/mask_rcnn_R_50_FPN_coco_procthor_reproduce_default500.yaml")
    parser.add_argument("--train-dataset", required=True, help="Already-built COCO dataset dir with a train.json, e.g. a round_NNN/ from a main.py run.")
    parser.add_argument("--val-dataset", required=True, help="Already-built COCO dataset dir with a val.json, e.g. datasets/procthor_reproduce_valdistinct_<scene>/.")
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", default="default")
    args = parser.parse_args()
    main(args)
