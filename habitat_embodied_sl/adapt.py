"""Unsupervised test-time adaptation baselines for habitat_embodied_sl/main.py's fine-tuning: the source model
(CustomFT_Detic as main.py builds it, before any training step), its FrozenBatchNorm2d layers adapted without labels by
--method (but source_layers, kept at their source statistics), then scored once on every DATASETS.TEST split of
--config-file, in scene order, each in its own vocabulary, the first (procthor8's test set) giving the classes.
Logged to logs/<--run-name, or the config-file stem and method>.

Methods (hyperparameters in --adapt-config):
  source -- no adaptation; scores as main.py's iteration 0 on the same --config-file.
  norm   -- NORM: each test batch of test_batch_size frames of one scene normalized by its statistics mixed with the
            source ones, the batch's weighted B / (prior_strength + B).

Usage (defaults: procthor8, config/r50_detic.yaml, config/r50_procthor8_unsup_norm.yaml):
  PYTHONPATH=. python habitat_embodied_sl/adapt.py --config-file habitat_embodied_sl/config/r50_detic.yaml --method norm \
      --run-name r50_procthor8_unsup_norm
"""
from itertools import groupby
from pathlib import Path
from typing import List

from detectron2.data import DatasetCatalog, DatasetMapper  # type: ignore
from detectron2.data.build import trivial_batch_collator  # type: ignore
from detectron2.data.common import MapDataset  # type: ignore
from detectron2.evaluation import inference_on_dataset  # type: ignore
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from common.utils.dataset_writers import frame_scene
from common.utils.detectron_trainer import build_evaluator, report_results
from common.vision.test_time_adaptation import AdaptiveBatchNorm2d, replace_frozen_batch_norms, thawed
from common.vision.detic import build_vocabulary_model
from habitat_embodied_sl.main import score_each_vocabulary, setup, train_argument_parser

DEFAULT_CONFIG = "habitat_embodied_sl/config/r50_detic.yaml"
DEFAULT_ADAPT_CONFIG = "habitat_embodied_sl/config/r50_procthor8_unsup_norm.yaml"


def adapt_norm(model: nn.Module, adapt_cfg: DictConfig) -> None:
    replace_frozen_batch_norms(
        model, lambda frozen: thawed(frozen, AdaptiveBatchNorm2d(frozen.num_features, adapt_cfg.prior_strength, frozen.eps)),
        adapt_cfg.source_layers,
    )


ADAPTATIONS = {"source": lambda model, adapt_cfg: None, "norm": adapt_norm}


def scene_batches(records: List[dict], batch_size: int) -> List[List[int]]:
    """Indices of records in order, in batches of up to batch_size consecutive records of one scene each."""
    scenes = [frame_scene(record["file_name"]) for record in records]
    groups = [list(indices) for _, indices in groupby(range(len(records)), key=scenes.__getitem__)]
    return [group[start:start + batch_size] for group in groups for start in range(0, len(group), batch_size)]


def scene_order_scores(cfg, model: nn.Module, dataset_name: str, batch_size: int) -> dict:
    """dataset_name's ClassAP50Evaluator scores of model, run on its scene_batches."""
    records = DatasetCatalog.get(dataset_name)
    loader = DataLoader(
        MapDataset(records, DatasetMapper(cfg, is_train=False)), batch_sampler=scene_batches(records, batch_size),
        collate_fn=trivial_batch_collator, num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    return inference_on_dataset(model, loader, build_evaluator(cfg, dataset_name, model))


def main(args):
    adapt_cfg = OmegaConf.load(args.adapt_config)
    cfg, vocabulary = setup(args, args.run_name or f"{Path(args.config_file).stem}_{args.method}")
    OmegaConf.save(adapt_cfg, Path(cfg.OUTPUT_DIR) / "adapt.yaml")

    model = build_vocabulary_model(cfg, vocabulary)
    ADAPTATIONS[args.method](model, adapt_cfg)
    batch_size = adapt_cfg.test_batch_size if args.method == "norm" else 1
    results = score_each_vocabulary(cfg, model, lambda dataset_name: scene_order_scores(cfg, model, dataset_name, batch_size))
    report_results(cfg, list(cfg.DATASETS.TEST[:1]), results, cfg.MODEL.WEIGHTS)
    print(f"Run '{Path(cfg.OUTPUT_DIR).name}' logged to {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    parser = train_argument_parser()
    parser.add_argument("--method", default="source", choices=list(ADAPTATIONS), help="Adaptation method.")
    parser.add_argument("--adapt-config", default=DEFAULT_ADAPT_CONFIG, help="Adaptation methods' hyperparameters.")
    args = parser.parse_args()
    args.config_file = args.config_file or DEFAULT_CONFIG
    main(args)
