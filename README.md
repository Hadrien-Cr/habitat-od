# habitat-od

Generates COCO-format object-detection datasets from simulated indoor scenes (HSSD-HAB, ProcTHOR-hab, MP3D,
Gibson-Semantic) in Habitat, fine-tunes detectors on them, and evaluates Detic-driven object search on IMPRINT's
HSSD-rare ObjectNav episodes. Ground truth comes from the simulator's semantic sensor. See `INSTALL.MD` for setup.

# Dataset collection

`habitat_embodied_sl/main.py` explores scenes with an exploration agent, maps Detic detections into object views,
and writes COCO train/test splits (e.g. `datasets/smoke_sl/{train,test}`). Its pipeline steps, thresholds and
outputs are described in `habitat_embodied_sl/README.md`.

# Training

Two entry points, both launched from the repo root with `PYTHONPATH=.`:

- **`pretrain.py`** -- trains a plain detectron2 COCO architecture from scratch or a COCO
  checkpoint (e.g. `BoxSup-C2_L_CLIP_SwinB_896b32_4x`), on `coco_testbench`'s local COCO copy or
  any registered COCO dataset. Supports multi-GPU via `--num-gpus`.
- **`train.py`** -- fine-tunes `habitat_embodied_sl/meta_arch.py`'s `CustomFT_Detic` on an
  `ObjectViewsDataset` (one of `habitat_embodied_sl/main.py`'s collection train variants:
  `pseudo`, `pseudo_gt_classes`, `recovered`), scored against that run's written test set (e.g.
  `datasets/smoke_sl/test`). Single-GPU only.

Both score against their test set via a training-time `EvalHook`. Dataset registration, the
evaluator-wired `Trainer`, periodic TensorBoard image logging (`PeriodicImageWriter`), and
final-checkpoint reporting are shared by both via `common/utils/detectron_trainer.py`.

- **`evaluate.py`** -- standalone evaluation of a Detic config/checkpoint against a COCO test
  split written by `habitat_embodied_sl/main.py` (e.g. `datasets/smoke_sl/test`), via Detic's own
  `train_net.py` `do_test` and `COCOEvaluator`. Resets Detic's zero-shot classifier to the CLIP
  text embeddings of the split's categories.

`habitat_embodied_sl/config/smoke_sl_ft.yaml` is a 100-iteration overlay on
`finetune_detic_swinb.yaml` (same model/data shape, `SOLVER.MAX_ITER`/`WARMUP_ITERS`/
`CHECKPOINT_PERIOD` and `TEST.EVAL_PERIOD` scaled down to fit), to check `train.py`'s fine-tuning
loop runs end-to-end on `smoke_sl`'s small train/test split in minutes:

```bash
PYTHONPATH=. python train.py --config smoke_sl.yaml \
    --config-file habitat_embodied_sl/config/smoke_sl_ft.yaml --variant pseudo
```

# Object search (`habitat_od_search`)

Steps, thresholds, metrics and outputs are described in `habitat_od_search/README.md`.

- **`habitat_od_search/main.py`** -- evaluates `SearchAgent` (`search_agent.py`) on IMPRINT's HSSD-rare episodes
  (`hssd_rare_dataset.py`). Each step, the agent scores its rgb against the CLIP text embedding of the episode's
  target category, fuses the best detection into a goal-score map over its own depth-built map, and then either
  heads to the believed goal or explores frontiers. The run is configured by `--config` (default
  `habitat_od_search/config/hssd_rare_eval.yaml`). There are two ablations: `agent.use_gt_detections` swaps in
  ground-truth goal detections, and `agent.use_oracle_path` follows the navmesh shortest path to the goal.
  Per-episode success, SPL, distance to goal and failure category (`evaluator.py`: exploration failure, FN/FP
  recognition failure, TP early stop) are appended to `habitat_od_search/logs/<run_name>/metrics.jsonl`. When
  the run ends, per-rarity-tier summary and failure tables go to `summary.md`. With `save_visuals`, it also
  writes per-step snapshots and an `episode.webp` per episode.
- **`habitat_od_search/enhanced_detector.py`** -- enhanced-detector baseline for one target in one scene
  (`--scene`, `--category`). It replaces the text embedding with an ensemble of Detic region embeddings of example
  views of the target: simulator renders of the isolated object and Bing-scraped web images. It then ranks the
  class-agnostic boxes collected from random poses against each query (`query_matches.png`). Outputs go to
  `habitat_od_search/logs/enhanced_detector/<scene_id>/<category>/`.

```bash
export HABITAT_DATA=$HOME/habitat_data/data BASE_DIR=$(pwd) PYTHONNOUSERSITE=1
PYTHONPATH=. python habitat_od_search/main.py --config habitat_od_search/config/hssd_rare_eval.yaml
PYTHONPATH=. python habitat_od_search/enhanced_detector.py --scene 102344094
```
