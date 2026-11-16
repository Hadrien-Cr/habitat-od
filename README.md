# habitat-od

Generates COCO-format object-detection datasets from simulated indoor scenes (HSSD-HAB, ProcTHOR-hab, MP3D,
Gibson-Semantic) in Habitat, fine-tunes detectors on them, and evaluates Detic-driven object search on IMPRINT's
HSSD-rare ObjectNav episodes. Ground truth comes from the simulator's semantic sensor. See `INSTALL.MD` for setup.

# Dataset collection

`habitat_embodied_sl/collect.py` explores scenes with an exploration agent, maps Detic detections into object views,
and writes COCO train/test splits (e.g. `datasets/smoke_sl/{train,test}`). Its pipeline steps, thresholds and
outputs are described in `habitat_embodied_sl/README.md`.

# Training

Two entry points, both launched from the repo root with `PYTHONPATH=.`:

- **`pretrain.py`** -- trains a plain detectron2 COCO architecture from scratch or a COCO
  checkpoint (e.g. `BoxSup-C2_L_CLIP_SwinB_896b32_4x`), on `coco_testbench`'s local COCO copy or
  any registered COCO dataset. Supports multi-GPU via `--num-gpus`.
- **`habitat_embodied_sl/main.py`** -- fine-tunes `habitat_embodied_sl/meta_arch.py`'s `CustomFT_Detic` on an
  `ObjectViewsDataset` (one of `habitat_embodied_sl/collect.py`'s collection train variants:
  `pseudo`, `gt`, `gt++`), scored against that run's written test set (e.g.
  `datasets/procthor_medium/test`), also once before training (`InitialEvalHook`, logged at iteration 0). Each evaluation also saves its confusion matrix as `eval/<dataset>/confusion_matrix_iter<N>.png` (`ConfusionMatrixWriter`). Before
  training it also logs to TensorBoard a bar chart of the train and test sets' annotation counts per class, each stacked by scene, then both stacked
  (`ClassCountsWriter`, image `class_counts`, also saved as `class_counts.png` in the run's log folder). Single-GPU only.

Both score against their test set via a training-time `EvalHook`. Dataset registration, the
evaluator-wired `Trainer`, periodic TensorBoard image logging (`PeriodicImageWriter`: ground truth as shaded boxes and one-class-per-box predictions as outlined boxes, on one frame under `images/`, on test frames, plus train frames for `main.py`), and
final-checkpoint reporting are shared by both via `common/utils/detectron_trainer.py`.

- **`evaluate.py`** -- standalone evaluation of a Detic config/checkpoint against a COCO test
  split written by `habitat_embodied_sl/collect.py` (e.g. `datasets/smoke_sl/test`), via Detic's own
  `train_net.py` `do_test` and `COCOEvaluator`. Resets Detic's zero-shot classifier to the CLIP
  text embeddings of the split's categories.

`habitat_embodied_sl/config/r50_procthor8_embodied_sl_gt++.yaml` (the reference R50 fine-tune on the `gt++` variant, 1,000 iterations)
and `config/procthor8_collection.yaml` are `main.py`'s defaults:

```bash
PYTHONPATH=. python habitat_embodied_sl/main.py
```

# Object search (`habitat_od_search`)

Steps, thresholds, metrics and outputs are described in `habitat_od_search/README.md`.

- **`habitat_od_search/main.py`** -- evaluates `SearchAgent` (`search_agent.py`) on IMPRINT's HSSD-rare episodes
  (`hssd_rare_dataset.py`). Each step, the agent scores its rgb against the CLIP text embedding of the episode's
  target category, fuses the best detection into a goal-score map over its own depth-built map, and then either
  heads to the believed goal or explores frontiers. The run is configured by `--config` (default
  `habitat_od_search/config/hssd_rare_eval.yaml`). `agent.use_gt_detections` is an ablation that swaps in
  ground-truth goal detections. Per-episode success, SPL, distance to goal and failure category (`evaluator.py`: exploration failure, FN/FP
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
