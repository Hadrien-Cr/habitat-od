# habitat-od

Utils to create benchmark datasets for object detection in AI2THOR

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

# Roadmap
- [x] habitat-od: Automatically generate an object detection dataset from HSSD-HAB scenes
    - [x] Base Implementation
    - [x] Implementation with co-visibility clustering

- [ ] habitat-c-od: Continual learning object detection dataset from ProcTHOR scenes

- [ ] habitat-act-od: Generate Active Vision dataset from ProcTHOR scenes
    - [ ] Base Implementation
    - [ ] Implement confusing examples
