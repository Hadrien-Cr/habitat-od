# Embodied SL experiments (R50, procthor8)

Runs are named `<backbone>_<data>_<method>` after their config (`config/<name>.yaml`) and log to `logs/<name>/`.
`procthor8` is `procthor8_collection.yaml`'s 8 train houses (dataset folder `datasets/procthor_medium`); `unsup` marks
methods that use no label. A run's previous logs are deleted before it is rerun.

## Experiments done

**Data (recollected 2026-11-16, `datasets/procthor_medium`):** test set ProcTHOR-Val-0 to -7, 369 frames, 1,877 boxes,
75 of the 95 classes present; train set (full ground truth of the train frames) 2,248 frames, 14,807 boxes. Variants
`pseudo` and `gt` 272 objects, 2,690 views; `gt++` 342 objects, 3,308 views.

| run | config | test AP | test AP50 | test AP50 agnostic | test recall50 agnostic | train AP | train AP50 |
|---|---|---|---|---|---|---|---|
| `r50_procthor8_source` | `r50_detic.yaml` (`adapt.py --method source`) | 30.2 | 36.1 | 66.9 | 95.0 | 28.3 | 33.7 |
| `r50_procthor8_unsup_norm` | `r50_detic.yaml`, `r50_procthor8_unsup_norm.yaml` (`adapt.py --method norm`) | to rerun | | | | | |
| `r50_procthor8_unsup_mean_teacher` | `r50_procthor8_unsup_mean_teacher.yaml` (`mean_teacher.py`) | to rerun | | | | | |
| `r50_procthor8_embodied_sl_{pseudo,gt,gt++}` | `r50_procthor8_embodied_sl_<variant>.yaml` (`main.py`) | to run | | | | | |

**On the previous collection** (same test set, `gt++` then 325 objects and 3,131 views; last evaluation): NORM 30.4 AP,
Mean-Teacher 30.3, embodied SL (`gt++`) **40.0** AP, 48.0 AP50, 29.4 APs.

- **Embodied SL (previous collection):** +9.8 AP over source; AP 30.2, 36.4, 37.5, 39.7, 40.0 every 250 iterations (doubling the iterations
  added 0.2). The 66 test classes with training objects go from 38.0 to 52.0 AP50; the 9 without any lose 4.2
  (22.5 → 18.3), safe and coffee machine the most.
- **The unsupervised baselines do nothing.** NORM +0.2 AP. Mean-Teacher (pseudo-labels at score ≥ 0.5, momentum
  0.999) +0.1 AP: it confirms what the source already predicts. Embodied SL's gain comes from its labels.
- **What made the reference:** AdamW rather than SGD (+5 AP), the student backbone at × 0.1 its learning rate, and
  training frames at the test scale (800), without which small-object AP falls below source.
- **Reading AP:** box regression is not trained and proposals come from the frozen teacher, so AP75 and above measure
  the source's localization; AP50 and per-class AP50 (`class_ap.png`, `class_table.csv`) measure what is trained. A
  class with one test box detected at IoU 0.90 to 0.95 scores exactly 90 AP.
- Class ids follow `CLASS_LABELS_PROCTHOR100`'s alphabetical order since 2026-11-16 (dataset remapped in place);
  checkpoints from before index the old order.

## Next experiments

- **H, hssd variant:** collect and test on HSSD scenes instead.
- **V, train variants:** `r50_procthor8_embodied_sl_{pseudo,gt,gt++}`, one config each: `pseudo` (Detic's classes),
  `gt` (the same objects, correct classes), `gt++` (also without false positives and with missed objects' views). The
  cost of label noise, and the value of recovery.
- **D, data volume** (all 3 variants): AP against the number of train houses, `r50_procthor{1,2,4,8,16}_<variant>`
  (needs a `train_scenes` config key, and 8 more houses collected for 16), and against the fraction of each house's
  exploration, `r50_procthor8_exploration_{25,50,100}_<variant>` (needs an `exploration_fraction` key).
- **S, static real datasets:** score source, NORM, Mean-Teacher and embodied SL also on NYU Depth V2's labeled subset
  (1,449 frames, boxes from its instance masks) and SUN RGB-D (2D boxes), each in its own vocabulary: the classifier
  is the class names' CLIP embeddings, reset per dataset, so fine-tuned checkpoints stay open-vocabulary. Stuff and
  catch-all classes dropped (wall, floor, otherprop). AP50 and class-agnostic AP50, split into classes in and out of
  `CLASS_LABELS_PROCTHOR100`: transfer on the classes embodied SL trains on, forgetting on the rest. NYU is ready
  (`common/env_utils/nyu_depth_v2.py` wrote `datasets/nyu_depth_v2/test`, scored by every run given
  `--static-test-dirs datasets/nyu_depth_v2/test`); SUN RGB-D needs its converter.
- **B, fewer classes:** the reference and source restricted to the classes with ≥ 3 (or ≥ 5) training objects and test
  boxes (`core33`, `core16`), so classes with 1–2 test boxes stop adding noise (needs a `kept_classes` key).
