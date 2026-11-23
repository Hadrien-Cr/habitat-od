# Embodied SL experiments (R50, procthor8)

Runs are named `<backbone>_<data>_<method>` after their config (`config/<name>.yaml`) and log to `logs/<name>/`.
`procthor8` is the run name of `procthor8_collection.yaml` and `procthor8_test.yaml`, which since 2026-11-19 cover 16
train and 16 test houses (`outputs/procthor8/launch.sh`'s `TRAIN`/`TEST`), not 8; `unsup` marks methods that use no
label. A run's previous logs are deleted before it is rerun.

## Experiments done

**Data (recollected 2026-11-19, 94 classes):** recorded in `raw/procthor8/`, built into one dataset per variant,
`datasets/procthor8_<variant>/`, and tested on `datasets/procthor8_test/`, named in the training configs' `DATASETS` as
registered splits (`procthor8_V2_drop_add_object_ds`, `procthor8_V4_full_train`, `procthor8_test_test`). Test set
ProcTHOR-Val-0 to -15, 724 frames, 4,467 boxes, 82 of the 94 classes present. Train houses ProcTHOR-Train-5, -6, -9,
-11, -14, -26, -30, -32, -33, -36, -38, -40, -41, -44, -45, -47: `V1_pseudo` 504 objects, 4,983 views; `V2_drop_add`
712 objects, 6,925 views; `V3_cls_box_drop_add` 712 objects, 6,859 views; `V4_full` 4,461 frames, 28,057 boxes, 81
classes. V3's maps (`evaluate_maps.py`, 3D box IoU ≥ 0.25): class-agnostic recall 61.7% (small objects 43.1%),
precision 90.7%, AP25 30.1, AP25 agnostic 60.1.

Run 2026-11-20 (`run.sh`, 1,000 iterations, evaluated at 0, 500 and 1,000), last evaluation, the best for every run but
`V1_pseudo` (29.0 AP at 500). Test cls is the test set's ground-truth boxes classified alone, given to the ROI heads as
proposals (`cls_top1acc`/`cls_top5acc`). Train is `procthor8_V4_full_train`, the train frames' full ground truth
(`V4_full`'s own training data). NYU-47 is NYU Depth V2's 47-class test set (`nyu_depth_v2_test`):

| run | test AP | test AP50 | test APs | test AP50 agnostic | test recall50 agnostic | test cls top-1 | test cls top-5 | train AP | train AP50 | NYU-47 AP50 | NYU-47 AP50 agnostic |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `r50_procthor8_source` | 27.6 | 33.6 | 23.9 | 65.6 | 93.9 | 53.2 | 84.3 | 27.3 | 32.4 | **28.2** | 32.9 |
| `r50_procthor8_unsup_norm` | 27.4 | 33.3 | 24.7 | 66.6 | 94.5 | 54.1 | 85.0 | 27.3 | 32.6 | 27.1 | 32.0 |
| `r50_procthor8_unsup_mean_teacher` | 28.7 | 35.4 | 24.2 | 61.0 | 94.8 | 51.1 | 85.5 | 28.3 | 34.1 | 27.5 | 28.7 |
| `r50_procthor8_embodied_sl_V1_pseudo` | 27.9 | 34.7 | 20.5 | 66.0 | 95.1 | 67.2 | 89.7 | 29.5 | 35.9 | 25.3 | 25.8 |
| `r50_procthor8_embodied_sl_V2_drop_add` | 30.1 | 37.5 | 22.8 | 69.6 | 95.5 | 69.4 | 90.4 | 33.8 | 41.2 | 25.6 | 25.6 |
| `r50_procthor8_embodied_sl_V3_cls_box_drop_add` | 37.7 | 46.5 | 27.9 | 70.4 | 95.2 | 75.9 | 92.6 | 51.6 | 62.5 | 25.7 | 25.2 |
| `r50_procthor8_embodied_sl_V4_full` | 39.9 | 50.0 | 30.0 | 71.4 | 95.1 | **82.0** | **95.2** | 58.8 | 72.6 | 24.0 | 23.3 |
| `r50_procthor8_embodied_sl_V4_full_bgneg` | **46.9** | **57.0** | **37.2** | **87.1** | **95.7** | 81.3 | 95.0 | 65.2 | 78.1 | 27.6 | **36.1** |

- **V, one step at a time:** the objects (`V1` → `V2`, a perfect mapper) +2.2 AP; their labels (`V2` → `V3`, a
  perfect labeler) +7.6, the largest step; full annotation over object views (`V3` → `V4_full`) +2.2; background
  negatives (`V4_full` → `V4_full_bgneg`) +7.0, and class-agnostic AP50 71.4 → 87.1: positive-only training never
  learns what is not an object. The pseudo-labels as classified (`V1`) gain +0.3 AP over source (+1.4 at 500
  iterations, then falling) and lose 3.4 APs, no better than the unsupervised baselines.
- **Classification alone:** every embodied SL variant classifies ground-truth boxes better, top-1 53.2 → 67.2 (`V1`),
  69.4, 75.9 and 82.0 (`V4_full`). `V1`'s +14.0 top-1 barely moves its AP (+0.3), so its detections lose elsewhere,
  in scores against background. `V4_full_bgneg` classifies no better than `V4_full` (81.3): its +7.0 AP is objectness.
  Mean-Teacher lowers top-1 (51.1).
- **The unsupervised baselines do nothing:** NORM −0.2 AP; Mean-Teacher +1.1 AP but class-agnostic AP50 65.6 → 61.0.
- **Per class:** 77 of the 82 test classes have V3 training objects. Their AP50 goes from 34.2 (source) to 35.8 (`V1`),
  48.3 (`V3`), 51.7 (`V4_full`) and 58.8 (`V4_full_bgneg`); the 5 without any go from 25.6 to 17.8, 18.8, 24.3 and 29.3.
- **NYU-47:** every fine-tune but `V4_full_bgneg` forgets (class-agnostic AP50 32.9 → 23.3 to 25.8; NORM 32.0,
  Mean-Teacher 28.7). `V4_full_bgneg` is the only run above source in class-agnostic AP50 (36.1), 0.6 below it in AP50.
  Split into classes in and out of `CLASS_LABELS_PROCTHOR100` in Questions below: the trained classes forget most.
- **Reading AP:** box regression is not trained and proposals come from the frozen teacher, so AP75 and above measure
  the source's localization; AP50 and per-class AP50 (`class_ap.png`, `class_table.csv`) measure what is trained. A
  class with one test box detected at IoU 0.90 to 0.95 scores exactly 90 AP.

## Questions

Per-class numbers from each run's `class_table.csv`, AP50 at the last evaluation against iteration 0.

- **Why does AP not improve when cls improves?** `V1`'s top-1 goes 53.2 → 67.2 and its AP 27.6 → 27.9;
  `V4_full_bgneg` is the reverse, top-1 −0.7 and AP +7.0. Top-1 is measured on ground-truth boxes alone, so it does not
  see how the head scores the teacher's background proposals against objects. It also averages over boxes (doorway,
  chair and window are 28% of the test boxes), while AP averages over classes. To check: top-1 per class, and whether
  each run's highest-scored false positives are background or a wrong class.
- **Why do more than 10 classes lose 10+ AP in `V1`?** 15 lose ≥ 10 and 15 gain ≥ 10 (losers: `V2` 11, `V3` 5,
  `V4_full` 8, `V4_full_bgneg` 2, Mean-Teacher 6). 6 of the 15 have no `V1` training object (lettuce −40.2, pen,
  coffee machine, safe, clothes dryer, credit card) and 4 have one or two (plunger, paper towel roll, pencil, baseball
  bat): are untrained classes crowded out by trained ones? The other 5 are trained: desk lamp (7 objects, 126 test
  boxes, −19.5), faucet (10, 28, −18.0), apple (10, 6, −15.6), pillow (6, 150, −14.5), dresser (18, 101, −11.6).
  `V3` has 2 apples where `V1` has 10, so most of `V1`'s apples are other objects. Are they losing to wrong
  pseudo-labels? To check: each `V1` view's pseudo class against its object's ground-truth class.
- **Why do some classes still lose with a perfect labeler V4?**
- **Why does every fine-tune forget NYU-47, and why does `V4_full_bgneg` not?** 
- **Why does Mean-Teacher gain AP (+1.1) but lose class-agnostic AP50 (65.6 → 61.0)?**

## Next experiments

- **H, hssd variant:** collect and test on HSSD scenes instead.
- **V, train variants (done 2026-11-20, above):** `r50_procthor8_embodied_sl_<variant>`, one config each, named `V<i>_` and the fixes applied to
  the pseudo-labels, following the pipeline's two stages: which objects the map finds (`drop_add`: false positives
  dropped, missed objects added), then how it labels them (`cls_box`: class and mask). One step at a time:

  | variant | train data | step from the previous isolates |
  |---|---|---|
  | `V1_pseudo` | object views, as classified | |
  | `V2_drop_add` | matched objects' views as classified, plus missed objects' views classified as the map's objects are | the objects (a perfect mapper) |
  | `V3_cls_box_drop_add` | the same objects with their ground truth's class and mask (F) | their labels (a perfect labeler) |
  | `V4_full` | the train set's frames, every ground-truth object annotated | object views vs full annotation |
  | `V4_full_bgneg` | `V4_full`, background proposals trained on too (Detic's own ROI heads) | positive-only training |

  `V2_drop_add` keeps the labels pseudo: missed objects are classified by `pseudo_classes` as `classify` does the map's,
  from their ground-truth masks and box extent; they keep ground-truth masks, having no pseudo one. `V4_full_bgneg` is a
  training setting, not data: background negatives are sound only where every object is labelled. On the 16-house
  collection the pseudo boxes of `V3`'s matched views had a median IoU of 0.94 with the ground truth's (64% ≥ 0.9, 12%
  < 0.7), so the box fix is folded into `cls_box`.
- **D, data volume** (all variants): AP against the number of train houses, `r50_procthor{1,2,4,8,16}_<variant>`
  (needs a `train_scenes` config key; `raw/procthor8/` already holds 16 houses), and against the fraction of each house's
  exploration, `r50_procthor8_exploration_{25,50,100}_<variant>` (needs an `exploration_fraction` key).
- **S, static real datasets:** score source, NORM, Mean-Teacher and embodied SL also on NYU Depth V2's labeled subset
  (1,449 frames, boxes from its instance masks) and SUN RGB-D (2D boxes), each in its own vocabulary: the classifier
  is the class names' CLIP embeddings, reset per dataset, so fine-tuned checkpoints stay open-vocabulary. Stuff and
  catch-all classes dropped (wall, floor, otherprop), and classes with fewer than 100 test objects
  (`static_vocabulary.py`, plurals and spelling variants merged first): NYU's 890 raw classes (360 with one box) made
  its mean AP a measure of the tail. AP50 and class-agnostic AP50, split into classes in and out of
  `CLASS_LABELS_PROCTHOR100`: transfer on the classes embodied SL trains on, forgetting on the rest.
  NYU-47 (`habitat_embodied_sl/nyu_depth_v2.py`): `datasets/nyu_depth_v2/test`, 47 classes, 21,972 boxes since
  2026-11-17 (the previous 890-class set, 29,095 boxes, kept as `datasets/nyu_depth_v2_890classes`); scored by every
  2026-11-20 run, split in and out of the vocabulary from its saved detections (Questions above). SUN RGB-D
  (`habitat_embodied_sl/sunrgbd.py`, its non-NYU frames, to `datasets/sunrgbd/test`): tested on a synthetic fixture
  only, the data not downloaded (rgbd.cs.princeton.edu/data/ answered 500 on 2026-11-17).
- **B, fewer classes:** the reference and source restricted to the classes with ≥ 3 (or ≥ 5) training objects and test
  boxes (`core33`, `core16`), so classes with 1–2 test boxes stop adding noise (needs a `kept_classes` key).
- **C, contrastive learning:** 

- **F, Fixing the box location to the gt:** when collecting, a new variant would be to use the gt box over the pseudo box.
  Implemented as part of V's `cls_box` fix (`V3_cls_box_drop_add`, `recover_gt_masks`). Box regression is not trained,
  so it changes which proposals a view labels with its class, not localization.