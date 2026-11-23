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
classes. V3's maps (`evaluate_maps.py`, 3D box IoU ≥ 0.25): AP25 30.1; TIDE dAP25/FN 62.0 and FP 2.3, the maps
missing objects far more than inventing them; Cls 28.9, Miss 19.0, Loc 0.8, Both 0.2, Bkg 0.1, Dupe 0.0.

Run 2026-11-20 (`run.sh`, 1,000 iterations, evaluated at 0, 500 and 1,000), last evaluation, the best for every run but
`V1_pseudo` (29.0 AP at 500). Test cls is the test set's ground-truth boxes classified alone, given to the ROI heads as
proposals (each box's rank in `coco_gt_box_classifications.json`). NYU-47 is NYU Depth V2's 47-class test set
(`nyu_depth_v2_test`).

TIDE (arXiv 2008.08115, `eval_utils.tide_daps`), recomputed 2026-11-23 from each run's saved last-evaluation detections:
dAP50/<error> is the AP50 gained once an oracle fixes every error of a type, lower is better. `Cls` wrong class (IoU ≥ 0.5),
`Loc` right class (IoU 0.1 to 0.5), `Both` neither, `Dupe` duplicate, `Bkg` background (IoU < 0.1 with everything),
`Miss` ground truth no detection comes near; `FP`/`FN` every false positive/negative at once. Lowest in bold, `Dupe`
aside (0.0 to 0.1 everywhere: per-class NMS).

| run | test AP50 | dAP50/Cls | dAP50/Loc | dAP50/Both | dAP50/Bkg | dAP50/Miss | dAP50/FP | dAP50/FN | NYU-47 AP50 | dAP50/Cls | dAP50/Loc | dAP50/Both | dAP50/Bkg | dAP50/Miss | dAP50/FP | dAP50/FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `r50_procthor8_source` | 33.6 | 16.4 | 1.6 | 1.2 | 3.9 | 1.2 | 50.7 | 4.5 | **28.2** | **9.5** | 5.7 | **1.8** | 6.2 | 1.9 | 40.9 | **8.4** |
| `r50_procthor8_unsup_norm` | 33.3 | 15.6 | 1.6 | 1.3 | 3.8 | 1.1 | 52.4 | **3.3** | 27.1 | 9.6 | 5.4 | **1.8** | 5.8 | 2.1 | 40.7 | 8.8 |
| `r50_procthor8_unsup_mean_teacher` | 35.4 | 18.3 | 2.1 | 1.4 | 4.1 | 0.8 | 46.8 | 4.8 | 27.5 | 10.1 | 6.4 | **1.8** | 6.7 | **1.8** | 39.3 | 8.8 |
| `r50_procthor8_embodied_sl_V1_pseudo` | 34.7 | 15.8 | 2.7 | 1.3 | 5.2 | **0.4** | 44.9 | 5.3 | 25.3 | 11.1 | 6.1 | 1.9 | 6.1 | 1.9 | 39.4 | 9.2 |
| `r50_procthor8_embodied_sl_V2_drop_add` | 37.5 | 16.8 | 2.5 | 1.7 | 4.4 | 0.5 | 44.1 | 4.8 | 25.6 | 10.5 | 6.2 | 2.0 | 6.0 | **1.8** | 40.1 | 9.1 |
| `r50_procthor8_embodied_sl_V3_cls_box_drop_add` | 46.5 | 9.8 | 2.4 | 1.4 | 4.8 | 0.5 | 40.7 | 4.7 | 25.7 | 10.4 | 6.1 | 2.1 | 5.7 | 1.9 | 39.8 | 9.2 |
| `r50_procthor8_embodied_sl_V4_full` | 50.0 | **7.0** | 2.9 | 1.6 | 6.6 | 0.5 | 38.9 | 4.2 | 24.0 | 10.3 | 5.9 | 2.1 | 5.8 | 1.9 | **39.1** | 9.0 |
| `r50_procthor8_embodied_sl_V4_full_bgneg` | **57.0** | 12.2 | **1.4** | **0.9** | **1.8** | 0.7 | **35.0** | 3.5 | 27.6 | 10.4 | **5.2** | 1.9 | **5.2** | 2.6 | 39.4 | 9.9 |

The same with one class per proposal (OCPP, `eval_utils.one_class_per_proposal`): Detic's box regression is
class-agnostic, so per-class NMS at score 0.02 emits each box once per class (source's 200,702 test detections are
35,653 boxes); OCPP keeps each box once, under its top-scored class, as an agent acting on one label per object would.

| run | test AP50_OCPP | dAP50_OCPP/Cls | dAP50_OCPP/Loc | dAP50_OCPP/Both | dAP50_OCPP/Bkg | dAP50_OCPP/Miss | dAP50_OCPP/FP | dAP50_OCPP/FN | NYU-47 AP50_OCPP | dAP50_OCPP/Cls | dAP50_OCPP/Loc | dAP50_OCPP/Both | dAP50_OCPP/Bkg | dAP50_OCPP/Miss | dAP50_OCPP/FP | dAP50_OCPP/FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `r50_procthor8_source` | 25.7 | 51.2 | 2.4 | 0.7 | **1.5** | 1.4 | 12.7 | 36.7 | **25.4** | 22.8 | 5.7 | 1.5 | 4.9 | 2.1 | 24.0 | 18.2 |
| `r50_procthor8_unsup_norm` | 25.8 | 48.9 | 2.2 | 0.9 | 1.6 | 1.2 | 14.0 | 34.5 | 24.1 | 22.9 | 5.5 | **1.4** | 4.7 | 2.3 | 23.6 | 18.6 |
| `r50_procthor8_unsup_mean_teacher` | 24.0 | 54.6 | 3.0 | **0.6** | 1.7 | **0.4** | 14.5 | 33.9 | 24.7 | 22.2 | 6.1 | 1.5 | 4.9 | **1.9** | 24.1 | 16.5 |
| `r50_procthor8_embodied_sl_V1_pseudo` | 28.7 | 44.7 | 3.6 | 0.7 | 2.3 | 0.9 | 14.5 | 30.1 | 22.8 | 22.6 | 6.0 | 1.5 | 4.8 | 2.1 | 25.0 | **16.3** |
| `r50_procthor8_embodied_sl_V2_drop_add` | 31.0 | 41.2 | 3.4 | 1.0 | 2.6 | 0.6 | 17.2 | 28.8 | 22.9 | 21.7 | 6.2 | 1.6 | 4.7 | 2.0 | 25.0 | 17.2 |
| `r50_procthor8_embodied_sl_V3_cls_box_drop_add` | 42.9 | 24.8 | 3.3 | 1.1 | 4.6 | 0.9 | 17.4 | 24.0 | 22.9 | 21.8 | 6.1 | 1.7 | 4.6 | 2.2 | 24.8 | 17.9 |
| `r50_procthor8_embodied_sl_V4_full` | 46.9 | **22.2** | 3.5 | 1.3 | 5.6 | 1.1 | 18.9 | **18.9** | 22.0 | **20.8** | 5.9 | 1.8 | 4.6 | 2.2 | 25.2 | 16.6 |
| `r50_procthor8_embodied_sl_V4_full_bgneg` | **52.8** | 29.3 | **1.7** | 0.9 | 1.7 | 2.0 | **10.9** | 26.2 | 24.2 | 24.5 | **5.1** | 1.6 | **4.2** | 3.0 | **22.4** | 21.3 |

- **V, one step at a time:** the objects (`V1` → `V2`, a perfect mapper) +2.2 AP; their labels (`V2` → `V3`, a
  perfect labeler) +7.6, the largest step, dAP50/Cls 16.8 → 9.8 with every other error type within 0.4: the pseudo-labels'
  wrong classes are what V2 loses; full annotation over object views (`V3` → `V4_full`) +2.2; background
  negatives (`V4_full` → `V4_full_bgneg`) +7.0, dAP50/Bkg 6.6 → 1.8, the lowest of every run: positive-only training
  never learns what is not an object, every other embodied SL variant above source in Bkg (4.4 to 6.6, `V4_full` the
  highest). The pseudo-labels as classified (`V1`) gain +0.3 AP over source (+1.4 at 500
  iterations, then falling) and lose 3.4 APs, no better than the unsupervised baselines.
- **One class per proposal:** the lower-scored class copies of each box are worth 3.1 (`V4_full`) to 11.4
  (Mean-Teacher) AP50: a box whose top class is wrong is still a true positive under its right one. Without them,
  embodied SL's lead over source grows (`V1` +1.1 → +3.0 AP50, `V3` +12.9 → +17.2, `V4_full_bgneg` +23.4 → +27.1) and
  Mean-Teacher falls below source (+1.8 → −1.7): its AP gain is in the copies, its top-1 label worse (top-1 51.1).
- **Classification alone:** every embodied SL variant classifies ground-truth boxes better, top-1 53.2 → 67.2 (`V1`),
  69.4, 75.9 and 82.0 (`V4_full`). `V1`'s +14.0 top-1 barely moves its AP (+0.3) nor its dAP50/Cls (16.4 → 15.8),
  while its Bkg rises 3.9 → 5.2 and Loc 1.6 → 2.7: a box whose top class is wrong still scores under its right one,
  so top-1 shows only one class per proposal (+3.0 AP50_OCPP over source, dAP50_OCPP/Cls 51.2 → 44.7). dAP50/Cls only
  falls with true labels, `V3` 9.8 and `V4_full` 7.0. `V4_full_bgneg` classifies no better than `V4_full` (81.3): its
  +7.0 AP is objectness (Bkg 6.6 → 1.8), and with background gone, more of its remaining loss is classification (Cls
  7.0 → 12.2). Mean-Teacher lowers top-1 (51.1).
- **The unsupervised baselines do nothing:** NORM −0.2 AP; Mean-Teacher +1.1 AP but the most lost to classification of
  every run, dAP50/Cls 16.4 → 18.3.
- **Per class:** 77 of the 82 test classes have V3 training objects. Their AP50 goes from 34.2 (source) to 35.8 (`V1`),
  48.3 (`V3`), 51.7 (`V4_full`) and 58.8 (`V4_full_bgneg`); the 5 without any go from 25.6 to 17.8, 18.8, 24.3 and 29.3.
- **NYU-47:** every embodied SL fine-tune but `V4_full_bgneg` forgets, AP50 28.2 → 24.0 to 25.7 (NORM 27.1,
  Mean-Teacher 27.5), as classification: dAP50/Cls 9.5 → 10.3 to 11.1, FN 8.4 → 9.0 to 9.2, while Loc and Bkg stay
  within 0.5 of source. `V4_full_bgneg` is 0.6 below source in AP50, with the lowest Loc and Bkg (5.2 each) but the
  highest Miss (2.6) and FN (9.9).
  Split into classes in and out of `CLASS_LABELS_PROCTHOR100` in Questions below: the trained classes forget most.
- **Reading AP:** box regression is not trained and proposals come from the frozen teacher, so AP75 and above measure
  the source's localization; AP50 and per-class AP50 (`class_ap.png`, `class_table.csv`) measure what is trained. A
  class with one test box detected at IoU 0.90 to 0.95 scores exactly 90 AP.

## Questions

Per-class numbers from each run's `class_table.csv`, AP50 at the last evaluation against iteration 0.

- **Why do more than 10 classes lose 10+ AP in `V1`?** 
- **Why do some classes still lose with a perfect labeler V4?**
- **Why does every fine-tune forget NYU-47, and why does `V4_full_bgneg` not?** 
- **What explains the gap between V2 and V3 ?** 

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