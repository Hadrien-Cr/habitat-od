# reproduce/

Reproduces the Mask R-CNN pretraining experiment from
[`habitat_embodied_al/third_party/embodied-active-learning-od`](../third_party/embodied-active-learning-od)
(`scripts/d2_pretraining.sh procthor`) on habitat-od's `ProcTHOR-hab` env, using
habitat-od's own data-collection and training mechanism.

## Running it

```bash
export HABITAT_DATA=$HOME/habitat_data/data
export BASE_DIR=$(pwd)
export PYTHONPATH=.
export PYTHONNOUSERSITE=1
```

Each collection step drives the simulator over dozens of scenes and can take
hours, writing tens of GB of raw sense files before being reduced to a COCO
json (auto-deleted afterwards — pass `--keep-raw` to keep them). Steps 1 and 2
are independent (either order, or in parallel); step 3 needs step 1; step 4
needs step 2 and a checkpoint from step 3; step 5 needs step 2 and a
checkpoint from step 3.

### 1. Collect the pretraining set — 80 scenes, train+val

```bash
python habitat_embodied_al/collect_dataset.py \
    --config habitat_embodied_al/reproduce/config/ds_procthor_reproduce.yaml
```

Writes `datasets/procthor_reproduce/{train,val}.json` + images.

### 2. Collect the distinct validation set — 10 held-out scenes, one dataset per scene

```bash
python habitat_embodied_al/collect_validation_dataset.py \
    --config habitat_embodied_al/reproduce/config/ds_procthor_reproduce_valdistinct.yaml
```

Writes `datasets/procthor_reproduce_valdistinct_ProcTHOR-Val-{0..9}/val.json` + images — one
independent dataset per scene rather than pooled, so step 4 can score each one separately.

### 3. Pretrain Mask R-CNN

```bash
python pretrain.py \
    --config-file habitat_embodied_al/reproduce/config/mask_rcnn_R_50_FPN_coco_procthor_reproduce.yaml \
    --ds-config habitat_embodied_al/reproduce/config/ds_procthor_reproduce.yaml
```

Logs to `habitat_embodied_al/reproduce/logs/mask_rcnn_R_50_FPN_coco_procthor_reproduce/`
(checkpoints, `metrics.png`, `results.json`, confusion matrix). `results.json`'s
`val` numbers are against `procthor_reproduce`'s own held-out 16 scenes, not the
distinct validation set from step 2.

### 4. Score the final checkpoint against a distinct validation scene

```bash
python eval.py \
    --config-file habitat_embodied_al/reproduce/config/mask_rcnn_R_50_FPN_coco_procthor_reproduce.yaml \
    --dataset datasets/procthor_reproduce_valdistinct_ProcTHOR-Val-0 \
    MODEL.WEIGHTS habitat_embodied_al/reproduce/logs/mask_rcnn_R_50_FPN_coco_procthor_reproduce/model_final.pth
```

`--dataset` scores exactly the one dataset directory it's pointed at (one call per scene). Writes
predictions + a confusion matrix under
`habitat_embodied_al/reproduce/logs/mask_rcnn_R_50_FPN_coco_procthor_reproduce/eval/procthor_reproduce_valdistinct_ProcTHOR-Val-0/`,
plus a `results.json` in that same run's log dir summarizing whichever `--dataset` was scored
last (each call overwrites it, so save/rename it between scenes if scoring all 10, e.g. with a
shell loop over `ProcTHOR-Val-{0..9}`). `NUM_CLASSES` is inferred automatically from the
registered vocab (49, matching `classes.py`) — no manual override needed.

### 5. Run an active-learning experiment

```bash
python habitat_embodied_al/reproduce/main.py \
    --config-file habitat_embodied_al/reproduce/config/mask_rcnn_R_50_FPN_coco_procthor_reproduce.yaml \
    --config habitat_embodied_al/reproduce/config/al_procthor_reproduce_ProcTHOR-Val-0.yaml
```

First scores `--config`'s `init_checkpoint` as-is (round `"init"`, no fine-tuning) to give
`history.json` a pre-AL baseline, logged to
`habitat_embodied_al/reproduce/logs/al_<run_name>/round_init/`. Then runs `rounds` rounds: each
round explores `scene_name` for `timesteps` steps, scoring every frame with the previous round's
own fine-tuned checkpoint (round 0 explores with `init_checkpoint` itself), keeps a
`samples`-sized random selection's ground truth (the rest are discarded, never written to disk),
rebuilds the growing labeled pool into one COCO dataset, and fine-tunes `init_checkpoint` itself
again from scratch on that whole pool (`--config`'s optional `MODEL` section — e.g.
`SOLVER.MAX_ITER`, `SOLVER.BASE_LR` — overlays the detectron2 cfg first). Retraining always
restarts from `init_checkpoint` rather than chaining round-to-round, so later rounds don't
compound drift/forgetting from earlier ones; only exploration/scoring uses each round's latest
checkpoint. Evaluated every round against `scene_name`'s own distinct validation set from step 2
(`datasets/procthor_reproduce_valdistinct_<scene_name>/` — collect and test scene are
deliberately the same scene here, not held out from each other). Logs to
`habitat_embodied_al/reproduce/logs/al_<run_name>/round_NNN/` (checkpoint, `metrics.json`) plus
a `history.json` (per-round candidate/selected counts, checkpoint path, eval AP) in
`habitat_embodied_al/reproduce/logs/al_<run_name>/`.

Only a random exploration policy and random sample selection are wired in so far (see
`common/agents/`, `common/samplers/`) — this reproduces the loop's mechanics, not yet the
paper's actual `ada`/`fbe` navigation or discrepancy-ranked sampling.
