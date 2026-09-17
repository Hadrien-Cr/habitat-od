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
    --scene-name ProcTHOR-Val-0 --rounds 5 --timesteps 400 --samples 20 \
    --agent random --sampler random \
    --init-checkpoint habitat_embodied_al/reproduce/logs/mask_rcnn_R_50_FPN_coco_procthor_reproduce/model_final.pth \
    SOLVER.IMS_PER_BATCH 32 SOLVER.BASE_LR 0.001 SOLVER.STEPS "(70, 90)" SOLVER.MAX_ITER 100 SOLVER.WARMUP_ITERS 0
```

Every flag has a `--help`; `--run-name` defaults to a name derived from
`--scene-name`/`--agent`/`--sampler`/`--scoring`/`--rounds`/`--samples`/`--timesteps`/`--seed`.
`--env-name`/`--vocab-name`/`--area-thr`/`--filter-classes`/etc. default to the `ProcTHOR-hab`/
`ProcTHOR-native` setup steps 1-4 already collected against (`--filter-classes` defaults to
`habitat_embodied_al/reproduce/classes.py`'s `PROCTHOR_SHORT_CLASSES`) — override them together
if targeting a different env/vocab. Trailing `KEY VALUE` pairs (detectron2's own `opts`
convention, as in `eval.py`) overlay the detectron2 cfg for both training and eval every round —
e.g. `SOLVER.MAX_ITER`, `SOLVER.BASE_LR`.

First scores `--init-checkpoint` as-is (round `"init"`, no fine-tuning) to give `history.json` a
pre-AL baseline, logged to `habitat_embodied_al/reproduce/logs/al_<run_name>/round_init/`. Then
runs `--rounds` rounds: each round explores `--scene-name` for `--timesteps` steps, scoring every
frame with the previous round's own fine-tuned checkpoint (round 0 explores with
`--init-checkpoint` itself), keeps a `--samples`-sized selection's ground truth (the rest are
discarded, never written to disk), rebuilds the growing labeled pool into one COCO dataset, and
fine-tunes `--init-checkpoint` itself again from scratch on that whole pool. Retraining always
restarts from `--init-checkpoint` rather than chaining round-to-round, so later rounds don't
compound drift/forgetting from earlier ones; only exploration/scoring uses each round's latest
checkpoint. Evaluated every round against `--scene-name`'s own distinct validation set from step 2
(`datasets/procthor_reproduce_valdistinct_<scene_name>/` — collect and test scene are
deliberately the same scene here, not held out from each other). Logs to
`habitat_embodied_al/reproduce/logs/al_<run_name>/round_NNN/` (checkpoint, `metrics.json`) plus
a `history.json` (per-round candidate/selected counts, checkpoint path, eval AP) and a
`map_by_round.png` (mAP vs. round, including round `"init"` as round 0) in
`habitat_embodied_al/reproduce/logs/al_<run_name>/`. `--sampler two-stage` also writes a
`topN_mosaic.png` next to each round's selection: the first stage's ranked pool (its own
`--budget-expanding-ratio`-widened top-N, highest score first), red-bordered wherever the second
(diversity) stage actually kept that candidate.

Exploration policy (`common/agents/`, `--agent`) supports:

- `random` — walks to a random navmesh point, replans once it arrives.
- `fbe` — frontier-based exploration: walks to the nearest not-yet-seen navmesh cell
  (a forward vision cone around the agent's live pose marks cells "seen" each step), replanning
  once it arrives or once that cell gets seen first from elsewhere; degrades to `random`'s
  behavior once every reachable cell has been seen (`common/agents/frontier_agent.py`).
- `sweep` — jumps to a random not-yet-visited navmesh cell and does a full rotation
  there, repeating (resets once every reachable cell has been swept once) —
  `common/agents/sweep_agent.py`.

`ada` (the reference's learned/Bayesian-optimization navigation policy) isn't ported — it needs
an online-fit value model and a reward signal, not just a geometric planner, so it's a
different scope of work than `fbe`/`sweep`. Sample selection (`common/samplers/`, `--sampler`)
supports:

- `random` — uniform random pick, no other flags needed.
- `greedy` — ranks the pool by `--scoring` and takes the top `--samples`. `--scoring` is
  one of `count` (raw predicted-instance count), `discrepancy-total-count`
  (mean predicted-instance-count discrepancy, at select time, against every other pool
  candidate one action away from it in the same collection round, each restricted to their
  overlapping field of view — `common/samplers/discrepancy.py`,
  `common/samplers/scoring.py::score_count_discrepancy`), `entropy-mean`/`entropy-total` (per-box softmax
  entropy, averaged or summed over the frame — `common/samplers/scoring.py`), or `oracle`
  (ground-truth difficulty, weighted by `1 - AP` of each GT box's class from the *previous*
  round's eval — needs the distinct validation set from step 2 to produce non-trivial AP).
- `diversity` — k-means clusters the pool together with every already-selected
  candidate from earlier rounds (by prediction similarity, `common/samplers/similarity.py`),
  drops clusters touching an already-selected candidate, and picks `--samples` cluster centroids
  from what's left — diverse from each other and from the growing labeled pool.
- `two-stage` — `--scoring`-ranks the pool down to `--samples * --budget-expanding-ratio`
  (default `2.0`) candidates, then runs `diversity` selection on that narrowed set; also writes
  `topN_mosaic.png` (see step 5 above).

`--scoring`/`--budget-expanding-ratio` are only read for `greedy`/`two-stage` (`--scoring` is
required for both). Run `python habitat_embodied_al/reproduce/main.py --help` for the full flag
list.
