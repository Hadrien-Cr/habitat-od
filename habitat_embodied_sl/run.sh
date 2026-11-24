#!/usr/bin/env bash
# Runs, for every class set of CLASS_SETS (default procthor30 procthor100), the source and unsupervised baselines
# (adapt.py, mean_teacher.py) and embodied SL on every variant dataset, V4_full also with background negatives (main.py),
# each on config/r50_<set>_embodied_sl.yaml, its train split named here, and scored on that config's DATASETS.TEST (the
# set's test and val sets, train frames, NYU Depth V2), each run's earlier logs deleted first: one worker per GPU of GPUS
# pulling runs off a shared queue, longest first. A RUN_STORE folder, when set, holds the runs' logs and checkpoints
# instead, each linked back as logs/<run>. No simulator, so no HABITAT_DATA. Launch from the repo root once the sets'
# datasets are built and merged (scripts/procthor_datasets.sh) and NYU converted (static_datasets/nyu_depth_v2.py):
#   mkdir -p outputs && GPUS="1 2 3" nohup bash habitat_embodied_sl/run.sh > outputs/run.log 2>&1 &
set -uf  # no globbing: a run's DATASETS.TRAIN tuple must reach its script as written
cd "$(dirname "$0")/.."

source "$(dirname "${CONDA_EXE:-$HOME/miniconda3/bin/conda}")/../etc/profile.d/conda.sh"
conda activate habitat_od
export BASE_DIR="$(pwd)" PYTHONPATH=. PYTHONNOUSERSITE=1

read -ra GPU_LIST <<< "${GPUS:-0}"
read -ra SETS <<< "${CLASS_SETS:-procthor30 procthor100}"
RUN_STORE=${RUN_STORE:-}
CONFIG_DIR=habitat_embodied_sl/config
LOG_DIR=outputs/runs
CLAIMS="$LOG_DIR/claims"
mkdir -p "$LOG_DIR"

RUNS=()
for set in "${SETS[@]}"; do
  config="--config-file $CONFIG_DIR/r50_${set}_embodied_sl.yaml"
  for variant in V1_pseudo V2_drop_add; do
    RUNS+=("r50_${set}_embodied_sl_$variant habitat_embodied_sl/main.py $config --run-name r50_${set}_embodied_sl_$variant DATASETS.TRAIN (\"${set}_${variant}_object_ds\",)")
  done
  RUNS+=(
    "r50_${set}_embodied_sl_V4_full habitat_embodied_sl/main.py $config --run-name r50_${set}_embodied_sl_V4_full DATASETS.TRAIN (\"${set}_V4_full_train\",)"
    "r50_${set}_embodied_sl_V4_full_bgneg habitat_embodied_sl/main.py $config --run-name r50_${set}_embodied_sl_V4_full_bgneg DATASETS.TRAIN (\"${set}_V4_full_train\",) MODEL.ROI_HEADS.NAME DeticCascadeROIHeads"
    "r50_${set}_unsup_mean_teacher habitat_embodied_sl/mean_teacher.py $config --run-name r50_${set}_unsup_mean_teacher DATASETS.TRAIN (\"${set}_V3_cls_box_drop_add_object_ds\",)"
    "r50_${set}_embodied_sl_V3_cls_box_drop_add habitat_embodied_sl/main.py $config --run-name r50_${set}_embodied_sl_V3_cls_box_drop_add DATASETS.TRAIN (\"${set}_V3_cls_box_drop_add_object_ds\",)"
  )
done
for set in "${SETS[@]}"; do
  RUNS+=(
    "r50_${set}_source habitat_embodied_sl/adapt.py --config-file $CONFIG_DIR/r50_${set}_embodied_sl.yaml --method source --run-name r50_${set}_source"
    "r50_${set}_unsup_norm habitat_embodied_sl/adapt.py --config-file $CONFIG_DIR/r50_${set}_embodied_sl.yaml --method norm --run-name r50_${set}_unsup_norm"
  )
done

run() {
  local gpu=$1 name=$2
  shift 2
  rm -rf "habitat_embodied_sl/logs/$name"
  if [ -n "$RUN_STORE" ]; then
    rm -rf "$RUN_STORE/$name" && mkdir -p "$RUN_STORE/$name" && ln -s "$RUN_STORE/$name" "habitat_embodied_sl/logs/$name"
  fi
  echo "$(date '+%F %T') GPU $gpu $name started"
  if CUDA_VISIBLE_DEVICES=$gpu python "$@" > "$LOG_DIR/$name.log" 2>&1; then
    echo "$(date '+%F %T') GPU $gpu $name done"
  else
    echo "$(date '+%F %T') GPU $gpu $name FAILED, see $LOG_DIR/$name.log"
  fi
}

worker() {
  local gpu=$1 job name
  for job in "${RUNS[@]}"; do
    read -r name _ <<< "$job"
    mkdir "$CLAIMS/$name" 2>/dev/null || continue
    run "$gpu" $job
  done
}

rm -rf "$CLAIMS" && mkdir -p "$CLAIMS"
for gpu in "${GPU_LIST[@]}"; do worker "$gpu" & done
wait
echo "$(date '+%F %T') all runs finished"
