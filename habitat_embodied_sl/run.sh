#!/usr/bin/env bash
# Runs the source and unsupervised baselines (adapt.py, mean_teacher.py) and embodied SL on every procthor8 variant
# dataset, V4_full also with background negatives (main.py), each scored on its config's DATASETS.TEST (procthor8's test
# set and train frames, NYU Depth V2), each run's earlier logs deleted first: one worker per GPU of GPUS pulling runs off
# a shared queue, longest first. No simulator, so no HABITAT_DATA. Launch from the repo root once procthor8's datasets
# are built and merged (habitat_embodied_sl/merge.py) and NYU converted (habitat_embodied_sl/nyu_depth_v2.py):
#   mkdir -p outputs && GPUS="1 2 3" nohup bash habitat_embodied_sl/run.sh > outputs/run.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

source "$(dirname "${CONDA_EXE:-$HOME/miniconda3/bin/conda}")/../etc/profile.d/conda.sh"
conda activate habitat_od
export BASE_DIR="$(pwd)" PYTHONPATH=. PYTHONNOUSERSITE=1

read -ra GPU_LIST <<< "${GPUS:-0}"
CONFIG_DIR=habitat_embodied_sl/config
LOG_DIR=outputs/procthor8_runs
CLAIMS="$LOG_DIR/claims"
mkdir -p "$LOG_DIR"

RUNS=()
for variant in V1_pseudo V2_drop_add V4_full V4_full_bgneg; do
  RUNS+=("r50_procthor8_embodied_sl_$variant habitat_embodied_sl/main.py --config-file $CONFIG_DIR/r50_procthor8_embodied_sl_$variant.yaml")
done
RUNS+=(
  "r50_procthor8_unsup_mean_teacher habitat_embodied_sl/mean_teacher.py --config-file $CONFIG_DIR/r50_procthor8_unsup_mean_teacher.yaml"
  "r50_procthor8_embodied_sl_V3_cls_box_drop_add habitat_embodied_sl/main.py --config-file $CONFIG_DIR/r50_procthor8_embodied_sl_V3_cls_box_drop_add.yaml"
  "r50_procthor8_source habitat_embodied_sl/adapt.py --method source --run-name r50_procthor8_source"
  "r50_procthor8_unsup_norm habitat_embodied_sl/adapt.py --method norm --run-name r50_procthor8_unsup_norm"
)

run() {
  local gpu=$1 name=$2
  shift 2
  rm -rf "habitat_embodied_sl/logs/$name"
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
