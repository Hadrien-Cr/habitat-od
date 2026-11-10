#!/usr/bin/env bash
# Collects habitat_embodied_sl/config/collection_procthor_medium.yaml's 8 collection scenes, 4 at a time: one worker per
# GPU runs its two scenes in turn, each a habitat_embodied_sl/collect.py call into datasets/procthor_medium/. Deletes
# that folder and the logs of an earlier run first. Launch from anywhere:
#   nohup bash scripts/collect_procthor_medium.sh > outputs/collect_procthor_medium.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

source "$(dirname "${CONDA_EXE:-$HOME/miniconda3/bin/conda}")/../etc/profile.d/conda.sh"
conda activate habitat_od
export HABITAT_DATA="${HABITAT_DATA:-$HOME/habitat_data/data}" BASE_DIR="$(pwd)" PYTHONPATH=. PYTHONNOUSERSITE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CONFIG=habitat_embodied_sl/config/collection_procthor_medium.yaml
LOG_DIR=outputs/collect_procthor_medium
SCENES=(
  ProcTHOR-Train-5 ProcTHOR-Train-6 ProcTHOR-Train-9 ProcTHOR-Train-11
  ProcTHOR-Train-14 ProcTHOR-Train-26 ProcTHOR-Train-30 ProcTHOR-Train-32
)
GPUS=(0 1 2 3)

rm -rf datasets/procthor_medium "$LOG_DIR"
mkdir -p "$LOG_DIR"

worker() {
  local gpu=$1
  shift
  for scene in "$@"; do
    echo "$(date '+%F %T') gpu $gpu: $scene started"
    # set before python starts too: habitat-sim may initialize CUDA before collect.py sets it itself
    if CUDA_VISIBLE_DEVICES=$gpu python habitat_embodied_sl/collect.py "$gpu" "$scene" "$CONFIG" \
        --sam-checkpoint data/sam/sam_vit_b_01ec64.pth > "$LOG_DIR/$scene.log" 2>&1; then
      echo "$(date '+%F %T') gpu $gpu: $scene done"
    else
      echo "$(date '+%F %T') gpu $gpu: $scene FAILED, see $LOG_DIR/$scene.log"
    fi
  done
}

for index in "${!GPUS[@]}"; do
  worker "${GPUS[$index]}" "${SCENES[$index]}" "${SCENES[$((index + ${#GPUS[@]}))]}" &
done
wait
echo "$(date '+%F %T') all scenes finished"
