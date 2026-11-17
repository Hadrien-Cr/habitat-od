#!/usr/bin/env bash
# Runs, one after another on GPU, the source and unsupervised baselines (adapt.py, mean_teacher.py) and embodied SL on
# every train variant (main.py), each scored on procthor8's test and train sets and on NYU Depth V2 (STATIC_TEST_DIRS),
# each run's earlier logs deleted first. Launch from anywhere once procthor8 is collected and merged:
#   GPU=0 nohup bash habitat_embodied_sl/run.sh > outputs/run.log 2>&1 &
set -u
cd "$(dirname "$0")/.."

source "$(dirname "${CONDA_EXE:-$HOME/miniconda3/bin/conda}")/../etc/profile.d/conda.sh"
conda activate habitat_od
export BASE_DIR="$(pwd)" PYTHONPATH=. PYTHONNOUSERSITE=1 HABITAT_DATA="${HABITAT_DATA:-$HOME/habitat_data/data}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

STATIC_TEST_DIRS=(datasets/nyu_depth_v2/test)
CONFIG_DIR=habitat_embodied_sl/config
LOG_DIR=outputs/procthor8_runs
mkdir -p "$LOG_DIR"

run() {
  local name=$1
  shift
  rm -rf "habitat_embodied_sl/logs/$name"
  echo "$(date '+%F %T') $name started"
  if python "$@" --static-test-dirs "${STATIC_TEST_DIRS[@]}" > "$LOG_DIR/$name.log" 2>&1; then
    echo "$(date '+%F %T') $name done"
  else
    echo "$(date '+%F %T') $name FAILED, see $LOG_DIR/$name.log"
  fi
}

run r50_procthor8_source habitat_embodied_sl/adapt.py --method source --run-name r50_procthor8_source
run r50_procthor8_unsup_norm habitat_embodied_sl/adapt.py --method norm --run-name r50_procthor8_unsup_norm
run r50_procthor8_unsup_mean_teacher habitat_embodied_sl/mean_teacher.py --config-file "$CONFIG_DIR/r50_procthor8_unsup_mean_teacher.yaml"
for variant in pseudo gt gt++; do
  run "r50_procthor8_embodied_sl_$variant" habitat_embodied_sl/main.py --config-file "$CONFIG_DIR/r50_procthor8_embodied_sl_$variant.yaml"
done
echo "$(date '+%F %T') all runs finished"
