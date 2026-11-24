#!/usr/bin/env bash
# Collects NUM_SCENES (default 20) scenes per split once, every class kept, and builds every class set of CLASS_SETS
# (default procthor30 procthor100, each <set>_build.yaml's and <set>_eval_build.yaml's vocab_name) from them: the train
# scenes ProcTHOR-Train-0.. (collect_train.py, raw/procthor_train20) into every variant's dataset,
# datasets/<set>_<variant>/ (build_dataset.py), and the evaluation scenes (collect_test.py, raw/procthor_val40)
# ProcTHOR-Val-0.. into the test split and ProcTHOR-Val-20.. into the val split of datasets/<set>_eval/ (build_eval.py),
# then merges them (merge.py) and scores each set's train maps (evaluate_maps.py). Each phase runs one worker per GPU of
# GPUS pulling jobs off a shared queue, into outputs/procthor/<job>.log; already recorded scenes are skipped, so
# NUM_SCENES=1 is a smoke run whose scenes the full run reuses. Launch from the repo root, HABITAT_DATA exported:
#   mkdir -p outputs && GPUS="1 2 3" nohup bash scripts/procthor_datasets.sh 20 > outputs/procthor.log 2>&1 &
set -uf  # no globbing: a job's variant=[...] list must reach build_dataset.py as written
cd "$(dirname "$0")/.."

source "$(dirname "${CONDA_EXE:-$HOME/miniconda3/bin/conda}")/../etc/profile.d/conda.sh"
conda activate habitat_od
export BASE_DIR="$(pwd)" PYTHONPATH=. PYTHONNOUSERSITE=1

NUM_SCENES=${1:-20}
read -ra GPU_LIST <<< "${GPUS:-1 2 3}"
read -ra SETS <<< "${CLASS_SETS:-procthor30 procthor100}"
CONFIG_DIR=habitat_embodied_sl/config
LOG_DIR=outputs/procthor
CLAIMS="$LOG_DIR/claims"
VARIANTS=(V1_pseudo V2_drop_add V3_cls_box_drop_add V4_full)
VAL_OFFSET=20
mkdir -p "$LOG_DIR"

run() {
  local gpu=$1 name=$2
  shift 2
  echo "$(date '+%F %T') GPU $gpu $name started"
  if CUDA_VISIBLE_DEVICES=$gpu python "$@" > "$LOG_DIR/$name.log" 2>&1; then
    echo "$(date '+%F %T') GPU $gpu $name done"
  else
    echo "$(date '+%F %T') GPU $gpu $name FAILED, see $LOG_DIR/$name.log"
  fi
}

worker() {
  local gpu=$1 job name
  for job in "${JOBS[@]}"; do
    read -r name _ <<< "$job"
    mkdir "$CLAIMS/$name" 2>/dev/null || continue
    run "$gpu" $job
  done
}

run_phase() {
  rm -rf "$CLAIMS" && mkdir -p "$CLAIMS"
  for gpu in "${GPU_LIST[@]}"; do worker "$gpu" & done
  wait
}

TRAIN_SCENES=() TEST_SCENES=() VAL_SCENES=()
for ((index = 0; index < NUM_SCENES; index++)); do
  TRAIN_SCENES+=("ProcTHOR-Train-$index")
  TEST_SCENES+=("ProcTHOR-Val-$index")
  VAL_SCENES+=("ProcTHOR-Val-$((VAL_OFFSET + index))")
done

JOBS=()
for scene in "${TRAIN_SCENES[@]}"; do
  JOBS+=("collect_$scene habitat_embodied_sl/collect_train.py $scene $CONFIG_DIR/procthor_train20_collection.yaml")
done
for scene in "${TEST_SCENES[@]}" "${VAL_SCENES[@]}"; do
  JOBS+=("collect_$scene habitat_embodied_sl/collect_test.py $scene $CONFIG_DIR/procthor_val40_collection.yaml")
done
run_phase

JOBS=()
for set in "${SETS[@]}"; do
  for scene in "${TRAIN_SCENES[@]}"; do
    JOBS+=("build_${set}_$scene habitat_embodied_sl/build_dataset.py $scene $CONFIG_DIR/${set}_build.yaml variant=[$(IFS=,; echo "${VARIANTS[*]}")]")
  done
  for scene in "${TEST_SCENES[@]}"; do
    JOBS+=("build_${set}_$scene habitat_embodied_sl/build_eval.py $scene $CONFIG_DIR/${set}_eval_build.yaml test")
  done
  for scene in "${VAL_SCENES[@]}"; do
    JOBS+=("build_${set}_$scene habitat_embodied_sl/build_eval.py $scene $CONFIG_DIR/${set}_eval_build.yaml val")
  done
done
run_phase

JOBS=()
for set in "${SETS[@]}"; do
  for dataset in "${VARIANTS[@]/#/${set}_}" "${set}_eval"; do
    JOBS+=("merge_$dataset habitat_embodied_sl/merge.py datasets/$dataset")
  done
done
run_phase

JOBS=()
for set in "${SETS[@]}"; do
  JOBS+=("evaluate_maps_$set habitat_embodied_sl/evaluate_maps.py datasets/${set}_V3_cls_box_drop_add")
done
run_phase
echo "$(date '+%F %T') ${SETS[*]} finished"
