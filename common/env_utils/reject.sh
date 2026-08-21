#!/usr/bin/env bash
# Interactive accept/reject review for hssd400_cross_vocab_mapping.csv's
# borderline CLIP matches. Run from the repo root:
#   common/env_utils/reject.sh <floatmin> <floatmax>
PYTHONPATH=. BASE_DIR=$(pwd) python common/env_utils/reject.py "$@"
