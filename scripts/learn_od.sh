export HABITAT_DATA=$HOME/habitat_data/data
PYTHONPATH=. BASE_DIR=$(pwd) python habitat_learn_od/detic_finetuning.py --config-name train_hssd80_100x30