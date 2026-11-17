# Installation

Every block runs from the repository root, and none changes directory: `pip`/`git` take paths (`git -C`, `pip install -e <dir>`).

```bash
# Environment. The configs and commands read data/... from the repo root, so the link is named data whatever HABITAT_DATA's name.
export PYTHONNOUSERSITE=1 BASE_DIR=$(pwd)
export HABITAT_DATA=<path/to/habitat/data>   # e.g. ~/habitat_data/data
ln -s $HABITAT_DATA data
conda env create -n habitat_od -f environment.yml
conda activate habitat_od
export CUDA_HOME=$CONDA_PREFIX
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
```

```bash
# habitat-lab v0.3.2, its structured configs replaced by this repo's (adds ObjectDetectorGTSensorConfig)
git clone --branch v0.3.2 https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
pip install -e third_party/habitat-lab/habitat-lab
pip install -e third_party/habitat-lab/habitat-baselines
cp default_structured_configs.py third_party/habitat-lab/habitat-lab/habitat/config/default_structured_configs.py
```

```bash
# detectron2, pinned, built in place: its build needs setuptools<70, so install it before Detic, which needs setuptools>80
git clone https://github.com/facebookresearch/detectron2 third_party/detectron2
git -C third_party/detectron2 checkout 02b5c4e295e990042a714712c21dc79b731e8833
pip install "setuptools<70"
pip install --no-build-isolation -e third_party/detectron2 -v
```

```bash
# Detic, pinned, with its CenterNet2 submodule (common/vision/detic.py puts both on sys.path)
git clone https://github.com/facebookresearch/Detic.git third_party/Detic
git -C third_party/Detic checkout 436cda2a2347df60a7c66daca0e8c59f93dc5e79
pip install -U "setuptools>80"
pip install -r third_party/Detic/requirements.txt
git -C third_party/Detic submodule update --init
```

```bash
# Checkpoints. Detic Swin-B: collect.py's detector. Detic R50: the source model of every config/r50_*.yaml
# (main.py, mean_teacher.py, adapt.py). SAM ViT-B: masks prompted with Detic's boxes during collection.
mkdir -p third_party/Detic/models
wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -P third_party/Detic/models/
wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth -P third_party/Detic/models/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P $HABITAT_DATA/sam/
```

```bash
# Data for reproducing Detic's BoxSup-C2_L_CLIP_SwinB_896b32_4x training (pretrain.py):
# LVIS v1 annotations, COCO 2017 images and the converted Swin-B ImageNet-22k backbone, under third_party/Detic/{datasets,models}.
# --coco-images symlinks an existing COCO copy's train2017/ and val2017/; omit it to download them (~19GB).
common/vision/setup_detic_data.sh --coco-images <path/to/coco/images>
```

```bash
# Test data: habitat-sim's test scenes, and its test_assets (shipped in its repo, not downloadable).
# Same habitat-sim clone as scripts/gen_gibson_semantics.sh's Datatool build.
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path $HABITAT_DATA
git clone --branch v0.3.2 --recurse-submodules --shallow-submodules https://github.com/facebookresearch/habitat-sim.git third_party/habitat-sim
ln -s $(pwd)/third_party/habitat-sim/data/test_assets $HABITAT_DATA/test_assets
PYTHONPATH=. python -m pytest tests
```

```bash
# HSSD scene data & ObjectNav dataset
git clone git@hf.co:datasets/hssd/hssd-hab $HABITAT_DATA/scene_datasets/hssd-hab
# download https://www.dropbox.com/scl/fi/n5m00eoydfedi0de1nh34/objectnav_hssd-hab_v0.2.5.zip?rlkey=zvosfrxu99si8xkfmkd623b4s&dl=0
# into $HABITAT_DATA/datasets/objectnav/hssd-hab, then:
unzip $HABITAT_DATA/datasets/objectnav/hssd-hab/objectnav_hssd-hab_v0.2.5.zip -d $HABITAT_DATA/datasets/objectnav/hssd-hab
```

```bash
# ProcTHOR scene data (ProcTHOR-hab): the code reads $HABITAT_DATA/scene_datasets/ai2thor-hab/ai2thor-hab/, the repo's own subfolder
git clone git@hf.co:datasets/hssd/ai2thor-hab $HABITAT_DATA/scene_datasets/ai2thor-hab
# Optional, not read by this repo: ProcTHOR ObjectNav episodes. Download
# https://www.dropbox.com/scl/fi/noizniosf3sjaolq54a6v/objectnav_procthor-hab_v0.2.5.zip?rlkey=g5q3mmpin8fqu66jqrkfqsy6l&e=1&dl=0
# into $HABITAT_DATA/datasets/objectnav/procthor-hab, then:
unzip $HABITAT_DATA/datasets/objectnav/procthor-hab/objectnav_procthor-hab_v0.2.5.zip -d $HABITAT_DATA/datasets/objectnav/procthor-hab
```

```
# Gibson scene data (mesh + semantics) -- see scripts/gen_gibson_semantics.sh for everything:
# downloads, the Datatool build, and the generation command
scripts/gen_gibson_semantics.sh --out $HABITAT_DATA/scene_datasets/gibson_semantic
```

```
# Download MP3D scene data
wget --no-check-certificate http://kaldir.vc.cit.tum.de/matterport/v1/tasks/mp3d_habitat.zip -O mp3d_habitat.zip
unzip mp3d_habitat.zip -d mp3d_habitat
cp mp3d_habitat/mp3d $HABITAT_DATA/scene_datasets/ -r 
wget --no-check-certificate https://dl.fbaipublicfiles.com/habitat/mp3d/config_v1/mp3d.scene_dataset_config.json -P $HABITAT_DATA/scene_datasets/mp3d/
```

```bash
# Download SUN RGB-D (2D/3D box eval, no simulator): images/depth under $HABITAT_DATA/sunrgbd/SUNRGBD,
# plus the v2 global metadata .mat files (2D+3D box ground truth, per-image Rtilt/K calibration) --
# read directly with scipy.io.loadmat, no MATLAB extraction step needed. See notebooks/sunrgbd_visualization.ipynb.
mkdir -p $HABITAT_DATA/sunrgbd
wget -c https://rgbd.cs.princeton.edu/data/SUNRGBD.zip -P $HABITAT_DATA/sunrgbd
unzip -q $HABITAT_DATA/sunrgbd/SUNRGBD.zip -d $HABITAT_DATA/sunrgbd && rm $HABITAT_DATA/sunrgbd/SUNRGBD.zip
wget -c https://rgbd.cs.princeton.edu/data/SUNRGBDMeta2DBB_v2.mat -P $HABITAT_DATA/sunrgbd
wget -c https://rgbd.cs.princeton.edu/data/SUNRGBDMeta3DBB_v2.mat -P $HABITAT_DATA/sunrgbd
```

```bash
# Download NYU Depth V2 (2D segmentation/instance eval, no simulator): the labeled subset
# (1449 RGB-D pairs with dense class/instance labels) as a single .mat file --
# read directly with scipy.io.loadmat, no MATLAB extraction step needed. See notebooks/nyu_depth_v2_visualization.ipynb.
mkdir -p $HABITAT_DATA/nyu_depth_v2
wget -c https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat -P $HABITAT_DATA/nyu_depth_v2
```

```bash
# Download OpenLex3D (open-vocab eval, no simulator) for one Replica scene, office2: tiered ground truth labels
# (all 23 scenes, 20 MB), Replica's own office_2 (habitat/mesh_semantic.ply's per-face object_id is what the labels
# index) and NICE-SLAM's office2 RGB-D sequence + poses (the trajectory OpenLex3D's predictions use).
# Replica ships as one split .tar.gz (~34 GB): it's streamed through and only office_2 (~1.5 GB) is kept.
# NICE-SLAM's Replica.zip (12.4 GB) holds all 8 scenes; only office2 is unzipped. See notebooks/openlex3d_visualization.ipynb.
mkdir -p $HABITAT_DATA/openlex3d $HABITAT_DATA/replica/nice_slam
wget -c http://aisdatasets.cs.uni-freiburg.de/openlex3d/gt_openlex3d.zip -P $HABITAT_DATA/openlex3d
unzip -q $HABITAT_DATA/openlex3d/gt_openlex3d.zip -d $HABITAT_DATA/openlex3d && rm $HABITAT_DATA/openlex3d/gt_openlex3d.zip
for p in {a..q}; do curl -sfL https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.parta$p; done \
    | unpigz | tar -xC $HABITAT_DATA/replica --wildcards 'office_2/*'
wget -c https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip -P $HABITAT_DATA/replica/nice_slam
unzip -q $HABITAT_DATA/replica/nice_slam/Replica.zip 'Replica/office2/*' 'Replica/cam_params.json' -d $HABITAT_DATA/replica/nice_slam \
    && rm $HABITAT_DATA/replica/nice_slam/Replica.zip
```

```bash
# Embodied SL datasets and runs (habitat_embodied_sl/README.md). procthor8: collect every scene, one collect.py call each
# (Running in that README), then pool the scenes' shards into each split's json -- or unzip a copy of datasets/procthor_medium,
# already merged, into datasets/. NYU Depth V2's test set is converted from the .mat above.
PYTHONPATH=. python habitat_embodied_sl/merge.py habitat_embodied_sl/config/procthor8_collection.yaml
PYTHONPATH=. python common/env_utils/nyu_depth_v2.py datasets/nyu_depth_v2
# Source, NORM, Mean-Teacher and embodied SL on each train variant, one after another (logs in outputs/procthor8_runs/)
mkdir -p outputs && GPU=0 nohup bash habitat_embodied_sl/run.sh > outputs/run.log 2>&1 &
```
