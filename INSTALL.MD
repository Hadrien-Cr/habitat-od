# Installation

Every block runs from the repository root, and none changes directory: `pip`/`git` take paths (`git -C`, `pip install -e <dir>`).

```bash
# Environment. All data paths (configs, scripts, tests) resolve through $HABITAT_DATA; put the export in ~/.bashrc.
export PYTHONNOUSERSITE=1 BASE_DIR=$(pwd)
export HABITAT_DATA=<path/to/habitat/data>
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
# Checkpoints. Detic Swin-B: collect_train.py's detector. Detic R50: the source model of every config/r50_*.yaml
# (main.py, mean_teacher.py, adapt.py). SAM ViT-B: masks prompted with Detic's boxes during collection.
mkdir -p third_party/Detic/models
wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -P third_party/Detic/models/
wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth -P third_party/Detic/models/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P $HABITAT_DATA/sam/
```

```bash
# Data for reproducing Detic's LVIS-only BoxSup-C2_L_CLIP_{SwinB_896b32,R5021k_640b64}_4x trainings (pretrain.py), and for
# lvis_class_mosaic.py: LVIS v1 train/val annotations (~1.3GB; 100,170 / 19,809 images, 1,203 classes, all taken from COCO 2017,
# so COCO's images are needed too) and the Swin-B ImageNet-22k and ResNet-50 ImageNet-21k-P backbones converted to detectron2's
# format, under third_party/Detic/{datasets,models}.
# --coco-images symlinks an existing COCO copy's train2017/ and val2017/; omit it to download them (~19GB).
# Detic's other LVIS configs (Detic_LI*, Detic_LCOCOI21k*) also train on ImageNet-21K (~1.3TB, image-net.org login), not set up here.
common/vision/setup_detic_data.sh --coco-images <path/to/coco/images>
# Published BoxSup R50 checkpoint, to check a reproduction against its LVIS val mask mAP (31.5, 25.6 on rare classes)
wget https://dl.fbaipublicfiles.com/detic/BoxSup-C2_L_CLIP_R5021k_640b64_4x.pth -P third_party/Detic/models/
```

```bash
# Objects365 v2 val (Detic's cross-dataset eval split objects365_v2_val; Detic needs no train split): annotations (~270MB) and
# 44 image patches (~17.5GB, each streamed into val/images/v{1,2}/patch<N>/ without keeping the archive) from the public mirror
# YOLOv5 also uses. Patches 0-15 are v1, 16-43 v2; Detic's README lists patches up to 49, the mirror has none past 43.
# Detic's fix_o365_names.py then fixes typos in category names into annotations/zhiyuan_objv2_val_fixname.json.
o365=third_party/Detic/datasets/objects365
o365_url="https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val"
mkdir -p $o365/annotations $o365/val/images/v1 $o365/val/images/v2
wget -c $o365_url/zhiyuan_objv2_val.json -P $o365/annotations
for patch in $(seq 0 43); do
    version=$([ $patch -lt 16 ] && echo v1 || echo v2)
    curl -sfL $o365_url/images/$version/patch$patch.tar.gz | tar -xzC $o365/val/images/$version
done
(cd third_party/Detic && python tools/fix_o365_names.py)
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
# Download EgoObjects v1.0 (egocentric indoor-object detection eval, no simulator): 114K frames (79K train, 5.7K val,
# 29.5K test) from 9K+ wearable-camera videos, 368 categories, annotations in LVIS's format (github.com/facebookresearch/EgoObjects).
# Its files are behind a license form: accept it at https://ai.meta.com/datasets/egoobjects-downloads to get each file's
# link (they expire), and paste them below. Images ~40GB.
mkdir -p $HABITAT_DATA/egoobjects/data
wget -c "<EgoObjectsV1_images.zip link>" -O $HABITAT_DATA/egoobjects/EgoObjectsV1_images.zip
unzip -q $HABITAT_DATA/egoobjects/EgoObjectsV1_images.zip -d $HABITAT_DATA/egoobjects && rm $HABITAT_DATA/egoobjects/EgoObjectsV1_images.zip
wget -c "<EgoObjectsV1_unified_train.json link>" -O $HABITAT_DATA/egoobjects/data/EgoObjectsV1_unified_train.json
wget -c "<EgoObjectsV1_unified_eval.json link>" -O $HABITAT_DATA/egoobjects/data/EgoObjectsV1_unified_eval.json
wget -c "<EgoObjectsV1_unified_metadata.json link>" -O $HABITAT_DATA/egoobjects/data/EgoObjectsV1_unified_metadata.json
```

```bash
# Embodied SL datasets and runs (habitat_embodied_sl/README.md). procthor8: record every train scene raw, one collect_train.py
# call each, build each into every variant's dataset, one build_dataset.py call each listing them, and collect every test scene into
# the test set, one collect_test.py call each (Running in that README), then pool each dataset's shards. NYU Depth V2's
# test set is converted from the .mat above.
for dataset in procthor8_V1_pseudo procthor8_V2_drop_add procthor8_V3_cls_box_drop_add procthor8_V4_full procthor8_test; do
  PYTHONPATH=. python habitat_embodied_sl/merge.py datasets/$dataset
done
PYTHONPATH=. python habitat_embodied_sl/static_datasets/nyu_depth_v2.py datasets/nyu_depth_v2
PYTHONPATH=. python habitat_embodied_sl/static_datasets/sunrgbd.py datasets/sunrgbd
# Source, NORM, Mean-Teacher and embodied SL on each train variant, one run per GPU at a time (logs in outputs/procthor8_runs/)
mkdir -p outputs && GPUS="1 2 3" nohup bash habitat_embodied_sl/run.sh > outputs/run.log 2>&1 &
```
