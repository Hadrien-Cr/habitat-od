```
export PYTHONNOUSERSITE=1
export HABITAT_DATA=<path/to/habitat/data>
ln -s $HABITAT_DATA . # create a symlink from the data folder to the root
conda env create -n habitat_od -f environment.yml
export CUDA_HOME=$CONDA_PREFIX
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc 
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
```

```
git clone --branch v0.3.2 https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
cd third_party/habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cp default_structured_configs.py third_party/habitat-lab/habitat-lab/habitat/config/default_structured_configs.py
```
```
git clone https://github.com/facebookresearch/detectron2 third_party/detectron2
cd third_party/detectron2
python -m pip install --no-build-isolation -e . -v # make sure to have setuptools<70
```
```
git clone https://github.com/facebookresearch/Detic.git third_party/Detic
cd third_party/Detic
pip install setuptools -U # make sure to have setuptools>80
pip install -r requirements.txt
git submodule init && git submodule update
```
```
# Download checkpoints for Detic
mkdir third_party/Detic/models
wget --no-check-certificate https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -P third_party/Detic/models/
```
```
# Data for reproducing Detic's BoxSup-C2_L_CLIP_SwinB_896b32_4x training (habitat_embodied_sl/pretrain.py):
# LVIS v1 annotations, COCO 2017 images and the converted Swin-B ImageNet-22k backbone, under third_party/Detic/{datasets,models}.
# --coco-images symlinks an existing COCO copy's train2017/ and val2017/; omit it to download them (~19GB).
common/vision/setup_detic_data.sh --coco-images <path/to/coco/images>
```
```
# Download SAM's ViT-B checkpoint (masks prompted with Detic's boxes)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P $HABITAT_DATA/sam/
```

```
# Test data: habitat-sim's test scenes, and its test_assets (shipped in its repo, not downloadable).
# Same habitat-sim clone as scripts/gen_gibson_semantics.sh's Datatool build.
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path $HABITAT_DATA
git clone --branch v0.3.2 --recurse-submodules --shallow-submodules https://github.com/facebookresearch/habitat-sim.git third_party/habitat-sim
ln -s $(pwd)/third_party/habitat-sim/data/test_assets $HABITAT_DATA/test_assets
PYTHONPATH=. BASE_DIR=$(pwd) python -m pytest tests
```

```
# Download HSSD scene data & ObjectNav Dataset
git clone git@hf.co:datasets/hssd/hssd-hab $HABITAT_DATA/scene_datasets/hssd-hab
# go to https://www.dropbox.com/scl/fi/n5m00eoydfedi0de1nh34/objectnav_hssd-hab_v0.2.5.zip?rlkey=zvosfrxu99si8xkfmkd623b4s&dl=0
# and paste to $HABITAT_DATA/datasets/objectnav/hssd-hab
unzip $HABITAT_DATA/datasets/objectnav/hssd-hab/objectnav_hssd-hab_v0.2.5.zip -d $HABITAT_DATA/datasets/objectnav/hssd-hab
```

```
# Download ProcTHOR scene data & ObjectNav Dataset
git clone git@hf.co:datasets/hssd/ai2thor-hab $HABITAT_DATA/scene_datasets/ai2thor-hab
# go to https://www.dropbox.com/scl/fi/noizniosf3sjaolq54a6v/objectnav_procthor-hab_v0.2.5.zip?rlkey=g5q3mmpin8fqu66jqrkfqsy6l&e=1&dl=0
# and paste to $HABITAT_DATA/datasets/objectnav/ai2thor-hab
unzip $HABITAT_DATA/datasets/objectnav/hssd-hab/objectnav_procthor-hab_v0.2.5.zip -d $HABITAT_DATA/datasets/objectnav/procthor-hab
```

```
# Gibson scene data (mesh + semantics) -- see scripts/gen_gibson_semantics.sh for everything:
# downloads, the Datatool build, and the generation command
scripts/gen_gibson_semantics.sh --out $HABITAT_DATA/scene_datasets/gibson_semantic
```

```
# Download MP3D scene data
wget --no-check-certificate http://kaldir.vc.cit.tum.de/matterport/v1/tasks/mp3d_habitat.zip -o mp3d_habitat.zip
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
# Download OpenLex3D (open-vocab eval, no simulator) for one Replica scene, office0: tiered ground truth labels
# (all 23 scenes, 20 MB), Replica's own office_0 (habitat/mesh_semantic.ply's per-face object_id is what the labels
# index) and NICE-SLAM's office0 RGB-D sequence + poses (the trajectory OpenLex3D's predictions use).
# Replica ships as one split .tar.gz (~34 GB): it's streamed through and only office_0 (~1.5 GB) is kept.
# NICE-SLAM's Replica.zip (12.4 GB) holds all 8 scenes; only office0 is unzipped. See notebooks/openlex3d_visualization.ipynb.
mkdir -p $HABITAT_DATA/openlex3d $HABITAT_DATA/replica/nice_slam
wget -c http://aisdatasets.cs.uni-freiburg.de/openlex3d/gt_openlex3d.zip -P $HABITAT_DATA/openlex3d
unzip -q $HABITAT_DATA/openlex3d/gt_openlex3d.zip -d $HABITAT_DATA/openlex3d && rm $HABITAT_DATA/openlex3d/gt_openlex3d.zip
for p in {a..q}; do curl -sfL https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.parta$p; done \
    | unpigz | tar -xC $HABITAT_DATA/replica --wildcards 'office_0/*'
wget -c https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip -P $HABITAT_DATA/replica/nice_slam
unzip -q $HABITAT_DATA/replica/nice_slam/Replica.zip 'Replica/office0/*' 'Replica/cam_params.json' -d $HABITAT_DATA/replica/nice_slam \
    && rm $HABITAT_DATA/replica/nice_slam/Replica.zip
```