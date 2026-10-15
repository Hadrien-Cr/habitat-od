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