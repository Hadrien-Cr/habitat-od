```
export PYTHONNOUSERSITE=1
export HABITAT_DATA=<path/to/habitat/data>
conda create -n thor_od python=3.9
conda activate thor_od
conda install habitat-sim==0.3.3 -c conda-forge -c aihabitat
# if headless: conda install habitat-sim==0.3.3 headless -c conda-forge -c aihabitat
```

```
git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
cd third_party/habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ../..
pip install numpy==1.23
ln -s $HABITAT_DATA . # create a symlink from the data folder to the root
```

```
# Download ProcTHOR-HAB scene data
git clone git@hf.co:datasets/hssd/ai2thor-hab $HABITAT_DATA/scene_datasets/ai2thor-hab
# go to https://www.dropbox.com/scl/fi/noizniosf3sjaolq54a6v/objectnav_procthor-hab_v0.2.5.zip?rlkey=g5q3mmpin8fqu66jqrkfqsy6l&dl=0 
# and paste to $HABITAT_DATA/datasets/objectnav/procthor-hab
unzip $HABITAT_DATA/datasets/objectnav/procthor-hab/objectnav_procthor-hab_v0.2.5.zip -d $HABITAT_DATA/datasets/objectnav/procthor-hab
```

```
# Download HSSD scene data
git clone git@hf.co:datasets/hssd/hssd-hab $HABITAT_DATA/scene_datasets/hssd-hab
# go to https://www.dropbox.com/scl/fi/n5m00eoydfedi0de1nh34/objectnav_hssd-hab_v0.2.5.zip?rlkey=zvosfrxu99si8xkfmkd623b4s&dl=0
# and paste to $HABITAT_DATA/datasets/objectnav/procthor-hab
unzip $HABITAT_DATA/datasets/objectnav/hssd-hab/objectnav_hssd-hab_v0.2.5.zip -d $HABITAT_DATA/datasets/objectnav/procthor-hab
```

```
# Download MP3D scene data
python download_mp.py --task habitat -o $HABITAT_DATA/data/scene_datasets/mp3d_data
unzip $HABITAT_DATA/scene_datasets/mp3d_habitat.zip  -d $HABITAT_DATA/scene_datasets/
wget http://dl.fbaipublicfiles.com/habitat/mp3d/config_v1/mp3d.scene_dataset_config.json -P $HABITAT_DATA/scene_datasets/mp3d
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip -P $HABITAT_DATA/datasets/pointnav/mp3d/v1
unzip $HABITAT_DATA/datasets/pointnav/mp3d/v1/pointnav_mp3d_v1.zip -d third_party/habitat-lab/datasets/pointnav/mp3d/v1
```