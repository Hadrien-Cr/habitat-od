```
export PYTHONNOUSERSITE=1
export HABITAT_DATA=<path/to/habitat/data>
conda env create -n habitat_od -f environment.yml
ln -s $HABITAT_DATA . # create a symlink from the data folder to the root
```

```
git clone --branch v0.2.5 https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
cd third_party/habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```

```
pip install -r requirements.txt
```

```
# Download HSSD scene data & ObjectNav Dataset
git clone git@hf.co:datasets/hssd/hssd-hab $HABITAT_DATA/scene_datasets/hssd-hab
# go to https://www.dropbox.com/scl/fi/n5m00eoydfedi0de1nh34/objectnav_hssd-hab_v0.2.5.zip?rlkey=zvosfrxu99si8xkfmkd623b4s&dl=0
# and paste to $HABITAT_DATA/datasets/objectnav/hssd-hab
unzip $HABITAT_DATA/datasets/objectnav/hssd-hab/objectnav_hssd-hab_v0.2.5.zip -d $HABITAT_DATA/datasets/objectnav/hssd-hab
```