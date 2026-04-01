```
conda create -n thor_od python=3.9
conda activate thor_od
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
```

```
git clone  https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
cd third_party/habitat-lab
pip install -e third_party/habitat-lab
pip install -e third_party/habitat-baselines
```

```
# Download ProcTHOR-HAB data
cd data
git clone git@hf.co:datasets/datasets/hssd/ai2thor-hab data/scene_datasets/ai2thor-hab
```

```
# Download HSSD data
cd data
git clone git@hf.co:datasets/hssd/hssd-hab data/scene_datasets/hssd-hab
```
