```
conda create -n thor_od python=3.9
conda activate thor_od
conda install habitat-sim==0.2.5 -c conda-forge -c aihabitat
```

```
git clone --branch v0.2.5 https://github.com/facebookresearch/habitat-lab.git third_party/habitat-lab
cd third_party/habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
```

```
# Download ProcTHOR-HAB data
cd data
git clone git@hf.co:datasets/hssd/ai2thor-hab data/scene_datasets/ai2thor-hab
```

```
# Download HSSD data
cd data
git clone git@hf.co:datasets/hssd/hssd-hab data/scene_datasets/hssd-hab
```
