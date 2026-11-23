"""
End-to-end run of habitat_embodied_sl's stages, as Running in habitat_embodied_sl/README.md does them, with procthor8's
collection, dataset and test configs (config/procthor8_collection.yaml, procthor8_build.yaml, procthor8_test.yaml)
renamed test_collect and saved under tests/testdump/test_collect/ as collection.yaml, build.yaml and test.yaml, its raw
records under raw/ there and its datasets under datasets/: collect_train.py records procthor_scene.py's ProcTHOR-Train-0,
build_dataset.py builds it into one dataset per variant in one call, datasets/test_collect_<variant>/, collect_test.py collects
ProcTHOR-Val-7 into datasets/test_collect_test/, merge.py pools each dataset's shards and evaluate_maps.py scores the
train scene's map into datasets/test_collect_V3_cls_box_drop_add/map_eval/ (metrics.json, confusion_matrix.png),
printed. Each stage's saved config is the config.yaml of the raw run and of each dataset, and every script's output is in
tests/testdump/test_collect/run.log.
The record must be complete, the agent never stuck, every dataset non-empty, and rebuilding the train scene from its
record into one variant alone must write the same shard.

Needs ProcTHOR-hab under HABITAT_DATA/scene_datasets/ai2thor-hab, a GPU, Detic's Swin-B checkpoint under
third_party/Detic/models and SAM's ViT-B checkpoint under $HABITAT_DATA/sam. Runs on the GPU CUDA_VISIBLE_DEVICES names.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import pytest
import torch
from omegaconf import OmegaConf

from common.utils.dataset_writers import FULL_FRAMES, OBJECT_DS_DIR, VARIANTS, scene_shard_path, split_json_path
from common.utils.raw_scene import is_recorded
from procthor_scene import DATASET_CONFIG, SCENE_ID

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTDUMP_DIR = Path(__file__).parent / "testdump" / "test_collect"
_RUN_NAME = "test_collect"
_RAW_RUN = _TESTDUMP_DIR / "raw" / _RUN_NAME
_DATASET_ROOT = _TESTDUMP_DIR / "datasets"
_TEST_SCENE = "ProcTHOR-Val-7"
_DETIC_CHECKPOINT = _REPO_ROOT / "third_party" / "Detic" / "models" / "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
_SAM_CHECKPOINT = Path(os.environ.get("HABITAT_DATA", "")) / "sam" / "sam_vit_b_01ec64.pth"

pytestmark = [
    pytest.mark.skipif(not os.path.exists(DATASET_CONFIG), reason="requires HABITAT_DATA pointing at a real ai2thor-hab dataset (see INSTALL.MD)"),
    pytest.mark.skipif(not _DETIC_CHECKPOINT.exists() or not torch.cuda.is_available(), reason="requires a GPU and Detic's Swin-B checkpoint"),
    pytest.mark.skipif(not _SAM_CHECKPOINT.exists(), reason="requires SAM's ViT-B checkpoint"),
]


def _run(script: str, *args: str) -> str:
    """The output of habitat_embodied_sl/<script> run with args from the repo root, streamed into run.log as it runs; it must succeed."""
    environment = {**os.environ, "PYTHONPATH": str(_REPO_ROOT), "BASE_DIR": str(_REPO_ROOT)}
    log_path = _TESTDUMP_DIR / "run.log"
    with open(log_path, "a") as log:
        log.write(f"=== {script} {' '.join(args)}\n")
        start = log.tell()
        log.flush()
        result = subprocess.run([sys.executable, f"habitat_embodied_sl/{script}", *args], cwd=_REPO_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    with open(log_path) as log:
        log.seek(start)
        output = log.read()
    assert result.returncode == 0, f"{script} {' '.join(args)} failed:\n{output[-3000:]}"
    return output


def _configs() -> Tuple[str, str, str]:
    """Paths of procthor8's collection, dataset and test configs renamed _RUN_NAME, their raw records and datasets under
    the testdump, saved there."""
    config_dir = _REPO_ROOT / "habitat_embodied_sl" / "config"
    configs = {name: OmegaConf.load(config_dir / f"procthor8_{name}.yaml") for name in ("collection", "build", "test")}
    configs["collection"].run_name, configs["collection"].raw_root = _RUN_NAME, str(_RAW_RUN.parent)
    configs["build"].run_name, configs["build"].raw_run, configs["build"].dataset_root = f"{_RUN_NAME}_${{variant}}", str(_RAW_RUN), str(_DATASET_ROOT)
    configs["test"].run_name, configs["test"].dataset_root = f"{_RUN_NAME}_test", str(_DATASET_ROOT)
    for name, cfg in configs.items():
        OmegaConf.save(cfg, _TESTDUMP_DIR / f"{name}.yaml")
    return tuple(str(_TESTDUMP_DIR / f"{name}.yaml") for name in configs)


def _dataset_dir(variant: str) -> Path:
    return _DATASET_ROOT / f"{_RUN_NAME}_{variant}"


def _dataset_sizes() -> dict:
    """{dataset: its number of objects, or of frames} of every merged dataset."""
    sizes = {variant: len(json.loads(split_json_path(_dataset_dir(variant) / OBJECT_DS_DIR).read_text())["retrieved_objects"]) for variant in VARIANTS}
    frames = {name: len(json.loads(split_json_path(_dataset_dir(name) / split).read_text())["images"]) for name, split in ((FULL_FRAMES, "train"), ("test", "test"))}
    return {**sizes, **frames}


def test_collect_and_build():
    shutil.rmtree(_TESTDUMP_DIR, ignore_errors=True)
    _TESTDUMP_DIR.mkdir(parents=True)
    collection, build, test = _configs()
    assert "got stuck" not in _run("collect_train.py", SCENE_ID, collection)
    assert is_recorded(_RAW_RUN / SCENE_ID)
    _run("collect_test.py", _TEST_SCENE, test)
    _run("build_dataset.py", SCENE_ID, build, f"variant=[{','.join((*VARIANTS, FULL_FRAMES))}]")
    shard = scene_shard_path(_dataset_dir(VARIANTS[-1]) / OBJECT_DS_DIR, SCENE_ID)
    built = shard.read_text()
    _run("build_dataset.py", SCENE_ID, build, f"variant={VARIANTS[-1]}")
    assert shard.read_text() == built, "rebuilding the train scene from its raw record, its variant alone, changed its shard"
    for name in (*VARIANTS, FULL_FRAMES, "test"):
        _run("merge.py", str(_dataset_dir(name)))
    _run("evaluate_maps.py", str(_dataset_dir(VARIANTS[-1])))
    sizes = _dataset_sizes()
    print("datasets:", sizes)
    assert all(sizes.values()), f"empty dataset: {sizes}"
    print("map metrics:", json.dumps(json.loads((_dataset_dir(VARIANTS[-1]) / "map_eval" / "metrics.json").read_text()), indent=1))


if __name__ == "__main__":
    test_collect_and_build()
