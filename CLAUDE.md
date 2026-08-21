# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
`habitat-od` generates COCO-format object-detection datasets from HSSD-HAB scenes inside the Habitat simulator, and fine-tunes a detectron2 Mask R-CNN on them. Ground-truth boxes/masks come from the simulator's semantic sensor (object IDs baked into `semantic_id`), not from a hand-labeled dataset.

## Environment setup

Heavy, order-dependent native build (habitat-sim, detectron2, Detic, CenterNet2) — see `INSTALL.MD` for the full sequence. When debugging environment issues rather than re-deriving them, know that:

- Conda env `habitat_od` from `environment.yml` (Python 3.9, PyTorch 1.13.1, CUDA 11.7, `habitat-sim=0.3.3`).
- Required env vars at runtime: `BASE_DIR` (repo root), `PYTHONPATH=.`, `HABITAT_DATA` (Habitat data root containing `scene_datasets/hssd-hab` and `datasets/objectnav/hssd-hab`), `PYTHONNOUSERSITE=1`. See `.vscode/launch.json` for the exact set used for debugging.
- `third_party/habitat-lab` (pinned `v0.3.3`) and `third_party/detectron2` are vendored git clones, not pip packages — imported directly by path. `third_party/Detic` is only needed for `common/env_utils/vocab_constants.py::generate_mappings`'s CLIP text encoder (cross-vocab mapping tooling; not part of the active data-gen/train path).
- `default_structured_configs.py` at repo root must be copied over habitat-lab's own copy (`third_party/habitat-lab/habitat-lab/habitat/config/default_structured_configs.py`) to add habitat-od's Hydra structured configs (e.g. `ObjectDetectorGTSensorConfig`). If sensor/action config fields seem to be missing, check whether this copy is stale.
- `third_party/`, `data/`, `datasets/`, `outputs/`, `habitat_embodied_al_data/`, `datadump/` are gitignored — local/generated, not tracked source.
- `tests/` has a small pytest suite (sensor-filter/visibility/occlusion checks against the simulator); no linter or CI beyond that.

## Running things

Data collection (needs the simulator) and training (plain detectron2, no simulator) are separate steps, both launched from the repo root with `PYTHONPATH=.`:

```bash
export HABITAT_DATA=$HOME/habitat_data/data
export BASE_DIR=$(pwd)
PYTHONPATH=. python habitat_embodied_al/collect_dataset.py --config habitat_embodied_al/pretrain/config/ds.yaml
PYTHONPATH=. python pretrain.py --config-file habitat_embodied_al/pretrain/config/mask_rcnn_R_50_FPN.yaml --ds-config habitat_embodied_al/pretrain/config/ds.yaml --num-gpus 2
```

`collect_dataset.py --config` (a `ds.yaml`-shaped file) sets `run_name` (keys `datasets/<run_name>/`), `object_params` (vocab + `filter_out_classes`), and collection params (scenes/steps_per_episode/trainer_name/filter_empty); it writes a COCO-format `train.json`/`val.json` + images.

`pretrain.py --ds-config` registers that dataset under fixed `"train"`/`"val"` names and trains against it; without `--ds-config` it instead trains against `coco_testbench`'s local COCO copy, to reproduce detectron2's own model-zoo baselines. `--config-file` selects the architecture/checkpoint (a small overlay on `Base-RCNN-FPN.yaml`); `OUTPUT_DIR` is derived from `--config-file`'s own stem, under `<config-file's-parent-dir>/logs/<stem>/`.

## Architecture

**Vocabularies** (`HSSD80`, `HSSD500`, `NYU40`, `MPCAT40`, `COCO80`, `SCANNET200`) are built in `common/env_utils/vocab_constants.py` and registered into detectron2's `MetadataCatalog` on import. Non-HSSD vocabularies are nearest-neighbor mapped from `HSSD500` WordNet-synset labels via CLIP similarity, cached in `common/env_utils/hssd500_cross_vocab_mapping.csv`. `common/env_utils/hssd_object_annotations.py` (`ObjectSemanticsHSSD`) turns a vocab name into a per-scene `object_name -> class_name` mapping from `$HABITAT_DATA/scene_datasets/hssd-hab/semantics/objects.csv`.

**Data generation** (`common/env_utils`, `common/baselines`, `habitat_embodied_al`): `common/env_utils/env_base.py` registers a custom `ExplorationTask`/`ExplorationEnv`. `common/env_utils/object_detector_sensors.py` registers `ObjectDetectorGTSensor` (`uuid = "bbsgt"`), which reads the simulator's semantic buffer, decodes `semantic_id - 1000` into an object id, maps it to a class, filters by area/occlusion/`filter_out_classes`, and returns a detectron2 `Instances` — the ground truth used everywhere downstream. `habitat_embodied_al/collection.py::collect_raw` drives this sensor via a habitat-baselines trainer's `.collect()` to dump raw per-step rgb+GT sense files; `habitat_embodied_al/coco_writer.py::build_coco_dataset` converts those into a COCO json (`categories` already filtered to the kept vocab, with a `"vocab"` field tracing the full vocab it was filtered from); `habitat_embodied_al/dataset.py::register_dataset` registers the result into detectron2's catalogs.

**Training** (`pretrain.py`): a thin wrapper around detectron2's own `train_net.py` shape (`setup`/`Trainer`/`main`) — plain `DefaultTrainer` + `COCOEvaluator`, no custom model, so whichever architecture `--config-file` names runs unmodified. `MODEL.ROI_HEADS.NUM_CLASSES` is set at runtime from the registered dataset's `thing_classes`.

## Code style

Avoid comments and docstrings unless they're mandatory (e.g. required by a linter/interface) or explain a genuinely non-obvious WHY (a hidden constraint, a workaround, a subtle invariant). Don't restate what the code already says.
