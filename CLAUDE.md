# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
`habitat-od` generates COCO-format object-detection datasets from HSSD-HAB scenes inside the Habitat simulator, and fine-tunes a detectron2 Mask R-CNN on them. Ground-truth boxes/masks come from the simulator's semantic sensor (object IDs baked into `semantic_id`), not from a hand-labeled dataset.

## Environment setup

Heavy, order-dependent native build (habitat-sim, detectron2, Detic, CenterNet2) — see `INSTALL.MD` for the full sequence. When debugging environment issues rather than re-deriving them, know that:

- Conda env `habitat_od` from `environment.yml` (Python 3.9, PyTorch 1.13.1, CUDA 11.7, `habitat-sim=0.3.2`). habitat-lab and habitat-sim are both pinned to `v0.3.2`: upstream habitat-sim removed the `datatool` C++ utility (`create_gibson_semantic_mesh`) in the tag right after, which `scripts/gen_gibson_semantics.sh` needs to bake 3DSceneGraph annotations onto raw Gibson meshes.
- Required env vars at runtime: `BASE_DIR` (repo root), `PYTHONPATH=.`, `HABITAT_DATA` (Habitat data root containing `scene_datasets/hssd-hab` and `datasets/objectnav/hssd-hab`), `PYTHONNOUSERSITE=1`. See `.vscode/launch.json` for the exact set used for debugging.
- `third_party/habitat-lab` and `third_party/habitat-sim` (both pinned `v0.3.2`; habitat-sim's submodules included -- source of `scripts/gen_gibson_semantics.sh`'s `datatool` build and `tools/npz2ids.py`/`npz2scn.py`), and `third_party/detectron2` are vendored git clones, not pip packages — imported/built directly by path. `third_party/Detic` is only needed for `common/env_utils/vocab_constants.py::generate_mappings`'s CLIP text encoder (cross-vocab mapping tooling; not part of the active data-gen/train path).
- `default_structured_configs.py` at repo root must be copied over habitat-lab's own copy (`third_party/habitat-lab/habitat-lab/habitat/config/default_structured_configs.py`) to add habitat-od's Hydra structured configs (e.g. `ObjectDetectorGTSensorConfig`). If sensor/action config fields seem to be missing, check whether this copy is stale.
- `third_party/`, `data/`, `datasets/`, `outputs/`, `habitat_embodied_al_data/`, `datadump/` are gitignored — local/generated, not tracked source.
- `tests/` has a small pytest suite (sensor-filter/visibility/occlusion checks against the simulator); no linter or CI beyond that.

## Running things

Data collection (needs the simulator) and training (plain detectron2, no simulator) are separate steps, both launched from the repo root with `PYTHONPATH=.`:

```bash
export HABITAT_DATA=$HOME/habitat_data/data
export BASE_DIR=$(pwd)
PYTHONPATH=. python habitat_embodied_al/collect_dataset.py --config habitat_embodied_al/pretrain/config/ds_hssd.yaml
PYTHONPATH=. python pretrain.py --config-file habitat_embodied_al/pretrain/config/mask_rcnn_R_50_FPN.yaml --ds-config habitat_embodied_al/pretrain/config/ds_hssd.yaml --num-gpus 2
```

`collect_dataset.py --config` (a `ds_hssd.yaml`-shaped file) sets `run_name` (keys `datasets/<run_name>/`), `object_params` (`env_name` + `vocab_name` + `filter_out_classes`), and collection params (scenes/steps_per_episode/trainer_name/filter_empty); it writes a COCO-format `train.json`/`val.json` + images.

`pretrain.py --ds-config` registers that dataset under fixed `"train"`/`"val"` names and trains against it; without `--ds-config` it instead trains against `coco_testbench`'s local COCO copy, to reproduce detectron2's own model-zoo baselines. `--config-file` selects the architecture/checkpoint (a small overlay on `Base-RCNN-FPN.yaml`); `OUTPUT_DIR` is derived from `--config-file`'s own stem, under `<config-file's-parent-dir>/logs/<stem>/`.

## Architecture

**Vocabularies** (`HSSD80`, `HSSD500`, `NYU40`, `MPCAT40`, `COCO80`, `SCANNET200`) are built in `common/env_utils/vocab_constants.py` and registered into detectron2's `MetadataCatalog` on import. Non-HSSD-HAB target vocabularies are nearest-neighbor mapped from `HSSD500` WordNet-synset labels via CLIP similarity, cached in `common/env_utils/hssd500_cross_vocab_mapping.csv`. `MPCAT40`/`COCO80` do double duty as MP3D's and Gibson-Semantic's own *native* vocabularies (see below), not just HSSD-HAB mapping targets.

**Data generation** (`common/env_utils`, `common/baselines`, `habitat_embodied_al`): `common/env_utils/env_base.py` registers a custom `ExplorationTask`/`ExplorationEnv`. `common/env_utils/object_annotations.py::setup_semantic_labels(sim, env_name, vocab_name)` builds the `ObjectAnnotation` (classes + per-object `class_id`/dimensions + decode) for whichever of the 4 supported `env_name`s (`HSSD-HAB`, `MP3D`, `Gibson-Semantic`, `ProcTHOR-hab`) is loaded — HSSD-HAB/ProcTHOR-hab bake `node.semantic_id = object_id * 1000 + class_id` onto each placed rigid/articulated object's scene nodes (`_load_hssd_vocab` reads HSSD-HAB's own object-handle → target-vocab CSV mapping; ProcTHOR-hab reuses its native per-category id as `class_id` directly, restoring per-instance separation it doesn't have out of the box); MP3D/Gibson-Semantic have no writable per-object scene node, so their already-unique native `semantic_id` (from `sim.semantic_scene.objects`) is read as-is instead. Vocab resolution (HSSD-HAB's CSV read, ProcTHOR-hab's `object_semantic_id_mapping.json` parse) is cached per `(env_name, vocab_name)` since `setup_semantic_labels` reruns on every scene change. `common/env_utils/object_detector_sensors.py` registers `ObjectDetectorGTSensor` (`uuid = "bbsgt"`) as a thin wrapper around this: it calls `setup_semantic_labels` whenever the scene changes, then `decompose_frame` reads the simulator's rendered semantic buffer, decodes each pixel value via `ObjectAnnotation.semantic_id_to_classid_obj_id`, filters by area/occlusion/`filter_out_classes`, and returns a detectron2 `Instances` — the ground truth used everywhere downstream. `habitat_embodied_al/collection.py::collect_raw` drives this sensor via a habitat-baselines trainer's `.collect()` to dump raw per-step rgb+GT sense files; `habitat_embodied_al/coco_writer.py::build_coco_dataset` converts those into a COCO json (`categories` already filtered to the kept vocab, with a `"vocab"` field tracing the full vocab it was filtered from); `habitat_embodied_al/dataset.py::register_dataset` registers the result into detectron2's catalogs.

**Training** (`pretrain.py`): a thin wrapper around detectron2's own `train_net.py` shape (`setup`/`Trainer`/`main`) — plain `DefaultTrainer` + `COCOEvaluator`, no custom model, so whichever architecture `--config-file` names runs unmodified. `MODEL.ROI_HEADS.NUM_CLASSES` is set at runtime from the registered dataset's `thing_classes`.

## Code style

Avoid comments and docstrings unless they're mandatory (e.g. required by a linter/interface) or explain a genuinely non-obvious WHY (a hidden constraint, a workaround, a subtle invariant). Don't restate what the code already says.
