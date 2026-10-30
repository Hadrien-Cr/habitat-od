# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
`habitat-od` generates COCO-format object-detection datasets from HSSD-HAB scenes inside the Habitat simulator, and fine-tunes a detectron2 Mask R-CNN on them. Ground-truth boxes/masks come from the simulator's semantic sensor (object IDs baked into `semantic_id`), not from a hand-labeled dataset.

## Code style (strict — applies to every change)

Do not edit the logic of `fmm_planner.py` or `discrete_planner.py`. Remove `if TYPE_CHECKING:` blocks — always import unconditionally.

**Reuse before writing.**
- Before adding a function, search the repo (`common/`, `habitat_embodied_sl/`, `tests/`) for an existing one that does the same or nearly the same thing. Call it, or generalize it with a parameter, instead of writing a new one.
- Never duplicate logic across files; move shared lines into the most specific common module (e.g. `common/utils/`, `common/env_utils/`). Utils belong in `common/`, rarely in `tests/`.
- Prefer what detectron2, habitat-sim, numpy and torch already provide over hand-written versions.

**Short, single-purpose functions.**
- Each function does one thing. Aim for ≤ 20 lines, 40 as a hard limit. Split anything longer into well-named helpers.
- Keep nesting to 2 levels at most; use early returns, guard clauses and comprehensions instead of deep `if`/`for` blocks.
- Keep signatures to ≤ 4–5 parameters; group related ones into a config/dataclass instead.
- Avoid boolean flag parameters that switch between two behaviors — write two functions instead.
- Don't write a function whose body is 1–2 lines with a single caller — inline it at that call site. This applies even if the same one-liner body recurs at several single-caller sites; each one still gets inlined, not consolidated into a shared one-line helper.

**Readable without comments.**
- Use descriptive names so the code explains itself; no single-letter names outside short comprehensions or math.
- New code MUST NOT add comments, except a genuinely non-obvious WHY (hidden constraint, workaround, subtle invariant) in one short line.
- Docstrings are optional; when used, one line, no `Args:`/`Returns:` blocks.
- Don't restate the code in prose, and don't leave commented-out code, TODOs or print/debug leftovers.

**No dead code.** Remove unused parameters, flags, imports, variables and functions right away, including ones your change makes obsolete — update every call site. Don't add speculative options, fallbacks or abstractions for cases that don't exist yet.

**Match the surrounding code.** Follow the existing naming, structure and idioms of the file you edit. Keep diffs minimal and focused on the task.

**Keep `habitat_embodied_sl/README.md` up to date.** It is the algorithm outline of `habitat_embodied_sl/main.py`'s mapping pipeline: its processing and filtering steps, their thresholds, outputs and files. Any change to a step it describes must update the README in the same change. That covers `habitat_embodied_sl/`, `common/vision/detector.py`, `common/utils/concept_graph.py`, `projection_utils.py`, `image_memory.py`, `similarity_utils.py`, `retrieval.py`, `eval_utils.py` and `common/hybrid_exploration/`. Add, remove or reorder steps to match the code, change any threshold or default that moved, and drop anything the code no longer does.


## Environment setup

Heavy, order-dependent native build (habitat-sim, detectron2, Detic, CenterNet2) — see `INSTALL.MD` for the full sequence. When debugging environment issues rather than re-deriving them, know that:

- Conda env `habitat_od` from `environment.yml` (Python 3.9, PyTorch 1.13.1, CUDA 11.7, `habitat-sim=0.3.2`). habitat-lab and habitat-sim are both pinned to `v0.3.2`: upstream habitat-sim removed the `datatool` C++ utility (`create_gibson_semantic_mesh`) in the tag right after, which `scripts/gen_gibson_semantics.sh` needs to bake 3DSceneGraph annotations onto raw Gibson meshes.
- Required env vars at runtime: `BASE_DIR` (repo root), `PYTHONPATH=.`, `HABITAT_DATA` (Habitat data root containing `scene_datasets/hssd-hab` and `datasets/objectnav/hssd-hab`), `PYTHONNOUSERSITE=1`. See `.vscode/launch.json` for the exact set used for debugging.
- `third_party/habitat-lab` and `third_party/habitat-sim` (both pinned `v0.3.2`; habitat-sim's submodules included -- source of `scripts/gen_gibson_semantics.sh`'s `datatool` build and `tools/npz2ids.py`/`npz2scn.py`), and `third_party/detectron2` are vendored git clones, not pip packages — imported/built directly by path. `third_party/Detic` (put on `sys.path` by `common/vision/detic.py`) backs `common/vision/detector.py`'s Detic inference, `evaluate.py`'s Detic evaluation, and `common/env_utils/vocab_constants.py::generate_mappings`'s CLIP text encoder.
- `default_structured_configs.py` at repo root must be copied over habitat-lab's own copy (`third_party/habitat-lab/habitat-lab/habitat/config/default_structured_configs.py`) to add habitat-od's Hydra structured configs (e.g. `ObjectDetectorGTSensorConfig`). If sensor/action config fields seem to be missing, check whether this copy is stale.
- `third_party/`, `data/`, `datasets/`, `outputs/`, `datadump/` are gitignored — local/generated, not tracked source.
- `tests/` has a small pytest suite (sensor-filter/visibility/occlusion checks against the simulator); no linter or CI beyond that.

## Running things

Dataset collection (needs the simulator, via `habitat_embodied_sl/main.py`) and training (plain detectron2, no simulator) are separate steps, both launched from the repo root with `PYTHONPATH=.`:

```bash
export HABITAT_DATA=$HOME/habitat_data/data
export BASE_DIR=$(pwd)
PYTHONPATH=. python habitat_embodied_sl/main.py --config habitat_embodied_sl/config/collection_procthor.yaml \
    --config-file third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
    --checkpoint third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
    --sam-checkpoint data/sam/sam_vit_b_01ec64.pth
PYTHONPATH=. python pretrain.py --config-file coco_testbench/config/faster_rcnn_R_50_FPN_1x.yaml --num-gpus 2
```

Standalone Detic evaluation against a COCO test split written by `habitat_embodied_sl/main.py` (Detic's own `train_net.py` `do_test`, zero-shot classifier reset to the split's categories):

```bash
PYTHONPATH=. python evaluate.py --test-dir datasets/smoke_sl/test \
    --config-file third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
    MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
```

`pretrain.py` registers `coco_testbench`'s local COCO copy under fixed `coco_testbench_{train2017,val2017}` names and trains against it, to reproduce detectron2's own model-zoo baselines. `--config-file` selects the architecture/checkpoint (a small overlay on `Base-RCNN-FPN.yaml`); `OUTPUT_DIR` is derived from `--config-file`'s own stem, under `<config-file's-parent-dir>/logs/<stem>/`. `train.py` instead fine-tunes `habitat_embodied_sl/meta_arch.py`'s `CustomFT_Detic` on an already-built COCO train/test pair (e.g. one `habitat_embodied_sl/main.py` wrote) via `register_single_dataset`.

## Architecture

**Vocabularies** (`HSSD400`, `NYU40`, `MPCAT40`, `COCO80`, `SCANNET200`, `ProcTHOR-native`) are built in `common/env_utils/vocab_constants.py` and registered into detectron2's `MetadataCatalog` on import. Non-HSSD-HAB target vocabularies are nearest-neighbor mapped from `HSSD400` WordNet-synset labels via CLIP similarity, cached in `common/env_utils/hssd400_cross_vocab_mapping.csv`. `MPCAT40`/`COCO80` do double duty as MP3D's and Gibson-Semantic's own *native* vocabularies (see below), not just HSSD-HAB mapping targets.

**Data generation** (`common/env_utils`): `common/env_utils/env_base.py` registers a custom `ExplorationTask`/`ExplorationEnv`, and also holds `configure_scene`/`build_env`/`build_agent` — shared helpers to point a habitat config at a scene, wire up `ObjectDetectorGTSensor` from `object_params`, and build an `ExplorationEnv` + one of `RandomAgent`/`FrontierAgent`/`HybridFrontierAgent` from it. `common/env_utils/object_annotations.py::setup_semantic_labels(sim, env_name, vocab_name)` builds the `ObjectAnnotation` (classes + per-object `class_id`/dimensions + decode) for whichever of the 4 supported `env_name`s (`HSSD-HAB`, `MP3D`, `Gibson-Semantic`, `ProcTHOR-hab`) is loaded — HSSD-HAB/ProcTHOR-hab bake `node.semantic_id = object_id * 1000 + class_id` onto each placed rigid/articulated object's scene nodes (`_load_hssd_vocab` reads HSSD-HAB's own object-handle → target-vocab CSV mapping; ProcTHOR-hab maps its native per-category id through `object_semantic_id_mapping.json` onto `CLASS_LABELS_PROCTHOR100` (`vocab_constants.py`: `PROCTHOR_NATIVE_TO_CLASS` spaces each native CamelCase name into lowercase words, `GarbageCan` → `garbage can`, state/size variants like `AppleSliced` folded into their base class; restricted to the ~95 categories ProcTHOR's own procedural generator actually places, excluding ones only placed by iTHOR/RoboTHOR/ArchitecTHOR's hand-authored scenes) as `class_id`, restoring per-instance separation it doesn't have out of the box); MP3D/Gibson-Semantic have no writable per-object scene node, so their already-unique native `semantic_id` (from `sim.semantic_scene.objects`) is read as-is instead. Vocab resolution (HSSD-HAB's CSV read, ProcTHOR-hab's `object_semantic_id_mapping.json` parse) is cached per `(env_name, vocab_name)` since `setup_semantic_labels` reruns on every scene change. `common/env_utils/object_detector_sensors.py` registers `ObjectDetectorGTSensor` (`uuid = "bbsgt"`) as a thin wrapper around this: it calls `setup_semantic_labels` whenever the scene changes, then `decompose_frame` reads the simulator's rendered semantic buffer, decodes each pixel value via `ObjectAnnotation.semantic_id_to_classid_obj_id`, filters by area/occlusion/`filter_classes` (an allow-list of kept classes; `None`/empty keeps everything), and returns a detectron2 `Instances` — the ground truth used everywhere downstream, including `habitat_embodied_sl/main.py`'s own mapping pipeline (see `habitat_embodied_sl/README.md`). `common/env_utils/dataset_registry.py::register_dataset` registers an already-built COCO dataset directory (`<dir>/<split>.json` + `<dir>/<split>/` images) into detectron2's catalogs, for `train.py` (via `register_single_dataset`) to score against.

**Training** (`pretrain.py`): a thin wrapper around detectron2's own `train_net.py` shape (`setup`/`Trainer`/`main`) — plain `DefaultTrainer` + `COCOEvaluator`, no custom model, so whichever architecture `--config-file` names runs unmodified. `train.py` instead sets `MODEL.ROI_HEADS.NUM_CLASSES` at runtime from the collection config's kept vocabulary, and fine-tunes `CustomFT_Detic` (`habitat_embodied_sl/meta_arch.py`) on an `ObjectViewsDataset` train variant. Both share dataset registration, the evaluator-wired `Trainer`, periodic TensorBoard image logging, and final-checkpoint reporting via `common/utils/detectron_trainer.py`.
