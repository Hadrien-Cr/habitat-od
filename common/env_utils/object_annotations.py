import itertools
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import magnum as mn
import numpy as np
import pandas as pd
import habitat_sim # type: ignore
import habitat.sims.habitat_simulator.sim_utilities as sutils # type: ignore

from common.env_utils.vocab_constants import VOCABULARIES, CLASS_LABELS_HSSD400, HSSD400_TO_VOCAB
from common.env_utils.visibility_utils import compute_obj_dimensions, compute_dimensions_from_obb
from common.utils.grid_utils import HabitatObjOccupancyGrid

HABITAT_DATA = os.environ.get("HABITAT_DATA")
BASE_DIR = os.environ.get("BASE_DIR")
assert BASE_DIR is not None, "BASE_DIR environment variable must be set to the repo root"


def get_obj_from_id(sim: habitat_sim.Simulator, obj_id: int):
    rom = sim.get_rigid_object_manager()
    if rom.get_library_has_id(obj_id):
        return rom.get_object_by_id(obj_id)
    return None

def object_shortname_from_handle(object_handle: str) -> str:
    """Strips the :_xxxx suffix habitat_sim appends to an object handle to distinguish
    multiple instances of the same placed object in the scene."""
    return object_handle.split("_:")[0]

def get_all_objects(sim: habitat_sim.Simulator):
    managers = [
        sim.get_rigid_object_manager(),
        sim.get_articulated_object_manager(),
    ]
    all_objects = []
    for mngr in managers:
        all_objects.extend(mngr.get_objects_by_handle_substring().values())
    return all_objects


def get_objects_info(sim, obj_name_to_class: dict[str, str], fallback_obj_name_to_class: dict[str, str]) -> list[dict]:
    out = []

    for obj_id, obj_handle in sutils.get_all_object_ids(sim).items():
        obj_name = object_shortname_from_handle(obj_handle)

        obj = get_obj_from_id(sim, obj_id)

        aabb = obj.collision_shape_aabb # type: ignore
        min_v = aabb.min
        max_v = aabb.max

        # 8 local box corners
        corners_local = [
            mn.Vector3(c) # type: ignore
            for c in itertools.product(
                [min_v.x, max_v.x],
                [min_v.y, max_v.y],
                [min_v.z, max_v.z],
            )
        ]

        corners_world = [
            obj.rotation.transform_vector(c) + obj.translation # type: ignore
            for c in corners_local
        ]
        center_world = obj.rotation.transform_vector(aabb.center()) + obj.translation # type: ignore

        if obj_name not in obj_name_to_class:
            class_name = "/u:" + fallback_obj_name_to_class.get(obj_name, "undefined")
        else:
            class_name = obj_name_to_class[obj_name]

        out.append({
            "object_id": obj_id,
            "obj_name": obj_name,
            "class_name": class_name,
            "position": obj.translation, # type: ignore
            "rotation": obj.rotation, # type: ignore
            "center": np.array(center_world),
            "corners": [
                (c.x, c.y, c.z)
                for c in corners_world
            ],
        })
    return out


@dataclass
class ObjectAnnotation:
    """Per-scene ground-truth object annotation for one env_name, as built by
    setup_semantic_labels() below. classes/obj_id_to_class_id/dimensions_by_obj_id and
    semantic_id_to_classid_obj_id() are what ObjectDetectorGTSensor.decompose_frame needs to
    turn a rendered semantic buffer into per-object detections, for any of the 4 envs.
    object_info_list/object_occupancy_grid (the top-down occupancy-map debug view used by
    common/baselines/agents.py's do_visualize) are HSSD-HAB-only - None for the other 3 envs,
    which have no equivalent handle-based object classification to build them from."""
    env_name: str
    classes: list[str]
    obj_id_to_class_id: dict[int, int]
    dimensions_by_obj_id: dict[int, np.ndarray]
    object_info_list: Optional[list[dict]] = None
    object_occupancy_grid: Optional[HabitatObjOccupancyGrid] = None

    def get_classes(self) -> list[str]:
        return self.classes

    def semantic_id_to_classid_obj_id(self, semantic_id: int) -> Optional[tuple[int, int]]:
        if self.env_name in ("HSSD-HAB", "ProcTHOR-hab"):
            return semantic_id % 1000, semantic_id // 1000
        elif self.env_name in ("MP3D", "Gibson-Semantic"):
            obj_id = semantic_id
            class_id = self.obj_id_to_class_id.get(obj_id)
            return (class_id, obj_id) if class_id is not None else None
        else:
            raise NotImplementedError(f"Environment {self.env_name} not supported for object annotations")


@lru_cache(maxsize=None)
def _load_hssd_vocab(vocab_name: str) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """HSSD-HAB has no native per-object category scheme, so every object's handle has to be
    cross-vocab mapped by hand: hssd_obj_semantics_condensed.csv maps object name -> HSSD400
    class, HSSD400_TO_VOCAB (vocab_constants.py) then maps that into vocab_name. Static and
    scene-independent, so cached per vocab_name - setup_semantic_labels() below reruns on
    every episode reset/scene change, and re-parsing this CSV on every one would be wasteful.
    Returns (classes, target_vocab_object_annotations, hssd400_object_annotations):
    target_vocab_object_annotations is object_name -> vocab_name class name, excluding objects
    whose HSSD400 class has no mapping into vocab_name ("unknown"); hssd400_object_annotations
    is object_name -> HSSD400 class for every object including "unknown" ones, used only as
    get_objects_info's fallback label for objects target_vocab_object_annotations excludes."""
    if vocab_name not in VOCABULARIES:
        raise ValueError(f"Vocabulary {vocab_name} not recognized. Must be one of {VOCABULARIES.keys()}")
    classes, _, _ = VOCABULARIES[vocab_name]

    df_hssd400 = pd.read_csv(os.path.join(str(BASE_DIR), "common", "env_utils", "hssd_obj_semantics_condensed.csv")).set_index("Object Hash")
    condensed_hssd400_vocab = df_hssd400.iloc[:,2].to_dict()

    target_vocab_object_annotations: dict[str, str] = {}
    hssd400_object_annotations: dict[str, str] = {}
    for obj_name, s in condensed_hssd400_vocab.items():
        hssd400_class = str(s).replace("/", "_")

        if "unknown" in hssd400_class or hssd400_class == "nan":
            hssd400_class = "unknown"

        assert hssd400_class in CLASS_LABELS_HSSD400, (hssd400_class, s)
        hssd400_object_annotations[str(obj_name)] = hssd400_class

        if vocab_name == "HSSD400":
            target_class = hssd400_class
        else:
            target_class = HSSD400_TO_VOCAB[vocab_name][hssd400_class]

        if target_class != "unknown":
            target_vocab_object_annotations[str(obj_name)] = target_class

    return classes, target_vocab_object_annotations, hssd400_object_annotations


@lru_cache(maxsize=None)
def _resolve_native_vocab(env_name: str, vocab_name: str) -> list[str]:
    """Static, scene-independent native class list for MP3D/Gibson-Semantic - cached per
    (env_name, vocab_name) for the same reason as _load_hssd_vocab above."""
    if env_name == "MP3D":
        assert vocab_name == "MPCAT40", f"MP3D's native vocabulary is MPCAT40, got vocab_name={vocab_name!r}"
        return VOCABULARIES["MPCAT40"][0]
    elif env_name == "Gibson-Semantic":
        assert vocab_name == "COCO80", f"Gibson-Semantic's native vocabulary is COCO80, got vocab_name={vocab_name!r}"
        return VOCABULARIES["COCO80"][0]
    else:
        raise NotImplementedError(f"Environment {env_name} not supported for object annotations")


@lru_cache(maxsize=None)
def _load_procthor_classes(mapping_path: str) -> list[str]:
    """Native per-category semantic ids habitat_sim applies to ProcTHOR-hab objects at load
    time (configs/object_semantic_id_mapping.json - see test_procthor.py). Cached per mapping
    path for the same reason as _load_hssd_vocab. Returns a dense class list indexed by that
    native id, so it doubles directly as class_id with no separate remapping; "Undefined" is
    renamed to "unknown" to match ObjectAnnotation's decode/filtering sentinel."""
    name_to_id: dict[str, int] = json.load(open(mapping_path))

    classes = ["unknown"] * (max(name_to_id.values()) + 1)
    for name, idx in name_to_id.items():
        classes[idx] = "unknown" if name == "Undefined" else name
    return classes


def _setup_hssd(sim, vocab_name: str) -> ObjectAnnotation:
    classes, target_vocab_object_annotations, hssd400_object_annotations = _load_hssd_vocab(vocab_name)
    class2int = {c: i for i, c in enumerate(classes)}

    obj_id_to_class_id: dict[int, int] = {}
    dimensions_by_obj_id: dict[int, np.ndarray] = {}

    for obj in get_all_objects(sim):
        obj_name = object_shortname_from_handle(obj.handle)

        if obj_name not in target_vocab_object_annotations:
            for node in obj.visual_scene_nodes:
                node.semantic_id = 0
            continue

        class_name = target_vocab_object_annotations[obj_name]
        class_id = class2int[class_name]

        for node in obj.visual_scene_nodes:
            node.semantic_id = obj.object_id * 1000 + class_id

        obj_id_to_class_id[obj.object_id] = class_id
        dimensions_by_obj_id[obj.object_id] = compute_obj_dimensions(obj)

    object_info_list = get_objects_info(sim, target_vocab_object_annotations, hssd400_object_annotations)
    object_occupancy_grid = HabitatObjOccupancyGrid(sim, meters_per_grid_pixel=0.125, list_object_info=object_info_list)

    return ObjectAnnotation(
        env_name="HSSD-HAB", classes=classes,
        obj_id_to_class_id=obj_id_to_class_id, dimensions_by_obj_id=dimensions_by_obj_id,
        object_info_list=object_info_list, object_occupancy_grid=object_occupancy_grid,
    )


def _setup_procthor(sim) -> ObjectAnnotation:
    # Places furniture as real rigid/articulated objects (like HSSD-HAB), each with its own
    # writable visual_scene_nodes - unlike its native per-category semantic id (which merges
    # same-class instances), we can bake a per-instance id here too.
    ai2_root = os.path.dirname(sim.config.sim_cfg.scene_dataset_config_file)
    mapping_path = os.path.join(ai2_root, "configs", "object_semantic_id_mapping.json")
    classes = _load_procthor_classes(mapping_path)

    obj_id_to_class_id: dict[int, int] = {}
    dimensions_by_obj_id: dict[int, np.ndarray] = {}

    for obj in get_all_objects(sim):
        class_id = obj.semantic_id  # native id, applied by habitat_sim at load time - read before we overwrite it below

        if class_id >= len(classes) or classes[class_id] == "unknown":
            for node in obj.visual_scene_nodes:
                node.semantic_id = 0
            continue

        assert class_id < 1000, f"ProcTHOR-hab class id {class_id} too large to pack as object_id * 1000 + class_id"

        for node in obj.visual_scene_nodes:
            node.semantic_id = obj.object_id * 1000 + class_id

        obj_id_to_class_id[obj.object_id] = class_id
        dimensions_by_obj_id[obj.object_id] = compute_obj_dimensions(obj)

    return ObjectAnnotation(env_name="ProcTHOR-hab", classes=classes, obj_id_to_class_id=obj_id_to_class_id, dimensions_by_obj_id=dimensions_by_obj_id)


def _setup_native_semantic_scene(sim, env_name: str, vocab_name: str) -> ObjectAnnotation:
    # Native semantic_id is already a per-instance id baked into the scene mesh at
    # asset-authoring time - SemanticObject exposes no writable node to rewrite it (unlike
    # HSSD-HAB/ProcTHOR-hab's rigid/articulated objects), so there's nothing to bake here,
    # just a class lookup keyed by the id as-is.
    classes = _resolve_native_vocab(env_name, vocab_name)
    class2int = {c: i for i, c in enumerate(classes)}

    obj_id_to_class_id: dict[int, int] = {}
    dimensions_by_obj_id: dict[int, np.ndarray] = {}

    for sem_obj in sim.semantic_scene.objects:
        if sem_obj is None or sem_obj.category is None:
            continue

        class_name = sem_obj.category.name()
        if class_name not in class2int:
            continue

        obj_id_to_class_id[sem_obj.semantic_id] = class2int[class_name]
        dimensions_by_obj_id[sem_obj.semantic_id] = compute_dimensions_from_obb(sem_obj.obb)

    return ObjectAnnotation(env_name=env_name, classes=classes, obj_id_to_class_id=obj_id_to_class_id, dimensions_by_obj_id=dimensions_by_obj_id)


def setup_semantic_labels(sim, env_name: str, vocab_name: str) -> ObjectAnnotation:
    """Builds the ObjectAnnotation for the scene currently loaded in `sim`, dispatching per
    env_name (HSSD-HAB/ProcTHOR-hab bake node.semantic_id = object_id * 1000 + class_id;
    MP3D/Gibson-Semantic read their already-unique native per-instance semantic_id as-is - see
    _setup_procthor/_setup_native_semantic_scene and CLAUDE.md for why). Call this again every
    time the scene changes; vocab resolution is cached internally so repeat calls only redo
    the genuinely per-scene work."""
    if env_name == "HSSD-HAB":
        return _setup_hssd(sim, vocab_name)
    elif env_name == "ProcTHOR-hab":
        return _setup_procthor(sim)
    elif env_name in ("MP3D", "Gibson-Semantic"):
        return _setup_native_semantic_scene(sim, env_name, vocab_name)
    else:
        raise NotImplementedError(f"Environment {env_name} not supported for object annotations")
