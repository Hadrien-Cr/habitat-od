import glob
import itertools
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import magnum as mn
import numpy as np
import pandas as pd
import trimesh
import habitat_sim # type: ignore
import habitat.sims.habitat_simulator.sim_utilities as sutils # type: ignore

from common.env_utils.vocab_constants import (
    CLASS_LABELS_HSSD400, CLASS_LABELS_PROCTHOR, HSSD400_TO_VOCAB, PROCTHOR_NATIVE_CLASSES, PROCTHOR_NATIVE_VOCAB, PROCTHOR_VARIANT_SUFFIXES, VOCABULARIES,
)
from common.env_utils.visibility_utils import compute_obj_dimensions, compute_obj_render_dimensions, compute_dimensions_from_obb, box_size_measures, SIZE_MEASURES
from common.utils.grid_utils import HabitatObjOccupancyGrid

HABITAT_DATA = os.environ.get("HABITAT_DATA")
BASE_DIR = os.environ.get("BASE_DIR")
assert BASE_DIR is not None, "BASE_DIR environment variable must be set to the repo root"

PROCTHOR_ROOT = os.path.join(str(HABITAT_DATA), "scene_datasets", "ai2thor-hab", "ai2thor-hab")
HSSD80_SIZES_PATH = os.path.join(str(BASE_DIR), "common", "env_utils", "hssd80_sizes.csv")
PROCTHOR_NATIVE_SIZES_PATH = os.path.join(str(BASE_DIR), "common", "env_utils", "procthor_native_sizes.csv")


def get_obj_from_id(sim: habitat_sim.Simulator, obj_id: int):
    rom = sim.get_rigid_object_manager()
    if rom.get_library_has_id(obj_id):
        return rom.get_object_by_id(obj_id)
    return None

def object_world_mesh(obj) -> Optional[trimesh.Trimesh]:
    """obj's render mesh scaled and posed in world coordinates, or None without a render asset on disk."""
    asset_path = obj.creation_attributes.render_asset_fullpath
    if not asset_path or not os.path.exists(asset_path):
        return None
    mesh = trimesh.load(asset_path, force="mesh")
    # the first visual node carries the COM recentering on top of obj.transformation, but not the template scale
    mesh.apply_scale(np.array(obj.scale))
    mesh.apply_transform(np.array(obj.visual_scene_nodes[0].absolute_transformation()))
    return mesh


def colored_box_asset(path: str, color: tuple) -> str:
    """Exports a unit cube of the flat (r, g, b) color as a .glb at path, returned; a mesh asset, since primitives ignore force_flat_shading."""
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=[*color, 255]))
    mesh.export(path)
    return path


def objects_world_mesh(objects) -> trimesh.Trimesh:
    """One mesh merging every object's object_world_mesh, skipping those without a render asset on disk."""
    meshes = [object_world_mesh(obj) for obj in objects]
    return trimesh.util.concatenate([mesh for mesh in meshes if mesh is not None])


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
    object_info_list/object_occupancy_grid (the top-down occupancy-map debug view exposed via
    ExplorationEnv.get_tdmap) are HSSD-HAB-only - None for the other 3
    envs, which have no equivalent handle-based object classification to build them from."""
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


def _procthor_base_class(name: str) -> str:
    """name without a PROCTHOR_VARIANT_SUFFIXES suffix (AppleSliced -> Apple) when what remains is a class of its own."""
    return next((name[:-len(suffix)] for suffix in PROCTHOR_VARIANT_SUFFIXES if name.endswith(suffix) and name[:-len(suffix)] in PROCTHOR_NATIVE_CLASSES), name)


@lru_cache(maxsize=None)
def _procthor_class_ids(mapping_path: str) -> dict[int, int]:
    """{native semantic id habitat_sim applies to ProcTHOR-hab objects at load time: its PROCTHOR_NATIVE_CLASSES (and CLASS_LABELS_PROCTHOR) index}, from
    configs/object_semantic_id_mapping.json, variants folded into their base class and "Undefined" left out."""
    name_to_id: dict[str, int] = json.load(open(mapping_path))
    unlisted = [name for name in name_to_id if name != "Undefined" and _procthor_base_class(name) not in PROCTHOR_NATIVE_CLASSES]
    assert not unlisted, f"ProcTHOR-hab classes missing from PROCTHOR_NATIVE_CLASSES: {unlisted}"
    return {native_id: PROCTHOR_NATIVE_CLASSES.index(_procthor_base_class(name)) for name, native_id in name_to_id.items() if name != "Undefined"}


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


def _setup_procthor(sim, vocab_name: str) -> ObjectAnnotation:
    # Places furniture as real rigid/articulated objects (like HSSD-HAB), each with its own
    # writable visual_scene_nodes - unlike its native per-category semantic id (which merges
    # same-class instances), we can bake a per-instance id here too.
    assert vocab_name == PROCTHOR_NATIVE_VOCAB, (
        f"ProcTHOR-hab's vocabulary is CLASS_LABELS_PROCTHOR, got vocab_name={vocab_name!r}, expected {PROCTHOR_NATIVE_VOCAB!r}"
    )
    ai2_root = os.path.dirname(sim.config.sim_cfg.scene_dataset_config_file)
    class_ids = _procthor_class_ids(os.path.join(ai2_root, "configs", "object_semantic_id_mapping.json"))

    obj_id_to_class_id: dict[int, int] = {}
    dimensions_by_obj_id: dict[int, np.ndarray] = {}

    for obj in get_all_objects(sim):
        class_id = class_ids.get(obj.semantic_id)  # native id, applied by habitat_sim at load time - read before we overwrite it below

        if class_id is None:
            for node in obj.visual_scene_nodes:
                node.semantic_id = 0
            continue

        assert class_id < 1000, f"ProcTHOR-hab class id {class_id} too large to pack as object_id * 1000 + class_id"

        for node in obj.visual_scene_nodes:
            node.semantic_id = obj.object_id * 1000 + class_id

        obj_id_to_class_id[obj.object_id] = class_id
        dimensions_by_obj_id[obj.object_id] = compute_obj_dimensions(obj)

    return ObjectAnnotation(env_name="ProcTHOR-hab", classes=CLASS_LABELS_PROCTHOR, obj_id_to_class_id=obj_id_to_class_id, dimensions_by_obj_id=dimensions_by_obj_id)


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
        return _setup_procthor(sim, vocab_name)
    elif env_name in ("MP3D", "Gibson-Semantic"):
        return _setup_native_semantic_scene(sim, env_name, vocab_name)
    else:
        raise NotImplementedError(f"Environment {env_name} not supported for object annotations")


def resolve_classes(env_name: str, vocab_name: str) -> list[str]:
    """Class list for (env_name, vocab_name) without needing a live sim, unlike
    setup_semantic_labels above -- for callers (coco_writer.py, collection.py's
    visualize_mosaic, agents.py's do_visualize) that only run once collection's simulator
    has closed. ProcTHOR-hab's only vocab_name is PROCTHOR_NATIVE_VOCAB."""
    if env_name == "ProcTHOR-hab":
        assert vocab_name == PROCTHOR_NATIVE_VOCAB, (
            f"ProcTHOR-hab's vocabulary is CLASS_LABELS_PROCTHOR, got vocab_name={vocab_name!r}, expected {PROCTHOR_NATIVE_VOCAB!r}"
        )
        return CLASS_LABELS_PROCTHOR
    elif env_name in ("MP3D", "Gibson-Semantic"):
        return _resolve_native_vocab(env_name, vocab_name)
    elif env_name == "HSSD-HAB":
        classes, _, _ = VOCABULARIES[vocab_name]
        return classes
    else:
        raise NotImplementedError(f"Environment {env_name} not supported for object annotations")


def _scene_placements(scene_glob: str) -> list[tuple[str, np.ndarray]]:
    """(template name, scale) of every object instance placed in the scene_glob scenes."""
    instances = [inst for path in glob.glob(scene_glob) for inst in json.load(open(path))["object_instances"]]
    return [(os.path.basename(inst["template_name"]), np.abs(inst["non_uniform_scale"])) for inst in instances]


def _template_placements(object_config_glob: str) -> list[tuple[str, np.ndarray]]:
    """(template name, unit scale) of every object_config_glob template, placed or not."""
    return [(os.path.basename(path).removesuffix(".object_config.json"), np.ones(3)) for path in glob.glob(object_config_glob)]


def _template_dims_and_semantic_ids(dataset_config: str, template_names: set[str]) -> dict[str, tuple[np.ndarray, int]]:
    """Render-mesh dimensions and native semantic_id of each template with an exact config, read by instancing it once in a stage-less sim."""
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = dataset_config
    backend_cfg.scene_id = "NONE"
    backend_cfg.enable_physics = True  # without physics the rigid object manager returns no objects
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [habitat_sim.agent.AgentConfiguration()]))
    rom = sim.get_rigid_object_manager()
    handles = {
        os.path.basename(handle).removesuffix(".object_config.json"): handle
        for handle in sim.get_object_template_manager().get_template_handles()
    }

    template_info: dict[str, tuple[np.ndarray, int]] = {}
    for name in template_names & handles.keys():
        obj = rom.add_object_by_template_handle(handles[name])
        template_info[name] = (compute_obj_render_dimensions(obj), obj.semantic_id)
        rom.remove_object_by_id(obj.object_id)
    sim.close()
    return template_info


def _class_size_table(placements: list[tuple[str, np.ndarray]], template_info: dict[str, tuple[np.ndarray, int]], class_of_template) -> pd.DataFrame:
    """Per-class placement count and min/max of each box_size_measures of the scaled render-mesh AABB, over every (template name, scale) placement."""
    rows = []
    for name, scale in placements:
        if name not in template_info:
            continue
        dims, semantic_id = template_info[name]
        class_name = class_of_template(name, semantic_id)
        if class_name is not None:
            rows.append({"class": class_name, **box_size_measures(dims * scale)})
    per_class = pd.DataFrame(rows).groupby("class")
    ranges = {f"{bound}_{name}": per_class[name].agg(bound) for name in SIZE_MEASURES for bound in ("min", "max")}
    return pd.DataFrame({"num_instances": per_class.size(), **ranges})


def _write_class_size_table(dataset_config: str, placements: list[tuple[str, np.ndarray]], class_of_template, output_path: str) -> None:
    template_info = _template_dims_and_semantic_ids(dataset_config, {name for name, _ in placements})
    _class_size_table(placements, template_info, class_of_template).round(6).to_csv(output_path)


def class_size_ranges(sizes_path: str) -> dict[str, dict[str, tuple[float, float]]]:
    """{class: {size measure: (min, max)}} of a hssd_hab_sizes/procthor_hab_sizes table."""
    table = pd.read_csv(sizes_path, index_col="class")
    return {
        class_name: {name: (row[f"min_{name}"], row[f"max_{name}"]) for name in SIZE_MEASURES}
        for class_name, row in table.iterrows()
    }


def hssd_hab_sizes(output_path: str = HSSD80_SIZES_PATH) -> None:
    _, target_vocab_object_annotations, _ = _load_hssd_vocab("HSSD80")
    hssd_root = os.path.join(str(HABITAT_DATA), "scene_datasets", "hssd-hab")
    _write_class_size_table(
        os.path.join(hssd_root, "hssd-hab.scene_dataset_config.json"),
        _scene_placements(os.path.join(hssd_root, "scenes", "*.scene_instance.json")),
        lambda name, _: target_vocab_object_annotations.get(name),
        output_path,
    )


def procthor_hab_sizes(output_path: str = PROCTHOR_NATIVE_SIZES_PATH) -> None:
    class_ids = _procthor_class_ids(os.path.join(PROCTHOR_ROOT, "configs", "object_semantic_id_mapping.json"))
    _write_class_size_table(
        os.path.join(PROCTHOR_ROOT, "ai2thor-hab.scene_dataset_config.json"),
        _template_placements(os.path.join(PROCTHOR_ROOT, "configs", "objects", "*.object_config.json")),
        lambda _, native_id: CLASS_LABELS_PROCTHOR[class_ids[native_id]] if native_id in class_ids else None,
        output_path,
    )


if __name__ == "__main__":
    hssd_hab_sizes()
    procthor_hab_sizes()
