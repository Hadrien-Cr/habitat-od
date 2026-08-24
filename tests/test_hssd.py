"""
Sanity check that the HSSD-HAB scene dataset (see INSTALL.MD) loads correctly in habitat_sim:
instantiates one of the standard HSSD-HAB ObjectNav val scenes (habitat_embodied_al/pretrain/
config/ds_hssd.yaml's val_scenes) directly via raw habitat_sim.Simulator (no habitat-lab Env/task)
and renders an rgb+semantic overlay, labeled with each visible instance's category name and
its world-space AABB wireframe, at random navigable positions - dumped as plain-rgb|overlay
side-by-side frames to tests/testdump/test_hssd/ for visual inspection.

Unlike Gibson-Semantic/MP3D, HSSD-HAB doesn't populate sim.semantic_scene.objects (see
test_isometric_scene.py) - there's no per-scene semantic mesh annotation to read class names
from. Instead, per-pixel class ids have to be baked in by hand, exactly like
common/env_utils/object_annotations.py::setup_semantic_labels does for ObjectDetectorGTSensor:
every placed rigid/articulated object's handle is looked up in a target_vocab_object_annotations
mapping (object name -> target-vocab class name, built from hssd_obj_semantics_condensed.csv -
_load_target_vocab_object_annotations below is a standalone copy of that same module's
_load_hssd_vocab), and its visual scene nodes get
semantic_id = object_id * 1000 + class_id. Objects with no vocab match, or a filtered-out
class, get semantic_id 0 instead (plain unlabeled background). Decoding a rendered pixel
reverses that: class_id = semantic_id % 1000, object_id = semantic_id // 1000 - so, unlike
Gibson/MP3D, a semantic_id here is never itself a lookup key into any habitat_sim object
list, only something this file's own baking step defined.

Each object's wireframe is its collision_shape_aabb rotated+translated into world space by
the object's own transform (same corners common/env_utils/object_annotations.py::get_objects_info
computes) - HSSD has no separately-fit oriented box like Gibson/MP3D's .obb. Articulated
objects (e.g. a cabinet with a door) have no single collision_shape_aabb, so they get a label
but no box.

Requires real HSSD-HAB data under HABITAT_DATA/scene_datasets/hssd-hab/
(hssd-hab.scene_dataset_config.json + per-scene assets) - skipped otherwise.
"""

import collections
import itertools
import math
import os

import habitat_sim
import magnum as mn
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from habitat_sim.utils.common import colorize_ids
from common.env_utils.object_annotations import get_all_objects, object_shortname_from_handle
from common.env_utils.vocab_constants import VOCABULARIES, HSSD400_TO_VOCAB
from common.env_utils.visibility_utils import camera_basis

HABITAT_DATA = os.environ.get("HABITAT_DATA")
BASE_DIR = os.environ.get("BASE_DIR")
_DATASET_CONFIG = (
    f"{HABITAT_DATA}/scene_datasets/hssd-hab/hssd-hab.scene_dataset_config.json"
    if HABITAT_DATA else None
)

pytestmark = pytest.mark.skipif(
    not _DATASET_CONFIG or not os.path.exists(_DATASET_CONFIG),
    reason="requires HABITAT_DATA pointing at a real hssd-hab dataset (see INSTALL.MD)",
)

# Standard HSSD-HAB ObjectNav val split (habitat_embodied_al/pretrain/config/ds_hssd.yaml's
# val_scenes); any one of the four works as a smoke test.
_VAL_SCENES = ["102343992", "102816756", "104348328_171513363", "105515379_173104395"]
_SCENE = '102816756'
_VOCAB = "COCO80"  # matches common/config/hssd-hab/default.yaml's object_params.vocab_name
_RESOLUTION = [480, 480]
_HFOV_DEG = 90.0
_AGENT_HEIGHT = 0.88  # matches common/config/hssd-hab/default.yaml's agent height
_OVERLAY_ALPHA = 0.5
_N_POINTS = 10
_FILTER_OUT_CLASSES: list[str] = []  # see object_detector_sensors.py's filter_out_classes
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_hssd")
os.system(f"rm -rf {_TESTDUMP_DIR}")

# Cartesian-product corner order - only the adjacency pattern (which pairs differ in exactly
# one coordinate) matters here, so this is reused verbatim for any per-object min/max box.
_BOX_CORNERS_LOCAL = list(itertools.product((-1.0, 1.0), repeat=3))
_BOX_EDGES = [
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if sum(a != b for a, b in zip(_BOX_CORNERS_LOCAL[i], _BOX_CORNERS_LOCAL[j])) == 1
]

_NEAR_PLANE = 0.05  # depth (m) along the camera's forward axis below which a point can't be projected

_HssdObject = collections.namedtuple("_HssdObject", ["class_name", "managed_obj"])


@pytest.fixture(scope="module")
def sim():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = _DATASET_CONFIG
    backend_cfg.scene_id = _SCENE  # resolved against the "stages" registered in _DATASET_CONFIG
    # Unlike Gibson/MP3D's single scene mesh, HSSD's furniture is placed as separate rigid
    # objects - without physics enabled, get_rigid_object_manager() resolves every one of
    # them to None (still rendered, just not queryable), so _bake_semantic_ids finds nothing.
    backend_cfg.enable_physics = True

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = _RESOLUTION
    rgb_spec.hfov = mn.Deg(_HFOV_DEG)
    rgb_spec.position = [0.0, _AGENT_HEIGHT, 0.0]

    semantic_spec = habitat_sim.CameraSensorSpec()
    semantic_spec.uuid = "semantic"
    semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_spec.resolution = _RESOLUTION
    semantic_spec.hfov = rgb_spec.hfov
    semantic_spec.position = rgb_spec.position

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, semantic_spec]

    s = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    if not s.pathfinder.is_loaded:  # HSSD scenes don't ship a precomputed .navmesh
        nav_settings = habitat_sim.NavMeshSettings()
        nav_settings.set_defaults()
        s.recompute_navmesh(s.pathfinder, nav_settings)
    yield s
    s.close()


def _random_agent_state(sim, rng) -> habitat_sim.AgentState:
    state = habitat_sim.AgentState()
    state.position = sim.pathfinder.get_random_navigable_point()
    yaw = rng.uniform(0, 2 * math.pi)
    state.rotation = np.quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
    return state


def _load_target_vocab_object_annotations(vocab_name: str) -> tuple[list[str], dict[str, str]]:
    """Standalone copy of the object_name -> target-vocab class name resolution
    common/env_utils/object_annotations.py::_load_hssd_vocab does for the real sensor: HSSD-HAB
    has no native per-object category scheme, so every object's handle is cross-vocab mapped by
    hand (hssd_obj_semantics_condensed.csv -> HSSD400 class -> HSSD400_TO_VOCAB -> vocab_name).
    Duplicated here since this file intentionally bakes HSSD-HAB semantic ids against a raw
    habitat_sim.Simulator, with no ObjectDetectorGTSensor involved."""
    classes, _, _ = VOCABULARIES[vocab_name]
    df = pd.read_csv(os.path.join(str(BASE_DIR), "common", "env_utils", "hssd_obj_semantics_condensed.csv")).set_index("Object Hash")
    condensed_hssd400_vocab = df.iloc[:, 2].to_dict()

    target_vocab_object_annotations: dict[str, str] = {}
    for obj_name, s in condensed_hssd400_vocab.items():
        hssd400_class = str(s).replace("/", "_")
        if "unknown" in hssd400_class or hssd400_class == "nan":
            hssd400_class = "unknown"
        target_class = hssd400_class if vocab_name == "HSSD400" else HSSD400_TO_VOCAB[vocab_name][hssd400_class]
        if target_class != "unknown":
            target_vocab_object_annotations[str(obj_name)] = target_class

    return classes, target_vocab_object_annotations


def _bake_semantic_ids(sim, classes: list[str], target_vocab_object_annotations: dict[str, str]) -> dict:
    """Same encoding as object_annotations.py::setup_semantic_labels bakes for
    ObjectDetectorGTSensor: every placed object's visual scene nodes get
    semantic_id = object_id * 1000 + class_id, so the rendered
    per-pixel buffer carries both which object and which class. Objects with no target-vocab
    match, or a filtered-out class, get semantic_id 0 (unlabeled background) instead. Returns
    each labeled object's baked semantic_id (as it will appear in the rendered buffer) mapped
    to its class name and ManagedObject (for the wireframe)."""
    class2int = {c: i for i, c in enumerate(classes)}
    objects_by_id = {}
    for obj in get_all_objects(sim):
        if obj is None:
            continue
        obj_name = object_shortname_from_handle(obj.handle)
        class_name = target_vocab_object_annotations.get(obj_name)
        if class_name is None or class_name in _FILTER_OUT_CLASSES:
            for node in obj.visual_scene_nodes:
                node.semantic_id = 0
            continue

        semantic_id = obj.object_id * 1000 + class2int[class_name]
        for node in obj.visual_scene_nodes:
            node.semantic_id = semantic_id
        objects_by_id[semantic_id] = _HssdObject(class_name, obj)
    return objects_by_id


def _project_point(world_point, cam_pos, forward, right, up, focal):
    v = world_point - cam_pos
    depth = np.dot(v, forward)
    x = _RESOLUTION[0] / 2 + focal * np.dot(v, right) / depth
    y = _RESOLUTION[1] / 2 - focal * np.dot(v, up) / depth
    return x, y


def _draw_aabb_wireframe(draw, obj, color, cam_pos, forward, right, up, focal):
    if not hasattr(obj, "collision_shape_aabb"):
        return  # articulated objects have no single collision AABB

    aabb = obj.collision_shape_aabb
    min_v, max_v = aabb.min, aabb.max
    corners_local = itertools.product([min_v.x, max_v.x], [min_v.y, max_v.y], [min_v.z, max_v.z])

    world_corners = []
    for c in corners_local:
        w = obj.rotation.transform_vector(mn.Vector3(*c)) + obj.translation
        world_corners.append(np.array([w.x, w.y, w.z]))
    depths = [np.dot(c - cam_pos, forward) for c in world_corners]

    for i, j in _BOX_EDGES:
        pi, pj, di, dj = world_corners[i], world_corners[j], depths[i], depths[j]
        if di <= _NEAR_PLANE and dj <= _NEAR_PLANE:
            continue  # whole edge is behind the camera
        if di <= _NEAR_PLANE or dj <= _NEAR_PLANE:
            # object is close enough that this edge crosses the near plane - clip it there
            # instead of dropping it, since that's common for objects right in front of the agent
            t = (_NEAR_PLANE - di) / (dj - di)
            clipped = pi + t * (pj - pi)
            pi, pj = (clipped, pj) if di <= _NEAR_PLANE else (pi, clipped)
        draw.line(
            [_project_point(pi, cam_pos, forward, right, up, focal), _project_point(pj, cam_pos, forward, right, up, focal)],
            fill=color,
            width=2,
        )


def _rgb_semantic_overlay(rgb, semantic, objects_by_id, cam_pos, forward, right, up, focal) -> Image.Image:
    """rgb blended with colorize_ids(semantic) over labeled pixels only (id 0 stays plain rgb);
    each visible instance gets its AABB wireframe and its category name at the mask centroid."""
    base = rgb[:, :, :3].astype(np.float32)
    color_ids = colorize_ids(semantic).astype(np.float32)
    labeled = np.isin(semantic, list(objects_by_id.keys()))  # excludes id 0 and any filtered-out class
    blended = base.copy()
    blended[labeled] = _OVERLAY_ALPHA * color_ids[labeled] + (1 - _OVERLAY_ALPHA) * base[labeled]
    img = Image.fromarray(blended.astype(np.uint8))

    draw = ImageDraw.Draw(img)
    for obj_id in np.unique(semantic):
        if obj_id == 0:
            continue
        hssd_obj = objects_by_id.get(int(obj_id))
        if hssd_obj is None:
            continue

        color = tuple(int(c) for c in colorize_ids(np.array([[obj_id]]))[0, 0])
        _draw_aabb_wireframe(draw, hssd_obj.managed_obj, color, cam_pos, forward, right, up, focal)

        ys, xs = np.where(semantic == obj_id)
        draw.text((int(xs.mean()), int(ys.mean())), hssd_obj.class_name, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    return img


def test_generates_rgb_semantic_overlay_frames(sim):
    if not sim.pathfinder.is_loaded:
        pytest.skip(f"no navmesh loaded for {_SCENE} - check the hssd-hab assets")

    classes, target_vocab_object_annotations = _load_target_vocab_object_annotations(_VOCAB)
    objects_by_id = _bake_semantic_ids(sim, classes, target_vocab_object_annotations)
    assert objects_by_id, (
        f"{_SCENE} loaded with no {_VOCAB} objects - check hssd_obj_semantics_condensed.csv coverage"
    )

    os.makedirs(_TESTDUMP_DIR, exist_ok=True)
    agent = sim.get_agent(0)
    rng = np.random.default_rng(0)

    for i in range(_N_POINTS):
        agent.set_state(_random_agent_state(sim, rng))

        obs = sim.get_sensor_observations()
        rgb, semantic = obs["rgb"], obs["semantic"]

        assert rgb.shape[:2] == tuple(_RESOLUTION)
        assert semantic.shape[:2] == tuple(_RESOLUTION)

        sensor_state = sim.get_agent(0).get_state().sensor_states["rgb"]
        forward, right, up, focal = camera_basis(sensor_state.rotation, _RESOLUTION[0], _HFOV_DEG)
        cam_pos = np.array(sensor_state.position)

        prefix = f"{_SCENE}_{i:02d}"
        overlay = _rgb_semantic_overlay(rgb, semantic, objects_by_id, cam_pos, forward, right, up, focal)
        side_by_side = Image.fromarray(np.concatenate([rgb[:, :, :3], np.array(overlay)], axis=1))
        side_by_side.save(os.path.join(_TESTDUMP_DIR, f"{prefix}_overlay.png"))

        classes_in_frame = {objects_by_id[v].class_name for v in np.unique(semantic) if v != 0 and v in objects_by_id}
        print(f"{prefix}: {classes_in_frame}")

    print(f"{_SCENE}: {len(objects_by_id)} {_VOCAB} objects in the scene, {_N_POINTS} frames dumped to {_TESTDUMP_DIR}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
