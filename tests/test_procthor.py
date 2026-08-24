"""
Sanity check that the ProcTHOR-hab scene dataset (see INSTALL.MD and
data/scene_datasets/ai2thor-hab/README.md) loads correctly in habitat_sim: instantiates one of
two known-good ProcTHOR-Train houses directly via raw habitat_sim.Simulator (no habitat-lab
Env/task) and renders an rgb+semantic overlay, labeled with each visible instance's category
name and its world-space AABB wireframe, at random navigable positions - dumped as
plain-rgb|overlay side-by-side frames to tests/testdump/test_procthor/ for visual inspection.

Unlike Gibson-Semantic/MP3D/HSSD, ProcTHOR's semantic ids are CATEGORY ids, not per-instance
ones: every ai2thor-hab/configs/objects/*.object_config.json ships a static "semantic_id" field
(looked up in configs/object_semantic_id_mapping.json - e.g. "ArmChair": 88, per the README),
and habitat_sim applies it to that object's render nodes at load time - confirmed directly
against a loaded scene: two ArmChair instances both carry obj.semantic_id == 88, and that's
exactly the pixel value the semantic sensor renders for both. This file renders that native
buffer as-is, so it can't tell two same-category instances apart here, only which category is
present where (common/env_utils/object_annotations.py::setup_semantic_labels does bake a real
per-instance id for ProcTHOR-hab - obj.semantic_id read here is exactly the native class_id it
packs into object_id * 1000 + class_id - but this standalone smoke test intentionally doesn't
go through the sensor).

Because of that, the wireframe+label pass gates purely on whether an object's class id shows up
anywhere in that frame's rendered semantic buffer (np.unique(semantic)) - the same "did this id
survive into the rendered buffer" test the other three files use, just at category instead of
instance granularity. No extra geometry/frustum check on top of that: a box's own corners get
individually clipped at the camera's near plane while drawing (same as the other files' obb
wireframes), and PIL simply doesn't render whatever falls outside the canvas, so a large object
that's only partially in frame still gets its box - an earlier version of this file additionally
required the object's *center* to project inside the frame, which dropped exactly that case: a
big, close ArmChair filling a corner of the view with its center just outside it. The class-in-
buffer check is still a necessary, not sufficient, stand-in for "this exact instance is visible"
- it can't rule out one same-class instance being visible while another, elsewhere in frame, is
fully hidden (e.g. behind a closed door). Boxes are still pure 3D-geometry projections with no
occlusion culling once drawn, same as the other files.

The rgb+color tint itself is still genuine per-pixel semantic segmentation (id 0 "Undefined" =
unlabeled background), just at category rather than instance granularity - so, unlike the other
three files, two overlapping same-class instances tint identically and blend into one blob.

Requires real ProcTHOR-hab data under HABITAT_DATA/scene_datasets/ai2thor-hab/ai2thor-hab/
(ai2thor-hab.scene_dataset_config.json + per-scene assets) - skipped otherwise.
"""

import collections
import itertools
import json
import math
import os

import habitat_sim
import magnum as mn
import numpy as np
import pytest
from PIL import Image, ImageDraw

from habitat_sim.utils.common import colorize_ids
from common.env_utils.object_annotations import get_all_objects
from common.env_utils.visibility_utils import camera_basis

HABITAT_DATA = os.environ.get("HABITAT_DATA")
_DATASET_CONFIG = (
    f"{HABITAT_DATA}/scene_datasets/ai2thor-hab/ai2thor-hab/ai2thor-hab.scene_dataset_config.json"
    if HABITAT_DATA else None
)

pytestmark = pytest.mark.skipif(
    not _DATASET_CONFIG or not os.path.exists(_DATASET_CONFIG),
    reason="requires HABITAT_DATA pointing at a real ai2thor-hab/ProcTHOR dataset (see INSTALL.MD)",
)

# No standard val split shipped for ProcTHOR in this repo - these are the same two known-good
# ProcTHOR-Train houses test_isometric_scene.py already exercises (one single-room, one
# multi-room), so any one of them works as a smoke test.
_SCENES = ["ProcTHOR-Train-9632", "ProcTHOR-Train-6340"]
_SCENE = _SCENES[0]
_RESOLUTION = [480, 480]
_HFOV_DEG = 90.0
_AGENT_HEIGHT = 0.88  # matches common/config/hssd-hab/default.yaml's agent height
_OVERLAY_ALPHA = 0.5
_N_POINTS = 10
_FILTER_OUT_CLASSES: list[str] = []  # see object_detector_sensors.py's filter_out_classes
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_procthor")
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

_ProcthorObject = collections.namedtuple("_ProcthorObject", ["class_name", "class_id", "center", "world_corners"])


def _load_id2class(dataset_config: str) -> dict:
    mapping_path = os.path.join(os.path.dirname(dataset_config), "configs", "object_semantic_id_mapping.json")
    name_to_id = json.load(open(mapping_path))
    return {v: k for k, v in name_to_id.items()}


@pytest.fixture(scope="module")
def sim():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = _DATASET_CONFIG
    backend_cfg.scene_id = _SCENE  # resolved against the "stages" registered in _DATASET_CONFIG
    # Same as HSSD (see test_hssd.py) - without physics, get_rigid_object_manager() resolves
    # every placed object to None, so there'd be nothing to enumerate for the wireframe pass.
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
    if not s.pathfinder.is_loaded:  # ProcTHOR scenes don't ship a precomputed .navmesh either
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


def _collect_objects(sim, id2class: dict) -> list:
    """One entry per placed rigid object with a recognized, non-filtered class - world AABB
    corners (or just its translation, for the rare object with no single collision AABB) are
    computed once here since objects don't move, instead of every frame."""
    objects = []
    for obj in get_all_objects(sim):
        if obj is None:
            continue
        class_name = id2class.get(obj.semantic_id)
        if class_name is None or class_name == "Undefined" or class_name in _FILTER_OUT_CLASSES:
            continue

        if hasattr(obj, "collision_shape_aabb"):
            aabb = obj.collision_shape_aabb
            min_v, max_v = aabb.min, aabb.max
            corners_local = itertools.product([min_v.x, max_v.x], [min_v.y, max_v.y], [min_v.z, max_v.z])
            world_corners = []
            for c in corners_local:
                w = obj.rotation.transform_vector(mn.Vector3(*c)) + obj.translation
                world_corners.append(np.array([w.x, w.y, w.z]))
            center = np.mean(world_corners, axis=0)
        else:  # e.g. an articulated object - no single collision AABB
            world_corners = None
            t = obj.translation
            center = np.array([t.x, t.y, t.z])

        objects.append(_ProcthorObject(class_name, obj.semantic_id, center, world_corners))
    return objects


def _project_point(world_point, cam_pos, forward, right, up, focal):
    v = world_point - cam_pos
    depth = np.dot(v, forward)
    x = _RESOLUTION[0] / 2 + focal * np.dot(v, right) / depth
    y = _RESOLUTION[1] / 2 - focal * np.dot(v, up) / depth
    return x, y


def _draw_wireframe(draw, world_corners, color, cam_pos, forward, right, up, focal):
    """Draws every box edge, clipping at the camera's near plane exactly like the other three
    files' obb wireframes - no other visibility gating here. Returns the average projected
    position of the corners that are in front of the camera, as a label anchor (or None if the
    whole box is behind it)."""
    depths = [np.dot(c - cam_pos, forward) for c in world_corners]
    projected = [
        _project_point(c, cam_pos, forward, right, up, focal) if d > _NEAR_PLANE else None
        for c, d in zip(world_corners, depths)
    ]

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

    visible_points = [p for p in projected if p is not None]
    if not visible_points:
        return None
    xs, ys = zip(*visible_points)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _rgb_semantic_overlay(rgb, semantic, objects, valid_ids, cam_pos, forward, right, up, focal) -> Image.Image:
    """rgb blended with colorize_ids(semantic) over labeled pixels only (id 0 stays plain rgb) -
    genuine per-pixel category segmentation. An object gets its wireframe and category name
    purely based on whether its class id shows up anywhere in this frame's rendered semantic
    buffer - the same "did this id survive into the rendered buffer" test the other three files
    use, just at category instead of instance granularity (so, as documented at the top of this
    file, it's a necessary but not sufficient stand-in for "this exact instance is visible")."""
    base = rgb[:, :, :3].astype(np.float32)
    color_ids = colorize_ids(semantic).astype(np.float32)
    labeled = np.isin(semantic, list(valid_ids))
    blended = base.copy()
    blended[labeled] = _OVERLAY_ALPHA * color_ids[labeled] + (1 - _OVERLAY_ALPHA) * base[labeled]
    img = Image.fromarray(blended.astype(np.uint8))

    visible_ids = {int(v) for v in np.unique(semantic)} & valid_ids

    draw = ImageDraw.Draw(img)
    for obj in objects:
        if obj.class_id not in visible_ids:
            continue

        color = tuple(int(c) for c in colorize_ids(np.array([[obj.class_id]]))[0, 0])
        if obj.world_corners is not None:
            label_pos = _draw_wireframe(draw, obj.world_corners, color, cam_pos, forward, right, up, focal)
        else:
            depth = np.dot(obj.center - cam_pos, forward)
            label_pos = _project_point(obj.center, cam_pos, forward, right, up, focal) if depth > _NEAR_PLANE else None

        if label_pos is not None:
            draw.text(label_pos, obj.class_name, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    return img


def test_generates_rgb_semantic_overlay_frames(sim):
    if not sim.pathfinder.is_loaded:
        pytest.skip(f"no navmesh loaded for {_SCENE} - check the ai2thor-hab assets")

    id2class = _load_id2class(_DATASET_CONFIG)
    valid_ids = {i for i, name in id2class.items() if name != "Undefined" and name not in _FILTER_OUT_CLASSES}
    objects = _collect_objects(sim, id2class)
    assert objects, (
        f"{_SCENE} loaded with no recognized ProcTHOR objects - check object_semantic_id_mapping.json coverage"
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
        overlay = _rgb_semantic_overlay(rgb, semantic, objects, valid_ids, cam_pos, forward, right, up, focal)
        side_by_side = Image.fromarray(np.concatenate([rgb[:, :, :3], np.array(overlay)], axis=1))
        side_by_side.save(os.path.join(_TESTDUMP_DIR, f"{prefix}_overlay.png"))

        classes_in_frame = {id2class[int(v)] for v in np.unique(semantic) if int(v) in valid_ids}
        print(f"{prefix}: {classes_in_frame}")

    print(f"{_SCENE}: {len(objects)} recognized objects in the scene, {_N_POINTS} frames dumped to {_TESTDUMP_DIR}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
