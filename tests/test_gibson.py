"""
Sanity check that the Gibson-Semantic scene dataset (see INSTALL.MD) loads correctly in
habitat_sim: instantiates one of the standard Gibson-tiny ObjectNav val scenes directly via
raw habitat_sim.Simulator (no habitat-lab Env/task yet - that integration into common/ comes
later, once HSSD-HAB and ProcTHOR are both wired up) and renders an rgb+semantic overlay,
labeled with each visible instance's category name and its 3D oriented-bounding-box
wireframe, at random navigable positions - dumped as plain-rgb|overlay side-by-side frames to
tests/testdump/test_gibson/ for visual inspection.

The semantic sensor's raw pixel values are per-instance ids (id 0 = unlabeled background),
not class ids directly - class name comes from sim.semantic_scene.objects[id].category.name().
Boxes come from that same object's .obb (its .aabb is unreliable for this dataset - e.g. a
single chair's aabb can come back several meters across, while its obb is a tight, correctly
oriented fit) and are projected as pure 3D geometry with no occlusion culling, so a box can
extend behind whatever's actually drawn in front of it. _FILTER_OUT_CLASSES drops matching
category names entirely - same semantics as object_detector_sensors.py's filter_out_classes -
so their pixels stay untinted and get no box/label.

Requires real Gibson-Semantic data under HABITAT_DATA/scene_datasets/gibson_semantic/
(gibson_semantic.scene_dataset_config.json + per-scene assets) - skipped otherwise.
"""

import itertools
import math
import os

import habitat_sim
import magnum as mn
import numpy as np
import pytest
from PIL import Image, ImageDraw

from habitat_sim.utils.common import colorize_ids
from common.env_utils.visibility_utils import camera_basis

HABITAT_DATA = os.environ.get("HABITAT_DATA")
_DATASET_CONFIG = (
    f"{HABITAT_DATA}/scene_datasets/gibson_semantic/gibson_semantic.scene_dataset_config.json"
    if HABITAT_DATA else None
)

pytestmark = pytest.mark.skipif(
    not _DATASET_CONFIG or not os.path.exists(_DATASET_CONFIG),
    reason="requires HABITAT_DATA pointing at a real gibson_semantic dataset (see INSTALL.MD)",
)

# Standard Gibson-tiny ObjectNav val split (SemExp/PONI); any one of the five works as a smoke test.
_VAL_SCENES = ["Collierville", "Corozal", "Darden", "Markleeville", "Wiconisco"]
_SCENE = _VAL_SCENES[0]
_RESOLUTION = [480, 480]
_HFOV_DEG = 90.0
_AGENT_HEIGHT = 0.88  # matches common/config/hssd-hab/default.yaml's agent height
_OVERLAY_ALPHA = 0.5
_N_POINTS = 10
_FILTER_OUT_CLASSES: list[str] = []  # class names to drop entirely (see object_detector_sensors.py's filter_out_classes)
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_gibson")
os.system(f"rm -rf {_TESTDUMP_DIR}")

# An obb's local_to_world maps the local unit cube [-1, 1]^3 (half-extents already baked
# into the matrix) to the object's 8 world-space corners.
_BOX_CORNERS_LOCAL = list(itertools.product((-1.0, 1.0), repeat=3))
_BOX_EDGES = [
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if sum(a != b for a, b in zip(_BOX_CORNERS_LOCAL[i], _BOX_CORNERS_LOCAL[j])) == 1
]

_NEAR_PLANE = 0.05  # depth (m) along the camera's forward axis below which a point can't be projected


def rearrange_center(v) -> np.ndarray:
    """Single point of control for the world-coordinate axis order coming out of a
    Gibson-semantic obb (center or corner) - every place in this file that turns obb
    data into a world-space (x, y, z) point routes through here. (x, z, -y) is
    confirmed correct for gibson_semantic/Collierville: it's what makes an object's
    obb wireframe line up with its actual rendered geometry, rather than a
    degenerate/floating box."""
    x, y, z = v
    return np.array([x, z, -y])


@pytest.fixture(scope="module")
def sim():
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_dataset_config_file = _DATASET_CONFIG
    backend_cfg.scene_id = _SCENE  # resolved against the "stages" registered in _DATASET_CONFIG
    backend_cfg.enable_physics = False

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
    yield s
    s.close()


def _random_agent_state(sim, rng) -> habitat_sim.AgentState:
    state = habitat_sim.AgentState()
    state.position = sim.pathfinder.get_random_navigable_point()
    yaw = rng.uniform(0, 2 * math.pi)
    state.rotation = np.quaternion(math.cos(yaw / 2), 0.0, math.sin(yaw / 2), 0.0)
    return state


def _objects_by_id(sim) -> dict:
    return {
        obj.semantic_id: obj
        for obj in sim.semantic_scene.objects
        if obj is not None and obj.category is not None and obj.category.name() not in _FILTER_OUT_CLASSES
    }


def _project_point(world_point, cam_pos, forward, right, up, focal):
    v = world_point - cam_pos
    depth = np.dot(v, forward)
    x = _RESOLUTION[0] / 2 + focal * np.dot(v, right) / depth
    y = _RESOLUTION[1] / 2 - focal * np.dot(v, up) / depth
    return x, y


def _draw_obb_wireframe(draw, obb, color, cam_pos, forward, right, up, focal):
    world_corners = []
    for corner in _BOX_CORNERS_LOCAL:
        c = obb.local_to_world.transform_point(mn.Vector3(*corner))
        world_corners.append(rearrange_center((c.x, c.y, c.z)))
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
    each visible instance gets its obb wireframe and its category name at the mask centroid."""
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
        obj = objects_by_id.get(int(obj_id))
        if obj is None:
            continue

        color = tuple(int(c) for c in colorize_ids(np.array([[obj_id]]))[0, 0])
        _draw_obb_wireframe(draw, obj.obb, color, cam_pos, forward, right, up, focal)

        ys, xs = np.where(semantic == obj_id)
        draw.text((int(xs.mean()), int(ys.mean())), obj.category.name(), fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    return img


def test_generates_rgb_semantic_overlay_frames(sim):
    if not sim.pathfinder.is_loaded:
        pytest.skip(f"no navmesh loaded for {_SCENE} - check the gibson_semantic assets")

    objects_by_id = _objects_by_id(sim)
    assert objects_by_id, (
        f"{_SCENE} loaded with no semantic annotations - check gibson_semantic.scene_dataset_config.json"
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

        classes_in_frame = {objects_by_id[v].category.name() for v in np.unique(semantic) if v != 0 and v in objects_by_id}
        print(f"{prefix}: {classes_in_frame}")

    print(f"{_SCENE}: {len(objects_by_id)} semantic objects in the scene, {_N_POINTS} frames dumped to {_TESTDUMP_DIR}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
