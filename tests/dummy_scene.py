"""
Builds a synthetic flat in an empty "NONE" scene, two 8 x 6 m rooms side by side: a floor slab per room and the walls
enclosing them, a doorway between them, all STATIC flat-shaded boxes, plus ProcTHOR-hab objects at their own scale: a
table with 4 bottles on it and a chair against a wall in the spawn room, a bed turned sideways to the doorway with a
cushion lying on it in the other room. recompute_navmesh carves the walls and objects out of the
floor; the ceiling is added after it, so its top face isn't walkable. Every object gets its own semantic_id,
so the semantic sensor sees them as separate objects.

Needs ProcTHOR-hab's objects under HABITAT_DATA/scene_datasets/ai2thor-hab.
"""

import os

import habitat_sim
import magnum as mn
import numpy as np
import trimesh

from common.env_utils.object_annotations import colored_box_asset, object_world_mesh, objects_world_mesh
from common.hybrid_exploration.exploration_map import MapConfig
from common.utils.projection_utils import DepthCamera

_WALL_HEIGHT = 2.0
_WALL_THICKNESS = 0.1
_SLAB_THICKNESS = 0.1
_FLOOR_ID = 1
_WALL_ID = 2
_CEILING_ID = 3
_BOX_COLORS = {"floor": (170, 160, 150), "wall": (235, 230, 220), "ceiling": (255, 255, 255)}
_DOOR_WIDTH = 1.0
# ((min x, min z), (max x, max z)): the spawn room z<0 and the room z>0
_ROOMS = [((0.0, -6.0), (8.0, 0.0)), ((0.0, 0.0), (8.0, 6.0))]
# (axis the wall runs along, its coordinate on the other axis, (start, end) along its axis, door centers)
_WALLS = [
    ("x", -6.0, (0.0, 8.0), ()), ("x", 6.0, (0.0, 8.0), ()), ("z", 0.0, (-6.0, 6.0), ()), ("z", 8.0, (-6.0, 6.0), ()),
    ("x", 0.0, (0.0, 8.0), (6.0,)),
]
_HABITAT_DATA = os.environ.get("HABITAT_DATA", "")
_PROCTHOR_OBJECTS_DIR = os.path.join(_HABITAT_DATA, "scene_datasets", "ai2thor-hab", "ai2thor-hab", "configs", "objects")
# (semantic_id, ground-truth class, ProcTHOR-hab object, center xz): models at their own scale, one per model family
_WALL_OBJECT = (13, "chair", "Chair_007_4", (0.32, -3.0))  # 0.1 m off the outer wall x = 0
_TABLE = (14, "table", "Dining_Table_205_1", (3.0, -3.0))
_BED = (16, "bed", "Bed_13", (4.0, 5.07))  # turned by _YAWS_DEG, 0.1 m off the outer wall z = 6
_FLOOR_OBJECTS = [_WALL_OBJECT, _TABLE, _BED]
_YAWS_DEG = {"Bed_13": 90.0}
# standing on the first surface below them: 4 bottles 0.25 m off _TABLE's center along both axes, the last one lying on
# its side, and a cushion at _BED's center
_TABLETOP_OBJECTS = [
    (15, "bottle", "Wine_Bottle_1", (2.75, -3.25)),
    (19, "bottle", "Spray_Bottle_1", (3.25, -3.25)),
    (20, "bottle", "Soap_Bottle_10", (2.75, -2.75)),
    (21, "bottle", "FloorPlan28_physics-Bottle", (3.25, -2.75)),
]
_SURFACE_OBJECTS = [*_TABLETOP_OBJECTS, (17, "pillow", "pillow_12", (4.0, 5.07))]
_OBJECTS = [*_FLOOR_OBJECTS, *_SURFACE_OBJECTS]
_DETIC_CLASSES = ["chair", "couch", "bench", "table", "bottle", "bed", "pillow", "object"]
_CAMERA_HEIGHT = 1.0
_RESOLUTION = [640, 640]
_MAX_DEPTH_M = 16.0
_DEPTH_CAMERA = DepthCamera(hfov=90.0, min_depth=0.1, max_depth=_MAX_DEPTH_M)
_MAP_CONFIG = MapConfig(fov_deg=90.0, resolution_m=0.05, map_size_m=40.0, visibility_dist_m=5.0)
_TURN_ANGLE_DEG = 45.0
_START = (6.0, 0.0, -4.0)
_FLOOR_RAY_START_M = 0.3


def _box_assets(directory: str) -> dict:
    """{surface: unit-cube .glb of its _BOX_COLORS color}."""
    return {surface: colored_box_asset(os.path.join(directory, f"{surface}.glb"), color) for surface, color in _BOX_COLORS.items()}


def _add_box(sim: habitat_sim.Simulator, asset: str, center: tuple, size: tuple, semantic_id: int) -> None:
    """Static, flat-shaded axis-aligned box of the unit-cube asset."""
    template_manager = sim.get_object_template_manager()
    template = template_manager.create_new_template(f"dummy_box_{semantic_id}")
    template.render_asset_handle = template.collision_asset_handle = asset
    template.scale = np.array(size)
    template.force_flat_shading = True
    template_id = template_manager.register_template(template, f"dummy_box_{semantic_id}")
    box = sim.get_rigid_object_manager().add_object_by_template_id(template_id)
    box.translation = mn.Vector3(*center)
    box.semantic_id = semantic_id
    box.motion_type = habitat_sim.physics.MotionType.STATIC


def _add_procthor_object(sim: habitat_sim.Simulator, name: str, center: tuple, semantic_id: int, base_y: float = 0.0):
    """Static ProcTHOR-hab object at its own scale, turned by its _YAWS_DEG entry, its bounding box standing at height
    base_y around center."""
    config = os.path.join(_PROCTHOR_OBJECTS_DIR, f"{name}.object_config.json")
    template_id = sim.get_object_template_manager().load_configs(config)[0]
    obj = sim.get_rigid_object_manager().add_object_by_template_id(template_id)
    obj.rotation = mn.Quaternion.rotation(mn.Deg(_YAWS_DEG.get(name, 0.0)), mn.Vector3.y_axis())
    box = obj.root_scene_node.cumulative_bb
    box_center = obj.rotation.transform_vector(box.center())
    obj.translation = mn.Vector3(center[0] - box_center.x, base_y - box.min.y, center[1] - box_center.z)
    for node in obj.visual_scene_nodes:
        node.semantic_id = semantic_id
    obj.motion_type = habitat_sim.physics.MotionType.STATIC
    return obj


def _room_slab(room: tuple, center_y: float) -> tuple:
    """(center, size) of a _SLAB_THICKNESS slab covering room, centered at height center_y."""
    (min_x, min_z), (max_x, max_z) = room
    return ((min_x + max_x) / 2, center_y, (min_z + max_z) / 2), (max_x - min_x, _SLAB_THICKNESS, max_z - min_z)


def _wall_pieces(axis: str, offset: float, span: tuple, doors: tuple) -> list:
    """(center, size) of the boxes of one wall, leaving a _DOOR_WIDTH gap around each door center."""
    edges = [span[0], *[edge for door in doors for edge in (door - _DOOR_WIDTH / 2, door + _DOOR_WIDTH / 2)], span[1]]
    pieces = list(zip(edges[::2], edges[1::2]))
    y = _WALL_HEIGHT / 2
    if axis == "x":
        return [(((start + end) / 2, y, offset), (end - start, _WALL_HEIGHT, _WALL_THICKNESS)) for start, end in pieces]
    return [((offset, y, (start + end) / 2), (_WALL_THICKNESS, _WALL_HEIGHT, end - start)) for start, end in pieces]


def _build_room(sim: habitat_sim.Simulator, assets: dict) -> None:
    for room in _ROOMS:
        _add_box(sim, assets["floor"], *_room_slab(room, -_SLAB_THICKNESS / 2), _FLOOR_ID)
    for center, size in (piece for wall in _WALLS for piece in _wall_pieces(*wall)):
        _add_box(sim, assets["wall"], center, size, _WALL_ID)
    floor_objects = [_add_procthor_object(sim, name, center, semantic_id) for semantic_id, _, name, center in _FLOOR_OBJECTS]
    supports = objects_world_mesh(floor_objects)
    for semantic_id, _, name, center in _SURFACE_OBJECTS:
        _add_procthor_object(sim, name, center, semantic_id, _surface_height(supports, center))


def _surface_height(supports: trimesh.Trimesh, center: tuple) -> float:
    """Height of the highest render-mesh surface of supports straight below center xz (collision hulls float above
    soft surfaces like bedding)."""
    triangles = supports.triangles
    flat = triangles * np.array([1.0, 0.0, 1.0])
    points = np.tile([center[0], 0.0, center[1]], (len(triangles), 1))
    barycentric = trimesh.triangles.points_to_barycentric(flat, points)
    below = (barycentric >= 0).all(axis=1)
    return float((barycentric[below] * triangles[below][:, :, 1]).sum(axis=1).max())


def _add_ceiling(sim: habitat_sim.Simulator, assets: dict) -> None:
    for room in _ROOMS:
        _add_box(sim, assets["ceiling"], *_room_slab(room, _WALL_HEIGHT + _SLAB_THICKNESS / 2), _CEILING_ID)


def _camera_spec(uuid: str, sensor_type: habitat_sim.SensorType) -> habitat_sim.CameraSensorSpec:
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.resolution = _RESOLUTION
    spec.hfov = mn.Deg(90)
    spec.position = [0.0, _CAMERA_HEIGHT, 0.0]
    return spec


def _navmesh_floor_offset(sim: habitat_sim.Simulator, position: np.ndarray) -> float:
    ray = habitat_sim.geo.Ray(position + np.array([0.0, _FLOOR_RAY_START_M, 0.0]), np.array([0.0, -1.0, 0.0]))
    return float(position[1] - sim.cast_ray(ray).hits[0].point[1])


def _placed_object(sim: habitat_sim.Simulator, name: str):
    return next(iter(sim.get_rigid_object_manager().get_objects_by_handle_substring(name).values()))


def _ground_truth_meshes(sim: habitat_sim.Simulator) -> list:
    """(object id, class, world mesh) of every placed object, as main.py's ground_truth_meshes."""
    class_ids = {name: idx for idx, name in enumerate(_DETIC_CLASSES)}
    objects = [(class_ids[category], _placed_object(sim, name)) for _, category, name, _ in _OBJECTS]
    return [(obj.object_id, class_id, object_world_mesh(obj)) for class_id, obj in objects]


def _dummy_sim(assets_dir: str) -> habitat_sim.Simulator:
    """Simulator of the synthetic flat, its box assets written to assets_dir."""
    assets = _box_assets(assets_dir)
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "NONE"
    backend_cfg.enable_physics = True
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        _camera_spec("rgb", habitat_sim.SensorType.COLOR),
        _camera_spec("semantic", habitat_sim.SensorType.SEMANTIC),
        _camera_spec("depth", habitat_sim.SensorType.DEPTH),
    ]
    for action in ("turn_left", "turn_right"):
        agent_cfg.action_space[action].actuation.amount = _TURN_ANGLE_DEG
    for action in ("move_left", "move_right", "move_backward"):
        agent_cfg.action_space[action] = habitat_sim.agent.ActionSpec(action, habitat_sim.agent.ActuationSpec(amount=agent_cfg.action_space["move_forward"].actuation.amount))
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    _build_room(sim, assets)
    nav_settings = habitat_sim.NavMeshSettings()
    nav_settings.set_defaults()
    nav_settings.include_static_objects = True
    assert sim.recompute_navmesh(sim.pathfinder, nav_settings)
    _add_ceiling(sim, assets)  # after the navmesh, so its top face isn't walkable
    return sim
