"""
Builds a synthetic flat in an empty "NONE" scene: a floor slab, 4 outer walls and inner walls with doorways
splitting it into 4 rooms of different rectangular sizes, all STATIC flat-shaded boxes, plus 5 ProcTHOR-hab objects
per room, 20 distinct chairs, armchairs, sofas and benches at their own scale. recompute_navmesh carves the walls and objects out of the
floor; the ceiling is added after it, so its top face isn't walkable. Every object gets its own semantic_id,
so the semantic sensor sees them as separate objects.

Runs common/hybrid_exploration's HybridFrontierAgent in exhaustive mode in the flat, fed by an oracle detector:
each visible object's points come from its semantic mask and depth.
Into tests/testdump/test_dummy_scene/ (emptied when the test starts) go the last step's decision (current view in
yellow, candidates labelled with u*t above and t below, chosen goal ringed) to decision.png, its believed object
occupancy (object points in white, explored area green, the rest gray) to object.png, and every step's
Detic predictions drawn on its rgb as one tile of predictions_mosaic.png, as main.py's. Walls, floor and ceiling are flat-shaded.

Alongside, as in habitat_embodied_sl/main.py, Detic (Swin-B, classifying against chair, couch and bench only)
detects in every frame and SAM masks its boxes, and its masks and region embeddings are fused into a concept-graphs
ConceptGraphObjectSet (common/utils/concept_graph.py). Once the run ends it is resolved, each object classified by
Detector.classify_resolved from its raytraced views (their trials to resolved_trial_mosaic.png), then drawn
by class on the agent's map to semantic_topdown.png next to the placed objects' mesh silhouettes in
ground_truth_topdown.png (both main.py's), and retrieval_mosaic.png gets one row per resolved object of the remembered frames showing it.

Needs ProcTHOR-hab's objects under HABITAT_DATA/scene_datasets/ai2thor-hab, a GPU, Detic's Swin-B checkpoint
under third_party/Detic/models and SAM's ViT-B checkpoint under data/sam.
"""

import os
import shutil

import habitat_sim
import magnum as mn
import numpy as np
import pytest
import trimesh
import torch
from PIL import Image
from tqdm import tqdm

from common.hybrid_exploration.exploration_map import MapConfig
from common.hybrid_exploration.hybrid_agent import HybridConfig, HybridFrontierAgent, ViewpointMode
from common.hybrid_exploration.object_viewpoints import ObjectBelief
from common.env_utils.object_annotations import object_world_mesh
from common.env_utils.vocab_constants import make_colors
from common.utils.concept_graph import ConceptGraphObjectSet
from common.utils.image_memory import Frame, ImageMemory
from common.utils.plot_utils import make_mosaic
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import DepthCamera, downsample_points, object_points
from common.utils.retrieval import SegmentationStyle, retrieval_mosaic, segmentation_tile
from detector import Detector
from habitat_embodied_sl.main import ground_truth_topdown, predicted_objects, semantic_topdown

_ROOM_HALF_EXTENTS = (8.0, 6.0)
_WALL_HEIGHT = 2.0
_WALL_THICKNESS = 0.1
_FLOOR_ID = 1
_WALL_IDS = (2, 3, 4, 5)
_CEILING_ID = 6
_INNER_WALL_ID = 7
_BOX_COLORS = {"floor": (170, 160, 150), "wall": (235, 230, 220), "ceiling": (255, 255, 255)}
_DOOR_WIDTH = 1.0
# (axis the wall runs along, its coordinate on the other axis, (start, end) along its axis, door centers):
# rooms x<-2 (6 x 12 m), x>-2 z<0 (10 x 6 m), -2<x<4 z>0 (6 x 6 m) and x>4 z>0 (4 x 6 m)
_INNER_WALLS = [("z", -2.0, (-6.0, 6.0), (-3.0, 3.0)), ("x", 0.0, (-2.0, 8.0), (6.0,)), ("z", 4.0, (0.0, 6.0), (3.0,))]
_HABITAT_DATA = os.environ.get("HABITAT_DATA", "")
_PROCTHOR_OBJECTS_DIR = os.path.join(_HABITAT_DATA, "scene_datasets", "ai2thor-hab", "ai2thor-hab", "configs", "objects")
# (semantic_id, ground-truth class, ProcTHOR-hab object, floor center xz): 20 models at their own scale, one per
# model family; sofas and benches stand along walls, clear of the doorways
_WALL_OBJECT = (13, "chair", "Chair_007_4", (-7.68, 3.0))  # 0.1 m off the outer wall x = -8
_CLOSE_OBJECTS = [(14, "chair", "Chair_205_1", (0.0, -4.0)), (15, "chair", "Chair_203_1", (0.81, -4.0))]  # 0.4 m apart
_OBJECTS = [
    _WALL_OBJECT,
    (16, "couch", "Sofa_221_1", (-5.0, -5.5)),
    (17, "bench", "FloorPlan527_physics-Bench_4", (-5.0, 5.4)),
    (11, "chair", "Armchair_204_1", (-4.0, -2.0)),
    (18, "chair", "Chair_002_1", (-3.5, 1.0)),
    *_CLOSE_OBJECTS,
    (19, "couch", "Sofa_201_3", (3.5, -5.45)),
    (20, "bench", "FloorPlan516_physics-Bench_6", (2.5, -0.5)),
    (21, "chair", "Armchair_227_1", (-1.0, -1.5)),
    (12, "couch", "Sofa_227_1", (1.0, 5.45)),
    (22, "chair", "Chair_204_1", (-1.0, 5.0)),
    (23, "chair", "Armchair_201_3", (3.0, 1.0)),
    (24, "chair", "Armchair_207_1", (-0.5, 1.5)),
    (25, "chair", "Chair_215_1", (1.5, 2.5)),
    (26, "bench", "FloorPlan515_physics-Bench_1", (6.0, 5.5)),
    (10, "chair", "Armchair_230_1", (6.0, 3.5)),
    (27, "chair", "Armchair_219_2", (7.0, 1.5)),
    (28, "chair", "Chair_210_1", (5.0, 1.5)),
    (29, "chair", "Chair_313_1", (7.3, 4.5)),
]
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DETIC_ROOT = os.path.join(_REPO_ROOT, "third_party", "Detic")
_DETIC_CONFIG = os.path.join(_DETIC_ROOT, "configs", "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
_DETIC_CHECKPOINT = os.path.join(_DETIC_ROOT, "models", "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
_DETIC_CLASSES = ["chair", "couch", "bench"]
_SAM_CHECKPOINT = os.path.join(_REPO_ROOT, "data", "sam", "sam_vit_b_01ec64.pth")

pytestmark = [
    pytest.mark.skipif(not os.path.isdir(_PROCTHOR_OBJECTS_DIR), reason="requires HABITAT_DATA pointing at a real ai2thor-hab dataset (see INSTALL.MD)"),
    pytest.mark.skipif(not os.path.exists(_DETIC_CHECKPOINT) or not torch.cuda.is_available(), reason="requires a GPU and Detic's Swin-B checkpoint"),
    pytest.mark.skipif(not os.path.exists(_SAM_CHECKPOINT), reason="requires SAM's ViT-B checkpoint"),
]
_CAMERA_HEIGHT = 1.0
_RESOLUTION = [256, 256]
_MAX_DEPTH_M = 2 * max(_ROOM_HALF_EXTENTS)
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_dummy_scene")
_DEPTH_CAMERA = DepthCamera(hfov=90.0, min_depth=0.1, max_depth=_MAX_DEPTH_M)
_FULL_CONFIDENCE_AREA = 0.05
_MIN_VISIBLE_PIXELS = 20
_POINTS_VOXEL_M = 0.05
_TURN_ANGLE_DEG = 45.0
_START = (6.0, 0.0, -4.0)
_MAX_STEPS = 50
_PREDICTION_MOSAIC_COLS = 5
_MIN_VISIBILITY_FRACTION = 0.2
_DETIC_STYLE = SegmentationStyle(_DETIC_CLASSES, make_colors(len(_DETIC_CLASSES), seed=0, ctype=0))


def _box_assets(directory: str) -> dict:
    """{surface: unit-cube .glb of its _BOX_COLORS color}; a mesh asset, since primitives ignore force_flat_shading."""
    assets = {}
    for surface, color in _BOX_COLORS.items():
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        mesh.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=[*color, 255]))
        assets[surface] = os.path.join(directory, f"{surface}.glb")
        mesh.export(assets[surface])
    return assets


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


def _add_procthor_object(sim: habitat_sim.Simulator, name: str, center: tuple, semantic_id: int) -> None:
    """Static ProcTHOR-hab object at its own scale, its bounding box standing on the floor around center."""
    config = os.path.join(_PROCTHOR_OBJECTS_DIR, f"{name}.object_config.json")
    template_id = sim.get_object_template_manager().load_configs(config)[0]
    obj = sim.get_rigid_object_manager().add_object_by_template_id(template_id)
    box = obj.root_scene_node.cumulative_bb
    obj.translation = mn.Vector3(center[0] - box.center().x, -box.min.y, center[1] - box.center().z)
    for node in obj.visual_scene_nodes:
        node.semantic_id = semantic_id
    obj.motion_type = habitat_sim.physics.MotionType.STATIC


def _wall_boxes() -> list:
    """(center, size) of the 4 walls enclosing the floor."""
    half_x, half_z = _ROOM_HALF_EXTENTS
    y = _WALL_HEIGHT / 2
    x_wall = (_WALL_THICKNESS, _WALL_HEIGHT, 2 * half_z)
    z_wall = (2 * half_x, _WALL_HEIGHT, _WALL_THICKNESS)
    return [((-half_x, y, 0.0), x_wall), ((half_x, y, 0.0), x_wall), ((0.0, y, -half_z), z_wall), ((0.0, y, half_z), z_wall)]


def _wall_pieces(axis: str, offset: float, span: tuple, doors: tuple) -> list:
    """(center, size) of the boxes of one inner wall, leaving a _DOOR_WIDTH gap around each door center."""
    edges = [span[0], *[edge for door in doors for edge in (door - _DOOR_WIDTH / 2, door + _DOOR_WIDTH / 2)], span[1]]
    pieces = list(zip(edges[::2], edges[1::2]))
    y = _WALL_HEIGHT / 2
    if axis == "x":
        return [(((start + end) / 2, y, offset), (end - start, _WALL_HEIGHT, _WALL_THICKNESS)) for start, end in pieces]
    return [((offset, y, (start + end) / 2), (_WALL_THICKNESS, _WALL_HEIGHT, end - start)) for start, end in pieces]


def _build_room(sim: habitat_sim.Simulator, assets: dict) -> None:
    half_x, half_z = _ROOM_HALF_EXTENTS
    _add_box(sim, assets["floor"], (0.0, -0.05, 0.0), (2 * half_x, 0.1, 2 * half_z), _FLOOR_ID)
    for semantic_id, (center, size) in zip(_WALL_IDS, _wall_boxes()):
        _add_box(sim, assets["wall"], center, size, semantic_id)
    for center, size in (piece for wall in _INNER_WALLS for piece in _wall_pieces(*wall)):
        _add_box(sim, assets["wall"], center, size, _INNER_WALL_ID)
    for semantic_id, _, name, center in _OBJECTS:
        _add_procthor_object(sim, name, center, semantic_id)


def _add_ceiling(sim: habitat_sim.Simulator, assets: dict) -> None:
    half_x, half_z = _ROOM_HALF_EXTENTS
    _add_box(sim, assets["ceiling"], (0.0, _WALL_HEIGHT + 0.05, 0.0), (2 * half_x, 0.1, 2 * half_z), _CEILING_ID)


def _camera_spec(uuid: str, sensor_type: habitat_sim.SensorType) -> habitat_sim.CameraSensorSpec:
    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.resolution = _RESOLUTION
    spec.hfov = mn.Deg(90)
    spec.position = [0.0, _CAMERA_HEIGHT, 0.0]
    return spec


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    assets = _box_assets(str(tmp_path_factory.mktemp("boxes")))
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
    s = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    _build_room(s, assets)
    nav_settings = habitat_sim.NavMeshSettings()
    nav_settings.set_defaults()
    nav_settings.include_static_objects = True
    assert s.recompute_navmesh(s.pathfinder, nav_settings)
    _add_ceiling(s, assets)  # after the navmesh, so its top face isn't walkable
    yield s
    s.close()


def _object_belief(obs: dict, pose: np.ndarray, semantic_id: int, previous: ObjectBelief) -> ObjectBelief:
    mask = obs["semantic"] == semantic_id
    points = object_points(obs["depth"], pose, mask, _DEPTH_CAMERA)
    confidence = min(1.0, float(mask.mean()) / _FULL_CONFIDENCE_AREA)
    if previous is None:
        return ObjectBelief(confidence, points)
    return ObjectBelief(max(confidence, previous.confidence), downsample_points(np.vstack([previous.points, points]), _POINTS_VOXEL_M))


def _oracle_beliefs(beliefs: dict, obs: dict, pose: np.ndarray) -> dict:
    """beliefs updated with every object covering at least _MIN_VISIBLE_PIXELS of obs."""
    visible = [semantic_id for semantic_id, _, _, _ in _OBJECTS if (obs["semantic"] == semantic_id).sum() >= _MIN_VISIBLE_PIXELS]
    return {**beliefs, **{semantic_id: _object_belief(obs, pose, semantic_id, beliefs.get(semantic_id)) for semantic_id in visible}}


def _explore(sim: habitat_sim.Simulator, config: HybridConfig) -> tuple:
    """(agent, every step's agent state) of one HybridFrontierAgent run."""
    sim_agent = sim.get_agent(0)
    start = habitat_sim.AgentState()
    start.position = np.array(_START)
    sim_agent.set_state(start)
    agent = HybridFrontierAgent(sim.pathfinder, config, np.random.default_rng(0))
    beliefs, states = {}, []
    for _ in tqdm(range(_MAX_STEPS)):
        state, obs = sim_agent.get_state(), sim.get_sensor_observations()
        states.append(state)
        beliefs = _oracle_beliefs(beliefs, obs, sensor_pose_matrix(state.sensor_states["depth"]))
        agent.update_objects(beliefs)
        action = agent.act(state)
        if action == "stop":
            break
        sim.step(action)
    return agent, states


def _observe(sim: habitat_sim.Simulator, state: habitat_sim.AgentState) -> Frame:
    """The frame seen from agent state."""
    sim_agent = sim.get_agent(0)
    sim_agent.set_state(state)
    obs = sim.get_sensor_observations()
    return Frame(obs["rgb"][:, :, :3], obs["depth"], sensor_pose_matrix(sim_agent.get_state().sensor_states["depth"]))


def _fuse_predictions(sim: habitat_sim.Simulator, states: list, detector: Detector) -> tuple:
    """(every state's labelled detector predictions drawn on its rgb, concept graph fused from them, every frame)."""
    fused, memory, prediction_tiles = ConceptGraphObjectSet(), ImageMemory(), []
    for step, state in enumerate(tqdm(states)):
        frame = _observe(sim, state)
        memory.add_frame(frame)
        instances = detector.predict(frame.rgb)
        prediction_tiles.append(segmentation_tile(frame.rgb, instances, _DETIC_STYLE, f"step{step:03d}"))
        fused = fused + predicted_objects(frame, instances, _DEPTH_CAMERA)
    return prediction_tiles, fused, memory


def _ground_truth_meshes(sim: habitat_sim.Simulator) -> list:
    """(class, world mesh) of every placed object, as main.py's ground_truth_meshes."""
    class_ids = {name: idx for idx, name in enumerate(_DETIC_CLASSES)}
    objects = sim.get_rigid_object_manager()
    return [
        (class_ids[category], object_world_mesh(next(iter(objects.get_objects_by_handle_substring(name).values()))))
        for _, category, name, _ in _OBJECTS
    ]


def _save_resolved_maps(sim: habitat_sim.Simulator, agent: HybridFrontierAgent, fused: ConceptGraphObjectSet, memory: ImageMemory, detector: Detector) -> None:
    """Resolves and classifies fused as main.py does, then saves its semantic_topdown.png, ground_truth_topdown.png and
    retrieval_mosaic.png (one row per object of the frames showing it)."""
    fused.resolve()
    fused = detector.classify_resolved(fused, memory, _DEPTH_CAMERA, _MIN_VISIBILITY_FRACTION, os.path.join(_TESTDUMP_DIR, "resolved_trial_mosaic.png"))
    colors = _DETIC_STYLE.colors
    Image.fromarray(semantic_topdown(fused, agent, _DETIC_CLASSES, colors)).save(os.path.join(_TESTDUMP_DIR, "semantic_topdown.png"))
    Image.fromarray(ground_truth_topdown(_ground_truth_meshes(sim), agent, _DETIC_CLASSES, colors)).save(os.path.join(_TESTDUMP_DIR, "ground_truth_topdown.png"))
    raytraced = memory.raytrace_frames(fused, _DEPTH_CAMERA, _MIN_VISIBILITY_FRACTION)
    retrieval_mosaic(fused, memory, raytraced, _DETIC_STYLE, len(_OBJECTS)).save(os.path.join(_TESTDUMP_DIR, "retrieval_mosaic.png"))


@pytest.fixture(scope="module", autouse=True)
def empty_testdump():
    shutil.rmtree(_TESTDUMP_DIR, ignore_errors=True)
    os.makedirs(_TESTDUMP_DIR)


def _save_maps_and_predictions(agent: HybridFrontierAgent, prediction_tiles: list) -> None:
    """decision.png and object.png of the last step, and predictions_mosaic.png of every step's predictions, as main.py's."""
    for stem, tile in {"decision": agent.map_tile(), "object": agent.object_map_tile()}.items():
        make_mosaic([tile], N_cols=1).save(os.path.join(_TESTDUMP_DIR, f"{stem}.png"))
    make_mosaic(prediction_tiles, N_cols=_PREDICTION_MOSAIC_COLS).save(os.path.join(_TESTDUMP_DIR, "predictions_mosaic.png"))


@pytest.fixture(scope="module")
def detector():
    return Detector(_DETIC_CONFIG, _DETIC_CHECKPOINT, _DETIC_CLASSES, _SAM_CHECKPOINT, agnostic_nms_thresh=0.7)


@pytest.fixture(scope="module")
def explored(sim):
    return _explore(sim, HybridConfig(MapConfig(fov_deg=90.0, resolution=512), _TURN_ANGLE_DEG, viewpoint_mode=ViewpointMode.EXHAUSTIVE))


def test_exhaustive_mode_reaches_viewpoints(sim, explored, detector):
    agent, states = explored
    prediction_tiles, fused, memory = _fuse_predictions(sim, states, detector)
    _save_maps_and_predictions(agent, prediction_tiles)
    _save_resolved_maps(sim, agent, fused, memory, detector)
    assert any(seen.any() for seen in agent.seen_sectors.values()), f"no object viewpoint reached in {_MAX_STEPS} steps"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
