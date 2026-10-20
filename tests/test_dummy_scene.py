"""
Builds a synthetic flat in an empty "NONE" scene: a floor slab, 4 outer walls and inner walls with doorways
splitting it into 4 rooms of different rectangular sizes, all STATIC flat-shaded boxes, plus ProcTHOR-hab objects at
their own scale: 5 distinct chairs, armchairs, sofas and benches in each of 3 rooms, and only a table with 4 bottles on it
in the spawn room. recompute_navmesh carves the walls and objects out of the
floor; the ceiling is added after it, so its top face isn't walkable. Every object gets its own semantic_id,
so the semantic sensor sees them as separate objects.

Runs common/hybrid_exploration's HybridFrontierAgent in exhaustive mode in the flat, fed by an oracle detector:
each visible object's points come from its semantic mask and depth.
Into tests/testdump/test_dummy_scene/ (emptied when the test starts) go every step's decision (current view in
yellow, candidates labelled with u*t above and t below, chosen goal ringed) as one frame of decision.gif, the last step's believed object
occupancy (object points in white, explored area green, the rest gray) to object.png, and every step's
Detic predictions drawn on its rgb as one tile of predictions_mosaic.png. Walls, floor and ceiling are flat-shaded.

Alongside, as in habitat_embodied_sl/main.py, Detic (Swin-B, classifying against chair, couch, bench, table and bottle only)
detects in every frame and SAM masks its boxes, and its masks and region embeddings are fused into a concept-graphs
ConceptGraphMemory (common/utils/concept_graph.py). Once the run ends it is resolved, carved by SAM masks of a few diverse views of each object (ConceptGraphMemory.refine), each object classified by
ConceptGraphMemory.classify from its raytraced views, then drawn by class on the agent's map to semantic_topdown.png
next to the placed objects' mesh silhouettes, saved as semantic_point_cloud.ply with its objects' box wireframes in semantic_wireframes.glb, in ground_truth_topdown.png (both main.py's), scored against the placed objects by
eval_utils.evaluate_scene's 3D box AP and confusion_matrix.png, ground_truth_mesh.glb merges every box and object but the ceiling, and retrieval_mosaic.png gets one row per resolved object of
the diverse remembered views of it common/utils/retrieval.py's retrieve_views selects, retrieval_first_views.png its first column as a single row.

Needs ProcTHOR-hab's objects under HABITAT_DATA/scene_datasets/ai2thor-hab, a GPU, Detic's Swin-B checkpoint
under third_party/Detic/models and SAM's ViT-B checkpoint under data/sam.
_MASK_SOURCE = "detic" fuses Detic's mask-head masks instead of SAM's.
"""

import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

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
from common.env_utils.object_annotations import object_world_mesh, objects_world_mesh
from common.env_utils.vocab_constants import make_colors
from common.utils.concept_graph import ConceptGraphMemory
from common.utils.eval_utils import GroundTruthObjects, evaluate_scene, save_resolved_wireframes, scene_confusion_matrix
from common.utils.image_memory import Frame
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import DepthCamera, downsample_points, object_points, save_resolved_ply
from common.utils.sam_refinement import RefinementConfig
from common.utils.retrieval import SegmentationStyle, ViewSelection, first_views_mosaic, retrieve_views, segmentation_tile, views_mosaic
from detector import Detector
from habitat_embodied_sl.main import (
    ground_truth_topdown, predicted_objects, save_exploration_visuals, semantic_topdown,
)

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
# (semantic_id, ground-truth class, ProcTHOR-hab object, floor center xz): 16 models at their own scale, one per
# model family; sofas and benches stand along walls, clear of the doorways
_WALL_OBJECT = (13, "chair", "Chair_007_4", (-7.68, 3.0))  # 0.1 m off the outer wall x = -8
_TABLE = (14, "table", "Dining_Table_205_1", (3.0, -3.0))
# on _TABLE's top, 0.25 m off its center along both axes; the last one lies on its side
_TABLETOP_OBJECTS = [
    (15, "bottle", "Wine_Bottle_1", (2.75, -3.25)),
    (19, "bottle", "Spray_Bottle_1", (3.25, -3.25)),
    (20, "bottle", "Soap_Bottle_10", (2.75, -2.75)),
    (21, "bottle", "FloorPlan28_physics-Bottle", (3.25, -2.75)),
]
_FLOOR_OBJECTS = [
    _WALL_OBJECT,
    (16, "couch", "Sofa_221_1", (-5.0, -5.5)),
    (17, "bench", "FloorPlan527_physics-Bench_4", (-5.0, 5.4)),
    (11, "chair", "Armchair_204_1", (-4.0, -2.0)),
    (18, "chair", "Chair_002_1", (-3.5, 1.0)),
    _TABLE,
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
_OBJECTS = [*_FLOOR_OBJECTS, *_TABLETOP_OBJECTS]
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DETIC_ROOT = os.path.join(_REPO_ROOT, "third_party", "Detic")
_DETIC_CONFIG = os.path.join(_DETIC_ROOT, "configs", "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
_DETIC_CHECKPOINT = os.path.join(_DETIC_ROOT, "models", "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
_DETIC_CLASSES = ["chair", "couch", "bench", "table", "bottle", "object"]
_SAM_CHECKPOINT = os.path.join(_REPO_ROOT, "data", "sam", "sam_vit_b_01ec64.pth")
_MASK_SOURCE = "sam"

pytestmark = [
    pytest.mark.skipif(not os.path.isdir(_PROCTHOR_OBJECTS_DIR), reason="requires HABITAT_DATA pointing at a real ai2thor-hab dataset (see INSTALL.MD)"),
    pytest.mark.skipif(not os.path.exists(_DETIC_CHECKPOINT) or not torch.cuda.is_available(), reason="requires a GPU and Detic's Swin-B checkpoint"),
    pytest.mark.skipif(not os.path.exists(_SAM_CHECKPOINT), reason="requires SAM's ViT-B checkpoint"),
]
_CAMERA_HEIGHT = 1.0
_RESOLUTION = [640, 640]
_MAX_DEPTH_M = 2 * max(_ROOM_HALF_EXTENTS)
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_dummy_scene")
_DEPTH_CAMERA = DepthCamera(hfov=90.0, min_depth=0.1, max_depth=_MAX_DEPTH_M)
_FULL_CONFIDENCE_AREA = 0.05
_MIN_VISIBLE_PIXELS = 125
_POINTS_VOXEL_M = 0.05
_TURN_ANGLE_DEG = 45.0
_START = (6.0, 0.0, -4.0)
_MAX_STEPS = 500
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


def _add_procthor_object(sim: habitat_sim.Simulator, name: str, center: tuple, semantic_id: int, base_y: float = 0.0):
    """Static ProcTHOR-hab object at its own scale, its bounding box standing at height base_y around center."""
    config = os.path.join(_PROCTHOR_OBJECTS_DIR, f"{name}.object_config.json")
    template_id = sim.get_object_template_manager().load_configs(config)[0]
    obj = sim.get_rigid_object_manager().add_object_by_template_id(template_id)
    box = obj.root_scene_node.cumulative_bb
    obj.translation = mn.Vector3(center[0] - box.center().x, base_y - box.min.y, center[1] - box.center().z)
    for node in obj.visual_scene_nodes:
        node.semantic_id = semantic_id
    obj.motion_type = habitat_sim.physics.MotionType.STATIC
    return obj


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
    placed = {name: _add_procthor_object(sim, name, center, semantic_id) for semantic_id, _, name, center in _FLOOR_OBJECTS}
    table = placed[_TABLE[2]]
    table_top_y = table.translation.y + table.root_scene_node.cumulative_bb.max.y
    for semantic_id, _, name, center in _TABLETOP_OBJECTS:
        _add_procthor_object(sim, name, center, semantic_id, table_top_y)


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


def _start_agent(sim: habitat_sim.Simulator, config: HybridConfig) -> HybridFrontierAgent:
    """HybridFrontierAgent of a run starting at _START."""
    start = habitat_sim.AgentState()
    start.position = np.array(_START)
    sim.get_agent(0).set_state(start)
    return HybridFrontierAgent(sim.pathfinder, config, np.random.default_rng(0))


def _explore_and_fuse(sim: habitat_sim.Simulator, config: HybridConfig, detector: Detector) -> tuple:
    """(agent, every step's decision map tile, concept graph fused from every step's predictions and frames, every step's
    predictions) of one HybridFrontierAgent run steered by oracle beliefs."""
    sim_agent, agent = sim.get_agent(0), _start_agent(sim, config)
    beliefs, decision_tiles, fused, predictions = {}, [], ConceptGraphMemory(), []
    for _ in tqdm(range(_MAX_STEPS)):
        state, obs = sim_agent.get_state(), sim.get_sensor_observations()
        frame = Frame(obs["rgb"][:, :, :3], obs["depth"], sensor_pose_matrix(state.sensor_states["depth"]))
        fused.image_memory.add_frame(frame)
        predictions.append(detector.predict(frame.rgb))
        fused = fused + predicted_objects(frame, predictions[-1], _DEPTH_CAMERA, state.position[1])
        beliefs = _oracle_beliefs(beliefs, obs, frame.pose)
        agent.update_objects(beliefs)
        action = agent.act(state)
        decision_tiles.append(agent.map_tile())
        if action == "stop":
            break
        sim.step(action)
    return agent, decision_tiles, fused, predictions


def _prediction_tiles(fused: ConceptGraphMemory, predictions: list) -> list:
    """Every step's labelled detector predictions drawn on its rgb."""
    return [
        segmentation_tile(fused.image_memory.get_frame(step).rgb, instances, _DETIC_STYLE, f"step{step:03d}")
        for step, instances in enumerate(predictions)
    ]


def _placed_object(sim: habitat_sim.Simulator, name: str):
    return next(iter(sim.get_rigid_object_manager().get_objects_by_handle_substring(name).values()))


def _ground_truth_meshes(sim: habitat_sim.Simulator) -> list:
    """(class, world mesh) of every placed object, as main.py's ground_truth_meshes."""
    class_ids = {name: idx for idx, name in enumerate(_DETIC_CLASSES)}
    return [(class_ids[category], object_world_mesh(_placed_object(sim, name))) for _, category, name, _ in _OBJECTS]


@contextmanager
def _timed(phase: str):
    print(f"{phase}...")
    start = time.perf_counter()
    yield
    print(f"{phase} done in {time.perf_counter() - start:.1f}s")


def _resolve_and_classify(fused: ConceptGraphMemory, detector: Detector) -> None:
    """Resolves, refines and classifies fused as main.py does."""
    with _timed("Resolving"):
        fused.resolve()
    if detector.masker is not None:
        with _timed("Refining"):
            fused.refine(detector.masker, _DEPTH_CAMERA, RefinementConfig(min_visibility_fraction=_MIN_VISIBILITY_FRACTION))
    with _timed("Classifying"):
        fused.classify(detector, _DEPTH_CAMERA, _MIN_VISIBILITY_FRACTION)


def _save_resolved_maps(sim: habitat_sim.Simulator, agent: HybridFrontierAgent, fused: ConceptGraphMemory) -> None:
    """Evaluates fused's resolved objects against the placed objects, then saves their semantic_point_cloud.ply,
    semantic_wireframes.glb, semantic_topdown.png, ground_truth_topdown.png and confusion_matrix.png."""
    resolved, colors = fused.resolved_objects, _DETIC_STYLE.colors
    with _timed("Saving point cloud"):
        save_resolved_ply(resolved, colors, os.path.join(_TESTDUMP_DIR, "semantic_point_cloud.ply"))
        save_resolved_wireframes(resolved, _DETIC_CLASSES, colors, os.path.join(_TESTDUMP_DIR, "semantic_wireframes.glb"))
    meshes = _ground_truth_meshes(sim)
    with _timed("Evaluating"):
        ground_truth = GroundTruthObjects.from_meshes(meshes)
        evaluate_scene(resolved, ground_truth, _DETIC_CLASSES)
        scene_confusion_matrix(resolved, ground_truth, _DETIC_CLASSES, os.path.join(_TESTDUMP_DIR, "confusion_matrix.png"))
    with _timed("Saving topdown maps"):
        Image.fromarray(semantic_topdown(resolved, agent, _DETIC_CLASSES, colors)).save(os.path.join(_TESTDUMP_DIR, "semantic_topdown.png"))
        Image.fromarray(ground_truth_topdown(meshes, agent, _DETIC_CLASSES, colors)).save(os.path.join(_TESTDUMP_DIR, "ground_truth_topdown.png"))


def _save_retrieval_mosaics(fused: ConceptGraphMemory) -> None:
    """retrieval_mosaic.png (one row per object of the views retrieve_views selects) and retrieval_first_views.png (its
    first column as a single row)."""
    resolved, memory = fused.resolved_objects, fused.image_memory
    raytraced = memory.raytrace_frames(resolved, _DEPTH_CAMERA, _MIN_VISIBILITY_FRACTION)
    object_frames = retrieve_views(resolved, memory, raytraced, _DEPTH_CAMERA, ViewSelection())
    views_mosaic(memory, raytraced, _DETIC_STYLE, object_frames).save(os.path.join(_TESTDUMP_DIR, "retrieval_mosaic.png"))
    first_views_mosaic(memory, raytraced, _DETIC_STYLE, object_frames).save(os.path.join(_TESTDUMP_DIR, "retrieval_first_views.png"))


def _save_ground_truth_mesh(sim: habitat_sim.Simulator) -> None:
    """ground_truth_mesh.glb merging the floor, walls and placed objects, without the ceiling."""
    objects = sim.get_rigid_object_manager().get_objects_by_handle_substring().values()
    objects_world_mesh(obj for obj in objects if obj.semantic_id != _CEILING_ID).export(os.path.join(_TESTDUMP_DIR, "ground_truth_mesh.glb"))


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
    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
    _build_room(sim, assets)
    nav_settings = habitat_sim.NavMeshSettings()
    nav_settings.set_defaults()
    nav_settings.include_static_objects = True
    assert sim.recompute_navmesh(sim.pathfinder, nav_settings)
    _add_ceiling(sim, assets)  # after the navmesh, so its top face isn't walkable
    return sim


def test_exhaustive_mode_reaches_viewpoints():
    shutil.rmtree(_TESTDUMP_DIR, ignore_errors=True)
    os.makedirs(_TESTDUMP_DIR)

    with tempfile.TemporaryDirectory() as assets_dir:
        sim = _dummy_sim(assets_dir)
        _save_ground_truth_mesh(sim)  # inside, since it reloads the box assets

    with _timed("Model loading"):
        detector = Detector(_DETIC_CONFIG, _DETIC_CHECKPOINT, _DETIC_CLASSES,
            sam_checkpoint=_SAM_CHECKPOINT if _MASK_SOURCE == "sam" else None, score_thresh=0.5, agnostic_nms_thresh=0.5, mask_thresh=0.5, objectness_weight=0.75)
    config = HybridConfig(MapConfig(fov_deg=90.0, resolution=512), _TURN_ANGLE_DEG, viewpoint_mode=ViewpointMode.EXHAUSTIVE)
    with _timed("Exploring and fusing"):
        agent, decision_tiles, fused, predictions = _explore_and_fuse(sim, config, detector)

    with _timed("Visualisation"):
        save_exploration_visuals(agent, decision_tiles, _prediction_tiles(fused, predictions), Path(_TESTDUMP_DIR))
    _resolve_and_classify(fused, detector)
    _save_resolved_maps(sim, agent, fused)
    with _timed("Retrieval mosaics"):
        _save_retrieval_mosaics(fused)
    sim.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
