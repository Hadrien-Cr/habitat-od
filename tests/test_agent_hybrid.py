"""
Runs common/hybrid_exploration's HybridFrontierAgent in exhaustive mode in dummy_scene.py's synthetic flat, fed by an
oracle detector: each visible object's points come from its semantic mask and depth.
Into tests/testdump/test_agent_hybrid/ (emptied when the test starts) go the last step's decision (current view in
yellow, candidates labelled with u*t above and t below, chosen goal ringed) as decision.png, the last step's believed object
occupancy (object points in white, explored area green, the rest gray) to object.png, every step's
class-agnostic predictions drawn on its rgb as one tile of predictions_mosaic.png and individually as
prediction_frames/step_<step>.png, every step's 3-tile (observation, belief map, ground truth) snapshot to
images/test_exploration_gif/snapshot_<step>.png and assembled into exploration.webp, and every fused object's center
and size at every step, then every classified object's, to object_tracks.csv. Walls, floor and ceiling are flat-shaded.
The run asserts it never got stuck (no net translation over _STUCK_WINDOW_STEPS consecutive steps).

Alongside, as in habitat_embodied_sl/main.py, a Detector (Detic Swin-B's proposals scored by objectness alone) detects
in every frame and SAM masks its boxes, and its masks and region embeddings are fused into a ConceptGraphMemory
(common/utils/concept_graph.py).
Once the run ends it is resolved, only its foreground objects kept (ConceptGraphMemory.keep_foreground), each classified
among _DUMMY_VOCABULARY's chair, couch, bench, table, bottle and object by ConceptGraphMemory.classify from its raytraced
views, only among the classes its size fits as in main.py (sizes of the PROCTHOR100 classes
_PROCTHOR_SIZE_CLASSES maps the classes to), then drawn by class on the agent's map to estimated_topdown.png next to the
placed objects' mesh silhouettes, saved as semantic_point_cloud.ply with its objects' box wireframes in
semantic_wireframes.glb, in ground_truth_topdown.png (both eval_utils.py's save_maps), scored against the placed objects by eval_utils.py's
evaluate_resolved (3D box AP, confusion_matrix.png, class-agnostic matching), ground_truth_mesh.glb merges every box and
object but the ceiling, and retrieval_mosaic.png gets one row per resolved object of the diverse remembered views of it
common/utils/retrieval.py's retrieve_views selects, retrieval_first_views.png its first column as a single row.
Those views are written as train/train.json (common/utils/dataset_writers.py's ObjectViewsWriter, with train_mosaic.png), read
back as a common/utils/object_views_dataset.py ObjectViewsDataset, of which _MOSAIC_BATCHES class-balanced batches are
drawn one per row into dataset_mosaic.png. _LABEL_SOURCE = "gt" first relabels every resolved object matching a placed
object (main.py's class-agnostic matching) with its ground-truth class and drops the others, after evaluation; "pseudo"
keeps ConceptGraphMemory.classify's.
Every RNG is seeded with _SEED.

Needs ProcTHOR-hab's objects under HABITAT_DATA/scene_datasets/ai2thor-hab, a GPU, Detic's Swin-B checkpoint
under third_party/Detic/models and SAM's ViT-B checkpoint under data/sam.
_MASK_SOURCE = "detic" fuses Detic's mask-head masks instead of SAM's.
"""

import itertools
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import habitat_sim
import numpy as np
import pandas as pd
import pytest
import torch
from detectron2.data import DatasetMapper
from detectron2.utils.env import seed_all_rng
from habitat.tasks.nav.nav import TopDownMap
from tqdm import tqdm

from common.hybrid_exploration.frontier_exp_agent import exploration_snapshot, save_exploration_visuals
from common.hybrid_exploration.ground_truth_map import GroundTruthMapTracker
from common.hybrid_exploration.hybrid_agent import HybridConfig, HybridFrontierAgent, ViewpointMode
from common.hybrid_exploration.object_viewpoints import ObjectBelief
from common.env_utils.object_annotations import PROCTHOR100_SIZES_PATH, class_size_ranges, objects_world_mesh
from common.env_utils.vocab_constants import make_colors
from common.utils.eval_utils import GroundTruthObjects, matched_ground_truth_classes
from common.utils.image_memory import Frame
from common.utils.plot_utils import save_gif
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.eval_utils import evaluate_resolved, save_maps
from common.utils.projection_utils import downsample_points, object_points, resolved_object_rows
from common.utils.dataset_writers import ObjectViewsWriter
from common.utils.object_views_dataset import ObjectViewsDataset, balanced_loader, batches_mosaic
from common.utils.retrieval import SegmentationStyle, ViewSelection, first_views_mosaic, retrieved_views, segmentation_tile, views_mosaic, retrieve_resolved_views
from common.vision.detic import Vocabulary
from common.utils.concept_graph import ConceptGraphMemory, fused_object_rows, object_size_ranges, predicted_objects
from common.vision.detector import Detector
from dummy_scene import _CEILING_ID, _DEPTH_CAMERA, _DETIC_CLASSES, _MAP_CONFIG, _OBJECTS, _PROCTHOR_OBJECTS_DIR, _START, _TURN_ANGLE_DEG, _dummy_sim, _ground_truth_meshes, _navmesh_floor_offset

_PROCTHOR_SIZE_CLASSES = {
    "chair": ("chair", "arm chair"), "couch": ("sofa",), "table": ("dining table",), "bottle": ("bottle", "wine bottle", "spray bottle", "soap bottle"),
    "bed": ("bed",), "pillow": ("pillow",),
}
_DUMMY_VOCABULARY = Vocabulary("dummy_voc", tuple(_DETIC_CLASSES))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DETIC_ROOT = os.path.join(_REPO_ROOT, "third_party", "Detic")
_DETIC_CONFIG = os.path.join(_DETIC_ROOT, "configs", "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
_DETIC_CHECKPOINT = os.path.join(_DETIC_ROOT, "models", "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
_SAM_CHECKPOINT = os.path.join(_REPO_ROOT, "data", "sam", "sam_vit_b_01ec64.pth")
_MASK_SOURCE = "sam"

pytestmark = [
    pytest.mark.skipif(not os.path.isdir(_PROCTHOR_OBJECTS_DIR), reason="requires HABITAT_DATA pointing at a real ai2thor-hab dataset (see INSTALL.MD)"),
    pytest.mark.skipif(not os.path.exists(_DETIC_CHECKPOINT) or not torch.cuda.is_available(), reason="requires a GPU and Detic's Swin-B checkpoint"),
    pytest.mark.skipif(not os.path.exists(_SAM_CHECKPOINT), reason="requires SAM's ViT-B checkpoint"),
]
_TESTDUMP_DIR = os.path.join(os.path.dirname(__file__), "testdump", "test_agent_hybrid")
_FULL_CONFIDENCE_AREA = 0.05
_MIN_VISIBLE_PIXELS = 125
_POINTS_VOXEL_M = 0.05
_MAX_STEPS = 200
_MIN_MESH_VISIBILITY_FRACTION = 0.2
_DETIC_STYLE = SegmentationStyle(_DETIC_CLASSES, make_colors(len(_DETIC_CLASSES), seed=0, ctype=0))
_DETECTION_STYLE = SegmentationStyle(["object"], make_colors(1, seed=0, ctype=0))
_SEED = 0
_LABEL_SOURCE = "pseudo"
_BATCH_SIZE = 8
_MOSAIC_BATCHES = 20
_GIF_FPS = 6
_STUCK_WINDOW_STEPS = 15
_STUCK_RADIUS_M = 0.3


def _is_stuck(trajectory: list, window: int, radius_m: float) -> bool:
    """True once the agent's last window positions all lie within radius_m of each other -- no net
    translation, even while its heading keeps oscillating between turns."""
    if len(trajectory) < window:
        return False
    recent = np.stack([position for position, _ in trajectory[-window:]])
    return bool(np.all(np.linalg.norm(recent - recent[0], axis=1) < radius_m))


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


def _start_agent(sim: habitat_sim.Simulator, config: HybridConfig, camera) -> HybridFrontierAgent:
    """HybridFrontierAgent of a run starting at _START."""
    start = habitat_sim.AgentState()
    start.position = np.array(_START)
    sim.get_agent(0).set_state(start)
    agent = HybridFrontierAgent(config, camera, np.random.default_rng(_SEED), sim.pathfinder)
    agent.reset(np.array(_START), _START[1] - _navmesh_floor_offset(sim, np.array(_START)))
    return agent


def _explore_and_fuse(sim: habitat_sim.Simulator, config: HybridConfig, detector: Detector, images_dir: Path) -> tuple:
    """(agent, concept graph fused from every step's predictions and frames, every step's predictions, every step's
    fused object rows, whether the agent got stuck) of one HybridFrontierAgent run steered by oracle beliefs, saving
    a 3-tile (observation, belief map, ground truth) snapshot per step to images_dir/snapshot_<step>.png."""
    sim_agent, agent = sim.get_agent(0), _start_agent(sim, config, _DEPTH_CAMERA)
    ground_truth = GroundTruthMapTracker(sim.pathfinder, np.array(_START), fov_deg=_MAP_CONFIG.fov_deg, visibility_dist_m=_MAP_CONFIG.visibility_dist_m)
    ground_truth.set_object_footprints(_ground_truth_meshes(sim))
    trajectory_px: list = []
    beliefs, fused, predictions, object_tracks = {}, ConceptGraphMemory(), [], []
    stuck = False
    for step in tqdm(range(_MAX_STEPS)):
        state, obs = sim_agent.get_state(), sim.get_sensor_observations()
        heading = float(TopDownMap.get_polar_angle(state))
        trajectory_px.append(ground_truth.to_pixel(np.array(state.position)))
        ground_truth.reveal(np.array(state.position), heading)
        mosaic = exploration_snapshot(obs, agent, ground_truth, trajectory_px)
        if mosaic is not None:
            mosaic.save(images_dir / f"snapshot_{step:03d}.png")
        frame = Frame(obs["rgb"][:, :, :3], obs["depth"], sensor_pose_matrix(state.sensor_states["depth"]))
        frame_id = fused.image_memory.add_frame(frame)
        predictions.append(detector.predict(frame.rgb))
        fused = fused + predicted_objects(frame_id, frame, predictions[-1], _DEPTH_CAMERA, state.position[1])
        object_tracks += fused_object_rows(f"step{step:03d}", fused)
        beliefs = _oracle_beliefs(beliefs, obs, frame.pose)
        agent.update_objects(beliefs)
        action = agent.act(state, obs["depth"])
        stuck = _is_stuck(agent.trajectory, _STUCK_WINDOW_STEPS, _STUCK_RADIUS_M)
        if action == "stop" or stuck:
            break
        sim.step(action)
    return agent, fused, predictions, object_tracks, stuck


def _prediction_tiles(fused: ConceptGraphMemory, predictions: list) -> list:
    """Every step's labelled detector predictions drawn on its rgb."""
    return [
        segmentation_tile(fused.image_memory.get_frame(step).rgb, instances, _DETECTION_STYLE, f"step{step:03d}")
        for step, instances in enumerate(predictions)
    ]


@contextmanager
def _timed(phase: str):
    print(f"{phase}...")
    start = time.perf_counter()
    yield
    print(f"{phase} done in {time.perf_counter() - start:.1f}s")


def _resolve_and_classify(fused: ConceptGraphMemory, detector: Detector) -> None:
    """Resolves fused, keeps its foreground objects and classifies them among the classes their size fits, as main.py does."""
    with _timed("Resolving"):
        fused.resolve()
    with _timed("Keeping foreground"):
        fused.keep_foreground(_DEPTH_CAMERA, _MIN_MESH_VISIBILITY_FRACTION)
    with _timed("Classifying"):
        fused.classify(detector, _DEPTH_CAMERA, _MIN_MESH_VISIBILITY_FRACTION, object_size_ranges(_DETIC_CLASSES, _dummy_class_size_ranges()))


def _dummy_class_size_ranges() -> dict:
    """{class: {size measure: (min, max)} spanning every PROCTHOR100 class _PROCTHOR_SIZE_CLASSES maps it to}."""
    native_ranges = class_size_ranges(PROCTHOR100_SIZES_PATH)
    return {name: _spanning_ranges([native_ranges[native] for native in natives]) for name, natives in _PROCTHOR_SIZE_CLASSES.items()}


def _spanning_ranges(class_ranges: list) -> dict:
    """{size measure: (smallest min, largest max)} over class_ranges."""
    return {
        measure: (min(ranges[measure][0] for ranges in class_ranges), max(ranges[measure][1] for ranges in class_ranges))
        for measure in class_ranges[0]
    }


def _save_resolved_maps(sim: habitat_sim.Simulator, agent: HybridFrontierAgent, fused: ConceptGraphMemory) -> None:
    """eval_utils.py's maps of fused's resolved objects and of the placed objects, then its evaluation against the placed objects."""
    meshes = _ground_truth_meshes(sim)
    with _timed("Saving maps"):
        save_maps(fused.resolved_objects, meshes, agent, _DETIC_STYLE, Path(_TESTDUMP_DIR))
    with _timed("Evaluating"):
        evaluate_resolved(fused.resolved_objects, meshes, _DETIC_CLASSES, Path(_TESTDUMP_DIR))


def _relabel_with_ground_truth(sim: habitat_sim.Simulator, fused: ConceptGraphMemory) -> None:
    """Gives every resolved object matching a placed object that object's ground-truth class, dropping the others."""
    ground_truth = GroundTruthObjects.from_meshes(_ground_truth_meshes(sim))
    fused.resolved_objects.relabeled(matched_ground_truth_classes(fused.resolved_objects, ground_truth))


def _save_retrieval_mosaics(fused: ConceptGraphMemory, raytraced: dict, object_frames: dict) -> None:
    """retrieval_mosaic.png (one row per object of the views retrieve_views selects) and retrieval_first_views.png (its
    first column as a single row)."""
    memory = fused.image_memory
    views_mosaic(memory, raytraced, _DETIC_STYLE, object_frames).save(os.path.join(_TESTDUMP_DIR, "retrieval_mosaic.png"))
    first_views = first_views_mosaic(retrieved_views(fused.resolved_objects, memory, raytraced, object_frames), _DETIC_STYLE)
    first_views.save(os.path.join(_TESTDUMP_DIR, "retrieval_first_views.png"))


def _save_dataset_mosaic(fused: ConceptGraphMemory, raytraced: dict, object_frames: dict) -> None:
    """train/train.json of the retrieved views, then dataset_mosaic.png, one row per batch of the first _MOSAIC_BATCHES the
    balanced_loader of the ObjectViewsDataset read back from it draws."""
    writer = ObjectViewsWriter(Path(_TESTDUMP_DIR) / "train", {"categories": _DETIC_CLASSES})
    writer.add_scene("dummy", retrieved_views(fused.resolved_objects, fused.image_memory, raytraced, object_frames))
    writer.write(np.random.default_rng(_SEED))
    mapper = DatasetMapper(is_train=True, augmentations=[], image_format="RGB", use_instance_mask=True, instance_mask_format="bitmask")
    loader = balanced_loader(ObjectViewsDataset.from_json(Path(_TESTDUMP_DIR) / "train" / "train.json"), _BATCH_SIZE, mapper)
    batches = itertools.islice(itertools.chain.from_iterable(itertools.repeat(loader)), _MOSAIC_BATCHES)
    batches_mosaic(batches, _DETIC_STYLE).save(os.path.join(_TESTDUMP_DIR, "dataset_mosaic.png"))


def _save_ground_truth_mesh(sim: habitat_sim.Simulator) -> None:
    """ground_truth_mesh.glb merging the floor, walls and placed objects, without the ceiling."""
    objects = sim.get_rigid_object_manager().get_objects_by_handle_substring().values()
    objects_world_mesh(obj for obj in objects if obj.semantic_id != _CEILING_ID).export(os.path.join(_TESTDUMP_DIR, "ground_truth_mesh.glb"))


def test_exhaustive_mode_reaches_viewpoints():
    seed_all_rng(_SEED)
    shutil.rmtree(_TESTDUMP_DIR, ignore_errors=True)
    os.makedirs(_TESTDUMP_DIR)

    with tempfile.TemporaryDirectory() as assets_dir:
        sim = _dummy_sim(assets_dir)
        _save_ground_truth_mesh(sim)  # inside, since it reloads the box assets

    with _timed("Model loading"):
        detector = Detector(_DETIC_CONFIG, _DETIC_CHECKPOINT, _DUMMY_VOCABULARY,
            sam_checkpoint=_SAM_CHECKPOINT if _MASK_SOURCE == "sam" else None, score_thresh=0.5, nms_thresh=0.5, mask_thresh=0.5,
            input_size=1024)
    config = HybridConfig(_MAP_CONFIG, _TURN_ANGLE_DEG, viewpoint_mode=ViewpointMode.EXHAUSTIVE)
    images_dir = Path(_TESTDUMP_DIR) / "images" / "test_exploration_gif"
    images_dir.mkdir(parents=True)
    with _timed("Exploring and fusing"):
        agent, fused, predictions, object_tracks, stuck = _explore_and_fuse(sim, config, detector, images_dir)
    assert not stuck, f"agent got stuck (no net translation over the last {_STUCK_WINDOW_STEPS} steps)"
    save_gif(images_dir, "exploration.webp", fps=_GIF_FPS)

    with _timed("Visualisation"):
        save_exploration_visuals(agent, _prediction_tiles(fused, predictions), Path(_TESTDUMP_DIR))
    _resolve_and_classify(fused, detector)
    pd.DataFrame(object_tracks + resolved_object_rows("classified", fused.resolved_objects)).to_csv(
        os.path.join(_TESTDUMP_DIR, "object_tracks.csv"), index=False,
    )
    _save_resolved_maps(sim, agent, fused)
    if _LABEL_SOURCE == "gt":
        _relabel_with_ground_truth(sim, fused)
    with _timed("Retrieval mosaics"):
        raytraced, object_frames = retrieve_resolved_views(fused, _DEPTH_CAMERA, _MIN_MESH_VISIBILITY_FRACTION, ViewSelection())
        _save_retrieval_mosaics(fused, raytraced, object_frames)
        _save_dataset_mosaic(fused, raytraced, object_frames)
    sim.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
