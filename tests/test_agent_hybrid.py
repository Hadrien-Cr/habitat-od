"""
Agent should satisfy stop conditions.
Runs common/hybrid_exploration's HybridFrontierAgent in exhaustive mode in procthor_scene.py's ProcTHOR-Train-0 from its
start_state, steered as in habitat_embodied_sl/main.py by its own ObjectTracker, fed every step's Detector predictions
lifted by concept_graph.py's predicted_objects.
Into tests/testdump/test_agent_hybrid/ (emptied when the test starts) go every step's class-agnostic predictions
drawn on its rgb as one tile of predictions_mosaic.png and individually as prediction_frames/step_<step>.png, and
every step's 3-tile (observation, belief map, ground truth) snapshot to
images/test_exploration_gif/snapshot_<step>.png and assembled into exploration.webp; ground_truth_mesh.glb merges the
scene's stage, without its ceilings, and every placed object.
The run asserts it never got stuck (no net translation, BaseExplorationAgent.is_stuck).

Alongside, as in habitat_embodied_sl/main.py, a Detector (Detic Swin-B's proposals scored by objectness alone) detects
in every frame and SAM masks its boxes, and its masks and region embeddings are fused into a ConceptGraphMemory
(common/utils/concept_graph.py).
Once the run ends it is resolved, only its foreground objects kept (ConceptGraphMemory.keep_foreground), each classified
among the PROCTHOR100 classes by ConceptGraphMemory.classify from its raytraced views, only among the classes its size
fits as in main.py, then drawn by class on the agent's map to estimated_topdown.png next to the labelled objects' mesh
silhouettes, saved as semantic_point_cloud.ply with its objects' box wireframes in semantic_wireframes.glb, in
ground_truth_topdown.png (both eval_utils.py's save_maps), scored against the labelled objects by eval_utils.py's
evaluate_resolved (3D box AP, confusion_matrix.png, class-agnostic matching), and retrieval_mosaic.png gets one row per
resolved object of the diverse remembered views of it common/utils/retrieval.py's retrieve_views selects,
retrieval_first_views.png its first column as a single row.
Those views are written as train/train.json (common/utils/dataset_writers.py's ObjectViewsWriter, with train_mosaic.png), read
back as a common/utils/object_views_dataset.py ObjectViewsDataset, of which _MOSAIC_BATCHES class-balanced batches are
drawn one per row into dataset_mosaic.png. _LABEL_SOURCE = "gt" first relabels every resolved object matching a labelled
object (main.py's class-agnostic matching) with its ground-truth class and drops the others, after evaluation; "pseudo"
keeps ConceptGraphMemory.classify's.
test_agent_hybrid_label_propagation.py reruns the same exploration into a LabelPropagationPointCloud instead.
Every RNG is seeded with _SEED.

Needs ProcTHOR-hab under HABITAT_DATA/scene_datasets/ai2thor-hab, a GPU, Detic's Swin-B checkpoint
under third_party/Detic/models and SAM's ViT-B checkpoint under data/sam.
_MASK_SOURCE = "detic" fuses Detic's mask-head masks instead of SAM's.
"""

import itertools
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Tuple

import habitat_sim
import numpy as np
import pytest
import torch
from detectron2.data import DatasetMapper
from detectron2.utils.env import seed_all_rng
from habitat.tasks.nav.nav import TopDownMap
from tqdm import tqdm

from common.hybrid_exploration.frontier_exp_agent import DiscretePlannerConfig, exploration_snapshot, save_exploration_visuals
from common.hybrid_exploration.ground_truth_map import GroundTruthMapTracker
from common.hybrid_exploration.hybrid_agent import HybridConfig, HybridFrontierAgent
from common.env_utils.object_annotations import PROCTHOR100_SIZES_PATH, ObjectAnnotation, annotated_object_meshes, class_size_ranges, scene_world_mesh
from common.env_utils.vocab_constants import make_colors
from common.utils.eval_utils import GroundTruthObjects, evaluate_resolved, matched_ground_truth_classes, save_maps
from common.utils.image_memory import Frame, ImageMemory
from common.utils.plot_utils import save_gif
from common.utils.pose_utils import sensor_pose_matrix
from common.utils.projection_utils import ResolvedObjects
from common.utils.dataset_writers import ObjectViewsWriter
from common.utils.object_views_dataset import ObjectViewsDataset, balanced_loader, batches_mosaic
from common.utils.retrieval import SegmentationStyle, ViewSelection, first_views_mosaic, retrieved_views, segmentation_tile, views_mosaic, retrieve_resolved_views
from common.vision.detic import Vocabulary
from common.utils.concept_graph import ConceptGraphMemory, object_size_ranges, predicted_objects
from common.vision.detector import Detector
from procthor_scene import CLASSES, DATASET_CONFIG, DEPTH_CAMERA, MAP_CONFIG, TURN_ANGLE_DEG, VOCAB_NAME, floor_height, procthor_sim, start_state

_VOCABULARY = Vocabulary(f"{VOCAB_NAME}_kept", tuple(CLASSES))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DETIC_ROOT = os.path.join(_REPO_ROOT, "third_party", "Detic")
_DETIC_CONFIG = os.path.join(_DETIC_ROOT, "configs", "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
_DETIC_CHECKPOINT = os.path.join(_DETIC_ROOT, "models", "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
_SAM_CHECKPOINT = os.path.join(_REPO_ROOT, "data", "sam", "sam_vit_b_01ec64.pth")
_MASK_SOURCE = "sam"

pytestmark = [
    pytest.mark.skipif(not os.path.exists(DATASET_CONFIG), reason="requires HABITAT_DATA pointing at a real ai2thor-hab dataset (see INSTALL.MD)"),
    pytest.mark.skipif(not os.path.exists(_DETIC_CHECKPOINT) or not torch.cuda.is_available(), reason="requires a GPU and Detic's Swin-B checkpoint"),
    pytest.mark.skipif(not os.path.exists(_SAM_CHECKPOINT), reason="requires SAM's ViT-B checkpoint"),
]
_TESTDUMP_DIR = Path(__file__).parent / "testdump" / "test_agent_hybrid"
_EXPLORATION_STEPS = 1000
_MIN_MESH_VISIBILITY_FRACTION = 0.2
_DETIC_STYLE = SegmentationStyle(CLASSES, make_colors(len(CLASSES), seed=0, ctype=0))
_DETECTION_STYLE = SegmentationStyle(["object"], make_colors(1, seed=0, ctype=0))
_SEED = 0
_LABEL_SOURCE = "pseudo"
_BATCH_SIZE = 8
_MOSAIC_BATCHES = 20
_GIF_FPS = 10


@dataclass
class HybridRun:
    """One exploration: its simulator and annotation, agent, detector, testdump folder, every frame and every
    step's predictions."""

    sim: habitat_sim.Simulator
    annotation: ObjectAnnotation
    agent: HybridFrontierAgent
    detector: Detector
    testdump_dir: Path
    memory: ImageMemory = field(default_factory=ImageMemory)
    predictions: list = field(default_factory=list)


@contextmanager
def timed(phase: str):
    print(f"{phase}...")
    start = time.perf_counter()
    yield
    print(f"{phase} done in {time.perf_counter() - start:.1f}s")


def build_detector() -> Detector:
    with timed("Model loading"):
        return Detector(_DETIC_CONFIG, _DETIC_CHECKPOINT, _VOCABULARY,
            sam_checkpoint=_SAM_CHECKPOINT if _MASK_SOURCE == "sam" else None, score_thresh=0.5, nms_thresh=0.5, mask_thresh=0.5,
            input_size=1024)


def _start_agent(sim: habitat_sim.Simulator, testdump_dir: Path) -> HybridFrontierAgent:
    """HybridFrontierAgent of a run starting at start_state."""
    start = start_state()
    sim.get_agent(0).set_state(start)
    config = HybridConfig(MAP_CONFIG, TURN_ANGLE_DEG, planner_config=DiscretePlannerConfig(TURN_ANGLE_DEG))
    agent = HybridFrontierAgent(config, DEPTH_CAMERA, np.random.default_rng(_SEED), str(testdump_dir), "test_exploration_gif")
    agent.reset(sim.get_agent(0).get_state(), floor_height(sim, start.position))
    return agent


def _explore(run: HybridRun, fused, lift: Callable, images_dir: Path) -> Tuple[object, bool]:
    """(fused with lift(detections, frame, predictions, floor height) of every step's frame added, whether the agent got
    stuck) of run's agent, its tracker fed every step's detections (predicted_objects), saving a 3-tile (observation,
    belief map, ground truth) snapshot per step to images_dir/snapshot_<step>.png."""
    sim, sim_agent = run.sim, run.sim.get_agent(0)
    start = np.array(sim_agent.get_state().position)
    ground_truth = GroundTruthMapTracker(sim.pathfinder, start, fov_deg=MAP_CONFIG.fov_deg, visibility_dist_m=MAP_CONFIG.visibility_dist_m)
    ground_truth.set_object_footprints(annotated_object_meshes(sim, run.annotation, CLASSES))
    floor, trajectory_px = floor_height(sim, start), []
    for step in tqdm(range(_EXPLORATION_STEPS), desc="Exploration"):
        state, obs = sim_agent.get_state(), sim.get_sensor_observations()
        trajectory_px.append(ground_truth.to_pixel(np.array(state.position)))
        ground_truth.reveal(np.array(state.position), float(TopDownMap.get_polar_angle(state)))
        frame = Frame(obs["rgb"][:, :, :3], obs["depth"], sensor_pose_matrix(state.sensor_states["depth"]))
        frame_id = run.memory.add_frame(frame)
        run.predictions.append(run.detector.predict(frame.rgb))
        detections = predicted_objects(frame_id, frame, run.predictions[-1], DEPTH_CAMERA, floor)
        fused = fused + lift(detections, frame, run.predictions[-1], floor)
        run.agent.update_detections(detections)
        action, _ = run.agent.act(state, obs["depth"])
        mosaic = exploration_snapshot(obs, run.agent, ground_truth, trajectory_px)
        if mosaic is not None:
            mosaic.save(images_dir / f"snapshot_gif_{step:03d}.png")
        if action == "stop":
            return fused, run.agent.is_stuck()
        sim.step(action)
    return fused, False


def _prediction_tiles(run: HybridRun) -> list:
    """Every step's labelled detector predictions drawn on its rgb."""
    return [
        segmentation_tile(run.memory.get_frame(step).rgb, instances, _DETECTION_STYLE, f"step{step:03d}")
        for step, instances in enumerate(run.predictions)
    ]


def explore_and_fuse(detector: Detector, testdump_dir: Path, fused, lift: Callable) -> Tuple[HybridRun, object]:
    """(HybridRun, fused) of an exploration from an emptied testdump_dir, fusing lift(detections, frame, predictions,
    floor height) of every step into fused; saves ground_truth_mesh.glb, exploration.webp and the
    prediction visuals there, and asserts the agent never got stuck."""
    seed_all_rng(_SEED)
    shutil.rmtree(testdump_dir, ignore_errors=True)
    images_dir = testdump_dir / "images" / "test_exploration_gif"
    images_dir.mkdir(parents=True)
    sim, annotation = procthor_sim()
    scene_world_mesh(sim).export(testdump_dir / "ground_truth_mesh.glb")
    run = HybridRun(sim, annotation, _start_agent(sim, testdump_dir), detector, testdump_dir)
    with timed("Exploring and fusing"):
        fused, stuck = _explore(run, fused, lift, images_dir)
    assert not stuck, "agent got stuck (no net translation)"
    save_gif(images_dir, "exploration.webp", fps=_GIF_FPS)
    with timed("Visualisation"):
        save_exploration_visuals(run.agent, _prediction_tiles(run), testdump_dir)
    return run, fused


def _resolve_and_classify(fused: ConceptGraphMemory, detector: Detector) -> None:
    """Resolves fused, keeps its foreground objects and classifies them among the classes their size fits, as main.py does."""
    with timed("Resolving"):
        fused.resolve()
    with timed("Keeping foreground"):
        fused.keep_foreground(DEPTH_CAMERA, _MIN_MESH_VISIBILITY_FRACTION)
    with timed("Classifying"):
        fused.classify(detector, DEPTH_CAMERA, _MIN_MESH_VISIBILITY_FRACTION, object_size_ranges(CLASSES, class_size_ranges(PROCTHOR100_SIZES_PATH)))


def save_resolved_maps(run: HybridRun, resolved: ResolvedObjects) -> None:
    """eval_utils.py's maps of resolved's objects and of the labelled objects, then its evaluation against them, into run's testdump."""
    meshes = annotated_object_meshes(run.sim, run.annotation, CLASSES)
    with timed("Saving maps"):
        save_maps(resolved, meshes, run.agent, _DETIC_STYLE, run.testdump_dir, run.sim.pathfinder)
    with timed("Evaluating"):
        evaluate_resolved(resolved, meshes, CLASSES, run.testdump_dir)


def _relabel_with_ground_truth(run: HybridRun, resolved: ResolvedObjects) -> None:
    """Gives every resolved object matching a labelled object that object's ground-truth class, dropping the others."""
    ground_truth = GroundTruthObjects.from_meshes(annotated_object_meshes(run.sim, run.annotation, CLASSES))
    resolved.relabeled(matched_ground_truth_classes(resolved, ground_truth))


def retrieve_and_save_mosaics(run: HybridRun, resolved: ResolvedObjects) -> Tuple[dict, dict]:
    """(run's frames raytraced with resolved's objects, the views retrieve_views selects of each), saving
    retrieval_mosaic.png (one row per object of those views) and retrieval_first_views.png (its first column as a single
    row) to run's testdump."""
    with timed("Retrieval mosaics"):
        raytraced, object_frames = retrieve_resolved_views(resolved, run.memory, DEPTH_CAMERA, _MIN_MESH_VISIBILITY_FRACTION, ViewSelection(k=5))
        views_mosaic(run.memory, raytraced, _DETIC_STYLE, object_frames).save(run.testdump_dir / "retrieval_mosaic.png")
        first_views = first_views_mosaic(retrieved_views(resolved, run.memory, raytraced, object_frames), _DETIC_STYLE)
        first_views.save(run.testdump_dir / "retrieval_first_views.png")
    return raytraced, object_frames


def _save_dataset_mosaic(fused: ConceptGraphMemory, raytraced: dict, object_frames: dict) -> None:
    """train/train.json of the retrieved views, then dataset_mosaic.png, one row per batch of the first _MOSAIC_BATCHES the
    balanced_loader of the ObjectViewsDataset read back from it draws."""
    writer = ObjectViewsWriter(_TESTDUMP_DIR / "train", {"categories": CLASSES})
    writer.add_scene("procthor", retrieved_views(fused.resolved_objects, fused.image_memory, raytraced, object_frames))
    writer.write(np.random.default_rng(_SEED))
    mapper = DatasetMapper(is_train=True, augmentations=[], image_format="RGB", use_instance_mask=True, instance_mask_format="bitmask")
    loader = balanced_loader(ObjectViewsDataset.from_json(_TESTDUMP_DIR / "train" / "train.json"), _BATCH_SIZE, mapper)
    batches = itertools.islice(itertools.chain.from_iterable(itertools.repeat(loader)), _MOSAIC_BATCHES)
    batches_mosaic(batches, _DETIC_STYLE).save(_TESTDUMP_DIR / "dataset_mosaic.png")


def test_exhaustive_mode_reaches_viewpoints():
    detector = build_detector()
    run, fused = explore_and_fuse(
        detector, _TESTDUMP_DIR, ConceptGraphMemory(),
        lambda detections, frame, instances, floor: detections,
    )
    fused.image_memory = run.memory
    try:
        _resolve_and_classify(fused, detector)
        save_resolved_maps(run, fused.resolved_objects)
        if _LABEL_SOURCE == "gt":
            _relabel_with_ground_truth(run, fused.resolved_objects)
        raytraced, object_frames = retrieve_and_save_mosaics(run, fused.resolved_objects)
        _save_dataset_mosaic(fused, raytraced, object_frames)
    finally:
        run.sim.close()


if __name__ == "__main__":
    test_exhaustive_mode_reaches_viewpoints()
