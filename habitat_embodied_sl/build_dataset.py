"""Stage 2: builds a train scene's shard of each of its variants' datasets, datasets/<run_name>/, from its record in the
dataset config's raw run (common/utils/raw_scene.py, written by habitat_embodied_sl/collect_train.py), without the simulator,
classifying among the raw run's recorded classes with its Detic (its config.yaml), so a dataset keeps the class ids its
records were made with. One call per scene, so calls can run in parallel; habitat_embodied_sl/merge.py then pools the
scenes' shards. What an earlier build of the scene wrote is deleted first (flush_scene). The config's variant, set in it or
on the command line (variant=V2_drop_add, like any of its keys), is the dataset's train split; a list of them
(variant=[V1_pseudo,V4_full]) builds each variant's dataset, each with its own config, from one resolve, classification
and retrieval of the scene. A dataset's first call saves its config, its resolve and retrieval settings completed with
every default and the raw run's classes added, as datasets/<run_name>/config.yaml, and a call with another config fails
(save_run_config): another config is another dataset, under its own run_name.

A scene's fused objects are resolved, their foreground objects kept and classified (common/utils/concept_graph.py,
ResolveConfig with the config's resolve overrides):
they go to datasets/<run_name>/<scene>/predictions.json (eval_utils.py's single-scene COCO dataset: 3D boxes and top-down
footprints on the agent's map grid), and the record's ground truth, without the objects no camera on the navmesh has in
line of sight (inside a closed fridge, say), to ground_truth.json there, both drawn by class to
estimated_topdown.png and ground_truth_topdown.png; the resolved objects' points and point-extent boxes, by class, to
semantic_point_cloud.ply and semantic_wireframes.glb there, with the record's exploration snapshot. Up to retrieval.k diverse views of each object are then retrieved
(common/utils/retrieval.py), and the three object-view variants derived from them, each on its own (common/utils/ground_truth_recovery.py): "V1_pseudo" as classified; "V2_drop_add" with
false positives dropped (recover_drop_fp) and false negatives' views added (recover_add_fn), those classified as the
resolved objects are (pseudo_classes); "V3_cls_box_drop_add" with false positives dropped, matches taking their ground
truth's class and masks (recover_gt_classes, recover_gt_masks) and false negatives' views added with their ground-truth
class. An object-view variant writes its views to datasets/<run_name>/object_ds/, with a retrieval mosaic of the scene,
retrieval_mosaics/<scene>.jpg, one row of views per object at half resolution, its objects relabelled or added by recovery
bordered red, split into <scene>_<part>.jpg of at most 32 objects each unless --no-split, <scene>_first_views.jpg, one
row of each object's first view, and a semantic top-down map, semantic_topdown_maps/<scene>.png, of its resolved objects
by their class in the variant. "V4_full" writes every frame some object-view variant's view is in, with its full ground
truth (ground_truth_frame), to the COCO split datasets/<run_name>/train/ (annotated_frames): the same frames whichever
variant it is built beside.

Usage (Detic classifies on the GPU CUDA_VISIBLE_DEVICES names):
  export BASE_DIR=$(pwd) PYTHONNOUSERSITE=1  # HABITAT_DATA already exported, see INSTALL.MD
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python habitat_embodied_sl/build_dataset.py <scene> habitat_embodied_sl/config/procthor8_build.yaml \
      variant=[V1_pseudo,V2_drop_add,V3_cls_box_drop_add,V4_full]
  PYTHONPATH=. python habitat_embodied_sl/merge.py datasets/procthor8_V2_drop_add  # once all scenes are built
"""
import argparse
import copy
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from detectron2.structures import Instances
from omegaconf import DictConfig, OmegaConf

from common.env_utils.object_annotations import PROCTHOR100_SIZES_PATH, class_size_ranges
from common.utils.coco_utils import write_json
from common.utils.concept_graph import ResolveConfig, classify, keep_foreground, object_size_ranges, resolve
from common.utils.dataset_writers import FULL_FRAMES, OBJECT_DS_DIR, VARIANTS, collection_root, flush_scene, split_writer
from common.utils.eval_utils import (
    GROUND_TRUTH_JSON, PREDICTIONS_JSON, GroundTruthObjects, ResolvedDetections, matched_ground_truth, navigable_viewpoints,
    observable_ground_truth, predictions_dataset, save_box_wireframes, save_topdowns, semantic_topdown,
)
from common.utils.ground_truth_recovery import (
    GroundTruthAnnotation, GroundTruthView, ground_truth_frame, missed_ground_truth, pseudo_classes, recover_add_fn, recover_drop_fp,
    recover_gt_classes, recover_gt_masks, recovered_objects, views_by_frame,
)
from common.utils.image_memory import Frame, object_views, raytrace_frames
from common.utils.plot_utils import MAX_MOSAIC_ROWS, MOSAIC_JPEG_QUALITY, PNG_COMPRESS_LEVEL, save_mosaic_parts
from common.utils.projection_utils import ResolvedObjects, save_resolved_ply
from common.utils.raw_scene import (
    EXPLORATION_SNAPSHOT_PNG, SCENE_MESH_GLB, RawScene, is_recorded, load_raw_scene, load_run_config, save_run_config,
)
from common.utils.retrieval import (
    RetrievedView, SegmentationStyle, ViewSelection, first_views_mosaic, objects_views_mosaics, retrieve_resolved_views, retrieved_views,
)
from common.vision.detector import Detector, build_detector

_TILE_DOWNSCALE = 2


def mosaic_max_rows(args: argparse.Namespace) -> Optional[int]:
    return None if args.no_split else MAX_MOSAIC_ROWS


def resolve_and_classify(raw: RawScene, detector: Detector, cfg: DictConfig, size_ranges: Dict[int, Dict[str, tuple]]) -> ResolvedObjects:
    """raw's fused objects resolved with cfg's resolve parameters, the foreground ones kept and classified among the
    classes their size fits."""
    resolve_config, min_mesh_visibility_fraction = ResolveConfig(**cfg.resolve), raw.min_mesh_visibility_fraction
    resolved, sources = resolve(raw.fused_objects, raw.config.downsample_voxel_size, resolve_config)
    raytraced = raytrace_frames(resolved, raw.frames, raw.camera, min_mesh_visibility_fraction)
    scores = keep_foreground(resolved, sources, raytraced, resolve_config.min_detection_rate)
    classify(resolved, scores, object_views(resolved, raw.frames, raw.camera, min_mesh_visibility_fraction), detector, size_ranges)
    return resolved


@dataclass(frozen=True)
class SceneRecovery:
    """What every train variant derives from: the scene's retrieved views as classified, their objects' ground-truth
    matches (matched_ground_truth), the annotation and the frames and view selection missed objects' views come from."""

    pseudo: List[RetrievedView]
    matches: Dict[int, int]
    annotation: GroundTruthAnnotation
    frames: Dict[int, Frame]
    selection: ViewSelection


def scene_recovery(resolved: ResolvedObjects, raw: RawScene, cfg: DictConfig) -> SceneRecovery:
    """The views retrieved of the resolved, classified objects, with what recovering them from raw's ground truth needs."""
    selection = ViewSelection(**cfg.retrieval)
    retrieved = retrieve_resolved_views(resolved, raw.frames, raw.camera, raw.min_mesh_visibility_fraction, selection)
    annotation = GroundTruthAnnotation(raw.ground_truth, raw.ground_truth_views)
    return SceneRecovery(retrieved_views(resolved, raw.frames, *retrieved), matched_ground_truth(resolved, raw.ground_truth), annotation, raw.frames, selection)


def drop_add_views(recovery: SceneRecovery, detector: Detector, size_ranges: Dict[int, Dict[str, tuple]]) -> List[RetrievedView]:
    """V2_drop_add: the matched objects' views as classified, plus missed ground truths' views (their masks) classified as
    the resolved objects are: the objects fixed, their labels pseudo."""
    missed = missed_ground_truth(recovery.annotation, recovery.matches)
    classes = pseudo_classes(recovery.annotation, missed, recovery.frames, detector, size_ranges)
    return recover_add_fn(recover_drop_fp(recovery.pseudo, recovery.matches), recovery.annotation, classes, recovery.frames, recovery.selection)


def cls_box_drop_add_views(recovery: SceneRecovery) -> List[RetrievedView]:
    """V3_cls_box_drop_add: the matched objects' views with their ground truth's class and masks, plus missed ground
    truths' views with their class: the objects and their labels fixed."""
    matched = recover_drop_fp(recovery.pseudo, recovery.matches)
    relabelled = recover_gt_masks(recover_gt_classes(matched, recovery.matches, recovery.annotation.objects), recovery.matches, recovery.annotation)
    missed = missed_ground_truth(recovery.annotation, recovery.matches)
    classes = {int(index): int(recovery.annotation.objects.class_ids[index]) for index in missed}
    return recover_add_fn(relabelled, recovery.annotation, classes, recovery.frames, recovery.selection)


def scene_variants(resolved: ResolvedObjects, raw: RawScene, cfg: DictConfig, detector: Detector, size_ranges: Dict[int, Dict[str, tuple]]) -> Dict[str, List[RetrievedView]]:
    """{variant: views} of every object-view train variant (VARIANTS) of the scene, each derived from the same retrieved views."""
    recovery = scene_recovery(resolved, raw, cfg)
    return dict(zip(VARIANTS, (recovery.pseudo, drop_add_views(recovery, detector, size_ranges), cls_box_drop_add_views(recovery))))


def save_retrieval_mosaics(views: List[RetrievedView], highlighted: Set[int], style: SegmentationStyle, out_path: Path, max_rows: Optional[int]) -> None:
    """out_path of a variant's views: one row per object of its views, tiles downscaled by _TILE_DOWNSCALE, bordered red
    when highlighted (relabelled or added by recovery), split into parts of at most max_rows objects (save_mosaic_parts);
    and <out_path's stem>_first_views.jpg, one row of each object's first view."""
    save_mosaic_parts(objects_views_mosaics(views, style, highlighted, _TILE_DOWNSCALE, max_rows), out_path)
    first_views = first_views_mosaic(views, style, highlighted)
    if first_views is not None:
        first_views.save(out_path.with_name(f"{out_path.stem}_first_views.jpg"), quality=MOSAIC_JPEG_QUALITY)


def save_semantic_topdown(raw: RawScene, views: List[RetrievedView], resolved_objects: ResolvedObjects, style: SegmentationStyle, out_path: Path) -> None:
    """out_path of a variant's views: the resolved objects they show, by their class there, on raw's map grid;
    recover_add_fn's objects have no points to draw."""
    object_classes = {view.object_id: view.class_id for view in views if view.object_id >= 0}
    resolved = copy.deepcopy(resolved_objects).relabeled(object_classes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_topdown(predictions_dataset(raw.scene, resolved, raw.grid, style.classes), style).save(out_path, compress_level=PNG_COMPRESS_LEVEL)


def annotated_frames(rgbs: Dict[int, np.ndarray], ground_truth_views: List[GroundTruthView]) -> List[Tuple[int, np.ndarray, Instances]]:
    """(frame id, rgb, ground truth) of every frame of rgbs some ground truth shows, in frame order."""
    frame_views = views_by_frame(ground_truth_views)
    return [(frame_id, rgbs[frame_id], ground_truth_frame(frame_views[frame_id], rgbs[frame_id].shape[:2])) for frame_id in sorted(rgbs) if frame_views[frame_id]]


def observable_ground_truth_dataset(raw: RawScene) -> dict:
    """raw's ground_truth.json without the objects no camera on the navmesh, at the recorded camera height, has in line of
    sight (observable_ground_truth)."""
    camera_height = float(np.median([frame.pose[1, 3] for frame in raw.frames.values()]))
    viewpoints = navigable_viewpoints(raw.ground_truth_dataset, raw.grid, camera_height)
    return observable_ground_truth(raw.ground_truth_dataset, raw.scene_mesh, viewpoints, raw.camera.max_depth)


def save_scene_maps(raw: RawScene, resolved: ResolvedObjects, style: SegmentationStyle, scene_dir: Path) -> None:
    """scene_dir's predictions.json of resolved's objects and ground_truth.json of raw's observable ground truth, their
    top-downs, and for viewing in 3D, resolved's semantic_point_cloud.ply, both box sets by class as
    semantic_wireframes.glb and ground_truth_wireframes.glb, and the scene's ground_truth_mesh.glb."""
    ground_truth = observable_ground_truth_dataset(raw)
    write_json(scene_dir / PREDICTIONS_JSON, predictions_dataset(raw.scene, resolved, raw.grid, style.classes))
    write_json(scene_dir / GROUND_TRUTH_JSON, ground_truth)
    save_topdowns(scene_dir, style)
    save_resolved_ply(resolved, style.colors, scene_dir / "semantic_point_cloud.ply")
    save_box_wireframes(ResolvedDetections.from_resolved(resolved), style.classes, style.colors, scene_dir / "semantic_wireframes.glb")
    save_box_wireframes(GroundTruthObjects.from_annotations(ground_truth["annotations"]), style.classes, style.colors, scene_dir / "ground_truth_wireframes.glb")
    raw.scene_mesh.export(scene_dir / SCENE_MESH_GLB)


def write_variant(raw: RawScene, resolved: ResolvedObjects, scene_views: Dict[str, List[RetrievedView]], cfg: DictConfig, max_rows: Optional[int]) -> None:
    """Writes raw's scene's shard of cfg's variant from scene_views (scene_variants), and an object-view variant's
    mosaics and top-down map."""
    style, root = SegmentationStyle.of(cfg.classes), collection_root(cfg)
    if cfg.variant == FULL_FRAMES:
        view_rgbs = {view.frame_id: view.rgb for views in scene_views.values() for view in views}
        split_writer(root / "train", cfg).write_scene(raw.scene, annotated_frames(view_rgbs, raw.ground_truth_views))
        return
    views = scene_views[cfg.variant]
    highlighted = recovered_objects(scene_views[VARIANTS[0]], views)
    save_retrieval_mosaics(views, highlighted, style, root / "retrieval_mosaics" / f"{raw.scene}.jpg", max_rows)
    save_semantic_topdown(raw, views, resolved, style, root / "semantic_topdown_maps" / f"{raw.scene}.png")
    split_writer(root / OBJECT_DS_DIR, cfg).write_scene(raw.scene, views)


def build_scene(raw: RawScene, cfgs: List[DictConfig], detector: Detector, max_rows: Optional[int]) -> None:
    """Resolves, classifies and retrieves raw's fused objects once, as cfgs' shared resolve and retrieval settings say,
    into its scene's shard of each of cfgs' variants (write_variant), each with the same map files (save_scene_maps)."""
    style, scene_dirs = SegmentationStyle.of(detector.classes), [collection_root(cfg) / raw.scene for cfg in cfgs]
    size_ranges = object_size_ranges(detector.classes, class_size_ranges(PROCTHOR100_SIZES_PATH))
    resolved = resolve_and_classify(raw, detector, cfgs[0], size_ranges)
    save_scene_maps(raw, resolved, style, scene_dirs[0])
    for scene_dir in scene_dirs[1:]:
        shutil.copytree(scene_dirs[0], scene_dir)
    scene_views = scene_variants(resolved, raw, cfgs[0], detector, size_ranges)
    for cfg in cfgs:
        write_variant(raw, resolved, scene_views, cfg, max_rows)


def build_train_scene(raw_dir: Path, raw_cfg: DictConfig, cfgs: List[DictConfig], max_rows: Optional[int]) -> None:
    """build_scene of the record under raw_dir, classified by its raw run's detector (no SAM: the build only classifies),
    with its exploration snapshot in each variant's scene folder."""
    raw = load_raw_scene(raw_dir)
    build_scene(raw, cfgs, build_detector(raw_cfg, None), max_rows)
    for cfg in cfgs:
        shutil.copy(raw_dir / EXPLORATION_SNAPSHOT_PNG, collection_root(cfg) / raw.scene)


def variant_configs(cfg: DictConfig) -> List[DictConfig]:
    """cfg once per variant it names, one or a list of them (variant=[V1_pseudo,V4_full]): each its variant's dataset config."""
    variants = list(cfg.variant) if OmegaConf.is_list(cfg.variant) else [cfg.variant]
    unknown = [variant for variant in variants if variant not in (*VARIANTS, FULL_FRAMES)]
    if unknown:
        raise ValueError(f"variants must be among {(*VARIANTS, FULL_FRAMES)}, got {unknown}")
    cfgs = [copy.deepcopy(cfg) for _ in variants]
    for variant_cfg, variant in zip(cfgs, variants):
        variant_cfg.variant = variant
    return cfgs


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    raw_dir = Path(cfg.raw_run) / args.scene
    if not is_recorded(raw_dir):
        raise ValueError(f"{raw_dir} holds no complete raw record: collect {args.scene} first")
    raw_cfg = load_run_config(raw_dir.parent)
    cfg.resolve, cfg.retrieval = asdict(ResolveConfig(**cfg.resolve)), asdict(ViewSelection(**cfg.retrieval))
    cfg.classes = list(raw_cfg.classes)
    cfgs = variant_configs(cfg)
    for variant_cfg in cfgs:
        save_run_config(collection_root(variant_cfg), variant_cfg)
    for variant_cfg in cfgs:
        flush_scene(collection_root(variant_cfg), args.scene)
    build_train_scene(raw_dir, raw_cfg, cfgs, mosaic_max_rows(args))
    print(f"{args.scene} built from {raw_dir} into {', '.join(str(collection_root(variant_cfg)) for variant_cfg in cfgs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="Train scene to build, e.g. ProcTHOR-Train-5.")
    parser.add_argument("config", help="Dataset config: run_name, raw_run, dataset_root, variant, resolve and retrieval settings.")
    parser.add_argument("overrides", nargs="*", help="The config's keys to override, as key=value, e.g. variant=V2_drop_add or variant=[V1_pseudo,V4_full].")
    parser.add_argument("--no-split", action="store_true", help="Save each mosaic whole instead of in parts of at most 32 rows.")
    main(parser.parse_args())
