"""A train scene's raw record (RawScene), written by habitat_embodied_sl/collect_train.py (stage 1) and read by
habitat_embodied_sl/build_dataset.py (stage 2), which builds the datasets from it without the simulator: every frame (frames/<id>.png rgb, frames/<id>.npz depth, pose and floor height, all lossless so a build from the record is a
build from collect_train.py's own memory), every frame's detections (detections/<id>.npz, masks RLE-encoded), the fused
ConceptGraphObjects (fused_objects.npz, ragged fields concatenated with offsets), the depth camera, the
agent's map grid, the annotated objects, their ground_truth.json on that grid (its navigable cells the simulator's
navmesh, which a build has no other way to know), the scene's mesh (ground_truth_mesh.glb, what the build tells the
objects in line of sight of the navmesh from) and the annotated objects' per-frame ground-truth views, and for viewing
only, their point cloud (ground_truth_point_cloud.ply), and its exploration's last snapshot
(exploration_snapshot.png). scene.json, the scene and its depth camera, is written last: a record without it is
incomplete. A test scene's record (save_test_scene, written by habitat_embodied_sl/collect_test.py and read by
habitat_embodied_sl/build_eval.py) holds only its kept frames' rgb and their ground-truth views, then its scene.json. The
records of a run share <raw_root>/<run_name>/config.yaml, the
collection config they were made with, completed with its full fusion settings and its kept classes, which their ground
truth's class ids index (save_run_config): a train scene's fusion and visibility settings are read back from it."""
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import quaternion
import torch
import trimesh
from detectron2.structures import Boxes, Instances
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from common.hybrid_exploration.exploration_map import MapGrid
from common.utils.coco_utils import encoded_masks, json_rle, rle_from_json, stacked_masks, write_json
from common.utils.concept_graph import ConceptGraphConfig, ConceptGraphObject
from common.utils.eval_utils import GROUND_TRUTH_JSON, GroundTruthObjects
from common.utils.ground_truth_recovery import GroundTruthView, encoded_views
from common.utils.image_memory import Frame
from common.utils.projection_utils import DepthCamera

FRAMES_DIR = "frames"
DETECTIONS_DIR = "detections"
RECORD_JSON = "scene.json"
VIEWS_JSON = "ground_truth_views.json"
SCENE_MESH_GLB = "ground_truth_mesh.glb"
EXPLORATION_SNAPSHOT_PNG = "exploration_snapshot.png"
RUN_CONFIG_YAML = "config.yaml"


@dataclass
class RawScene:
    scene: str
    camera: DepthCamera
    fused_objects: List[ConceptGraphObject]
    config: ConceptGraphConfig
    min_mesh_visibility_fraction: float
    ground_truth: GroundTruthObjects
    ground_truth_views: List[GroundTruthView]
    grid: MapGrid
    ground_truth_dataset: dict
    scene_mesh: trimesh.Trimesh
    frames: Dict[int, Frame] = field(default_factory=dict)


def save_run_config(run_dir: Path, cfg: DictConfig) -> None:
    """Saves cfg as run_dir/config.yaml, the config every scene of a raw run (collect_train.py, collect_test.py) or of a
    dataset (build_dataset.py, build_eval.py) shares, or checks that it is the one already saved there."""
    path = run_dir / RUN_CONFIG_YAML
    if path.exists():
        if OmegaConf.to_container(OmegaConf.load(path)) != OmegaConf.to_container(cfg):
            raise ValueError(f"{run_dir} was made with another config, {path}: give this one its own run_name, or delete {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    OmegaConf.save(cfg, temporary)
    temporary.replace(path)  # atomic, so a parallel call never reads it half-written


def raw_scene_dir(cfg: DictConfig, scene: str) -> Path:
    return Path(cfg.raw_root) / cfg.run_name / scene


def load_run_config(run_dir: Path) -> DictConfig:
    return OmegaConf.load(run_dir / RUN_CONFIG_YAML)


def is_recorded(raw_dir: Path) -> bool:
    """Whether raw_dir holds a complete record."""
    return (raw_dir / RECORD_JSON).exists()


def save_frame(raw_dir: Path, frame_id: int, frame: Frame, floor_height: float, instances: Instances) -> None:
    """frame's rgb, depth, pose and the floor height its detections were lifted above (what replaying their fusion needs),
    and the detector's instances on it, under raw_dir."""
    (raw_dir / FRAMES_DIR).mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.rgb).save(raw_dir / FRAMES_DIR / f"{frame_id}.png")
    np.savez_compressed(raw_dir / FRAMES_DIR / f"{frame_id}.npz", depth=frame.depth, pose=frame.pose, floor_height=floor_height)
    (raw_dir / DETECTIONS_DIR).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_dir / DETECTIONS_DIR / f"{frame_id}.npz", **_instance_arrays(instances))


def _instance_arrays(instances: Instances) -> dict:
    """Every field of instances as an array, pred_masks as their RLE counts (mask_counts) and size (mask_size)."""
    fields = {name: value.tensor if isinstance(value, Boxes) else value for name, value in instances.get_fields().items() if name != "pred_masks"}
    rles = encoded_masks(instances.pred_masks) if len(instances) else []
    return {
        **{name: value.numpy() for name, value in fields.items()},
        "mask_counts": np.array([rle["counts"] for rle in rles], dtype=bytes), "mask_size": np.array(instances.image_size),
    }


def load_detections(raw_dir: Path, frame_id: int) -> Instances:
    """The detector instances save_frame recorded for frame_id under raw_dir, their masks decoded."""
    arrays = np.load(raw_dir / DETECTIONS_DIR / f"{frame_id}.npz")
    size = tuple(int(side) for side in arrays["mask_size"])
    masks = stacked_masks([{"size": list(size), "counts": bytes(counts)} for counts in arrays["mask_counts"]], size)
    fields = {name: torch.from_numpy(arrays[name]) for name in arrays.files if name not in ("mask_counts", "mask_size", "pred_boxes")}
    return Instances(size, **fields, pred_boxes=Boxes(torch.from_numpy(arrays["pred_boxes"])), pred_masks=masks)


def save_raw_scene(raw_dir: Path, raw: RawScene) -> None:
    """Everything of raw but its frames, which save_frame writes as they come, under raw_dir, its scene.json last."""
    _save_fused_objects(raw_dir / "fused_objects.npz", raw.fused_objects)
    np.savez_compressed(raw_dir / "ground_truth_objects.npz", **asdict(raw.ground_truth))
    _save_views(raw_dir, raw.ground_truth_views)
    grid = dict(asdict(raw.grid), ref_rotation=quaternion.as_float_array(raw.grid.ref_rotation))
    np.savez_compressed(raw_dir / "map_grid.npz", **grid)
    write_json(raw_dir / GROUND_TRUTH_JSON, raw.ground_truth_dataset)
    raw.scene_mesh.export(raw_dir / SCENE_MESH_GLB)
    write_json(raw_dir / RECORD_JSON, {"scene": raw.scene, "camera": asdict(raw.camera)})


def save_test_scene(raw_dir: Path, scene: str, frames: List[Tuple[int, np.ndarray, Instances]]) -> None:
    """Each (frame id, rgb, ground truth) of frames as its frames/<id>.png and its ground truth's views, under raw_dir, then its scene.json."""
    (raw_dir / FRAMES_DIR).mkdir(parents=True, exist_ok=True)
    for frame_id, rgb, _ in frames:
        Image.fromarray(rgb).save(raw_dir / FRAMES_DIR / f"{frame_id}.png")
    _save_views(raw_dir, [view for frame_id, _, ground_truth in frames for view in encoded_views(frame_id, ground_truth)])
    write_json(raw_dir / RECORD_JSON, {"scene": scene})


def _save_views(raw_dir: Path, views: List[GroundTruthView]) -> None:
    write_json(raw_dir / VIEWS_JSON, [
        {"frame_id": int(view.frame_id), "object_id": int(view.object_id), "class_id": int(view.class_id), "visibility": float(view.visibility), "rle": json_rle(view.rle)}
        for view in views
    ])


def _save_fused_objects(path: Path, objects: List[ConceptGraphObject]) -> None:
    np.savez_compressed(
        path,
        points=np.concatenate([obj.points for obj in objects]), point_num_dets=np.concatenate([obj.point_num_dets for obj in objects]),
        point_offsets=np.cumsum([0, *(len(obj.points) for obj in objects)]),
        region_features=torch.stack([obj.region_feature for obj in objects]).numpy(), objectness=np.array([obj.objectness for obj in objects]),
        object_num_dets=np.array([obj.object_num_dets for obj in objects]),
        detected_frames=np.concatenate([sorted(obj.detected_frames) for obj in objects]),
        frame_offsets=np.cumsum([0, *(len(obj.detected_frames) for obj in objects)]),
    )


def load_raw_scene(raw_dir: Path) -> RawScene:
    """The RawScene save_frame and save_raw_scene wrote under raw_dir, its fusion and visibility settings its run's."""
    scene, run = json.loads((raw_dir / RECORD_JSON).read_text()), load_run_config(raw_dir.parent)
    grid = dict(np.load(raw_dir / "map_grid.npz"))
    return RawScene(
        scene["scene"], DepthCamera(**scene["camera"]), _load_fused_objects(raw_dir / "fused_objects.npz"), ConceptGraphConfig(**run.fusion),
        run.object_params.min_mesh_visibility_fraction,
        GroundTruthObjects(**np.load(raw_dir / "ground_truth_objects.npz")), load_views(raw_dir),
        MapGrid(**{**grid, "meters_per_pixel": float(grid["meters_per_pixel"]), "ref_rotation": quaternion.from_float_array(grid["ref_rotation"])}),
        json.loads((raw_dir / GROUND_TRUTH_JSON).read_text()), trimesh.load(raw_dir / SCENE_MESH_GLB, force="mesh"), _load_frames(raw_dir),
    )


def _load_frames(raw_dir: Path) -> Dict[int, Frame]:
    frames = {}
    for frame_id, rgb in load_rgbs(raw_dir).items():
        arrays = np.load(raw_dir / FRAMES_DIR / f"{frame_id}.npz")
        frames[frame_id] = Frame(rgb, arrays["depth"], arrays["pose"])
    return frames


def load_rgbs(raw_dir: Path) -> Dict[int, np.ndarray]:
    frame_ids = sorted(int(path.stem) for path in (raw_dir / FRAMES_DIR).glob("*.png"))
    return {frame_id: np.asarray(Image.open(raw_dir / FRAMES_DIR / f"{frame_id}.png")) for frame_id in frame_ids}


def load_views(raw_dir: Path) -> List[GroundTruthView]:
    views = json.loads((raw_dir / VIEWS_JSON).read_text())
    return [GroundTruthView(**dict(view, rle=rle_from_json(view["rle"]))) for view in views]


def _load_fused_objects(path: Path) -> List[ConceptGraphObject]:
    arrays = dict(np.load(path))
    point_offsets, frame_offsets = arrays["point_offsets"], arrays["frame_offsets"]
    return [
        ConceptGraphObject(
            arrays["points"][point_offsets[index]:point_offsets[index + 1]], arrays["point_num_dets"][point_offsets[index]:point_offsets[index + 1]],
            torch.from_numpy(arrays["region_features"][index]), float(arrays["objectness"][index]),
            frozenset(arrays["detected_frames"][frame_offsets[index]:frame_offsets[index + 1]].tolist()), int(arrays["object_num_dets"][index]),
        )
        for index in range(len(arrays["objectness"]))
    ]
