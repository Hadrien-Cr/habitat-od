"""Tunes each query's confidence threshold offline, on the frames a main.py run of --config saved with save_frames
(typically a use_gt_detections run over episodes outside the evaluation's own, via first_episode, with its query banks
built by build_query_banks.py): every frame is re-detected with Detector.predict_target against each of --queries,
and its best detection labelled by the ground-truth object under its mask (evaluator.detection_hit). Then, for
each threshold of _THRESHOLDS:
- first-hit success: the share of episodes whose first frame scoring at least the threshold lands on a goal -- the
  agent commits to its first confident detection, so this approximates its success along the saved trajectories;
- first-hit synonym: the share whose first such frame lands on an object synonym;
- frame precision: the share of frames scoring at least the threshold that land on a goal;
- episode recall: the share of episodes with some goal frame scoring at least the threshold.
Use it as agent.confidence_thresh and agent.goal_score_thresh. Per-frame (score, hit) pairs are cached in
<run_dir>/tuning/<query>.json (delete it to re-detect); the table (every _PRINT_EVERY-th threshold, plus the best
first-hit one) goes to <run_dir>/tuning/<query>.md and is printed.

Usage:
  export BASE_DIR=$(pwd) PYTHONNOUSERSITE=1  # HABITAT_DATA already exported, see INSTALL.MD
  PYTHONPATH=. python habitat_od_search/tune_threshold.py --config habitat_od_search/logs/configs/tune_gt_frames_max3.yaml \
      --queries text_full web web+text_cat gt_views
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from omegaconf import OmegaConf
from tabulate import tabulate
from tqdm import tqdm

from common.vision.detic import Vocabulary
from common.vision.detector import Detector
from habitat_od_search.evaluator import EpisodeObjects, detection_hit
from habitat_od_search.main import RUNS_ROOT
from habitat_od_search.query_bank import target_query

_THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
_PRINT_EVERY = 5

Detections = List[Tuple[float, str]]


def episode_detections(detector: Detector, frames_dir: Path, query: str) -> Detections:
    """(best score, detection_hit) of every saved frame of frames_dir's episode, against query."""
    episode = json.loads((frames_dir / "episode.json").read_text())
    objects = EpisodeObjects(set(episode["goal_ids"]), set(episode["synonym_ids"]), {int(k): v for k, v in episode["classes"].items()}, {})
    target = target_query(query, episode["scene_id"], episode["object_category"])
    detections = []
    for rgb_path in sorted(frames_dir.glob("rgb_*.jpg")):
        rgb = np.ascontiguousarray(cv2.imread(str(rgb_path))[:, :, ::-1])
        id_image = cv2.imread(str(frames_dir / rgb_path.name.replace("rgb_", "ids_").replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED).astype(np.int32) - 1
        instances, best_score = detector.predict_target(rgb, target.text_embedding, target.region_embeddings)
        detections.append((best_score, detection_hit(objects, instances, id_image)))
    return detections


def run_detections(detector: Detector, run_dir: Path, query: str) -> Dict[str, Detections]:
    """{episode frames dir name: episode_detections} of every episode of run_dir, cached in run_dir/tuning/<query>.json."""
    cache = run_dir / "tuning" / f"{query}.json"
    if cache.exists():
        return {name: [tuple(detection) for detection in detections] for name, detections in json.loads(cache.read_text()).items()}
    frames_dirs = sorted(path for path in (run_dir / "frames").iterdir() if path.is_dir())
    detections = {path.name: episode_detections(detector, path, query) for path in tqdm(frames_dirs, desc=query, ncols=80)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(detections))
    return detections


def threshold_row(episodes: List[Detections], threshold: float) -> Tuple[float, ...]:
    """(threshold, first-hit success, first-hit synonym, frame precision, episode recall), shares in %."""
    first_hits = [next((hit for score, hit in episode if score >= threshold), None) for episode in episodes]
    confident = [hit for episode in episodes for score, hit in episode if score >= threshold]
    return (
        threshold,
        100 * np.mean([hit == "goal" for hit in first_hits]),
        100 * np.mean([hit == "synonym" for hit in first_hits]),
        100 * np.mean([hit == "goal" for hit in confident]) if confident else float("nan"),
        100 * np.mean([any(score >= threshold and hit == "goal" for score, hit in episode) for episode in episodes]),
    )


def tuning_table(episodes: List[Detections]) -> Tuple[str, Tuple[float, ...]]:
    """The threshold sweep's markdown table and its best first-hit-success row (the lowest threshold among ties)."""
    rows = [threshold_row(episodes, threshold) for threshold in _THRESHOLDS]
    best = max(rows, key=lambda row: row[1])
    shown = [row for index, row in enumerate(rows) if (index + 1) % _PRINT_EVERY == 0 or row is best]
    table = tabulate(
        [(f"**{row[0]:.2f}**" if row is best else f"{row[0]:.2f}", *row[1:]) for row in shown], tablefmt="pipe", floatfmt=".1f",
        headers=["threshold", "first-hit success", "first-hit synonym", "frame precision", "episode recall"], numalign="left",
    )
    return table, best


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    run_dir = RUNS_ROOT / cfg.run_name
    detector = Detector(
        args.config_file, args.checkpoint, Vocabulary("tuning_unused", ("object",)), None,
        score_thresh=args.score_thresh, nms_thresh=args.nms_thresh, device=args.device, input_size=cfg.agent.input_image_size,
    )
    for query in args.queries:
        table, best = tuning_table(list(run_detections(detector, run_dir, query).values()))
        report = f"# {query}: best threshold {best[0]:.2f} ({best[1]:.1f}% first-hit success)\n\n{table}\n"
        (run_dir / "tuning" / f"{query}.md").write_text(report)
        print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="main.py run config whose frames/ were saved with save_frames.")
    parser.add_argument("--queries", nargs="+", required=True, help="query_bank.py query names, e.g. text_full web web+text_cat.")
    parser.add_argument("--config-file", default="third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    parser.add_argument("--checkpoint", default="third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--score-thresh", type=float, default=0.4)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
