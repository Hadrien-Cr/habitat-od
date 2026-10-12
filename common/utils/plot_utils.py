import json
from pathlib import Path
from typing import Optional

from matplotlib.pyplot import text
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from detectron2.utils.visualizer import Visualizer as DetVisualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import Metadata

# Fixed-order categorical hues (validated for CVD/contrast) -- reused as-is,
# never cycled per-key, so the same slot always means the same series.
_SERIES_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
_MIN_LABEL_AREA_M2 = 0.01


def load_metrics(metrics_json: Path) -> list[dict]:
    """Parses a detectron2 `EventStorage` JSONWriter log: one JSON object per
    line, only carrying whatever scalars were logged at that iteration (loss
    terms every step, eval metrics like `bbox/AP` only every `eval_period`)."""
    with open(metrics_json) as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_metrics(metrics_json: Path, out_path: Path, keys: Optional[list[str]] = None) -> Path:
    """Plots scalars logged during a detectron2 `DefaultTrainer` run (losses,
    lr, and any periodic `EvalHook` results such as `bbox/AP`) against
    `iteration`, one subplot per metric -- never dual-axis, since loss/mAP/lr
    live on incomparable scales. Defaults to every numeric key in the log
    except `iteration`/`lr`/`time`/`data_time`/`eta_seconds`; pass `keys` to
    restrict to a subset (e.g. `["bbox/AP", "bbox/AP50"]`).

    When `cfg.DATASETS.TEST` has more than one dataset (e.g. val + train),
    detectron2 prefixes eval keys with the dataset name (`"<dataset>/bbox/AP"`);
    those are detected and overlaid as separate lines on the same subplot
    instead of splitting into their own panels.
    """
    records = load_metrics(metrics_json)
    if not records:
        raise ValueError(f"No metrics found in {metrics_json}")

    if keys is None:
        skip = {"iteration", "lr", "time", "data_time", "eta_seconds"}
        keys = sorted({k for r in records for k in r} - skip)

    # key -> (series_label, metric_name); a key like "<dataset>/bbox/AP" (2+
    # slashes) is an eval metric for a specific dataset in a multi-dataset
    # TEST set -- split off the dataset name so same-metric/different-dataset
    # keys share one panel instead of each getting its own.
    groups: dict[str, dict[str, str]] = {}
    for key in keys:
        parts = key.split("/")
        series, metric = (parts[0], "/".join(parts[1:])) if len(parts) >= 3 else (key, key)
        groups.setdefault(metric, {})[series] = key

    n_cols = min(4, len(groups)) or 1
    n_rows = -(-len(groups) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for ax, (metric, series_map) in zip(axes.flat, groups.items()):
        for i, (series, key) in enumerate(series_map.items()):
            xs = [r["iteration"] for r in records if key in r]
            ys = [r[key] for r in records if key in r]
            # marker is required, not cosmetic: eval metrics (e.g. bbox/AP) are only
            # logged every eval_period iterations, so a run with few eval points would
            # otherwise render as an invisible line (matplotlib draws nothing for an
            # unmarked single point).
            ax.plot(xs, ys, color=_SERIES_COLORS[i % len(_SERIES_COLORS)], linewidth=2, marker="o", markersize=4, label=series)
        ax.set_title(metric, fontsize=9)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
        if len(series_map) > 1:
            ax.legend(fontsize=7)

    for ax in axes.flat[len(groups):]:
        ax.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_al_history(history: list[dict], out_path: Path) -> Path:
    rounds, maps = [], []
    for entry in history:
        ap = entry.get("eval", {}).get("bbox", {}).get("AP")
        if ap is None or np.isnan(ap):
            continue
        rounds.append(0 if entry["round"] == "init" else entry["round"])
        maps.append(ap)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rounds, maps, color=_SERIES_COLORS[0], linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("round")
    ax.set_ylabel("mAP")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_semantic_2d_map(
    bg_grid,
    footprints: list[tuple[int, np.ndarray]],
    colors: dict[int, tuple[int,int,int]],
    classes: dict[int, str],
    meters_per_grid_pixel: float = 0.01,
    scale=1.0
) -> Image.Image:
    """bg_grid's navigable map with each object's (class, top-down mask) footprint colored by class, labelled once at its centroid."""
    footprints = sorted(footprints, key=lambda footprint: footprint[1].sum(), reverse=True)
    colored = np.zeros((*bg_grid.shape, 3), dtype=np.uint8)
    colored[bg_grid == 1] = [100, 100, 100]
    for class_id, mask in footprints:
        colored[mask] = colors[class_id]

    img = Image.fromarray(colored)
    img = img.resize(
        (int(img.width * scale), int(img.height * scale)),
        resample=Image.NEAREST  # type: ignore
    )
    draw = ImageDraw.Draw(img)
    for class_id, mask in footprints:
        area_m = mask.sum() * meters_per_grid_pixel ** 2
        if area_m < _MIN_LABEL_AREA_M2:
            continue
        y, x = np.argwhere(mask).mean(axis=0) * scale
        draw.text((x, y), classes[class_id], fill=(255,255,255), font_size= 4 * np.clip(np.sqrt(area_m), 1, 4) * scale, anchor="mm")

    return img

def draw_line(
    start: tuple[int, int],
    end: tuple[int, int],
    mat: np.ndarray,
    steps: int = 25,
    w: int = 1,
) -> np.ndarray:
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w : x + w, y - w : y + w] = 1
    return mat

def get_contour_points(
    pos: tuple[float, float, float],
    origin: tuple[float, float],
    size: int = 20,
) -> np.ndarray:
    x, y, o = pos
    pt1 = (int(x) + origin[0], int(y) + origin[1])
    pt2 = (
        int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1],
    )
    pt3 = (int(x + size * np.cos(o)) + origin[0], int(y + size * np.sin(o)) + origin[1])
    pt4 = (
        int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1],
    )

    return np.array([pt1, pt2, pt3, pt4])

def plot_array(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)
    

def plot_mask(mask) -> Image.Image:
    colored = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    colored[mask == 1] = [255, 255, 255]
    return Image.fromarray(colored)

def plot_segmentation_pred(rgb: np.ndarray, pred_instances, classes: list[str], colors: list[tuple[float, float, float]], scale: float = 1.0) -> Image.Image:
    det_visualizer = DetVisualizer(
        rgb,
        scale=scale,
        instance_mode=ColorMode.SEGMENTATION,
        font_size_scale=1.0
    )

    for i, (box, class_id, score) in enumerate(zip(
        pred_instances.pred_boxes.tensor.cpu().numpy(),
        pred_instances.pred_classes.cpu().numpy(),
        pred_instances.scores.cpu().numpy()
    )):
        x1, y1, x2, y2 = box
        height_ratio = (y2 - y1) / rgb.shape[0]
        font_size = (
            np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * det_visualizer._default_font_size
        )
        det_visualizer.draw_text(
            f"{i}: pt={classes[class_id]} {round(100*score)}%",
            font_size=font_size,
            position=(x1, y1),
            horizontal_alignment="left",
            color="#{:02x}{:02x}{:02x}".format(*colors[class_id])
        )

    vis_img = det_visualizer.overlay_instances(
        boxes=pred_instances.pred_boxes.tensor.cpu().numpy(),
        masks=pred_instances.pred_masks.cpu().numpy() if pred_instances.has("pred_masks") else None,
        labels=None,
        assigned_colors=[(x/255, y/255, z/255) for class_id in pred_instances.pred_classes.cpu().numpy() for x, y, z in [colors[class_id]]],
    )
    result = vis_img.get_image()
    im = Image.fromarray(result)
    return im


def plot_segmentation_gt(rgb: np.ndarray, gt_instances, classes: list[str], colors: list[tuple[float, float, float]]) -> Image.Image:
    det_visualizer = DetVisualizer(
        rgb,
        scale=1.0,
        instance_mode=ColorMode.SEGMENTATION,
        font_size_scale=1.0
    )

    for i, (box, class_id, info) in enumerate(zip(
        gt_instances.gt_boxes.tensor.cpu().numpy(),
        gt_instances.gt_classes.cpu().numpy(),
        gt_instances.infos if hasattr(gt_instances, "infos") else [{"filtered_low_area": False, "filtered_low_visibility": False, "visibility_fraction": 0.0}] * len(gt_instances.gt_boxes))
    ):
        x1, y1, x2, y2 = box
        height_ratio = (y2 - y1) / rgb.shape[0]
        font_size = (
            np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * det_visualizer._default_font_size
        )
        det_visualizer.draw_text(
            f"{i}: gt={classes[class_id]}",
            font_size=font_size,
            position=(x1, y2 - 1.5 * font_size),
            horizontal_alignment="left",
            color="#{:02x}{:02x}{:02x}".format(*colors[class_id])
        )

    vis_img = det_visualizer.overlay_instances(
        boxes=gt_instances.gt_boxes.tensor.cpu().numpy(),
        masks=gt_instances.gt_masks.cpu().numpy() if gt_instances.has("gt_masks") else None,
        labels=None,
        assigned_colors=[(x/255, y/255, z/255) for class_id in gt_instances.gt_classes.cpu().numpy() for x, y, z in [colors[class_id]]],
        alpha=0.5
    )
    result = vis_img.get_image()
    im = Image.fromarray(result)
    return im


def plot_segmentation(
    rgb: np.ndarray, 
    pred_instances, 
    gt_instances, 
    classes: list[str], 
    colors: list[tuple[float, float, float]], 
    title: str = ""
) -> Image.Image:
    
    if gt_instances is not None:
        im = plot_segmentation_gt(rgb, gt_instances, classes, [(255, 0, 0) for _ in classes] if pred_instances is not None else colors)
        rgb = np.array(im)

    if pred_instances is not None:
        im = plot_segmentation_pred(rgb, pred_instances, classes, colors)
        rgb = np.array(im)

    if title:
        fontscale = 0.5
        cv2.putText(
            rgb,
            title,
            (int((rgb.shape[1] / 2)), int(fontscale * 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontscale,
            (255, 255, 255),
            thickness=2,
        )
        im = Image.fromarray(rgb)
    return im

def plot_exploration_map(top_down_map: np.ndarray, revealed_mask: np.ndarray, trajectory_px: list, frontier_px: Optional[np.ndarray] = None) -> np.ndarray:
    """BGR top-down map: unrevealed navigable area gray, revealed green, non-navigable black, (row, col) trajectory as red dots linked by arrows,
    current position circled in red, frontier_px as a purple cross."""
    image = np.full(top_down_map.shape + (3,), 100, dtype=np.uint8)
    image[(revealed_mask > 0) & (top_down_map > 0)] = (127, 200, 127)
    image[top_down_map == 0] = (0, 0, 0)
    points = [(int(col), int(row)) for row, col in trajectory_px]

    for point in points:
        cv2.circle(image, point, 1, (0, 0, 255), -1)

    for start, end in zip(points, points[1:]):
        cv2.arrowedLine(image, start, end, (255, 255, 0), 2, tipLength=0.2)
    if points:
        cv2.circle(image, points[-1], 5, (0, 0, 255), 2)
    if frontier_px is not None:
        cv2.drawMarker(image, (int(frontier_px[1]), int(frontier_px[0])), (128, 0, 128), cv2.MARKER_CROSS, 10, 2)
    return image


def _padded_row(tiles: list[np.ndarray], width: int) -> np.ndarray:
    return np.hstack(tiles + [np.zeros_like(tiles[0])] * (width - len(tiles)))


def make_mosaic(
    list_fnames_images: list[tuple[str, np.ndarray]],
    target_size: int = 1_000_000,
    N_cols: int = 4,
    tile_colors: Optional[list[Optional[tuple[int, int, int]]]] = None,
    row_sizes: Optional[list[int]] = None,
) -> Image.Image:
    """Grid of the labelled tiles, N_cols per row, or row_sizes[i] tiles on row i (padded to the longest row) when given."""
    n =  len(list_fnames_images)
    processed_images = []

    for i, (filename, img) in enumerate(list_fnames_images):
        # add text overlay with filename
        cv2.putText(
            img,
            filename,
            (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if tile_colors and tile_colors[i] is not None:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), tile_colors[i], 10)
        processed_images.append(img)


    width = max(row_sizes) if row_sizes else N_cols
    if row_sizes is None:
        row_sizes = [min(N_cols, n - start) for start in range(0, n, N_cols)]
    bounds = np.cumsum([0, *row_sizes])
    rows = [_padded_row(processed_images[start:end], width) for start, end in zip(bounds[:-1], bounds[1:]) if end > start]

    final_mosaic = np.vstack(rows)

    downscale_factor = np.ceil(np.sqrt(target_size / (final_mosaic.shape[0] * final_mosaic.shape[1])))
    final_mosaic = cv2.resize(
        final_mosaic,
        (
            int(final_mosaic.shape[1] * downscale_factor),
            int(final_mosaic.shape[0] * downscale_factor),
        ),
    )
    return Image.fromarray(final_mosaic)