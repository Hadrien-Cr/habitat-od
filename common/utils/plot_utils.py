import json
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from matplotlib.pyplot import text
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from PIL import Image, ImageDraw
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from sklearn.manifold import TSNE
import cv2
from detectron2.utils.visualizer import Visualizer as DetVisualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import Metadata

from common.env_utils.object_detector_sensors import FILTER_KEYS
from common.env_utils.sense import filter_reason
from common.hybrid_exploration.plotting import plot_occupancy

# Fixed-order categorical hues (validated for CVD/contrast) -- reused as-is,
# never cycled per-key, so the same slot always means the same series.
_SERIES_COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
_CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_TILE_BORDER_COLOR = (255, 255, 0)
_TILE_BORDER_THICKNESS = 2


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


def class_ap_history(metrics_json: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    """(each evaluation's iteration, {class: its AP50 at each evaluation}) of a run's metrics_json."""
    evaluations = [record for record in load_metrics(metrics_json) if "bbox/AP" in record]
    class_keys = [key for key in evaluations[0] if key.startswith("bbox/AP50-")]
    aps = {key[len("bbox/AP50-"):]: np.array([evaluation[key] for evaluation in evaluations]) for key in class_keys}
    return [evaluation["iteration"] for evaluation in evaluations], aps


def plot_class_ap(iterations: list[int], class_aps: dict[str, np.ndarray], out_path: Path) -> Path:
    """Heatmap of each class's AP50 (row) at each evaluation (column), written in its cell, rows sorted by their change
    since the first."""
    ordered = sorted(class_aps.items(), key=lambda item: item[1][-1] - item[1][0])
    fig, ax = plt.subplots(figsize=(2.5 + 0.6 * len(iterations), 0.22 * len(ordered) + 1.2))
    values = np.stack([aps for _, aps in ordered])
    image = ax.imshow(values, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    _annotate_cells(image, values, fontsize=6)
    ax.set_xticks(np.arange(len(iterations)), iterations, fontsize=7)
    ax.set_yticks(np.arange(len(ordered)), [f"{name} ({aps[-1] - aps[0]:+.1f})" for name, aps in ordered], fontsize=7)
    ax.set_xlabel("iteration")
    fig.colorbar(image, ax=ax, label="AP50")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_box_areas(split_areas: dict[str, np.ndarray], out_path: Path) -> Path:
    """Each split's histogram of box side sqrt(area), with COCO's small/medium/large bounds and shares."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.geomspace(4, 2048, 40)
    for index, (split, areas) in enumerate(split_areas.items()):
        shares = np.histogram(areas, bins=[0, 32 ** 2, 96 ** 2, np.inf])[0] / len(areas)
        label = f"{split} ({len(areas)}): " + "/".join(f"{share:.0%}" for share in shares) + " S/M/L"
        ax.hist(np.sqrt(areas), bins=bins, density=True, histtype="step", linewidth=2, color=_SERIES_COLORS[index], label=label)
    for bound in (32, 96):
        ax.axvline(bound, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("box side sqrt(area), pixels")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_embedding_tsne(embeddings: np.ndarray, names: list[str], out_path: Path) -> Path:
    """t-SNE of embeddings' (N, D) rows by cosine distance, each point labelled by its name (labels pushed apart until
    none overlap) and colored by its group in the cosine Ward tree cut into len(_CLUSTER_COLORS) groups."""
    points = TSNE(perplexity=min(15, len(names) - 1), metric="cosine", init="pca", random_state=0).fit_transform(embeddings)
    groups = fcluster(_cosine_linkage(embeddings), len(_CLUSTER_COLORS), criterion="maxclust")
    fig, ax = plt.subplots(figsize=(16, 16))
    for group in np.unique(groups):
        ax.scatter(*points[groups == group].T, s=24, zorder=4, color=_CLUSTER_COLORS[group - 1], label=f"group {group} ({(groups == group).sum()})")
    ax.legend(loc="upper left", fontsize=8, title="cosine Ward group", title_fontsize=8)
    labels = [
        ax.annotate(name, point, xytext=(0, 4), textcoords="offset points", fontsize=7, ha="center", va="bottom", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", linewidth=0.5, color="0.6", shrinkA=0, shrinkB=5))
        for point, name in zip(points, names)
    ]
    ax.set_xticks([])
    ax.set_yticks([])
    _separate_labels(fig, labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _separate_labels(fig, labels: list, max_steps: int = 500) -> None:
    """Shifts each overlapping pair of labels apart along the axis they overlap least on, until no two text boxes overlap."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points_per_pixel = 72 / fig.dpi
    for _ in range(max_steps):
        boxes = [Text.get_window_extent(label, renderer) for label in labels]
        pairs = [(first, second) for first, second in combinations(range(len(labels)), 2) if boxes[first].overlaps(boxes[second])]
        if not pairs:
            return
        for first, second in pairs:
            shift = _separating_shift(boxes[first], boxes[second]) * points_per_pixel / 2
            labels[first].xyann = tuple(np.add(labels[first].xyann, -shift))
            labels[second].xyann = tuple(np.add(labels[second].xyann, shift))


def _separating_shift(first, second) -> np.ndarray:
    """The pixel shift moving second off first along the axis they overlap least on, with a 1-pixel margin."""
    overlap = np.array([min(first.x1, second.x1) - max(first.x0, second.x0), min(first.y1, second.y1) - max(first.y0, second.y0)]) + 1
    direction = np.where(np.array([second.x0 + second.x1 - first.x0 - first.x1, second.y0 + second.y1 - first.y0 - first.y1]) >= 0, 1, -1)
    axis = int(np.argmin(overlap))
    return np.eye(2)[axis] * overlap[axis] * direction[axis]


def plot_similarity_triangle(embeddings: np.ndarray, names: list[str], out_path: Path) -> Path:
    """Lower triangle of embeddings' (N, D) rows' pairwise cosine similarity × 100, written in each cell and colored over
    its off-diagonal range, rows ordered by the cosine Ward tree's leaves so similar ones form blocks along the diagonal."""
    order = leaves_list(_cosine_linkage(embeddings))
    names = [names[index] for index in order]
    unit = embeddings[order] / np.linalg.norm(embeddings[order], axis=1, keepdims=True)
    similarity = np.ma.masked_where(np.triu(np.ones((len(names), len(names)), dtype=bool)), 100 * unit @ unit.T)
    fig, ax = plt.subplots(figsize=(0.2 * len(names) + 3, 0.2 * len(names) + 2))
    image = ax.imshow(similarity, cmap="Blues", interpolation="nearest")
    _annotate_cells(image, similarity, fontsize=3.5)
    ax.set_xticks(np.arange(len(names)), names, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(names)), names, fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.5, label="cosine similarity × 100")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _annotate_cells(image, values: np.ndarray, fontsize: float) -> None:
    """Writes each unmasked, non-NaN cell's value, rounded, at its center, in black or white by its color's luminance."""
    for (row, column), value in np.ndenumerate(np.ma.filled(values.astype(float), np.nan)):
        if np.isnan(value):
            continue
        red, green, blue, _ = image.cmap(image.norm(value))
        color = "black" if 0.299 * red + 0.587 * green + 0.114 * blue > 0.5 else "white"
        image.axes.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=fontsize, color=color)


def _cosine_linkage(embeddings: np.ndarray) -> np.ndarray:
    """Ward linkage of embeddings' L2-normalized rows, whose Euclidean distance is monotonic in their cosine distance."""
    return linkage(embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True), method="ward")


def plot_class_counts(counts: dict[str, dict[str, np.ndarray]], classes: list[str]) -> np.ndarray:
    """RGB image of one horizontal bar chart per split of its count per class, stacked by scene, then one of
    every split's total stacked, side by side on a shared class axis."""
    fig, axes = plt.subplots(1, len(counts) + 1, figsize=(5 * (len(counts) + 1), 0.18 * len(classes) + 1), sharey=True, squeeze=False)
    totals = {
        split: _stacked_barh(ax, scene_counts, plt.get_cmap("BuPu" if split == "train" else "GnBu")(np.linspace(0.5, 1.0, len(scene_counts))))
        for ax, (split, scene_counts) in zip(axes.flat, counts.items())
    }
    _stacked_barh(axes[0, -1], totals, _SERIES_COLORS)
    titles = [*counts, " + ".join(counts)]
    for ax, title, total in zip(axes.flat, titles, [*totals.values(), sum(totals.values())]):
        ax.set_title(f"{title} ({int(total.sum())})", fontsize=9)
        ax.set_xlabel("count")
        ax.grid(True, axis="x", alpha=0.3)
    axes[0, 0].set_yticks(np.arange(len(classes)), classes, fontsize=7)
    axes[0, 0].set_ylim(len(classes) - 0.5, -0.5)
    fig.tight_layout()
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return image


def _stacked_barh(ax, series_counts: dict[str, np.ndarray], colors: list) -> np.ndarray:
    """Draws series_counts on ax as horizontal bars stacked per class, with a legend; returns their per-class total."""
    stacked = np.zeros(len(next(iter(series_counts.values()))))
    for index, (series, counts) in enumerate(series_counts.items()):
        ax.barh(np.arange(len(stacked)), counts, left=stacked, color=colors[index % len(colors)], label=series)
        stacked = stacked + counts
    ax.legend(fontsize=6, loc="lower right")
    return stacked


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


_UNEXPLORED_COLOR = (255, 255, 255)
_EXPLORED_FREE_COLOR = (225, 225, 225)
_EXPLORED_OBSTACLE_COLOR = (90, 90, 90)


def plot_semantic_2d_map(
    bg_grid,
    footprints: list[tuple[int, np.ndarray]],
    colors: dict[int, tuple[int,int,int]],
    classes: dict[int, str],
    meters_per_grid_pixel: float = 0.01,
    scale=1.0,
    explored: Optional[np.ndarray] = None,
    text_color: tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """bg_grid's navigable map with each object's (class, top-down mask) footprint colored by class, labelled once at its
    centroid. Without explored, non-navigable cells are black and navigable ones flat grey; with it (the agent's own
    explored grid), cells split three ways: white unexplored, light grey explored-free, dark grey explored-obstacle."""
    footprints = sorted(footprints, key=lambda footprint: footprint[1].sum(), reverse=True)
    if explored is None:
        colored = np.zeros((*bg_grid.shape, 3), dtype=np.uint8)
    else:
        colored = np.full((*bg_grid.shape, 3), _UNEXPLORED_COLOR, dtype=np.uint8)
        colored[explored.astype(bool) & ~bg_grid.astype(bool)] = _EXPLORED_OBSTACLE_COLOR
    colored[bg_grid.astype(bool)] = _EXPLORED_FREE_COLOR
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
        y, x = np.argwhere(mask).mean(axis=0) * scale
        draw.text((x, y), classes[class_id], fill=text_color, font_size= 4 * np.clip(np.sqrt(area_m), 1, 4) * scale, anchor="mm")

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
        font_size_scale=scale
    )

    for i, (box, class_id, score) in enumerate(zip(
        pred_instances.pred_boxes.tensor.cpu().numpy(),
        pred_instances.pred_classes.cpu().numpy(),
        pred_instances.scores.cpu().numpy()
    )):
        x1, y1, x2, y2 = box
        height_ratio = (y2 - y1) / rgb.shape[0]
        font_size = (
            np.clip((height_ratio - 0.02) / 0.08 + 1, 1, 2) * 0.5 * det_visualizer._default_font_size
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


def plot_segmentation_gt(rgb: np.ndarray, gt_instances, classes: list[str], colors: list[tuple[float, float, float]], scale: float = 1.0) -> Image.Image:
    det_visualizer = DetVisualizer(
        rgb,
        scale=scale,
        instance_mode=ColorMode.SEGMENTATION,
        font_size_scale=scale
    )

    for i, (box, class_id, info) in enumerate(zip(
        gt_instances.gt_boxes.tensor.cpu().numpy(),
        gt_instances.gt_classes.cpu().numpy(),
        gt_instances.infos if hasattr(gt_instances, "infos") else [dict.fromkeys(FILTER_KEYS, False)] * len(gt_instances.gt_boxes))
    ):
        x1, y1, x2, y2 = box
        height_ratio = (y2 - y1) / rgb.shape[0]
        font_size = (
            np.clip((height_ratio - 0.02) / 0.08 + 1, 1, 2) * 0.5 * det_visualizer._default_font_size
        )
        det_visualizer.draw_text(
            f"{i}: gt={classes[class_id]} {filter_reason(info)}",
            font_size=font_size,
            position=(x1, min(y2-1.5*font_size, rgb.shape[0])),
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


def _label_font_size(visualizer: DetVisualizer, box_height: float) -> float:
    """detectron2's overlay_instances label size, growing with the box's height relative to the image, its floor halved:
    0.3 to 1 times the visualizer's default size."""
    height_ratio = box_height / np.sqrt(visualizer.output.height * visualizer.output.width)
    return float(np.clip((height_ratio - 0.02) / 0.08 + 1, 0.6, 2) * 0.5 * visualizer._default_font_size)


def plot_boxes_overlay(rgb: np.ndarray, gt_instances, pred_instances, metadata: Metadata, scale: float) -> Image.Image:
    """rgb resized by scale, gt_instances' boxes shaded, dash-outlined and labelled `gt=<class>` under them, pred_instances'
    boxes outlined and labelled `<class> <score>` inside their top, each in its class's metadata.thing_colors, labels sized by
    their box (_label_font_size); each set drawn largest box first, so smaller boxes and their labels stay on top."""
    visualizer = DetVisualizer(rgb, scale=scale)
    gt_instances = gt_instances[gt_instances.gt_boxes.area().argsort(descending=True)]
    pred_instances = pred_instances[pred_instances.pred_boxes.area().argsort(descending=True)]
    for (x1, y1, x2, y2), class_id in zip(gt_instances.gt_boxes.tensor.tolist(), gt_instances.gt_classes.tolist()):
        color = np.array(metadata.thing_colors[class_id]) / 255
        visualizer.output.ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=color, edgecolor="none", alpha=0.35))
        visualizer.draw_box([x1, y1, x2, y2], alpha=1.0, edge_color=color, line_style="--")
        visualizer.draw_text(f"gt={metadata.thing_classes[class_id]}", (x1, y2), color=color, horizontal_alignment="left", font_size=_label_font_size(visualizer, y2 - y1))
    for box, class_id, score in zip(pred_instances.pred_boxes.tensor.tolist(), pred_instances.pred_classes.tolist(), pred_instances.scores.tolist()):
        color = np.array(metadata.thing_colors[class_id]) / 255
        visualizer.draw_box(box, alpha=1.0, edge_color=color)
        label = f"{metadata.thing_classes[class_id]} {score:.2f}"
        visualizer.draw_text(label, (box[0], box[1]), color=color, horizontal_alignment="left", font_size=_label_font_size(visualizer, box[3] - box[1]))
    return Image.fromarray(visualizer.output.get_image())


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

def plot_exploration_map(
    top_down_map: np.ndarray, revealed_mask: np.ndarray, trajectory_px: list, frontier_px: Optional[np.ndarray] = None, goal_px: Optional[np.ndarray] = None,
) -> np.ndarray:
    """BGR top-down map: unrevealed navigable area gray, revealed green, non-navigable black, (row, col) trajectory as red dots linked by arrows,
    current position circled in red, frontier_px as a purple cross, goal_px (e.g. a detected search target) as a green cross."""
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
    if goal_px is not None:
        cv2.drawMarker(image, (int(goal_px[1]), int(goal_px[0])), (0, 200, 0), cv2.MARKER_CROSS, 14, 3)
    return image


def box_crop(rgb: np.ndarray, box: list[int], size: int) -> np.ndarray:
    """The xyxy box of rgb, stretched to size x size."""
    x0, y0, x1, y1 = box
    return cv2.resize(rgb[y0:y1, x0:x1], (size, size))


def draw_ranked_classes(image: np.ndarray, names: list[str], probs: list[float], top: int = 50) -> np.ndarray:
    """image with one "name prob" line per class, each over a bar as wide as its probability, from pixel row top down."""
    line_height = 28
    for rank, (name, prob) in enumerate(zip(names, probs)):
        y = top + rank * line_height
        cv2.rectangle(image, (0, y), (image.shape[1] - 1, y + line_height - 4), (0, 0, 0), -1)
        cv2.rectangle(image, (0, y), (int(prob * (image.shape[1] - 1)), y + line_height - 4), (40, 110, 200), -1)
        cv2.putText(image, f"{name} {prob:.2f}", (5, y + line_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def _padded_row(tiles: list[np.ndarray], width: int) -> np.ndarray:
    return np.hstack(tiles + [np.zeros_like(tiles[0])] * (width - len(tiles)))


_LABEL_FONT_SCALE = 1.1
_LABEL_LINE_HEIGHT = 44


def _draw_label(img: np.ndarray, label: str) -> np.ndarray:
    """label's lines (split on "\\n"), stacked in a black bar at img's top."""
    lines = label.split("\n")
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, _LABEL_LINE_HEIGHT * len(lines)), (0, 0, 0), -1)
    for row, line in enumerate(lines):
        y = _LABEL_LINE_HEIGHT * row + 32
        cv2.putText(img, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, _LABEL_FONT_SCALE, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def _framed_tile(img: np.ndarray, label: str, tile_color: Optional[tuple[int, int, int]] = None) -> np.ndarray:
    """A copy of img with its label and yellow border drawn on it, plus a tile_color border when given."""
    img = img.copy()
    _draw_label(img, label)
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), _TILE_BORDER_COLOR, _TILE_BORDER_THICKNESS)
    if tile_color is not None:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), tile_color, 10)
    return img


MAX_MOSAIC_SIDE = 16384  # JPEG holds at most 65,500 pixels a side; larger mosaics aren't viewable anyway


def make_mosaic(
    list_fnames_images: list[tuple[str, np.ndarray]],
    target_size: int = 1_000_000,
    N_cols: int = 4,
    tile_colors: Optional[list[Optional[tuple[int, int, int]]]] = None,
    row_sizes: Optional[list[int]] = None,
) -> Image.Image:
    """Grid of the labelled, yellow-bordered tiles, N_cols per row, or row_sizes[i] tiles on row i (padded to the longest row) when
    given, scaled (aspect kept) to at least target_size pixels and at most MAX_MOSAIC_SIDE a side."""
    n =  len(list_fnames_images)
    processed_images = [
        _framed_tile(img, filename, tile_colors[i] if tile_colors else None)
        for i, (filename, img) in enumerate(list_fnames_images)
    ]

    width = max(row_sizes) if row_sizes else N_cols
    if row_sizes is None:
        row_sizes = [min(N_cols, n - start) for start in range(0, n, N_cols)]
    bounds = np.cumsum([0, *row_sizes])
    rows = [_padded_row(processed_images[start:end], width) for start, end in zip(bounds[:-1], bounds[1:]) if end > start]

    final_mosaic = np.vstack(rows)

    upscale = np.ceil(np.sqrt(target_size / (final_mosaic.shape[0] * final_mosaic.shape[1])))
    scale = min(upscale, MAX_MOSAIC_SIDE / max(final_mosaic.shape[:2]))
    size = (int(final_mosaic.shape[1] * scale), int(final_mosaic.shape[0] * scale))
    return Image.fromarray(cv2.resize(final_mosaic, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR))


def rows_mosaic(rows: list[list[tuple[str, np.ndarray]]]) -> Optional[Image.Image]:
    """make_mosaic of one row per list of labelled tiles, or None without any tile."""
    tiles = [tile for row in rows for tile in row]
    return make_mosaic(tiles, row_sizes=[len(row) for row in rows]) if tiles else None


PNG_COMPRESS_LEVEL = 1  # PIL's default zlib level 6 encodes ~2x slower for files only ~1.5x smaller
MOSAIC_JPEG_QUALITY = 90


def chunked(items: list, size: Optional[int]) -> list[list]:
    """items in consecutive chunks of at most size each, or one chunk of them all without size."""
    step = size or max(len(items), 1)
    return [items[start:start + step] for start in range(0, len(items), step)]


MAX_MOSAIC_ROWS = 32


def save_mosaic_parts(mosaics: list[Image.Image], out_path: Path) -> None:
    """A single mosaic to out_path, several each to out_path's stem suffixed with its index, in out_path's format (PNG or JPEG)."""
    paths = [out_path] if len(mosaics) == 1 else [out_path.with_name(f"{out_path.stem}_{index:02d}{out_path.suffix}") for index in range(len(mosaics))]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for mosaic, path in zip(mosaics, paths):
        mosaic.save(path, compress_level=PNG_COMPRESS_LEVEL, quality=MOSAIC_JPEG_QUALITY)


def save_tile(tile: tuple, out_path: Path) -> None:
    """(label, image) tile as a single-tile mosaic, saved to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_mosaic([tile], N_cols=1).save(out_path, compress_level=PNG_COMPRESS_LEVEL)


def centered_crop(image: np.ndarray, center: np.ndarray, size: int) -> np.ndarray:
    """size x size window of image around (row, col) center, shifted inward to stay whole at image borders."""
    row0 = int(np.clip(center[0] - size // 2, 0, image.shape[0] - size))
    col0 = int(np.clip(center[1] - size // 2, 0, image.shape[1] - size))
    return image[row0:row0 + size, col0:col0 + size]


def letterboxed(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """image scaled to fit within (width, height) without distorting its aspect ratio, centered on black bars."""
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y0, x0 = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas



def save_gif(images_dir: Path, out_name: str = "episode.webp", fps: int = 6) -> None:
    """Assembles every snapshot_<step>.png in images_dir into images_dir/out_name, in step order, as lossless
    animated WebP -- GIF's 256-color-palette cap otherwise crowds out or merges colors across frames full of
    distinct anti-aliased shades (e.g. a rare green tint quantized into a nearby gray)."""
    frames = sorted(images_dir.glob("snapshot_gif_*.png"))
    if not frames:
        return
    images = [Image.open(frame).convert("RGB") for frame in frames]
    images[0].save(images_dir / out_name, format="WEBP", lossless=True, save_all=True, append_images=images[1:], duration=1000 // fps, loop=0)
