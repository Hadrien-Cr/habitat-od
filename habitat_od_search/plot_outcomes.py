"""Plots one donut chart per habitat_od_search/main.py run directory, splitting its episodes by outcome (success and
each failure category from metrics.jsonl), with outer leader-line labels, into a single figure.

Example:
    PYTHONPATH=. python habitat_od_search/plot_outcomes.py habitat_od_search/logs/eval_200_gt \
        habitat_od_search/logs/eval_200_vanilla --out habitat_od_search/logs/failure_modes_pies.png
"""
import argparse
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.utils.plot_utils import _CLUSTER_COLORS, load_metrics
from habitat_od_search.evaluator import FailureCategory

_SURFACE, _INK, _MUTED = "#fcfcfb", "#1f1f1e", "#6b6a64"
_LABEL_MIN_GAP = 0.34


def outcome_order(run_counts: list[Counter]) -> list[str]:
    """FailureCategory's order, then any category only older runs logged, so each keeps one color across charts."""
    known = [category.value for category in FailureCategory]
    logged = sorted(set().union(*run_counts) - set(known))
    return [category for category in known + logged if any(counts[category] for counts in run_counts)]


def spread(heights: list[float]) -> np.ndarray:
    order = np.argsort(heights)
    placed = np.array(heights, dtype=float)
    for prev, cur in zip(order, order[1:]):
        placed[cur] = max(placed[cur], placed[prev] + _LABEL_MIN_GAP)
    return placed


def label_side(ax, wedges, texts: list[str], side: int) -> None:
    angles = [np.deg2rad((wedge.theta1 + wedge.theta2) / 2) for wedge in wedges]
    on_side = [i for i, angle in enumerate(angles) if (np.cos(angle) >= 0) == (side > 0)]
    heights = spread([1.25 * np.sin(angles[i]) for i in on_side])
    for i, height in zip(on_side, heights):
        ax.annotate(texts[i], xy=(np.cos(angles[i]), np.sin(angles[i])), xytext=(1.45 * side, height),
                    ha="left" if side > 0 else "right", va="center", fontsize=10, color=_INK,
                    arrowprops=dict(arrowstyle="-", color=_MUTED, lw=0.8,
                                    connectionstyle=f"angle,angleA=0,angleB={np.rad2deg(angles[i]):.0f}"))


def draw_donut(ax, counts: Counter, colors: dict[str, str], title: str) -> None:
    total = sum(counts.values())
    shown = [category for category in colors if counts[category]]
    wedges, _ = ax.pie([counts[c] for c in shown], colors=[colors[c] for c in shown], startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor=_SURFACE, linewidth=2))
    texts = [f"{outcome_name(c)}\n{100 * counts[c] / total:.1f}% ({counts[c]})" for c in shown]
    label_side(ax, wedges, texts, 1)
    label_side(ax, wedges, texts, -1)
    ax.text(0, 0.06, f"{100 * counts['success'] / total:.1f}%", ha="center", va="center", fontsize=17, fontweight="bold", color=_INK)
    ax.text(0, -0.17, "success", ha="center", va="center", fontsize=9.5, color=_MUTED)
    ax.set_title(f"{title}\nn = {total} episodes", fontsize=12, color=_INK, pad=14)
    ax.set(xlim=(-2.3, 2.3), ylim=(-1.6, 1.6))


def outcome_name(category: str) -> str:
    return category.removesuffix("_failure").replace("_", " ")


def plot_outcomes(run_dirs: list[Path], out_path: Path) -> Path:
    run_counts = [Counter(row["failure_category"] for row in load_metrics(run_dir / "metrics.jsonl")) for run_dir in run_dirs]
    colors = dict(zip(outcome_order(run_counts), _CLUSTER_COLORS))
    rows = math.ceil(len(run_dirs) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(15, 5.5 * rows), facecolor=_SURFACE, squeeze=False)
    for ax, run_dir, counts in zip(axes.flat, run_dirs, run_counts):
        draw_donut(ax, counts, colors, run_dir.name)
    for ax in axes.flat[len(run_dirs):]:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values()]
    fig.legend(handles, [outcome_name(c) for c in colors], loc="lower center", ncol=len(colors), frameon=False, fontsize=10.5, labelcolor=_INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="habitat_od_search/main.py run directories, each with a metrics.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="output image path")
    args = parser.parse_args()
    print(plot_outcomes(args.run_dirs, args.out))
