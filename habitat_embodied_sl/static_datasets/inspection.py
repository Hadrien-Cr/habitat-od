"""Views of a detection_sources.py source for its notebook (notebooks/{lvis,objects365,egoobjects}_visualization.ipynb):
example records, its classes CLIP-matched to a vocabulary, their object counts against MIN_OCCURRENCES, and, for a
federated source, how often each matched class is verified on the images showing it."""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from detectron2.data.catalog import Metadata  # type: ignore
from detectron2.data.detection_utils import read_image  # type: ignore
from detectron2.utils.visualizer import Visualizer  # type: ignore

from common.vision.detic import Vocabulary
from habitat_embodied_sl.static_datasets.detection_sources import category_ids
from habitat_embodied_sl.static_datasets.vocabulary import MIN_OCCURRENCES

VERIFICATION_STATES = ("annotated exhaustively", "annotated, not exhaustively", "listed absent", "unverified")


def show_records(records: list, classes: tuple, count: int = 4, seed: int = 0) -> None:
    """count of records picked at random, each drawn with its annotations (masks where it has them) and classes."""
    metadata = Metadata(thing_classes=list(classes))
    picked = np.random.default_rng(seed).choice(len(records), size=count, replace=False)
    _, axes = plt.subplots(1, count, figsize=(6 * count, 6))
    for ax, index in zip(axes, picked):
        drawn = Visualizer(read_image(records[index]["file_name"], format="RGB"), metadata).draw_dataset_dict(records[index])
        ax.imshow(drawn.get_image())
        ax.set_title(f"image {records[index]['image_id']}, {len(records[index]['annotations'])} objects")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def matched_table(vocabulary: Vocabulary, matches: list) -> pd.DataFrame:
    """matches (common/vision/detic.py's matched_classes of vocabulary) as a table, most similar first."""
    return pd.DataFrame([(vocabulary.classes[index], word, similarity) for index, word, similarity in matches], columns=["class", "most similar word", "similarity"])


def plot_matched_counts(records: list, classes: tuple, matched: set) -> None:
    """Object count of each matched class among records, log scale, the classes category_ids keeps in blue."""
    counts = Counter(classes[annotation["category_id"]] for record in records for annotation in record["annotations"] if annotation["category_id"] in matched)
    kept = {classes[index] for index in category_ids(records, classes, matched)[0]}
    names = sorted(counts, key=counts.get, reverse=True)
    _, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.18)))
    ax.barh(names, [counts[name] for name in names], color=["#2a78d6" if name in kept else "#b0b0b0" for name in names])
    ax.axvline(MIN_OCCURRENCES, color="black", linestyle="--", label=f"MIN_OCCURRENCES ({MIN_OCCURRENCES})")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("objects (log scale)")
    ax.set_title(f"matched classes' objects in the {len(records)} images showing one: {len(kept)} of {len(names)} kept")
    ax.legend()
    plt.tight_layout()
    plt.show()


def verification_state(record: dict, class_index: int) -> str:
    """class_index's VERIFICATION_STATES entry on record's federated image."""
    if class_index in record["neg_category_ids"]:
        return "listed absent"
    if all(annotation["category_id"] != class_index for annotation in record["annotations"]):
        return "unverified"
    return "annotated, not exhaustively" if class_index in record["not_exhaustive_category_ids"] else "annotated exhaustively"


def plot_verification(records: list, classes: tuple, matched: set) -> None:
    """Share of records in each VERIFICATION_STATES entry, per matched class: an unverified class's detections are ignored."""
    states = pd.DataFrame({classes[index]: Counter(verification_state(record, index) for record in records) for index in matched}).T
    shares = states.reindex(columns=list(VERIFICATION_STATES), fill_value=0).fillna(0)
    shares = shares.div(shares.sum(axis=1), axis=0).sort_values("unverified")
    shares.plot.barh(stacked=True, figsize=(8, max(4, len(shares) * 0.18)), color=["#2a78d6", "#8fb8ea", "#3aa676", "#d0d0d0"], width=0.85)
    plt.gca().invert_yaxis()
    plt.xlabel(f"share of the {len(records)} images showing a matched class")
    plt.title("where each matched class is verified")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.show()
