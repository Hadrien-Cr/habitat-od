"""Mosaic of LVIS v1 train images: one row of up to 10 per LVIS class whose CLIP text embedding is at least --sim-word-thresh
cosine-similar to some --vocab word's (both prompted as `a <name>`; common/vision/detic.py's matched_classes, the classes
lvis_val.py's test set starts from), rows from most to least similar, each image with the class's boxes drawn. Saved to
--output, split into parts of at most MAX_MOSAIC_ROWS rows.

Usage:
  BASE_DIR=$(pwd) PYTHONPATH=. python habitat_embodied_sl/static_datasets/lvis_class_mosaic.py --vocab "trash can" sofa --sim-word-thresh 0.9 \
      --output outputs/lvis_class_mosaic.jpg
"""
import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from detectron2.data import DatasetCatalog  # type: ignore
from detectron2.data.detection_utils import read_image  # type: ignore

from common.utils.plot_utils import MAX_MOSAIC_ROWS, chunked, letterboxed, rows_mosaic, save_mosaic_parts
from common.vision.detic import LVIS, matched_classes, register_lvis_v1

IMAGES_PER_CLASS = 10
TILE_WIDTH, TILE_HEIGHT = 480, 360
BOX_COLOR = (0, 255, 0)


def class_records(dataset_dicts: list[dict], class_ids: set[int]) -> dict[int, list[dict]]:
    """Each of class_ids' records among dataset_dicts, those with at least one of its boxes."""
    records = defaultdict(list)
    for record in dataset_dicts:
        for class_id in {annotation["category_id"] for annotation in record["annotations"]} & class_ids:
            records[class_id].append(record)
    return records


def class_tile(record: dict, class_id: int, label: str) -> tuple[str, np.ndarray]:
    """record's image with class_id's boxes drawn, letterboxed to a tile, under label."""
    image = read_image(record["file_name"], format="RGB").copy()
    for annotation in record["annotations"]:
        if annotation["category_id"] == class_id:
            x, y, width, height = map(int, annotation["bbox"])
            cv2.rectangle(image, (x, y), (x + width, y + height), BOX_COLOR, 4)
    return label, letterboxed(image, TILE_WIDTH, TILE_HEIGHT)


def class_row(records: list[dict], class_id: int, label: str) -> list[tuple[str, np.ndarray]]:
    """Tiles of up to IMAGES_PER_CLASS of records, picked at random (seed 0), the first under label, the others under their image id."""
    picked = np.random.default_rng(0).choice(len(records), size=min(IMAGES_PER_CLASS, len(records)), replace=False)
    labels = [label] + [str(records[index]["image_id"]) for index in picked[1:]]
    return [class_tile(records[index], class_id, tile_label) for index, tile_label in zip(picked, labels)]


def main(args: argparse.Namespace) -> None:
    matches = matched_classes(LVIS, args.vocab, args.sim_word_thresh)
    if not matches:
        raise SystemExit(f"No LVIS class is at least {args.sim_word_thresh}-similar to any of {args.vocab}.")
    register_lvis_v1(["lvis_v1_train"])
    records = class_records(DatasetCatalog.get("lvis_v1_train"), {class_id for class_id, _, _ in matches})
    rows = [class_row(records[class_id], class_id, f"{LVIS.classes[class_id]}\n~{word} {similarity:.2f}") for class_id, word, similarity in matches]
    save_mosaic_parts([rows_mosaic(part) for part in chunked(rows, MAX_MOSAIC_ROWS)], args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab", nargs="+", required=True, help="Words to look up among LVIS's classes.")
    parser.add_argument("--sim-word-thresh", type=float, default=0.9, help="Minimum CLIP text cosine similarity to a --vocab word.")
    parser.add_argument("--output", type=Path, default=Path("outputs/lvis_class_mosaic.jpg"))
    main(parser.parse_args())
