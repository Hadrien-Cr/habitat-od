"""Static COCO test sets of detection datasets with a broad vocabulary (lvis_val.py, objects365.py, egoobjects.py): a source's
records (detectron2's dataset dicts, category_id a class index), up to --max-images of those showing a class CLIP-matched to a
--vocab word (common/vision/detic.py's matched_classes), written as <out_dir>/test/test.json and its images in
<out_dir>/images/ by common/utils/dataset_writers.py's CocoFramesWriter, one scene named after the source; registered as
<out_dir's name>_test (common/env_utils/dataset_registry.py). Its vocabulary is vocabulary.py's frequent classes of the
matched classes' objects in the written images, sorted; the other classes' objects are left out.

A federated source (LVIS, EgoObjects) verifies a class on an image only where it is annotated exhaustively or listed as
absent: its records carry neg_category_ids and not_exhaustive_category_ids as class indices, and each category with a class
unverified on an image gets a whole-image iscrowd box there (federated), which COCOeval matches that category's unmatched
detections to and ignores, as LVIS's own evaluation does."""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from detectron2.data.detection_utils import read_image  # type: ignore
from detectron2.structures import Instances  # type: ignore

from common.utils.dataset_writers import IMAGES_DIR, CocoFramesWriter, box_instances
from common.vision.detic import Vocabulary, matched_classes
from habitat_embodied_sl.static_datasets.vocabulary import MIN_OCCURRENCES, frequent_classes


FrameInstances = Callable[[dict, tuple, dict], Instances]  # (record, image size, {class index: category}) -> ground truth


@dataclass(frozen=True)
class Source:
    vocabulary: Vocabulary
    records: list
    frame_instances: FrameInstances


def source_parser(description: str) -> argparse.ArgumentParser:
    """The arguments every source's builder takes: its out_dir, the words its classes are matched to, and its image count."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="Dataset directory; its test set goes to out_dir/test.")
    parser.add_argument("--vocab", nargs="+", required=True, help="Words to look up among the source's classes.")
    parser.add_argument("--sim-word-thresh", type=float, default=0.9, help="Minimum CLIP text cosine similarity to a --vocab word.")
    parser.add_argument("--max-images", type=int, help="Images to write, picked at random (default: every one showing a matched class).")
    return parser


def showing_records(records: list, class_indices: set, max_images) -> list:
    """Up to max_images (every one for None) of records showing one of class_indices, picked at random (seed 0), in order."""
    showing = [record for record in records if any(annotation["category_id"] in class_indices for annotation in record["annotations"])]
    return [showing[index] for index in sorted(np.random.default_rng(0).permutation(len(showing))[:max_images])]


def category_ids(records: list, classes: tuple, class_indices: set) -> tuple:
    """({class index: category index}, sorted categories) of class_indices' frequent classes among records' objects."""
    occurrences = (classes[annotation["category_id"]] for record in records for annotation in record["annotations"] if annotation["category_id"] in class_indices)
    named = frequent_classes(occurrences, ())
    categories = sorted(set(named.values()))
    return {index: categories.index(named[classes[index]]) for index in class_indices if classes[index] in named}, categories


def box_frame_instances(record: dict, image_size: tuple, class_ids: dict) -> Instances:
    """Ground truth of record's boxes whose class is in class_ids, each its filled rectangle, crowd boxes as iscrowd."""
    kept = [(index, annotation) for index, annotation in enumerate(record["annotations"]) if annotation["category_id"] in class_ids]
    boxes = np.array([annotation["bbox"] for _, annotation in kept], dtype=float).reshape(-1, 4)
    infos = [{"object_id": index, "iscrowd": annotation.get("iscrowd", 0)} for index, annotation in kept]
    return box_instances(boxes, [class_ids[annotation["category_id"]] for _, annotation in kept], infos, image_size)


def unverified_instances(record: dict, class_ids: dict, image_size: tuple) -> Instances:
    """A whole-image iscrowd box for each category of class_ids with a class unverified on record's federated image."""
    positives = {annotation["category_id"] for annotation in record["annotations"]}
    verified = (positives - set(record["not_exhaustive_category_ids"])) | set(record["neg_category_ids"])
    unverified = sorted({category for index, category in class_ids.items() if index not in verified})
    whole_image = np.tile([0, 0, image_size[1], image_size[0]], (len(unverified), 1))
    return box_instances(whole_image, unverified, [{"object_id": -1, "iscrowd": 1}] * len(unverified), image_size)


def federated(frame_instances: FrameInstances) -> FrameInstances:
    """frame_instances of a federated source, each record's unverified_instances added."""
    return lambda record, image_size, class_ids: Instances.cat([frame_instances(record, image_size, class_ids), unverified_instances(record, class_ids, image_size)])


def write_test_set(source: Source, args: argparse.Namespace) -> Path:
    """Writes source's images showing a class matched to args.vocab as args.out_dir/test, and returns that test directory."""
    matched = {index for index, _, _ in matched_classes(source.vocabulary, args.vocab, args.sim_word_thresh)}
    if not matched:
        raise SystemExit(f"No {source.vocabulary.name} class is at least {args.sim_word_thresh}-similar to any of {args.vocab}.")
    records = showing_records(source.records, matched, args.max_images)
    class_ids, categories = category_ids(records, source.vocabulary.classes, matched)
    info = {"source": source.vocabulary.name, "vocab": args.vocab, "sim_word_thresh": args.sim_word_thresh, "min_occurrences": MIN_OCCURRENCES}
    writer = CocoFramesWriter(args.out_dir / "test", info, categories, args.out_dir / IMAGES_DIR)
    rgbs = ((record, read_image(record["file_name"], format="RGB")) for record in records)
    writer.write_scene(source.vocabulary.name, ((record["image_id"], rgb, source.frame_instances(record, rgb.shape[:2], class_ids)) for record, rgb in rgbs))
    writer.merge(np.random.default_rng(0))
    return args.out_dir / "test"
