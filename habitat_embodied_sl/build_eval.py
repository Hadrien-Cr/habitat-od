"""Builds an evaluation scene's shard of a split, val or test, of the evaluation dataset <dataset_root>/<run_name>/ from its
record in the config's raw run (common/utils/raw_scene.py's save_test_scene, written by habitat_embodied_sl/collect_test.py),
without the simulator: the dataset's classes are the raw run's recorded ones its vocab_name has (vocabulary_classes), and
every recorded frame showing some ground truth of them goes to <split>/, with that ground truth, indexed by them, as its
annotations. One call per scene, so calls run in parallel, without a GPU; habitat_embodied_sl/merge.py then pools each
split's shards. A call first deletes what an earlier one wrote for its scene (flush_scene). The dataset's first call saves
the config, its classes added, as <dataset_root>/<run_name>/config.yaml, and a call with another config fails
(save_run_config), so the val and test splits of a dataset share its classes.

Usage:
  PYTHONPATH=. python habitat_embodied_sl/build_eval.py <scene> habitat_embodied_sl/config/procthor30_eval_build.yaml test
  PYTHONPATH=. python habitat_embodied_sl/merge.py datasets/procthor30_eval  # once every scene of both splits is built
"""
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from common.env_utils.object_annotations import vocabulary_classes
from common.utils.dataset_writers import collection_root, flush_scene, split_writer
from common.utils.ground_truth_recovery import annotated_frames, class_id_mapping, restricted_views
from common.utils.raw_scene import is_recorded, load_rgbs, load_run_config, load_views, save_run_config

SPLITS = ("val", "test")


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    raw_dir = Path(cfg.raw_run) / args.scene
    if not is_recorded(raw_dir):
        raise ValueError(f"{raw_dir} holds no complete raw record: collect {args.scene} first")
    raw_classes = list(load_run_config(raw_dir.parent).classes)
    cfg.classes = vocabulary_classes(raw_classes, cfg.vocab_name)
    root = collection_root(cfg)
    save_run_config(root, cfg)
    flush_scene(root, args.scene)
    views = restricted_views(load_views(raw_dir), class_id_mapping(raw_classes, list(cfg.classes)))
    frames = annotated_frames(load_rgbs(raw_dir), views)
    split_writer(root / args.split, cfg).write_scene(args.scene, frames)
    print(f"{args.scene}: {len(frames)} frames written into {root / args.split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="Evaluation scene to build, e.g. ProcTHOR-Val-0.")
    parser.add_argument("config", help="Evaluation dataset config: run_name, raw_run, dataset_root, vocab_name.")
    parser.add_argument("split", choices=SPLITS, help="The split the scene belongs to.")
    main(parser.parse_args())
