"""Test set of evaluation scenes, collected straight into its own dataset, datasets/<run_name>/test/, without a raw record
(see habitat_embodied_sl/config/procthor8_test.yaml and habitat_embodied_sl/README.md): a FrontierExplorationAgent explores
the scene (habitat_embodied_sl/evaluation_set.py), and every frame_stride-th step's frame showing some ground truth goes,
with every valid ground-truth instance of the config's kept classes, to the scene's shard of the COCO test split. One
scene per call, so calls run in parallel, each on the GPU its caller's CUDA_VISIBLE_DEVICES names;
habitat_embodied_sl/merge.py then pools the shards. A call first deletes what an earlier one wrote for its scene
(flush_scene), and writes the explorer's final map and planner dumps to datasets/<run_name>/<scene>/. The dataset's first
call saves the config, its kept classes added as classes, as datasets/<run_name>/config.yaml, and a call with another
config fails (save_run_config).

Usage:
  export BASE_DIR=$(pwd) PYTHONNOUSERSITE=1  # HABITAT_DATA already exported, see INSTALL.MD
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python habitat_embodied_sl/collect_test.py <scene> habitat_embodied_sl/config/procthor8_test.yaml
  PYTHONPATH=. python habitat_embodied_sl/merge.py datasets/procthor8_test  # once every scene is collected
"""
import argparse

from omegaconf import OmegaConf

from common.env_utils.object_annotations import kept_classes_from_args
from common.utils.dataset_writers import collection_root, flush_scene, split_writer
from common.utils.raw_scene import save_run_config
from habitat_embodied_sl.evaluation_set import evaluation_frames


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    cfg.classes = kept_classes_from_args(cfg.object_params)
    root = collection_root(cfg)
    save_run_config(root, cfg)
    flush_scene(root, args.scene)
    frames = evaluation_frames(args.scene, cfg, list(cfg.classes), root / args.scene)
    kept = [(step, rgb, ground_truth) for step, rgb, ground_truth in frames if step % cfg.frame_stride == 0 and len(ground_truth)]
    split_writer(root / "test", cfg).write_scene(args.scene, kept)
    print(f"{args.scene}: {len(kept)} test frames written into {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="Test scene to collect, e.g. ProcTHOR-Val-0.")
    parser.add_argument("config", help="Test config: run_name, dataset_root, seed, image size, object_params, frontier_agent, frame_stride.")
    main(parser.parse_args())
