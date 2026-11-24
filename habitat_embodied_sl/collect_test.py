"""Records an evaluation scene raw (common/utils/raw_scene.py's save_test_scene) under <raw_root>/<run_name>/<scene>/ for
habitat_embodied_sl/build_eval.py to build its val or test split from, without the simulator, for any class set (see
habitat_embodied_sl/config/procthor_val40_collection.yaml and habitat_embodied_sl/README.md): a FrontierExplorationAgent
explores the scene (habitat_embodied_sl/evaluation_set.py), and every frame_stride-th step's frame showing some ground
truth is recorded with every valid ground-truth instance of the config's kept classes. One scene per call, so calls run in
parallel, each on the GPU its caller's CUDA_VISIBLE_DEVICES names. A call whose scene already has a complete record skips
it unless --overwrite; otherwise it first deletes the scene's record, and writes the explorer's final map and planner
dumps there too. The run's first call saves the config, its kept classes added as classes, as
<raw_root>/<run_name>/config.yaml, and a call with another config fails (save_run_config).

Usage:
  export BASE_DIR=$(pwd) PYTHONNOUSERSITE=1  # HABITAT_DATA already exported, see INSTALL.MD
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python habitat_embodied_sl/collect_test.py <scene> habitat_embodied_sl/config/procthor_val40_collection.yaml
"""
import argparse
import shutil

from omegaconf import OmegaConf

from common.env_utils.object_annotations import kept_classes_from_args
from common.utils.raw_scene import is_recorded, raw_scene_dir, save_run_config, save_test_scene
from habitat_embodied_sl.evaluation_set import evaluation_frames


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    cfg.classes = kept_classes_from_args(cfg.object_params)
    raw_dir = raw_scene_dir(cfg, args.scene)
    save_run_config(raw_dir.parent, cfg)
    if is_recorded(raw_dir) and not args.overwrite:
        print(f"{args.scene} already recorded in {raw_dir}, skipped (--overwrite recollects it)")
        return
    shutil.rmtree(raw_dir, ignore_errors=True)
    frames = evaluation_frames(args.scene, cfg, list(cfg.classes), raw_dir)
    kept = [(step, rgb, ground_truth) for step, rgb, ground_truth in frames if step % cfg.frame_stride == 0 and len(ground_truth)]
    save_test_scene(raw_dir, args.scene, kept)
    print(f"{args.scene}: {len(kept)} frames recorded in {raw_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", help="Evaluation scene to collect, e.g. ProcTHOR-Val-0.")
    parser.add_argument("config", help="Collection config: run_name, raw_root, seed, image size, object_params, frontier_agent, frame_stride.")
    parser.add_argument("--overwrite", action="store_true", help="Recollect the scene even if its raw record is complete.")
    main(parser.parse_args())
