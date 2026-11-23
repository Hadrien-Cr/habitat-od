"""Saves the query banks habitat_od_search/query_bank.py loads, for every episode a main.py run of --config would run
(the same selection, first_episode and max_episodes): for each distinct (scene, object_category), enhanced_detector.py's
save_gt_views ("gt_views": simulator views of its goal objects, every other object removed) and/or save_scraped_views
("web": Bing images of its fine-grained string) into query_bank.bank_dir, with their crops and mosaics. Banks already
saved are kept, so an interrupted run resumes; targets that fail (e.g. no goal view passes the ground-truth filters, no
web image found) are listed in QUERY_BANKS_DIR/failures_<run_name>.tsv.

Usage:
  export BASE_DIR=$(pwd) PYTHONNOUSERSITE=1  # HABITAT_DATA already exported, see INSTALL.MD
  PYTHONPATH=. python habitat_od_search/build_query_banks.py --config habitat_od_search/logs/configs/<run>.yaml [--banks gt_views web]
"""
import argparse
import itertools
from typing import List, Tuple

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from common.env_utils.env_base import build_env
from common.vision.detic import Vocabulary
from common.vision.detector import Detector
from habitat_od_search.enhanced_detector import load_target, save_gt_views, save_scraped_views
from habitat_od_search.hssd_rare_dataset import available_scenes, load_scene_episodes
from habitat_od_search.main import selected_episodes
from habitat_od_search.query_bank import BANK_FILES, QUERY_BANKS_DIR, bank_dir


def run_targets(cfg: DictConfig) -> List[Tuple[str, str]]:
    """Distinct (scene_id, object_category) of the episodes a main.py run of cfg runs, in run order."""
    first = cfg.get("first_episode", 0)
    end = None if cfg.max_episodes is None else first + cfg.max_episodes
    targets, n_selected = {}, 0
    for scene_id in cfg.scenes or available_scenes():
        env, _ = build_env(scene_id, cfg.seed, cfg.agent.max_steps, cfg.object_params, cfg.agent.input_image_size)
        episodes = selected_episodes(cfg, load_scene_episodes(scene_id), env.get_pathfinder())
        env.close()
        window = episodes[max(first - n_selected, 0):None if end is None else max(end - n_selected, 0)]
        targets.update({(scene_id, episode.info["object_category"]): None for episode in window})
        n_selected += len(episodes)
        if end is not None and n_selected >= end:
            break
    return list(targets)


def build_bank(cfg: DictConfig, detector: Detector, target: Tuple[str, str], bank: str) -> None:
    """Saves target's bank to its bank_dir, unless already saved."""
    scene_id, category = target
    out_dir = bank_dir(scene_id, category)
    if (out_dir / BANK_FILES[bank]).exists():
        return
    if bank == "gt_views":
        save_gt_views(cfg, load_target(scene_id, category), detector, out_dir)
    else:
        save_scraped_views(out_dir, detector, category)


def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)
    detector = Detector(args.config_file, args.checkpoint, Vocabulary("bank_unused", ("object",)), None, input_size=cfg.agent.input_image_size)
    failures = []
    for target, bank in tqdm(list(itertools.product(run_targets(cfg), args.banks)), desc="query banks", ncols=80):
        try:
            build_bank(cfg, detector, target, bank)
        except (AssertionError, ValueError, OSError) as error:
            failures.append("\t".join([*target, bank, str(error)]))
    QUERY_BANKS_DIR.mkdir(parents=True, exist_ok=True)
    (QUERY_BANKS_DIR / f"failures_{cfg.run_name}.tsv").write_text("".join(f"{line}\n" for line in failures))
    print(f"{len(failures)} failed banks, listed in {QUERY_BANKS_DIR / f'failures_{cfg.run_name}.tsv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="main.py run config whose episodes need banks.")
    parser.add_argument("--banks", nargs="+", default=list(BANK_FILES), choices=list(BANK_FILES))
    parser.add_argument("--config-file", default="third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    parser.add_argument("--checkpoint", default="third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth")
    main(parser.parse_args())
