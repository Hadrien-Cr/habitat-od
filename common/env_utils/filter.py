"""The PROCTHOR30 class set (vocab_constants.py): the ProcTHOR-hab object classes large and thick enough, furniture
included, most placed in a set of scenes (procthor100_scene_classes.csv), by default the 60 of procthor30's train, test and
val sets, ProcTHOR-Train-0 to -19 and ProcTHOR-Val-0 to -39.

Usage:
  PYTHONPATH=. python common/env_utils/filter.py
"""
import argparse

import pandas as pd

from common.env_utils.object_annotations import PROCTHOR100_SCENE_CLASSES_PATH, PROCTHOR100_SIZES_PATH
from common.env_utils.vocab_constants import CLASS_LABELS_PROCTHOR100


def realistic_classes(scenes: list[str], count: int = 30, min_largest_edge: float = 0.2, min_shortest_edge: float = 0.1) -> list[str]:
    """The count object classes most placed in scenes whose largest instance is at least min_largest_edge long and
    min_shortest_edge thick, in CLASS_LABELS_PROCTHOR100's order."""
    # a doorway is its wall's opening and a faucet part of its sink, not objects of their own
    sizes = pd.read_csv(PROCTHOR100_SIZES_PATH, index_col="class").drop(index=["doorway", "faucet"])
    # counter-top clutter (fruit, mugs, cutlery, plates) rarely passes the area filter nor maps apart from the counter
    candidates = sizes.index[(sizes.max_largest_edge >= min_largest_edge) & (sizes.max_shortest_edge >= min_shortest_edge)]
    placed = pd.read_csv(PROCTHOR100_SCENE_CLASSES_PATH, index_col="scene").loc[scenes].sum()
    chosen = set(placed[placed.index.isin(candidates)].nlargest(count).index)
    return [name for name in CLASS_LABELS_PROCTHOR100 if name in chosen]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=int, default=20, help="Count ProcTHOR-Train-0 to -<train - 1>.")
    parser.add_argument("--val", type=int, default=40, help="Count ProcTHOR-Val-0 to -<val - 1>.")
    args = parser.parse_args()
    print(realistic_classes([f"ProcTHOR-Train-{index}" for index in range(args.train)] + [f"ProcTHOR-Val-{index}" for index in range(args.val)]))
