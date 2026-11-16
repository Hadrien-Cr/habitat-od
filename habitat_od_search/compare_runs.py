"""Paired comparison of habitat_od_search/main.py runs on the episodes they share: for every subset (overall, with and
without object synonyms), each run's success, and its paired difference to the first run with a 95% bootstrap
confidence interval (_N_BOOTSTRAP resamples of the shared episodes) and McNemar's exact p-value (a binomial test on
the episodes only one of the two runs succeeds on). Printed, and saved to --out when given.

Example:
    PYTHONPATH=. python habitat_od_search/compare_runs.py habitat_od_search/logs/eval_200_vanilla_full_max3 \
        habitat_od_search/logs/eval_200_web_max3 --out habitat_od_search/logs/comparison.md
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binomtest
from tabulate import tabulate

from common.utils.plot_utils import load_metrics

_N_BOOTSTRAP = 10000
_SUBSETS = {"overall": lambda n_synonyms: True, "synonyms": lambda n_synonyms: n_synonyms > 0, "no synonyms": lambda n_synonyms: n_synonyms == 0}


def run_episodes(run_dir: Path) -> Dict[Tuple[str, str], dict]:
    """{(scene_id, episode_id): metrics.jsonl row} of run_dir."""
    return {(row["scene_id"], row["episode_id"]): row for row in load_metrics(run_dir / "metrics.jsonl")}


def paired_difference(baseline: np.ndarray, other: np.ndarray, rng: np.random.Generator) -> Tuple[float, float, float, float]:
    """(mean difference, its 95% bootstrap interval's bounds, McNemar's exact p-value) of other's successes over
    baseline's, both in percentage points."""
    resamples = rng.integers(0, len(baseline), (_N_BOOTSTRAP, len(baseline)))
    differences = (other[resamples].astype(float) - baseline[resamples]).mean(axis=1)
    only_baseline, only_other = int((baseline & ~other).sum()), int((~baseline & other).sum())
    p_value = binomtest(only_other, only_baseline + only_other).pvalue if only_baseline + only_other else 1.0
    low, high = np.percentile(differences, [2.5, 97.5])
    return 100 * (other.mean() - baseline.mean()), 100 * low, 100 * high, p_value


def subset_rows(name: str, runs: List[Tuple[str, np.ndarray]], rng: np.random.Generator) -> List[Tuple]:
    """One row per run of the subset: its success, then its paired_difference to the first run (blank for the first)."""
    baseline = runs[0][1]
    rows: List[Tuple[Optional[object], ...]] = [(name, runs[0][0], len(baseline), 100 * baseline.mean(), None, None, None, None)]
    return rows + [(name, run_name, len(success), 100 * success.mean(), *paired_difference(baseline, success, rng)) for run_name, success in runs[1:]]


def comparison_table(run_dirs: List[Path]) -> str:
    episodes = [run_episodes(run_dir) for run_dir in run_dirs]
    shared = sorted(set.intersection(*(set(run) for run in episodes)))
    rng = np.random.default_rng(0)
    rows = []
    for name, keep in _SUBSETS.items():
        keys = [key for key in shared if keep(episodes[0][key]["n_synonyms"])]
        runs = [(run_dir.name, np.array([run[key]["success"] for key in keys], dtype=bool)) for run_dir, run in zip(run_dirs, episodes)]
        rows += subset_rows(name, runs, rng) if keys else []
    return tabulate(
        rows, tablefmt="pipe", floatfmt=".2f", missingval="", numalign="left",
        headers=["episodes", "run", "n", "success", f"Δ vs {run_dirs[0].name}", "95% low", "95% high", "McNemar p"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="main.py run directories, the first one the baseline")
    parser.add_argument("--out", type=Path, default=None, help="markdown file to save the table to")
    args = parser.parse_args()
    table = comparison_table(args.run_dirs)
    if args.out is not None:
        args.out.write_text(f"{table}\n")
    print(table)
