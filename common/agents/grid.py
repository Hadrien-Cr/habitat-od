"""Reachable-cell enumeration shared by grid-based exploration agents (frontier_agent.py,
sweep_agent.py) -- both build their visited/unexplored bookkeeping off the same navmesh cell
set (ExplorationEnv.get_navmesh_grid())."""
import numpy as np


def reachable_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    rows, cols = np.nonzero(grid)
    return set(zip(rows.tolist(), cols.tolist()))
