"""Unit tests for common/planning/skeleton.py's grid_path -- no simulator needed, pure
synthetic occupancy masks. Checks the "step walking" claim: that consecutive waypoints are
close together (roughly one skeleton-node step apart), not long straight-line jumps that
skip over intermediate grid points."""
import numpy as np
import pytest

from common.planning.skeleton import grid_path

_ROOM = np.zeros((120, 160), dtype=np.uint8)
_ROOM[2:-2, 2:-2] = 1  # bordered, open, obstacle-free room


def _consecutive_dists(path: list[tuple[int, int]]) -> list[float]:
    return [
        float(np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]))
        for i in range(len(path) - 1)
    ]


def test_all_waypoints_are_walkable():
    for step_size in (None, 10.0):
        path = grid_path(_ROOM, (5, 5), (150, 100), step_size=step_size)
        for col, row in path:
            assert _ROOM[row, col] == 1, f"waypoint {(col, row)} not on a walkable cell (step_size={step_size})"


def test_without_step_size_a_clear_line_of_sight_is_one_long_hop():
    """Documents the raw A* behavior grid_path builds on: shortest-hop-count is the wrong
    objective for "visit every grid point" -- an obstacle-free line of sight collapses to a
    single direct hop covering the whole distance, skipping every point in between."""
    path = grid_path(_ROOM, (5, 5), (150, 100))
    assert path == [(5, 5), (150, 100)]


def test_step_size_prevents_long_jumps():
    """With step_size set, no single hop should cover more than roughly one step -- this is
    what makes grid_path actually "step walk" instead of jumping straight to a distant node."""
    step_size = 10.0
    path = grid_path(_ROOM, (5, 5), (150, 100), step_size=step_size)
    dists = _consecutive_dists(path)

    print("path:", path)
    print("consecutive distances:", dists)

    assert max(dists) <= step_size * 1.5, (
        f"a single hop ({max(dists):.1f}px) is much longer than step_size ({step_size}px) -- "
        f"grid_path is still skipping over intermediate grid points, path={path}"
    )


def test_step_size_scales_waypoint_count_with_distance():
    step_size = 10.0
    short_path = grid_path(_ROOM, (5, 5), (25, 5), step_size=step_size)
    long_path = grid_path(_ROOM, (5, 5), (150, 100), step_size=step_size)
    print("short path:", short_path, "long path:", long_path)
    assert len(long_path) > len(short_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
