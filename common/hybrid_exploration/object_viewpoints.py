"""Per-object sector viewpoints: a ring of candidate standing points around each object's footprint, one per sector."""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ObjectBelief:
    confidence: float
    points: np.ndarray


@dataclass(frozen=True, eq=False)
class Candidate:
    """A goal the agent may pick; object_id/sector/look_at (the object's center) are None for a frontier."""
    position: np.ndarray
    gain: float
    object_id: Optional[int] = None
    sector: Optional[int] = None
    look_at: Optional[np.ndarray] = None

    def key(self) -> Tuple:
        return ("frontier",) if self.object_id is None else (self.object_id, self.sector)


def footprint(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """(xyz center, xz radius) of the points' axis-aligned bounding box."""
    low, high = points.min(axis=0), points.max(axis=0)
    return (low + high) / 2, float(np.linalg.norm((high - low)[[0, 2]]) / 2)


def sector_positions(center: np.ndarray, radius: float, n_sectors: int, height: float) -> np.ndarray:
    """(n_sectors, 3) points at radius around center, at each sector's middle bearing."""
    bearings = (np.arange(n_sectors) + 0.5) * 2 * np.pi / n_sectors
    return np.stack([center[0] + radius * np.cos(bearings), np.full(n_sectors, height), center[2] + radius * np.sin(bearings)], axis=1)


def sector_of(center: np.ndarray, position: np.ndarray, n_sectors: int) -> int:
    bearing = np.arctan2(position[2] - center[2], position[0] - center[0]) % (2 * np.pi)
    return int(bearing // (2 * np.pi / n_sectors))
