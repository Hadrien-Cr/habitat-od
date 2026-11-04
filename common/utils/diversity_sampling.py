from typing import Optional

import numpy as np


def k_centroid_greedy(dis_matrix: np.ndarray, k: int, rng: np.random.Generator, initial: Optional[list] = None) -> list:
    """Greedily picks k indices from dis_matrix maximizing each new pick's distance to the
    closest already-picked centroid, starting from the initial indices, or a random index without any."""
    n = dis_matrix.shape[0]
    centroids = list(initial) if initial else [int(rng.integers(0, n))]

    while len(centroids) < k:
        centroid_dis = dis_matrix[:, centroids].min(axis=1)
        centroid_dis[centroids] = -1
        centroids.append(int(np.argmax(centroid_dis)))
    return centroids
