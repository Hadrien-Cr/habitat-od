"""Diversity-maximizing sample selection -- see abstract_sampler.py. Ported from
embodied-active-learning-od's baselines/samplers/diversity_sampler.py: k-means (k-centroid
init, see k_centroid_greedy/kmeans below) over a pairwise distance matrix built from
common/samplers/similarity.py's similarity_score, clustering `candidates + annotated`
together so that clusters containing an already-annotated (already selected in an earlier
round) sample get dropped before picking centroids -- new picks end up diverse from both
each other and from what's already in the labeled pool."""
from typing import Optional

import numpy as np
from tqdm import tqdm  # type: ignore

from common.samplers.abstract_sampler import Sampler
from common.samplers.similarity import similarity_score
from common.utils.interface import Candidate


def k_centroid_greedy(dis_matrix: np.ndarray, k: int, rng: np.random.Generator) -> list:
    n = dis_matrix.shape[0]
    centroids = [int(rng.integers(0, n))]

    while len(centroids) < k:
        centroid_dis = dis_matrix[:, centroids].min(axis=1)
        centroid_dis[centroids] = -1
        centroids.append(int(np.argmax(centroid_dis)))
    return centroids


def kmeans(dis_matrix: np.ndarray, k: int, n_iter: int, rng: np.random.Generator) -> tuple:
    n = dis_matrix.shape[0]
    assert k <= n, "k must be <= number of points"

    if k == n:
        return {i: [i] for i in range(n)}, list(range(n))

    centroids = k_centroid_greedy(dis_matrix, k, rng)
    data_indices = np.arange(n)

    for _ in range(n_iter):
        cluster_assign = np.argmin(dis_matrix[:, centroids], axis=1)
        old_centroids = centroids
        new_centroids = []

        for i in range(k):
            cluster_i = data_indices[cluster_assign == i]

            if len(cluster_i) == 1:
                new_centroid_i = centroids[i]
            elif len(cluster_i) == 2:
                c1, c2 = cluster_i[0], cluster_i[1]
                other_centroids = [c for c in centroids if c != centroids[i]]
                d1 = min(dis_matrix[c1, c] for c in other_centroids)
                d2 = min(dis_matrix[c2, c] for c in other_centroids)
                new_centroid_i = c1 if d1 > d2 else c2
            else:
                dis_mat_i = dis_matrix[cluster_i][:, cluster_i]
                intra_cost = dis_mat_i.mean(axis=1)
                other_centroids = [c for c in centroids if c not in cluster_i]
                if other_centroids:
                    malus = np.array([np.mean([1.0 / (dis_matrix[p, c] + 1e-8) for c in other_centroids]) for p in cluster_i])
                else:
                    malus = np.zeros(len(cluster_i))
                new_centroid_i = cluster_i[np.argmin(intra_cost + 0.01 * malus)]

            new_centroids.append(int(new_centroid_i))

        centroids = new_centroids
        if centroids == old_centroids:
            break

    cluster_assign = np.argmin(dis_matrix[:, centroids], axis=1)
    partition: dict = {i: [] for i in range(k)}
    for idx, c in enumerate(cluster_assign):
        partition[int(c)].append(idx)

    return partition, centroids


class DiversitySampler(Sampler):
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def _distance_matrix(self, samples: list[Candidate]) -> np.ndarray:
        n = len(samples)
        dist = np.zeros((n, n), dtype=np.float32)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        for i, j in tqdm(pairs, desc="Computing similarity distance matrix", mininterval=10.0):
            distance = 1.0 - similarity_score(samples[i].pred_instances, samples[j].pred_instances)
            dist[i, j] = dist[j, i] = distance
        return dist

    def select(self, candidates: list[Candidate], budget: int, *, annotated: Optional[list] = None, classwise_ap: Optional[dict] = None) -> list:
        annotated = annotated or []
        n_collect, n_annot = len(candidates), len(annotated)
        assert n_collect >= budget, f"{n_collect} candidates, budget={budget}"

        samples = candidates + annotated
        partition, centroid_ids = kmeans(self._distance_matrix(samples), k=n_annot + budget, n_iter=100, rng=self.rng)

        annotated_clusters = {
            c: any(idx >= n_collect for idx in members) for c, members in partition.items()
        }
        remaining = [centroid_ids[c] for c in range(n_annot + budget) if not annotated_clusters[c]]
        assert len(set(remaining)) >= budget, "not enough non-annotated clusters left to satisfy budget"

        chosen = self.rng.choice(remaining, size=budget, replace=False)
        return [samples[i] for i in chosen]
