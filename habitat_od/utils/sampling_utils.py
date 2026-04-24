from collections import Counter
from pathlib import Path
import numpy as np
from habitat_sim.agent.agent import AgentState


def kmeans(inputs: list, k: int, rng_gen) -> list[list[int]]:
    n = len(inputs)

    if k == n:
        return [[i] for i in range(n)]

    centers = [inputs[i] for i in rng_gen.choice(range(n), size=k, replace=False)]
    
    converged = False

    while not converged:
        clusters: list[list[int]] = [[] for _ in range(k)]

        for i in range(n):
            closest_center = np.argmin([np.linalg.norm(inputs[i] - centers[j]) for j in range(k)])
            clusters[closest_center].append(i)

        old_centers = centers
        centers = [np.mean([inputs[i] for i in cluster], axis=0) for cluster in clusters]
        converged = all(np.allclose(old, new) for old, new in zip(old_centers, centers))
        
    return clusters


def balanced_supsampling(samples: list[tuple[AgentState, dict, list[dict]]], num_samples: int, rng_gen) -> list[int]:
    """
    Uses balanced class sampling to select a diverse set of samples. https://proceedings.mlr.press/v143/olivier21a/olivier21a.pdf
    """
    from scipy.optimize import minimize

    def get_objects(sample):
        _, _, label = sample
        return list(label.keys())

    all_objects = set()


    for sample in samples:
        all_objects.update(get_objects(sample))

    C = len(all_objects)
    N = len(samples)
    E = np.zeros((N, C))

    for i, sample in enumerate(samples):
        for obj in get_objects(sample):
            j = list(all_objects).index(obj)
            E[i, j] = 1
    
    A_prime = 2 * C * np.eye(C) - 2 * np.ones((C, C))
    A = E @ A_prime @ E.T


    def objective(p):return 0.5 * p @ A @ p

    def gradient(p): return A @ p

    constraints = [{'type': 'eq', 'fun': lambda p: np.sum(p) - 1}]
    bounds = [(1/N**2, None)] * N  # cleaner way to enforce p_i >= alpha
    p0 = np.ones(N) / N  # initial guess

    sampling_probs = minimize(
        objective,
        p0,
        jac=gradient,         # analytical gradient (optional but faster)
        method='SLSQP',       # handles equality + inequality constraints
        bounds=bounds,
        constraints=constraints
    ).x

    return rng_gen.choice(N, num_samples, p=sampling_probs, replace=False)


def coverage_subsampling(samples: list[tuple[AgentState, dict, list[dict]]], num_samples: int, rng_gen) ->  list[int]:
    """
    Subsamples by covering multiple (x,z) positions as most
    """

    def projection_fn(sample):
        agent_state, _, _ = sample
        return agent_state.position
    
    partitionned_indices = kmeans(
        inputs=[projection_fn(sample) for sample in samples],
        k=num_samples,
        rng_gen=rng_gen
    )
    assert len(partitionned_indices) == num_samples
    
    return [cluster[0] for cluster in partitionned_indices]


def covisibility_subsampling(samples: list[tuple[AgentState, np.ndarray, list[dict]]], num_samples: int, rng_gen) ->  list[int]:
    """
    Repeats multiple times covisibility filtering steps, until the amount of samples is reached. 
    At each step, selects and remove a set of samples that covers the set of objects
    """
    N = len(samples)
    out = []

    while len(out) < num_samples:
        remaining_idx = [sample_id for sample_id in list(range(N)) if sample_id not in out]
        selected = covisibility_subset(
            [samples[i] for i in remaining_idx],
            rng_gen
        )
        assert not any([remaining_idx[x] in out for x in selected])
        out.extend([remaining_idx[x] for x in selected])

    return out[:num_samples]


def covisibility_subset(samples: list[tuple[AgentState, dict, list[dict]]], rng_gen) ->  list[int]:
    """
    samples : list of tuples (fname, image, labels)
    Follow Co-Visibility Clustering algorithm from https://arxiv.org/pdf/2411.17735
    """
    def get_objects(sample):
        _, _, label = sample
        return list(label.keys())

    def cover(sample, object_cluster: list[str]):
        objects = get_objects(sample)
        return all(obj in objects for obj in object_cluster)

    def projection_fn(o):
        c1, x1, y1, z1 = get_object_class_position(o)
        return np.array([x1, y1, z1])
 
    all_objects = set()

    for sample in samples:
        all_objects.update(get_objects(sample))

    clusters = [list(all_objects)]
    snapshots = []

    while clusters:
        largest_cluster = max(clusters, key=len)

        sample_idx_covering = [i for i, s in enumerate(samples) if cover(s, largest_cluster)]

        if len(sample_idx_covering) > 0:
            best_sample_idx = max(sample_idx_covering, key=lambda idx: len(get_objects(samples[idx])))
            snapshots.append((largest_cluster, best_sample_idx))

        else:
            assert len(largest_cluster) > 1, (clusters, [cover(s, largest_cluster) for s in samples])
            partitionned_indices = kmeans(
                inputs=[projection_fn(o) for o in largest_cluster],
                k=2,
                rng_gen=rng_gen
            )
            c1 = [largest_cluster[i] for i in partitionned_indices[0]]
            c2 = [largest_cluster[i] for i in partitionned_indices[1]]
            clusters.extend([c1, c2])

        clusters.remove(largest_cluster)

    samples_idx = list(set([sample_idx for _, sample_idx in snapshots]))
    
    all_snapshoted_objects = set()

    for sample_idx in samples_idx:
        sample = samples[sample_idx]
        all_snapshoted_objects.update(get_objects(sample))
    
    assert len(all_snapshoted_objects) == len(all_objects)
    return samples_idx



def area_bin_sampling(
    list_of_samples: list[tuple[AgentState,dict, list[dict]]],
    rng_gen,
    num_samples = 20,
    num_bins = 10,
    min_area = 50,
    keep_top_k_bins = 5,
)-> list[int]:
    """Sample from the data to evenly spread the largest_mask area values accross bins values"""
    
    def get_area(obs: dict, masks: list[dict]) -> int:
        largest_mask_area = max([mask["mask_area"]  for mask in masks])
        return largest_mask_area
    
    areas = [get_area(obs, masks) for agent_state,obs,masks in list_of_samples]
    valid_indices = [i for i in range(len(areas)) if areas[i] > min_area]
    valid_areas = [areas[i] for i in range(len(areas)) if areas[i] > min_area]

    if len(valid_indices) <= num_samples:
        return valid_indices
    
    # construct bin edges
    bin_edges = np.quantile(
        valid_areas,
        np.linspace(0, 1, num_bins + 1)
    )
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) <= 2:
        return rng_gen.sample(valid_indices, num_samples)

    K = len(bin_edges) - 1

    # fill bin 
    bins = [[] for _ in range(K)]

    for i in valid_indices:
        idx = np.searchsorted(
            bin_edges,
            areas[i],
            side="right"
        ) - 1
        idx = min(max(idx, 0), K - 1)
        bins[idx].append(i)
    

    per_bin = num_samples // keep_top_k_bins
    sampled = []
    leftovers = []

    for b in bins[keep_top_k_bins:]:
        if len(b) >= per_bin:
            sampled.extend(
                rng_gen.sample(b, per_bin)
            )
            leftovers.extend(
                [x for x in b if x not in sampled]
            )
        else:
            sampled.extend(b)

    # Fill remaining slots from leftovers
    remaining = num_samples - len(sampled)

    if remaining > 0:
        pool = [
            x for b in bins[keep_top_k_bins:]
            for x in b
            if x not in sampled
        ]

        if len(pool) >= remaining:
            sampled.extend(
                rng_gen.sample(pool, remaining)
            )
        else:
            sampled.extend(pool)

    return sampled