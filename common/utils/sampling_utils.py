from collections import Counter
from pathlib import Path
import numpy as np
from habitat_sim.agent.agent import AgentState


def kmeans(inputs: list, k: int, rng_gen, max_iter: int = 20) -> tuple[list[int], list[list[int]]]:
    n = len(inputs)

    if k >= n:
        return list(range(n)), [[i] for i in range(n)]

    X = np.vstack(inputs)

    init_idx = rng_gen.choice(n, size=k, replace=False)
    centers = X[init_idx].copy()

    assignments = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        old_centers = centers.copy()

        distances = np.linalg.norm(
            X[:, None, :] - centers[None, :, :],
            axis=2
        )

        assignments = np.argmin(distances, axis=1)

        partition = [[] for _ in range(k)]
        for i, c in enumerate(assignments):
            partition[c].append(i)

        for j in range(k):
            if partition[j]:
                centers[j] = X[partition[j]].mean(axis=0)
            else:
                idx = rng_gen.integers(n)
                centers[j] = X[idx]

        # Convergence check
        if np.allclose(old_centers, centers):
            break

    center_indices = list(set([
        int(np.argmin(np.linalg.norm(X - center, axis=1)))
        for center in centers
    ]))

    return center_indices, partition


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
    
    centers_indices, partitionned_indices = kmeans(
        inputs=[projection_fn(sample) for sample in samples],
        k=num_samples,
        rng_gen=rng_gen
    )
    assert len(partitionned_indices) == num_samples
    
    return centers_indices


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
            centers_indices, partitionned_indices = kmeans(
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
    list_of_samples,
    rng_gen,
    mask_filtering_fn,
    num_samples=20,
    num_bins=10,
    keep_top_k_bins=5,
)-> list[int]:
    """
    Stratified sampling over largest mask area, focusing on top-k largest bins.
    """
    indices = np.array(list(range(len(list_of_samples))))

    if len(indices) < num_samples:
        return indices.tolist()
    
    areas = np.array([
        max([inst["mask_area"] for inst in instances if mask_filtering_fn(inst)], default=0)
        for _, instances in list_of_samples
    ])


    edges = np.unique(
        np.quantile(
            areas,
            np.linspace(0, 1, num_bins + 1)
        )
    )

    # Degenerate case
    if len(edges) < 3:
        return list(
            rng_gen.choice(
                indices,
                num_samples,
                replace=False
            )
        )

    bin_ids = np.digitize(areas, edges[1:-1])
    n_bins = len(edges) - 1

    # Keep top-k largest bins
    selected_bins = range(
        max(0, n_bins - keep_top_k_bins),
        n_bins
    )

    sampled = []
    leftovers = []

    per_bin = max(1, num_samples // len(selected_bins))

    for b in selected_bins:
        members = indices[bin_ids == b]

        if len(members) <= per_bin:
            sampled.extend(members.tolist())
        else:
            chosen = rng_gen.choice(
                members,
                per_bin,
                replace=False
            )
            sampled.extend(chosen.tolist())

            leftovers.extend(
                x for x in members
                if x not in chosen
            )

    # Fill any remaining slots
    remaining = num_samples - len(sampled)

    if remaining > 0 and leftovers:
        extra = rng_gen.choice(
            leftovers,
            min(remaining, len(leftovers)),
            replace=False
        )
        sampled.extend(extra.tolist())

    return sampled[:num_samples]