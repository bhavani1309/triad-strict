import numpy as np # type: ignore


def average_precision_at_k(sim_row, source_id, target_ids, ground_truth, k=50):

    ranked_indices = np.argsort(-sim_row)[:k]

    hits = 0
    sum_precisions = 0.0

    for rank, idx in enumerate(ranked_indices, start=1):
        if (source_id, target_ids[idx]) in ground_truth:
            hits += 1
            sum_precisions += hits / rank

    if hits == 0:
        return 0.0

    return sum_precisions / hits


def mean_average_precision(sim_matrix, source_ids, target_ids, ground_truth, k=50):

    aps = []

    for i, source_id in enumerate(source_ids):
        ap = average_precision_at_k(
            sim_matrix[i],
            source_id,
            target_ids,
            ground_truth,
            k=k
        )
        aps.append(ap)

    return np.mean(aps)