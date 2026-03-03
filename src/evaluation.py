import numpy as np


def precision_at_k(relevant_ids, predicted_ids, k: int):
    """
    relevant_ids: set or list of relevant product ids
    predicted_ids: list of predicted product ids
    """
    predicted_k = predicted_ids[:k]
    relevant_set = set(relevant_ids)

    relevant_found = sum(1 for pid in predicted_k if pid in relevant_set)

    return relevant_found / k


def recall_at_k(relevant_ids, predicted_ids, k: int):
    predicted_k = predicted_ids[:k]
    relevant_set = set(relevant_ids)

    relevant_found = sum(1 for pid in predicted_k if pid in relevant_set)

    if len(relevant_set) == 0:
        return 0.0

    return relevant_found / len(relevant_set)


def mean_reciprocal_rank(all_relevant, all_predictions):
    """
    all_relevant: list of sets of relevant ids for each query
    all_predictions: list of predicted id lists for each query
    """
    rr_scores = []

    for relevant_ids, predicted_ids in zip(all_relevant, all_predictions):
        rank = 0
        for idx, pid in enumerate(predicted_ids):
            if pid in relevant_ids:
                rank = idx + 1
                break

        if rank > 0:
            rr_scores.append(1 / rank)
        else:
            rr_scores.append(0)

    return np.mean(rr_scores)
