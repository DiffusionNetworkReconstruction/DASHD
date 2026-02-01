from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import List, Tuple

import numpy as np

from dashd.core.graph import GraphRelation, Edge


def edge_f1(pred_edges: set[Edge], true_edges: set[Edge]) -> float:
    tp = len(pred_edges & true_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


@dataclass(frozen=True)
class EvalMetrics:
    f1_avg_best: float
    best_perm: tuple[int, ...]      # mapping pred k -> gt best_perm[k]
    f1_per_pair: np.ndarray         # F1 for each matched pair under best_perm
    K_eval: int


def evaluate_against_ground_truth(
    pred_graphs: List[GraphRelation],
    gt_graphs: List[GraphRelation],
) -> EvalMetrics:
    """
    Evaluate by best matching (permutation) between predicted and GT relations using average edge F1.

    If K differs, evaluates on K=min(K_pred, K_gt) and ignores extra relations.
    """
    Kp = len(pred_graphs)
    Kg = len(gt_graphs)
    K = min(Kp, Kg)
    if K == 0:
        return EvalMetrics(0.0, tuple(), np.array([], dtype=np.float64), 0)

    pred_edges = [pred_graphs[k].to_edge_set() for k in range(K)]
    gt_edges = [gt_graphs[k].to_edge_set() for k in range(K)]

    F = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            F[i, j] = edge_f1(pred_edges[i], gt_edges[j])

    best_score = -1.0
    best_perm: tuple[int, ...] = tuple(range(K))
    best_pair = np.zeros(K, dtype=np.float64)

    for perm in itertools.permutations(range(K)):
        vals = np.array([F[i, perm[i]] for i in range(K)], dtype=np.float64)
        score = float(np.mean(vals))
        if score > best_score:
            best_score = score
            best_perm = tuple(int(x) for x in perm)
            best_pair = vals

    return EvalMetrics(best_score, best_perm, best_pair, K)
