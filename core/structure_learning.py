from __future__ import annotations

import itertools
import numpy as np

from dashd.core.graph import GraphRelation, Edge
from dashd.core.utils import sigmoid, safe_log


def _node_weighted_loglik(
    S: np.ndarray,
    r: np.ndarray,
    i: int,
    parents: list[int],
    weights: dict[Edge, float],
    base_q: float,
    eps: float,
) -> float:
    beta, _n = S.shape
    si = S[:, i].astype(np.float64)

    if not parents:
        p = np.full(beta, base_q, dtype=np.float64)
    else:
        p_no = np.ones(beta, dtype=np.float64)
        for j in parents:
            w = float(weights[(j, i)])
            active = S[:, j].astype(bool)
            p_no[active] *= (1.0 - w)
        p = 1.0 - p_no
        no_active = np.isclose(p_no, 1.0)
        p = p.copy()
        p[no_active] = base_q

    ll = si * safe_log(p, eps=eps) + (1.0 - si) * safe_log(1.0 - p, eps=eps)
    return float(np.sum(r * ll))


def fit_noisy_or_weights_for_parents(
    S: np.ndarray,
    r: np.ndarray,
    i: int,
    parents: list[int],
    base_q: float = 1e-3,
    iters: int = 40,
    lr: float = 0.35,
    eps: float = 1e-12,
) -> dict[Edge, float]:
    """
    Fit Noisy-OR edge probabilities for a fixed parent set by optimizing logits:
      w = sigmoid(a) in (0,1)
    Lightweight gradient ascent; designed for engineering clarity.
    """
    beta, _n = S.shape
    if not parents:
        return {}

    a = np.zeros(len(parents), dtype=np.float64)
    si = S[:, i].astype(np.float64)
    X = np.stack([S[:, j].astype(np.float64) for j in parents], axis=1)  # (beta, p)

    w_r = r.astype(np.float64)
    w_sum = float(np.sum(w_r))
    if w_sum <= eps:
        return {(parents[t], i): 0.2 for t in range(len(parents))}

    for _ in range(iters):
        w = sigmoid(a)

        # log p_no = sum_t X[:,t] * log(1 - w_t)
        log_p_no = np.zeros(beta, dtype=np.float64)
        for t in range(len(parents)):
            log_p_no += X[:, t] * np.log(np.clip(1.0 - w[t], 1e-12, 1.0))
        p_no = np.exp(log_p_no)
        p = 1.0 - p_no
        no_active = np.isclose(p_no, 1.0)
        p = p.copy()
        p[no_active] = base_q

        dlogp_dp = si / np.clip(p, eps, None) - (1.0 - si) / np.clip(1.0 - p, eps, None)
        dw_da = w * (1.0 - w)

        grad = np.zeros_like(a)
        for t in range(len(parents)):
            one_minus = np.clip(1.0 - w[t], 1e-12, 1.0)
            dp_dwt = np.zeros(beta, dtype=np.float64)
            active = X[:, t] > 0.5
            # dp/dw = p_no/(1-w) for active
            dp_dwt[active] = p_no[active] / one_minus
            grad[t] = float(np.sum(w_r * dlogp_dp * dp_dwt) * dw_da[t])

        a += lr * grad / max(1.0, w_sum)

    w = sigmoid(a)
    return {(parents[t], i): float(np.clip(w[t], 1e-6, 1.0 - 1e-6)) for t in range(len(parents))}


def nodewise_parent_set_search(
    S: np.ndarray,
    r: np.ndarray,
    candidates: list[list[int]],
    max_indegree: int,
    lambda_sparsity: float,
    base_q: float,
    weight_fit_iters: int,
    eps: float = 1e-12,
) -> GraphRelation:
    """
    M-step: relation-specific structure learning.
    For each node i, enumerate parent subsets from candidates[i] up to max_indegree,
    fit weights, and pick the best by:
      weighted log-likelihood - lambda * |parents|
    """
    beta, n = S.shape
    g = GraphRelation.empty(n)
    r_eff = r.astype(np.float64)
    if float(np.sum(r_eff)) <= eps:
        return g

    for i in range(n):
        cand_i = [j for j in candidates[i] if j != i]
        best_score = -np.inf
        best_parents: list[int] = []
        best_weights: dict[Edge, float] = {}

        subsets: list[tuple[int, ...]] = [()]
        for k in range(1, max_indegree + 1):
            if len(cand_i) >= k:
                subsets.extend(itertools.combinations(cand_i, k))

        for sub in subsets:
            parents = list(sub)
            weights = fit_noisy_or_weights_for_parents(
                S=S,
                r=r_eff,
                i=i,
                parents=parents,
                base_q=base_q,
                iters=weight_fit_iters,
                eps=eps,
            )
            ll_i = _node_weighted_loglik(
                S=S, r=r_eff, i=i, parents=parents, weights=weights, base_q=base_q, eps=eps
            )
            score = ll_i - lambda_sparsity * len(parents)
            if score > best_score:
                best_score = score
                best_parents = parents
                best_weights = weights

        g.set_parents(i, best_parents, best_weights)

    return g
