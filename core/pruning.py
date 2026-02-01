from __future__ import annotations

import numpy as np

from dashd.core.utils import safe_log


def weighted_mutual_information_matrix(
    S: np.ndarray,  # (beta, n) binary
    r: np.ndarray,  # (beta,) nonnegative weights
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Weighted mutual information MI(i,j) for binary snapshot variables with weights r.
    Returns symmetric MI matrix (n x n).
    """
    beta, n = S.shape
    w = r.astype(np.float64)
    w_sum = float(np.sum(w))
    if w_sum <= eps:
        return np.zeros((n, n), dtype=np.float64)

    S_f = S.astype(np.float64, copy=False)
    p1 = (w[:, None] * S_f).sum(axis=0) / w_sum
    p0 = 1.0 - p1

    MI = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        xi = S_f[:, i]
        for j in range(i + 1, n):
            xj = S_f[:, j]

            p11 = float(np.sum(w * (xi * xj)) / w_sum)
            p10 = float(np.sum(w * (xi * (1.0 - xj))) / w_sum)
            p01 = float(np.sum(w * ((1.0 - xi) * xj)) / w_sum)
            p00 = float(np.sum(w * ((1.0 - xi) * (1.0 - xj))) / w_sum)

            pij = np.clip(np.array([p00, p01, p10, p11], dtype=np.float64), eps, 1.0)
            pi0, pi1 = np.clip(p0[i], eps, 1.0), np.clip(p1[i], eps, 1.0)
            pj0, pj1 = np.clip(p0[j], eps, 1.0), np.clip(p1[j], eps, 1.0)
            denom = np.array([pi0 * pj0, pi0 * pj1, pi1 * pj0, pi1 * pj1], dtype=np.float64)

            mij = float(np.sum(pij * (safe_log(pij, eps) - safe_log(denom, eps))))
            MI[i, j] = MI[j, i] = max(0.0, mij)

    return MI


def top_c_candidates_from_mi(MI: np.ndarray, c: int) -> list[list[int]]:
    """For each node i, return top-c candidate parents by MI(i,j)."""
    n = MI.shape[0]
    out: list[list[int]] = []
    for i in range(n):
        scores = MI[i].copy()
        scores[i] = -1.0
        idx = np.argsort(-scores)[:c]
        out.append([int(x) for x in idx if scores[int(x)] > 0])
    return out
