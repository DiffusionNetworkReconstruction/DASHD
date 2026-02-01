from __future__ import annotations

import numpy as np

from dashd.core.graph import GraphRelation
from dashd.core.utils import safe_log, logsumexp


def loglik_noisy_or_terminal(
    S: np.ndarray,
    g: GraphRelation,
    base_q: float = 1e-3,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Per-cascade log-likelihood under a Noisy-OR terminal snapshot model.
    For node i:
      active parents = {j in Pa(i): S_l[j]=1}
      p_inf = base_q if none active else 1 - Π_{j in active} (1 - w_{j->i})
      contrib = log(p_inf) if S_l[i]=1 else log(1-p_inf)
    """
    if not np.issubdtype(S.dtype, np.integer):
        S = S.astype(np.int8, copy=False)
    beta, n = S.shape
    if n != g.n:
        raise ValueError("S.shape[1] must match g.n")

    ll = np.zeros(beta, dtype=np.float64)

    for i in range(n):
        pa = g.parents[i]
        if not pa:
            p = np.full(beta, base_q, dtype=np.float64)
        else:
            p_no = np.ones(beta, dtype=np.float64)
            for j in pa:
                w = float(g.weights.get((j, i), 0.0))
                active = S[:, j].astype(bool)
                p_no[active] *= (1.0 - w)
            p = 1.0 - p_no
            no_active = np.isclose(p_no, 1.0)
            p = p.copy()
            p[no_active] = base_q

        si = S[:, i].astype(np.float64)
        ll += si * safe_log(p, eps=eps) + (1.0 - si) * safe_log(1.0 - p, eps=eps)

    return ll


def mixture_loglik(loglik_mat: np.ndarray, pi: np.ndarray) -> float:
    """Sum_l log( sum_k pi_k * exp(loglik_lk) )."""
    log_pi = np.log(np.clip(pi, 1e-12, 1.0))
    return float(np.sum(logsumexp(loglik_mat + log_pi[None, :], axis=1)))
