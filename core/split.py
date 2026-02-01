from __future__ import annotations

import numpy as np


def cascade_embedding_terminal_snapshot(S: np.ndarray) -> np.ndarray:
    """
    Splitting features for cascades.
    Engineering default: use the terminal snapshot vector itself.
    """
    return S.astype(np.float64, copy=False)


def weighted_kmeans_2(
    X: np.ndarray,  # (beta, p)
    w: np.ndarray,  # (beta,)
    iters: int = 40,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted 2-means clustering.
    Returns:
      labels: (beta,) in {0,1}
      centers: (2,p)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    beta, p = X.shape
    w = w.astype(np.float64)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        labels = np.zeros(beta, dtype=np.int64)
        c0 = np.mean(X, axis=0)
        return labels, np.stack([c0, c0], axis=0)

    probs = w / w_sum
    idx = rng.choice(beta, size=2, replace=False, p=probs)
    centers = X[idx].astype(np.float64)

    labels = np.zeros(beta, dtype=np.int64)
    for _ in range(iters):
        d0 = np.sum((X - centers[0]) ** 2, axis=1)
        d1 = np.sum((X - centers[1]) ** 2, axis=1)
        new_labels = (d1 < d0).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for c in (0, 1):
            mask = labels == c
            ww = w[mask]
            sww = float(np.sum(ww))
            if ww.size == 0 or sww <= 1e-12:
                j = int(rng.choice(beta, p=probs))
                centers[c] = X[j]
            else:
                centers[c] = (ww[:, None] * X[mask]).sum(axis=0) / sww

    return labels, centers
