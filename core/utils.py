from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


def logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    s = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(s, axis=axis)


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, 1.0))


def normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = np.sum(mat, axis=1, keepdims=True)
    return mat / np.clip(s, eps, None)
