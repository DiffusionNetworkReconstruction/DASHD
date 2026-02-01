from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from dashd.core.dashd import DASHDResult
from dashd.core.graph import GraphRelation


def load_snapshots(path: str) -> np.ndarray:
    """Load terminal infection snapshots S (beta x n) from .npy or .npz (key 'S')."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    if p.suffix.lower() == ".npy":
        S = np.load(p, allow_pickle=False)
    elif p.suffix.lower() == ".npz":
        z = np.load(p, allow_pickle=False)
        if "S" not in z:
            raise KeyError("Expected key 'S' in .npz file.")
        S = z["S"]
    else:
        raise ValueError("Unsupported snapshots format. Use .npy or .npz containing key 'S'.")

    if S.ndim != 2:
        raise ValueError("S must be 2D (beta, n).")
    if np.min(S) < 0 or np.max(S) > 1:
        raise ValueError("S must be binary (0/1).")
    if not np.issubdtype(S.dtype, np.integer) and S.dtype != np.bool_:
        raise ValueError("S must be integer/bool array.")
    return S.astype(np.int8, copy=False)


def load_ground_truth_relations(folder: str, n: int) -> List[GraphRelation]:
    """
    Load ground-truth relations from rel*.txt files (0-indexed edges).
    Ground truth is used ONLY for evaluation (F1), not for training.
    """
    d = Path(folder)
    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(str(d))

    files = sorted(d.glob("rel*.txt"))
    if not files:
        raise FileNotFoundError("No files matching rel*.txt found in gt-folder.")

    gt: List[GraphRelation] = []
    for fp in files:
        parents = [[] for _ in range(n)]
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad edge line in {fp}: '{line}' (expected 'src dst')")
            j = int(parts[0])
            i = int(parts[1])
            if j < 0 or j >= n or i < 0 or i >= n:
                raise ValueError(f"Edge out of bounds in {fp}: {j}->{i} for n={n}")
            if j == i:
                continue
            parents[i].append(j)

        gt.append(GraphRelation(n=n, parents=parents, weights={}))
    return gt


def save_model_npz(path: str, res: DASHDResult) -> None:
    """
    Save learned model to .npz:
      - parents as object array
      - edges as (src,dst,w) flat arrays with pointers per relation
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    K = res.K_final
    n = res.graphs[0].n

    parents_obj = np.empty((K,), dtype=object)
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_w: list[float] = []
    edge_ptr = [0]

    for k, g in enumerate(res.graphs):
        parents_obj[k] = [list(p) for p in g.parents]
        for (j, i), w in g.weights.items():
            edge_src.append(int(j))
            edge_dst.append(int(i))
            edge_w.append(float(w))
        edge_ptr.append(len(edge_src))

    np.savez_compressed(
        out,
        K=np.int64(K),
        n=np.int64(n),
        pi=res.pi.astype(np.float64),
        objective=np.float64(res.objective),
        parents=parents_obj,
        edge_src=np.array(edge_src, dtype=np.int64),
        edge_dst=np.array(edge_dst, dtype=np.int64),
        edge_w=np.array(edge_w, dtype=np.float64),
        edge_ptr=np.array(edge_ptr, dtype=np.int64),
    )
