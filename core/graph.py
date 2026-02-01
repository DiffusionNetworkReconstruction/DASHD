from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

Edge = Tuple[int, int]  # (parent, child)


@dataclass
class GraphRelation:
    """
    Relation-specific directed structure.
    parents[i]: list of parent indices for node i
    weights[(j,i)]: edge parameter in (0,1) for Noisy-OR
    """
    n: int
    parents: List[List[int]]
    weights: Dict[Edge, float]

    @staticmethod
    def empty(n: int) -> "GraphRelation":
        return GraphRelation(n=n, parents=[[] for _ in range(n)], weights={})

    def copy(self) -> "GraphRelation":
        return GraphRelation(n=self.n, parents=[list(p) for p in self.parents], weights=dict(self.weights))

    def edge_count(self) -> int:
        return int(sum(len(p) for p in self.parents))

    def set_parents(self, i: int, new_parents: list[int], new_weights: dict[Edge, float]) -> None:
        # remove old
        for j in self.parents[i]:
            self.weights.pop((j, i), None)
        self.parents[i] = list(new_parents)
        # add new
        for j in new_parents:
            self.weights[(j, i)] = float(new_weights[(j, i)])

    def to_edge_set(self) -> set[Edge]:
        es: set[Edge] = set()
        for i in range(self.n):
            for j in self.parents[i]:
                es.add((j, i))
        return es

    def to_adjacency(self) -> np.ndarray:
        A = np.zeros((self.n, self.n), dtype=np.float64)
        for (j, i), w in self.weights.items():
            A[j, i] = w
        return A
