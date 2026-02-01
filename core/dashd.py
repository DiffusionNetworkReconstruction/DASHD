from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

from dashd.core.graph import GraphRelation
from dashd.core.likelihood import loglik_noisy_or_terminal, mixture_loglik
from dashd.core.pruning import weighted_mutual_information_matrix, top_c_candidates_from_mi
from dashd.core.structure_learning import nodewise_parent_set_search
from dashd.core.split import weighted_kmeans_2, cascade_embedding_terminal_snapshot
from dashd.core.utils import normalize_rows

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DASHDConfig:
    # Structure learning / pruning
    max_indegree: int = 3          # d
    prune_top_c: int = 20          # c (top candidates per node)

    # Alternating optimization
    alternating_max_iter: int = 15
    alternating_tol: float = 1e-4

    # Splitting / growth
    max_relations: int = 8
    accept_epsilon: float = 1e-3
    split_local_refine_iters: int = 8
    split_kmeans_iters: int = 40

    # Regularization
    lambda_sparsity: float = 0.2

    # Likelihood model
    base_q: float = 1e-3
    eps: float = 1e-12

    # Weight fitting
    weight_fit_iters: int = 40

    # Random seed
    random_seed: int = 7


@dataclass
class DASHDResult:
    graphs: list[GraphRelation]
    R: np.ndarray            # (beta, K)
    pi: np.ndarray           # (K,)
    objective: float
    K_final: int


class DASHD:
    """
    DASHD end-to-end for snapshots-only input.

    Steps:
      1) Initialize K=1
      2) Repeat alternating optimization at fixed K:
         - E-step: soft responsibilities via mixture posterior
         - M-step: pruning (weighted MI) + nodewise structure search
      3) Incremental splitting:
         - propose split on a selected component
         - split cascades via weighted 2-means
         - locally refine K+1 model
         - accept if penalized objective improves by epsilon
      4) Terminate when split rejected / invalid or K reaches max_relations
    """

    def __init__(self, cfg: DASHDConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.random_seed)

    def fit(self, S: np.ndarray) -> DASHDResult:
        S = np.asarray(S)
        if S.ndim != 2:
            raise ValueError("S must be a 2D array of shape (beta, n).")
        if np.min(S) < 0 or np.max(S) > 1:
            raise ValueError("S must be binary (0/1).")
        if not np.issubdtype(S.dtype, np.integer) and S.dtype != np.bool_:
            raise ValueError("S must be an integer/bool array.")
        S = S.astype(np.int8, copy=False)

        beta, n = S.shape
        logger.info("Fit DASHD: beta=%d, n=%d", beta, n)

        # Initialize K=1
        graphs = [GraphRelation.empty(n)]
        R = np.ones((beta, 1), dtype=np.float64)
        pi = np.ones(1, dtype=np.float64)

        # Optimize at K=1
        graphs, R, pi, obj = self._optimize_fixed_K(S, graphs, R, pi, max_iter=self.cfg.alternating_max_iter)

        # Incremental splitting
        while len(graphs) < self.cfg.max_relations:
            proposal = self._propose_and_refine_split(S, graphs, R, pi)
            if proposal is None:
                logger.info("No valid split proposal; terminating.")
                break

            graphs_new, R_new, pi_new, obj_new = proposal
            if obj_new >= obj + self.cfg.accept_epsilon:
                graphs, R, pi, obj = graphs_new, R_new, pi_new, obj_new
                logger.info("Split accepted: K=%d objective=%.6f", len(graphs), obj)
            else:
                logger.info("Split rejected: obj_new=%.6f < obj+eps=%.6f; terminating.",
                            obj_new, obj + self.cfg.accept_epsilon)
                break

        return DASHDResult(graphs=graphs, R=R, pi=pi, objective=obj, K_final=len(graphs))

    # -----------------------
    # Objective
    # -----------------------
    def _penalized_objective(self, S: np.ndarray, graphs: list[GraphRelation], pi: np.ndarray) -> float:
        """
        J_beta = log P(S | mixture) - lambda * total_edges
        """
        beta, _n = S.shape
        K = len(graphs)
        loglik_mat = np.zeros((beta, K), dtype=np.float64)
        for k in range(K):
            loglik_mat[:, k] = loglik_noisy_or_terminal(S, graphs[k], base_q=self.cfg.base_q, eps=self.cfg.eps)
        ll = mixture_loglik(loglik_mat, pi)
        edges = sum(g.edge_count() for g in graphs)
        return float(ll - self.cfg.lambda_sparsity * edges)

    # -----------------------
    # Fixed-K alternating optimization
    # -----------------------
    def _optimize_fixed_K(
        self,
        S: np.ndarray,
        graphs: list[GraphRelation],
        R: np.ndarray,
        pi: np.ndarray,
        max_iter: int,
    ) -> tuple[list[GraphRelation], np.ndarray, np.ndarray, float]:
        K = len(graphs)
        beta, _n = S.shape
        prev_obj = -np.inf

        for it in range(max_iter):
            # E-step: log-likelihoods
            loglik_mat = np.zeros((beta, K), dtype=np.float64)
            for k in range(K):
                loglik_mat[:, k] = loglik_noisy_or_terminal(S, graphs[k], base_q=self.cfg.base_q, eps=self.cfg.eps)

            # posterior responsibilities: R_lk ∝ pi_k * exp(loglik_lk)
            log_pi = np.log(np.clip(pi, 1e-12, 1.0))
            a = loglik_mat + log_pi[None, :]
            a -= np.max(a, axis=1, keepdims=True)
            R = normalize_rows(np.exp(a), eps=self.cfg.eps)

            # update mixing proportions
            pi = np.mean(R, axis=0)
            pi = pi / np.sum(pi)

            # M-step: prune + structure learning per relation
            new_graphs: list[GraphRelation] = []
            for k in range(K):
                r_k = R[:, k]
                MI = weighted_mutual_information_matrix(S, r_k, eps=self.cfg.eps)
                candidates = top_c_candidates_from_mi(MI, c=self.cfg.prune_top_c)
                gk = nodewise_parent_set_search(
                    S=S,
                    r=r_k,
                    candidates=candidates,
                    max_indegree=self.cfg.max_indegree,
                    lambda_sparsity=self.cfg.lambda_sparsity,
                    base_q=self.cfg.base_q,
                    weight_fit_iters=self.cfg.weight_fit_iters,
                    eps=self.cfg.eps,
                )
                new_graphs.append(gk)
            graphs = new_graphs

            obj = self._penalized_objective(S, graphs, pi)
            if it > 0 and abs(obj - prev_obj) <= self.cfg.alternating_tol * (1.0 + abs(prev_obj)):
                return graphs, R, pi, obj
            prev_obj = obj

        obj = self._penalized_objective(S, graphs, pi)
        return graphs, R, pi, obj

    # -----------------------
    # Splitting
    # -----------------------
    def _propose_and_refine_split(
        self,
        S: np.ndarray,
        graphs: list[GraphRelation],
        R: np.ndarray,
        pi: np.ndarray,
    ) -> tuple[list[GraphRelation], np.ndarray, np.ndarray, float] | None:
        beta, _n = S.shape
        K = len(graphs)

        # current logliks to choose a split component
        loglik_mat = np.zeros((beta, K), dtype=np.float64)
        for k in range(K):
            loglik_mat[:, k] = loglik_noisy_or_terminal(S, graphs[k], base_q=self.cfg.base_q, eps=self.cfg.eps)

        # heuristic split score: mass * expected negative loglik
        scores = np.full(K, -np.inf, dtype=np.float64)
        for k in range(K):
            mass = float(np.sum(R[:, k]))
            if mass <= self.cfg.eps:
                continue
            avg_nll = -float(np.sum(R[:, k] * loglik_mat[:, k]) / mass)
            scores[k] = mass * avg_nll

        k_split = int(np.argmax(scores))
        r_split = R[:, k_split]
        if float(np.sum(r_split)) <= self.cfg.eps:
            return None

        # weighted 2-means on cascade embeddings
        X = cascade_embedding_terminal_snapshot(S)
        labels, _ = weighted_kmeans_2(X=X, w=r_split, iters=self.cfg.split_kmeans_iters, rng=self._rng)

        w0 = float(np.sum(r_split[labels == 0]))
        w1 = float(np.sum(r_split[labels == 1]))
        if min(w0, w1) <= 1e-8:
            return None

        # initialize K+1 graphs: duplicate the split graph into two
        keep = [k for k in range(K) if k != k_split]
        graphs_new: list[GraphRelation] = [graphs[k].copy() for k in keep]
        graphs_new.append(graphs[k_split].copy())
        graphs_new.append(graphs[k_split].copy())

        # initialize responsibilities: keep others; split r_split by labels
        R_new = np.zeros((beta, K + 1), dtype=np.float64)
        R_new[:, : len(keep)] = R[:, keep]
        R_new[:, len(keep)] = r_split * (labels == 0)
        R_new[:, len(keep) + 1] = r_split * (labels == 1)
        R_new = normalize_rows(R_new, eps=self.cfg.eps)

        pi_new = np.mean(R_new, axis=0)
        pi_new = pi_new / np.sum(pi_new)

        # local refinement at K+1
        graphs_ref, R_ref, pi_ref, obj_ref = self._optimize_fixed_K(
            S, graphs_new, R_new, pi_new, max_iter=self.cfg.split_local_refine_iters
        )
        return graphs_ref, R_ref, pi_ref, obj_ref
