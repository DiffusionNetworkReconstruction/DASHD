from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt


ArrayFloat = npt.NDArray[np.float64]
ArrayInt = npt.NDArray[np.int64]
ArrayBool = npt.NDArray[np.bool_]



@dataclass
class RelationStructure:
    """Stores one relation-specific subnetwork."""

    parents: List[Tuple[int, ...]]
    candidate_parents: List[Tuple[int, ...]]
    threshold_wmi: float = 0.0


@dataclass
class FitResult:
    """Training result container."""

    num_relations: int
    priors: ArrayFloat
    responsibilities: ArrayFloat
    structures: List[RelationStructure]
    objective_history: List[float] = field(default_factory=list)


class DASHDModel:

    def __init__(self, config: DASHDConfig) -> None:
        """Initializes the model.

        Args:
            config: DASHD hyperparameters.
        """
        self.config = config
        self.rng = np.random.default_rng(config.random_state)

        self.num_nodes_: Optional[int] = None
        self.num_snapshots_: Optional[int] = None

        self.priors_: Optional[ArrayFloat] = None
        self.responsibilities_: Optional[ArrayFloat] = None
        self.structures_: Optional[List[RelationStructure]] = None
        self.objective_history_: List[float] = []


    def fit(self, snapshots: ArrayInt) -> FitResult:
        x = self._validate_snapshots(snapshots)
        self.num_snapshots_, self.num_nodes_ = x.shape

        current = self._fit_fixed_k(x, num_relations=1)

        while current.num_relations < self.config.max_relations:
            proposal = self._propose_split(x, current)
            if proposal is None:
                break

            candidate = self._fit_fixed_k(
                x,
                num_relations=current.num_relations + 1,
                init_responsibilities=proposal,
            )

            if candidate.objective_history[-1] > current.objective_history[-1] + self.config.split_epsilon:
                current = candidate
            else:
                break

        self.priors_ = current.priors
        self.responsibilities_ = current.responsibilities
        self.structures_ = current.structures
        self.objective_history_ = current.objective_history
        return current

    def fit_fixed_k(
        self,
        snapshots: ArrayInt,
        num_relations: int,
        init_responsibilities: Optional[ArrayFloat] = None,
    ) -> FitResult:
        x = self._validate_snapshots(snapshots)
        self.num_snapshots_, self.num_nodes_ = x.shape
        return self._fit_fixed_k(x, num_relations, init_responsibilities)


    def _fit_fixed_k(
        self,
        snapshots: ArrayInt,
        num_relations: int,
        init_responsibilities: Optional[ArrayFloat] = None,
    ) -> FitResult:
        beta, n = snapshots.shape
        gamma = (
            self._initialize_responsibilities(beta, num_relations)
            if init_responsibilities is None
            else self._normalize_rows(init_responsibilities)
        )
        priors = gamma.mean(axis=0)
        structures = self._initialize_empty_structures(num_relations, n)

        objective_history: List[float] = []

        for _ in range(self.config.em_max_iters):
            priors = gamma.mean(axis=0)  

            structures = self._update_structures(snapshots, gamma, num_relations)

            log_snapshot_lik = self._compute_relation_log_likelihoods(snapshots, structures, gamma)
            gamma = self._update_responsibilities(log_snapshot_lik, priors)

            obj = self._objective(snapshots, structures, gamma, priors)
            objective_history.append(obj)

            if len(objective_history) >= 2:
                improvement = objective_history[-1] - objective_history[-2]
                if improvement <= self.config.em_tol:
                    break

        return FitResult(
            num_relations=num_relations,
            priors=priors,
            responsibilities=gamma,
            structures=structures,
            objective_history=objective_history,
        )


    def _update_responsibilities(
        self,
        log_snapshot_lik: ArrayFloat,
        priors: ArrayFloat,
    ) -> ArrayFloat:
        log_priors = np.log(np.clip(priors, 1e-12, None))
        logits = log_snapshot_lik + log_priors[None, :]
        logits = logits - self._logsumexp(logits, axis=1, keepdims=True)
        return np.exp(logits)


    def _update_structures(
        self,
        snapshots: ArrayInt,
        gamma: ArrayFloat,
        num_relations: int,
    ) -> List[RelationStructure]:
        _, n = snapshots.shape
        structures: List[RelationStructure] = []

        for k in range(num_relations):
            candidates, tau = self._prune_candidates_wmi(snapshots, gamma[:, k])
            parents: List[Tuple[int, ...]] = []
            for target in range(n):
                best = self._select_best_parent_set(
                    snapshots=snapshots,
                    weights=gamma[:, k],
                    target=target,
                    candidate_parents=candidates[target],
                )
                parents.append(best)
            structures.append(
                RelationStructure(
                    parents=parents,
                    candidate_parents=candidates,
                    threshold_wmi=tau,
                )
            )

        return structures

    def _select_best_parent_set(
        self,
        snapshots: ArrayInt,
        weights: ArrayFloat,
        target: int,
        candidate_parents: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        max_size = min(self.config.max_parent_set_size, len(candidate_parents))
        best_score = -np.inf
        best_parents: Tuple[int, ...] = ()

        for r in range(max_size + 1):
            for parent_set in itertools.combinations(candidate_parents, r):
                score = self._local_parent_score(
                    snapshots=snapshots,
                    weights=weights,
                    target=target,
                    parents=parent_set,
                )
                if score > best_score:
                    best_score = score
                    best_parents = tuple(sorted(parent_set))

        return best_parents

    def _local_parent_score(
        self,
        snapshots: ArrayInt,
        weights: ArrayFloat,
        target: int,
        parents: Tuple[int, ...],
    ) -> float:
        x_target = snapshots[:, target].astype(np.int64)

        if len(parents) == 0:
            cfg_codes = np.zeros(snapshots.shape[0], dtype=np.int64)
            num_cfgs = 1
        else:
            parent_matrix = snapshots[:, list(parents)].astype(np.int64)
            bit_weights = (1 << np.arange(len(parents), dtype=np.int64))
            cfg_codes = (parent_matrix * bit_weights[None, :]).sum(axis=1)
            num_cfgs = 1 << len(parents)

        ll = 0.0
        alpha = self.config.alpha

        for cfg in range(num_cfgs):
            cfg_mask = cfg_codes == cfg
            weighted_total = weights[cfg_mask].sum()
            if weighted_total <= 0.0:
                continue

            weighted_x1 = weights[cfg_mask & (x_target == 1)].sum()
            weighted_x0 = weights[cfg_mask & (x_target == 0)].sum()

            p0 = (weighted_x0 + alpha) / (weighted_total + 2.0 * alpha)
            p1 = (weighted_x1 + alpha) / (weighted_total + 2.0 * alpha)

            if weighted_x0 > 0:
                ll += weighted_x0 * np.log(p0)
            if weighted_x1 > 0:
                ll += weighted_x1 * np.log(p1)

        return ll - self.config.lambda_sparse * len(parents)


    def _compute_relation_log_likelihoods(
        self,
        snapshots: ArrayInt,
        structures: Sequence[RelationStructure],
        gamma: ArrayFloat,
    ) -> ArrayFloat:
        beta, n = snapshots.shape
        num_relations = len(structures)
        out = np.zeros((beta, num_relations), dtype=np.float64)

        for k, structure in enumerate(structures):
            weights = gamma[:, k]
            for target in range(n):
                parents = structure.parents[target]
                node_logp = self._node_conditional_log_probs(
                    snapshots=snapshots,
                    weights=weights,
                    target=target,
                    parents=parents,
                )
                out[:, k] += node_logp

        return out

    def _node_conditional_log_probs(
        self,
        snapshots: ArrayInt,
        weights: ArrayFloat,
        target: int,
        parents: Tuple[int, ...],
    ) -> ArrayFloat:
        beta = snapshots.shape[0]
        x_target = snapshots[:, target].astype(np.int64)
        alpha = self.config.alpha

        if len(parents) == 0:
            cfg_codes = np.zeros(beta, dtype=np.int64)
            num_cfgs = 1
        else:
            parent_matrix = snapshots[:, list(parents)].astype(np.int64)
            bit_weights = (1 << np.arange(len(parents), dtype=np.int64))
            cfg_codes = (parent_matrix * bit_weights[None, :]).sum(axis=1)
            num_cfgs = 1 << len(parents)

        log_probs = np.zeros(beta, dtype=np.float64)

        for cfg in range(num_cfgs):
            cfg_mask = cfg_codes == cfg
            weighted_total = weights[cfg_mask].sum()
            weighted_x1 = weights[cfg_mask & (x_target == 1)].sum()
            weighted_x0 = weights[cfg_mask & (x_target == 0)].sum()

            p0 = (weighted_x0 + alpha) / (weighted_total + 2.0 * alpha)
            p1 = (weighted_x1 + alpha) / (weighted_total + 2.0 * alpha)

            log_probs[cfg_mask & (x_target == 0)] = np.log(p0)
            log_probs[cfg_mask & (x_target == 1)] = np.log(p1)

        return log_probs


    def _prune_candidates_wmi(
        self,
        snapshots: ArrayInt,
        weights: ArrayFloat,
    ) -> Tuple[List[Tuple[int, ...]], float]:
        beta, n = snapshots.shape
        ek = float(weights.sum())
        if ek <= self.config.min_relation_mass:
            return [tuple(j for j in range(n) if j != i) for i in range(n)], 0.0

        x = snapshots.astype(np.float64)
        w = weights[:, None]

        p1 = (w * x).sum(axis=0) / ek
        p0 = 1.0 - p1

        joint11 = (x.T @ (x * weights[:, None])) / ek
        joint10 = (x.T @ ((1.0 - x) * weights[:, None])) / ek
        joint01 = ((1.0 - x).T @ (x * weights[:, None])) / ek
        joint00 = ((1.0 - x).T @ ((1.0 - x) * weights[:, None])) / ek

        mi = np.zeros((n, n), dtype=np.float64)
        terms = [
            (joint00, np.outer(p0, p0)),
            (joint01, np.outer(p0, p1)),
            (joint10, np.outer(p1, p0)),
            (joint11, np.outer(p1, p1)),
        ]
        for joint, marg in terms:
            valid = joint > 0.0
            mi[valid] += joint[valid] * np.log(joint[valid] / np.clip(marg[valid], 1e-12, None))

        np.fill_diagonal(mi, 0.0)
        tau = self._fixed_zero_two_means_threshold(mi[np.triu_indices(n, k=1)])

        candidates: List[Tuple[int, ...]] = []
        for i in range(n):
            cand = tuple(np.where((mi[i] > tau) & (np.arange(n) != i))[0].tolist())
            candidates.append(cand)

        return candidates, tau

    def _fixed_zero_two_means_threshold(self, values: ArrayFloat) -> float:
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0

        c0 = 0.0
        c1 = float(np.max(values))
        assign_zero = np.ones(values.shape[0], dtype=bool)

        for _ in range(self.config.kmeans_max_iters):
            dist0 = np.abs(values - c0)
            dist1 = np.abs(values - c1)
            new_assign_zero = dist0 <= dist1

            if np.array_equal(new_assign_zero, assign_zero):
                break
            assign_zero = new_assign_zero

            if np.any(~assign_zero):
                c1 = float(values[~assign_zero].mean())

        if np.any(assign_zero):
            return float(values[assign_zero].max())
        return 0.0


    def _propose_split(
        self,
        snapshots: ArrayInt,
        current: FitResult,
    ) -> Optional[ArrayFloat]:
        gamma = current.responsibilities
        structures = current.structures
        beta, _ = snapshots.shape
        k = current.num_relations

        log_lik = self._compute_relation_log_likelihoods(snapshots, structures, gamma)
        neg_log_lik = -log_lik 
        ek = gamma.sum(axis=0)

        if np.any(ek <= self.config.min_relation_mass):
            return None

        weighted_means = (gamma * neg_log_lik).sum(axis=0) / np.clip(ek, 1e-12, None)
        weighted_vars = (gamma * (neg_log_lik - weighted_means[None, :]) ** 2).sum(axis=0) / np.clip(ek, 1e-12, None)
        split_scores = ek * weighted_vars
        k_star = int(np.argmax(split_scores))

        weights = gamma[:, k_star]
        labels = self._weighted_two_means_binary_snapshots(snapshots, weights)

        if labels.sum() == 0 or labels.sum() == beta:
            return None

        new_gamma = np.zeros((beta, k + 1), dtype=np.float64)
        insert_idx = k_star

        new_col = 0
        for old_col in range(k):
            if old_col == k_star:
                continue
            new_gamma[:, new_col] = gamma[:, old_col]
            new_col += 1

        new_gamma[:, k - 1] = gamma[:, k_star] * labels
        new_gamma[:, k] = gamma[:, k_star] * (1.0 - labels)

        return self._normalize_rows(new_gamma)

    def _weighted_two_means_binary_snapshots(
        self,
        snapshots: ArrayInt,
        weights: ArrayFloat,
    ) -> ArrayFloat:
        x = snapshots.astype(np.float64)
        beta = x.shape[0]

        idx = np.argsort(weights)[-2:] if beta >= 2 else np.array([0, 0])
        c_a = x[idx[0]].copy()
        c_b = x[idx[-1]].copy()

        labels = np.zeros(beta, dtype=np.float64)
        for _ in range(self.config.kmeans_max_iters):
            d_a = ((x - c_a[None, :]) ** 2).sum(axis=1)
            d_b = ((x - c_b[None, :]) ** 2).sum(axis=1)
            new_labels = (d_a <= d_b).astype(np.float64)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            wa = weights * labels
            wb = weights * (1.0 - labels)

            if wa.sum() > 0:
                c_a = (wa[:, None] * x).sum(axis=0) / wa.sum()
            if wb.sum() > 0:
                c_b = (wb[:, None] * x).sum(axis=0) / wb.sum()

        return labels


    def _objective(
        self,
        snapshots: ArrayInt,
        structures: Sequence[RelationStructure],
        gamma: ArrayFloat,
        priors: ArrayFloat,
    ) -> float:
        log_lik = self._compute_relation_log_likelihoods(snapshots, structures, gamma)
        log_priors = np.log(np.clip(priors, 1e-12, None))[None, :]
        expected_complete = float((gamma * (log_lik + log_priors)).sum())
        penalty = self.config.lambda_sparse * sum(len(p) for s in structures for p in s.parents)
        return expected_complete - penalty

    def _initialize_responsibilities(self, beta: int, num_relations: int) -> ArrayFloat:
        raw = self.rng.random((beta, num_relations))
        return self._normalize_rows(raw)

    def _initialize_empty_structures(self, num_relations: int, num_nodes: int) -> List[RelationStructure]:
        all_candidates = [tuple(j for j in range(num_nodes) if j != i) for i in range(num_nodes)]
        return [
            RelationStructure(
                parents=[tuple() for _ in range(num_nodes)],
                candidate_parents=all_candidates,
                threshold_wmi=0.0,
            )
            for _ in range(num_relations)
        ]

    def _validate_snapshots(self, snapshots: ArrayInt) -> ArrayInt:
        x = np.asarray(snapshots)
        if x.ndim != 2:
            raise ValueError("`snapshots` must be a 2D array of shape (num_snapshots, num_nodes).")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("`snapshots` must be non-empty.")
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("`snapshots` must contain only binary values in {0, 1}.")
        return x.astype(np.int64, copy=False)

    @staticmethod
    def _normalize_rows(x: ArrayFloat) -> ArrayFloat:
        denom = np.clip(x.sum(axis=1, keepdims=True), 1e-12, None)
        return x / denom

    @staticmethod
    def _logsumexp(x: ArrayFloat, axis: int, keepdims: bool = False) -> ArrayFloat:
        x_max = np.max(x, axis=axis, keepdims=True)
        out = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
        if not keepdims:
            out = np.squeeze(out, axis=axis)
        return out