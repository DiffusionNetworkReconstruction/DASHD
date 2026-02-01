from __future__ import annotations

import argparse
import logging
from typing import Sequence

import numpy as np

from dashd.config import load_config
from dashd.logging_utils import setup_logging
from dashd.core.dashd import DASHD, DASHDConfig
from dashd.core.eval import evaluate_against_ground_truth
from dashd.core.io import load_snapshots, load_ground_truth_relations, save_model_npz

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dashd",
        description="DASHD: adaptive disentanglement of heterogeneous diffusion structures (snapshots-only).",
    )
    p.add_argument("--config", type=str, default=None, help="Path to JSON config file.")
    sub = p.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="Fit DASHD on terminal infection snapshots.")
    fit.add_argument("--snapshots", type=str, required=True, help="Path to .npy or .npz containing S.")
    fit.add_argument("--out", type=str, required=True, help="Output path for model .npz.")
    fit.add_argument("--gt-folder", type=str, default=None, help="Optional folder with rel*.txt for evaluation only.")

    # Hyperparameters (safe defaults; you can tune)
    fit.add_argument("--max-indegree", type=int, default=3)
    fit.add_argument("--prune-top-c", type=int, default=20)
    fit.add_argument("--alternating-max-iter", type=int, default=15)
    fit.add_argument("--alternating-tol", type=float, default=1e-4)

    fit.add_argument("--max-relations", type=int, default=8)
    fit.add_argument("--accept-epsilon", type=float, default=1e-3)
    fit.add_argument("--split-local-refine-iters", type=int, default=8)
    fit.add_argument("--split-kmeans-iters", type=int, default=40)

    fit.add_argument("--lambda-sparsity", type=float, default=0.2)
    fit.add_argument("--base-q", type=float, default=1e-3)
    fit.add_argument("--weight-fit-iters", type=int, default=40)

    fit.add_argument("--seed", type=int, default=7)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app_cfg = load_config(args.config)
    setup_logging(app_cfg.log_level)

    if args.command != "fit":
        return 2

    S = load_snapshots(args.snapshots)
    cfg = DASHDConfig(
        max_indegree=args.max_indegree,
        prune_top_c=args.prune_top_c,
        alternating_max_iter=args.alternating_max_iter,
        alternating_tol=args.alternating_tol,
        max_relations=args.max_relations,
        accept_epsilon=args.accept_epsilon,
        split_local_refine_iters=args.split_local_refine_iters,
        split_kmeans_iters=args.split_kmeans_iters,
        lambda_sparsity=args.lambda_sparsity,
        base_q=args.base_q,
        weight_fit_iters=args.weight_fit_iters,
        random_seed=args.seed,
    )

    model = DASHD(cfg)
    res = model.fit(S)

    print("=== DASHD fit summary ===")
    print(f"K_final: {res.K_final}")
    print(f"objective: {res.objective:.6f}")
    print(f"pi: {np.round(res.pi, 6)}")

    if args.gt_folder is not None:
        gt_graphs = load_ground_truth_relations(args.gt_folder, n=S.shape[1])
        metrics = evaluate_against_ground_truth(res.graphs, gt_graphs)
        print("\n=== Evaluation (ground truth used only for assessment) ===")
        print(f"F1_avg (best matching): {metrics.f1_avg_best:.6f}")
        print(f"Best matching (pred->gt): {metrics.best_perm}")
        print(f"Per-pair F1 under best matching: {np.round(metrics.f1_per_pair, 6)}")

    save_model_npz(args.out, res)
    print(f"\nSaved model to: {args.out}")
    return 0
