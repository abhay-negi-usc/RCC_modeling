#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Ablation / Sample Efficiency Analysis for Time-Series CNN (TCN)

This script evaluates how validation loss and validation classification accuracy
scale with the amount of training data used. It uses the same dataset loading,
chunking, normalization, and model definition from train_CNN.py, and keeps the
validation set fixed across all runs.

Key compatibility updates with train_CNN.py:
- Uses trial-aware train/val split by unique `trial_col` values (no cross-trial leakage)
- Builds datasets using the updated WrenchPoseChunkDataset signature (requires trial_col and allowed_trials)

For each requested training fraction, it samples that fraction of training
chunks (with a fixed seed for reproducibility), trains for a limited number of
epochs, and records the best validation loss and best validation classification
accuracy observed.

Outputs:
- A CSV summary of results per fraction (and repeat if >1)
- Optional per-fraction confusion matrices saved under the ablation out_dir

Example:
  python CNN_model/run_data_ablation.py \
    --fractions 0.02 0.05 0.1 0.2 0.5 1.0 \
    --epochs 100 --repeats 1

Note: If you execute this script manually in a shell, activate your env first:
  conda activate icp_icra
"""

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Set

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

# Reuse model, dataset, and utilities from the training script
from CNN_model.train_CNN import (
    CONFIG as BASE_CONFIG,
    TimeSeriesCNN,
    WrenchPoseChunkDataset,
    ChunkingConfig,
    zscore_fit,
    zscore_apply,  # noqa: F401 (imported for consistency)
    make_loss_fn,
    train_one_epoch,
    evaluate,
    evaluate_classification,
)


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_trial_splits(df: pd.DataFrame, cfg: dict, seed: int) -> Tuple[Set, Set]:
    """Create train/val trial ID sets using the same approach as train_CNN.py."""
    trial_col = cfg["trial_col"]
    if trial_col not in df.columns:
        raise ValueError(f"Trial column '{trial_col}' not found in CSV. Columns: {list(df.columns)}")
    trials_arr = df[trial_col].values
    unique_trials = pd.unique(trials_arr)  # preserves order of appearance
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_trials))
    n_val_trials = max(1, int(math.ceil(len(unique_trials) * cfg["val_split"])))
    val_trial_ids = set(unique_trials[perm[:n_val_trials]])
    train_trial_ids = set(unique_trials) - val_trial_ids
    return train_trial_ids, val_trial_ids


def build_datasets(
    df: pd.DataFrame,
    cfg: dict,
    train_trial_ids: Set,
    val_trial_ids: Set,
) -> Tuple[WrenchPoseChunkDataset, WrenchPoseChunkDataset, Tuple[np.ndarray, np.ndarray]]:
    # Fit normalization on the whole file (consistent with train_CNN.py)
    wrench_all = df[cfg["wrench_cols"]].values.astype(np.float32)
    mu, sd = zscore_fit(wrench_all)

    chunk_cfg = ChunkingConfig(window=cfg["window"], stride=cfg["stride"], drop_last=True)

    N = len(df)
    trial_col = cfg["trial_col"]

    train_ds = WrenchPoseChunkDataset(
        df,
        cfg["wrench_cols"],
        cfg["pose_cols"],
        cfg["label_cols"],
        chunk_cfg,
        mu,
        sd,
        start=0,
        end=N,
        trial_col=trial_col,
        allowed_trials=train_trial_ids,
    )
    val_ds = WrenchPoseChunkDataset(
        df,
        cfg["wrench_cols"],
        cfg["pose_cols"],
        cfg["label_cols"],
        chunk_cfg,
        mu,
        sd,
        start=0,
        end=N,
        trial_col=trial_col,
        allowed_trials=val_trial_ids,
    )

    return train_ds, val_ds, (mu, sd)


def run_one_fraction(
    fraction: float,
    repeat_idx: int,
    df: pd.DataFrame,
    base_cfg: dict,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    class_boundaries: Tuple[float, float],
    out_dir: str,
    seed: int,
    eval_every: int = 5,
    patience: int = 0,
    train_trial_ids: Set | None = None,
    val_trial_ids: Set | None = None,
):
    """Train a model using a fraction of training chunks and return metrics."""
    set_seeds(seed + repeat_idx)

    if train_trial_ids is None or val_trial_ids is None:
        # Fallback: compute splits deterministically from base seed
        train_trial_ids, val_trial_ids = compute_trial_splits(df, base_cfg, seed)

    train_ds, val_ds, _ = build_datasets(df, base_cfg, train_trial_ids, val_trial_ids)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"Insufficient data after chunking. Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}"
        )

    # Determine subset size by fraction of chunks (stride-based chunks within train trials)
    total_chunks = len(train_ds)
    k = max(1, int(round(fraction * total_chunks)))
    # Build stable index list in [0, total_chunks)
    rng = np.random.default_rng(seed + repeat_idx)
    subset_indices = rng.choice(total_chunks, size=k, replace=False)
    subset_indices = sorted(int(i) for i in subset_indices)

    sub_train = Subset(train_ds, subset_indices)

    train_loader = DataLoader(sub_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = TimeSeriesCNN(
        input_dim=12,
        hidden=base_cfg["cnn_hidden"],
        layers=base_cfg["cnn_layers"],
        kernel_size=base_cfg["cnn_kernel"],
        dropout=base_cfg["cnn_dropout"],
        dilation_base=base_cfg["cnn_dilation_base"],
        bidirectional=base_cfg["cnn_bidirectional"],
        num_tasks=6,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = make_loss_fn(base_cfg)

    best_val_loss = float("inf")
    best_val_loss_epoch = 0
    best_mae_mean = None
    best_rmse_mean = None

    best_class_acc = -1.0
    best_class_epoch = 0

    no_improve = 0

    # Fraction-specific output dir
    frac_tag = f"frac_{fraction:.4f}_rep_{repeat_idx}"
    frac_dir = os.path.join(out_dir, frac_tag)
    os.makedirs(frac_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device, loss_fn)
        vl_loss, _, _, mae_mean, rmse_mean = evaluate(model, val_loader, device, loss_fn)

        # Track best validation loss
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_val_loss_epoch = epoch
            best_mae_mean = mae_mean
            best_rmse_mean = rmse_mean
            no_improve = 0
        else:
            no_improve += 1

        # Periodic classification eval
        if (epoch % max(1, eval_every)) == 0:
            class_mean_acc, _, _, fig_path = evaluate_classification(
                model, val_loader, device, frac_dir, boundaries=tuple(class_boundaries)
            )
            if class_mean_acc > best_class_acc:
                best_class_acc = class_mean_acc
                best_class_epoch = epoch

        if patience > 0 and no_improve >= patience:
            break

    # Run a final classification eval if we didn't hit it at the last epoch boundary
    if (epochs % max(1, eval_every)) != 0:
        class_mean_acc, _, _, fig_path = evaluate_classification(
            model, val_loader, device, frac_dir, boundaries=tuple(class_boundaries)
        )
        if class_mean_acc > best_class_acc:
            best_class_acc = class_mean_acc
            best_class_epoch = epochs

    result = {
        "fraction": fraction,
        "repeat": repeat_idx,
        "num_train_chunks_total": len(train_ds),
        "num_train_chunks_used": len(sub_train),
        "epochs_run": epoch,
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "best_mae_mean": best_mae_mean,
        "best_rmse_mean": best_rmse_mean,
        "best_val_class_mean_acc": best_class_acc,
        "best_val_class_epoch": best_class_epoch,
        "out_dir": frac_dir,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Data Ablation / Sample Efficiency Analysis for TSCNN")
    parser.add_argument("--csv", type=str, default=BASE_CONFIG["csv"], help="Path to processed CSV")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs per fraction")
    parser.add_argument("--batch_size", type=int, default=BASE_CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--lr", type=float, default=BASE_CONFIG["lr"], help="Learning rate")
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Fractions of training chunks to use",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per fraction")
    parser.add_argument("--seed", type=int, default=BASE_CONFIG["seed"], help="Random seed")
    parser.add_argument("--eval_every", type=int, default=5, help="Classification eval frequency (epochs)")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience on val loss (0=off)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(BASE_CONFIG["out_dir"], "ablation"),
        help="Directory to write ablation outputs",
    )

    args = parser.parse_args()

    # Build config based on BASE_CONFIG but do not use wandb or checkpoints here
    cfg = dict(BASE_CONFIG)
    cfg["csv"] = args.csv

    os.makedirs(args.out_dir, exist_ok=True)
    run_dir = os.path.join(args.out_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    print("Loading CSV:", cfg["csv"])
    df = pd.read_csv(cfg["csv"])
    # Validate columns exist
    for c in cfg["wrench_cols"] + cfg["pose_cols"] + cfg["label_cols"]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV. Available: {list(df.columns)}")
    if cfg.get("trial_col") not in df.columns:
        raise ValueError(f"Trial column '{cfg.get('trial_col')}' not found in CSV. Columns: {list(df.columns)}")

    # Trial-aware split (fixed across all runs/repeats)
    train_trial_ids, val_trial_ids = compute_trial_splits(df, cfg, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = []
    fractions = sorted(set(max(0.0, min(1.0, f)) for f in args.fractions))

    print(
        f"Will run ablation for fractions={fractions}, repeats={args.repeats}, epochs={args.epochs}, batch_size={args.batch_size}"
    )
    print(f"Train trials: {len(train_trial_ids)} | Val trials: {len(val_trial_ids)}")

    for f in fractions:
        for r in range(args.repeats):
            print(f"\n=== Fraction {f:.4f} | Repeat {r} ===")
            res = run_one_fraction(
                fraction=f,
                repeat_idx=r,
                df=df,
                base_cfg=cfg,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                class_boundaries=tuple(cfg.get("class_boundaries", [0.5, 0.75])),
                out_dir=run_dir,
                seed=args.seed,
                eval_every=args.eval_every,
                patience=args.patience,
                train_trial_ids=train_trial_ids,
                val_trial_ids=val_trial_ids,
            )
            print(
                "Result:",
                {k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items() if k not in {"out_dir"}},
            )
            results.append(res)

    # Save results CSV
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(run_dir, "ablation_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved ablation results to: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
