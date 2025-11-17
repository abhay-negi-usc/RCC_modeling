#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid search wrapper to find the best classification parameters (b0, chunk_window, margin, persistence_length)
that maximize the worst-case (minimum) row-normalized diagonal entry across all dimensions and classes
of the 3-class confusion matrices produced by eval_CNN_refactor.

Search space:
- b0 in {0.75, 0.80, 0.85, 0.90, 0.95}
- chunk_window in {10, 25, 50, 100, 200, 250, 500}
- margin in {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}
- persistence_length in {round(frac * chunk_window) for frac in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]}

Scoring objective:
- For each dimension (6 total), build the 3-class confusion matrix using the given params.
- Row-normalize each confusion matrix (per true class) to get per-class accuracies.
- Collect all diagonal elements across all dimensions and classes, discard rows with zero counts.
- Score = min of these diagonal fractions (maximize this worst-case correct-class fraction).

Optional:
- If eval_CNN_refactor.CONFIG['gt_persistence'] is True, apply persistence to GT labels as well (no margin),
  mirroring eval_CNN_refactor's behavior.

Outputs:
- Prints best parameters and score.
- Writes a JSON summary next to the checkpoint containing the best params and scores.
"""

import os
import json
import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Reuse model, dataset, and helpers from eval_CNN_refactor
from CNN_model.eval_CNN_refactor import (
    TimeSeriesCNN,
    ChunkingConfig,
    WrenchPoseChunkDataset,
    FixedStartIndexChunkDataset,
    zscore_fit,
    find_checkpoint,
    load_split_trials,
    load_window_split,
    predict_all,
    classify_window_three_class,
)

# Also import CONFIG for paths/columns
import CNN_model.eval_CNN_refactor as eval_mod


def build_val_loader(cfg_eval) -> Tuple[DataLoader, Dict]:
    """Build validation dataloader exactly like eval_CNN_refactor, and return (loader, meta).
    meta contains: df, mu, sd, ckpt_path, ckpt_dir, train_conf, dim_names
    """
    # Locate checkpoint & load training config
    ckpt_path = find_checkpoint(cfg_eval["out_dir"], cfg_eval["checkpoint"])
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found via '{cfg_eval['checkpoint']}' under {cfg_eval['out_dir']}"
        )
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_conf = ckpt.get("config", {})

    # Instantiate model
    model = TimeSeriesCNN(
        input_dim=12,
        hidden=train_conf.get("cnn_hidden", 36),
        layers=train_conf.get("cnn_layers", 8),
        kernel_size=train_conf.get("cnn_kernel", 11),
        dropout=train_conf.get("cnn_dropout", 0.1),
        dilation_base=train_conf.get("cnn_dilation_base", 16),
        bidirectional=train_conf.get("cnn_bidirectional", True),
        num_tasks=6,
    )
    model.load_state_dict(ckpt["model"])  # type: ignore[index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load CSV and compute normalization (as in training: fit on whole file)
    df = pd.read_csv(cfg_eval["csv"])
    for c in cfg_eval["wrench_cols"] + cfg_eval["pose_cols"] + cfg_eval["label_cols"]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in CSV.")
    wrench_all = df[cfg_eval["wrench_cols"]].values.astype(np.float32)
    mu, sd = zscore_fit(wrench_all)

    # Build validation dataset (window split preferred)
    dcfg = ChunkingConfig(
        window=train_conf.get("window", cfg_eval["window"]),
        stride=train_conf.get("stride", cfg_eval["stride"]),
        drop_last=True,
    )

    val_window_starts = load_window_split(cfg_eval["out_dir"])
    if val_window_starts is not None and val_window_starts.size > 0:
        val_ds = FixedStartIndexChunkDataset(
            df,
            cfg_eval["wrench_cols"],
            cfg_eval["pose_cols"],
            cfg_eval["label_cols"],
            start_indices=val_window_starts,
            window=dcfg.window,
            wrench_norm_mu=mu,
            wrench_norm_sd=sd,
        )
    else:
        val_trials, _ = load_split_trials(cfg_eval["out_dir"])  # sets of strings
        N = len(df)
        val_ds = WrenchPoseChunkDataset(
            df,
            cfg_eval["wrench_cols"],
            cfg_eval["pose_cols"],
            cfg_eval["label_cols"],
            dcfg,
            mu,
            sd,
            0,
            N,
            cfg_eval["trial_col"],
            allowed_trials=val_trials,
        )

    if len(val_ds) == 0:
        raise RuntimeError("No validation chunks; check window/stride or split.")

    val_loader = DataLoader(
        val_ds,
        batch_size=train_conf.get("batch_size", cfg_eval["batch_size"]),
        shuffle=False,
        drop_last=False,
    )

    meta = {
        "df": df,
        "mu": mu,
        "sd": sd,
        "ckpt_path": ckpt_path,
        "ckpt_dir": ckpt_dir,
        "train_conf": train_conf,
        "device": device,
        "model": model,
        "dim_names": ["X", "Y", "Z", "A", "B", "C"],
    }
    return val_loader, meta


def compute_cm3_for_params(
    preds: np.ndarray,
    gts: np.ndarray,
    chunk_window: int,
    b0: float,
    margin: float,
    persistence_len: int,
    apply_gt_persistence: bool = False,
) -> Dict[str, np.ndarray]:
    """Compute 3-class confusion matrices per dimension for given params.
    Returns dict: dim_name -> (3x3) confusion counts array (int64).
    """
    dim_names = ["X", "Y", "Z", "A", "B", "C"]
    _, T, D = preds.shape
    assert D == 6, "Expect 6 output dimensions"
    cm_by_dim: Dict[str, np.ndarray] = {}
    cw = int(chunk_window)

    for di, dn in enumerate(dim_names):
        cm = np.zeros((3, 3), dtype=np.int64)
        for b in range(preds.shape[0]):
            series_gt = gts[b, :, di]
            series_pr = preds[b, :, di]
            nwin = series_gt.shape[0] // cw
            if nwin <= 0:
                continue
            for w in range(nwin):
                s = w * cw
                e = s + cw
                g3, p3 = classify_window_three_class(
                    series_gt[s:e], series_pr[s:e], b0, margin, persistence_len, apply_gt_persistence
                )
                cm[g3, p3] += 1
        cm_by_dim[dn] = cm
    return cm_by_dim


def row_normalized_diagonal_entries(cm: np.ndarray) -> List[float]:
    """Return list of diagonal fractions (per row) for rows with nonzero counts."""
    diag = []
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sum = cm.sum(axis=1)
        for i in range(cm.shape[0]):
            if row_sum[i] > 0:
                diag.append(float(cm[i, i] / row_sum[i]))
    return diag


def main():
    torch.manual_seed(eval_mod.CONFIG.get("seed", 42))
    np.random.seed(eval_mod.CONFIG.get("seed", 42))

    # Build val loader, model, and get predictions once
    val_loader, meta = build_val_loader(eval_mod.CONFIG)
    model: torch.nn.Module = meta["model"]
    device: torch.device = meta["device"]

    preds, gts = predict_all(model, val_loader, device)  # [B,T,6]
    if preds.size == 0:
        raise RuntimeError("Empty predictions; aborting.")

    # Define grid
    b0_vals = [0.85] #[round(x, 2) for x in np.arange(0.8, 0.9 + 1e-9, 0.05)]
    cw_vals = [1, 100]
    margin_vals = [0.1, 0.2, 0.4] #[round(x, 2) for x in np.arange(0.3, 0.4 + 1e-9, 0.1)]

    gt_persist = False #bool(eval_mod.CONFIG.get("gt_persistence", False))

    best = {
        "score": -1.0,
        "mean_diag": -1.0,
        "params": None,
    }

    total_trials = 0
    for cw in cw_vals:
        k_fracs = [0.0, 0.1]
        k_vals = sorted({int(round(frac * cw)) for frac in k_fracs})
        for b0 in b0_vals:
            for margin in margin_vals:
                for k in k_vals:
                    total_trials += 1
                    cm_by_dim = compute_cm3_for_params(preds, gts, cw, b0, margin, k, gt_persist)

                    # Collect all diagonal fractions across dims/classes
                    diag_fracs: List[float] = []
                    for dn, cm in cm_by_dim.items():
                        diag_fracs.extend(row_normalized_diagonal_entries(cm))

                    if not diag_fracs:
                        # If no windows/materialized rows, skip
                        continue

                    score = float(np.min(diag_fracs))
                    mean_diag = float(np.mean(diag_fracs))

                    if (
                        score > best["score"]
                        or (
                            math.isclose(score, best["score"], rel_tol=1e-9, abs_tol=1e-9)
                            and mean_diag > best["mean_diag"]
                        )
                    ):
                        best = {
                            "score": score,
                            "mean_diag": mean_diag,
                            "params": {
                                "b0": b0,
                                "chunk_window": cw,
                                "margin": margin,
                                "persistence_length": k,
                                "gt_persistence": gt_persist,
                            },
                        }

    # Report
    print("Total combinations evaluated:", total_trials)
    print("Best parameters:")
    print(json.dumps(best["params"], indent=2))
    print(
        f"Best worst-case diagonal (row-normalized) across all dims/classes: {best['score']:.4f}"
    )
    print(f"Mean diagonal across all dims/classes for best: {best['mean_diag']:.4f}")

    # Save summary next to checkpoint
    out_dir = meta["ckpt_dir"]
    summary_path = os.path.join(out_dir, "best_3class_param_search.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params": best["params"],
                "best_worst_case_diag": best["score"],
                "best_mean_diag": best["mean_diag"],
                "grid_sizes": {
                    "b0": len(b0_vals),
                    "chunk_window": len(cw_vals),
                    "margin": len(margin_vals),
                    "k_per_cw": len({int(round(frac * 1)) for frac in [0.0, 0.01, 0.025]}),
                    "total": total_trials,
                },
                "checkpoint": meta["ckpt_path"],
                "gt_persistence": gt_persist,
            },
            f,
            indent=2,
        )
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
