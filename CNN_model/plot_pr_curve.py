#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot precision-recall curves by sweeping margin and persistence for 3-class window classification.

Baseline hyperparameters:
- b0 = 0.85
- chunk_window (cw) = 25
- baseline margin = 0.40
- baseline persistence_length = 0

Sweeps:
- margin in [0.0, 0.5] with step 0.05
- persistence_length as a fraction of cw in [0.0, 0.25] with step 0.05

For each persistence fraction, we vary margin and compute a PR curve for the "Pos" class (class index 2)
aggregated across all validation windows and dimensions.
If eval_CNN_refactor.CONFIG['gt_persistence'] is True, GT persistence is also applied (no margin) consistently.

Outputs:
- PNG figure saved under the checkpoint directory
- CSV with the PR points per (persistence_frac, margin)
"""

import os
import json
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
import CNN_model.eval_CNN_refactor as eval_mod


def build_val_loader(cfg_eval) -> Tuple[DataLoader, Dict]:
    ckpt_path = find_checkpoint(cfg_eval["out_dir"], cfg_eval["checkpoint"])
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint found via '{cfg_eval['checkpoint']}' under {cfg_eval['out_dir']}"
        )
    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_conf = ckpt.get("config", {})

    # model
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

    # data
    df = pd.read_csv(cfg_eval["csv"])
    for c in cfg_eval["wrench_cols"] + cfg_eval["pose_cols"] + cfg_eval["label_cols"]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in CSV.")
    mu, sd = zscore_fit(df[cfg_eval["wrench_cols"]].values.astype(np.float32))

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
        "ckpt_path": ckpt_path,
        "ckpt_dir": ckpt_dir,
        "train_conf": train_conf,
        "device": device,
        "model": model,
    }
    return val_loader, meta


def compute_confusion3(
    preds: np.ndarray,
    gts: np.ndarray,
    cw: int,
    b0: float,
    margin: float,
    k: int,
    apply_gt_persistence: bool,
) -> np.ndarray:
    """Aggregate 3x3 confusion matrix across all dims and windows for given params."""
    _, T, D = preds.shape
    assert D == 6
    cm = np.zeros((3, 3), dtype=np.int64)
    for di in range(D):
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
                    series_gt[s:e], series_pr[s:e], b0, margin, k, apply_gt_persistence
                )
                cm[g3, p3] += 1
    return cm


def precision_recall_from_cm(cm: np.ndarray, pos_class: int = 2) -> Tuple[float, float, Dict[str, int]]:
    """Compute precision and recall for a given positive class from a 3x3 confusion matrix."""
    tp = int(cm[pos_class, pos_class])
    fp = int(cm[:, pos_class].sum() - tp)
    fn = int(cm[pos_class, :].sum() - tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall, {"tp": tp, "fp": fp, "fn": fn, "support": int(cm[pos_class, :].sum())}


def main():
    torch.manual_seed(eval_mod.CONFIG.get("seed", 42))
    np.random.seed(eval_mod.CONFIG.get("seed", 42))

    # Build val loader and get predictions once
    val_loader, meta = build_val_loader(eval_mod.CONFIG)
    model: torch.nn.Module = meta["model"]
    device: torch.device = meta["device"]

    preds, gts = predict_all(model, val_loader, device)
    if preds.size == 0:
        raise RuntimeError("Empty predictions; aborting.")

    # Baseline and sweeps
    b0 = 0.7
    cw = 10
    base_margin = 0.40
    # base_k = 0  # only for reference, not used directly

    margin_vals = [round(x, 2) for x in np.arange(0.0, 0.5 + 1e-9, 0.05)]
    k_fracs = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]

    gt_persist = False #bool(eval_mod.CONFIG.get("gt_persistence", False))

    # Collect results for plotting and CSV
    rows: List[Dict] = []
    # Store (recall, precision, margin) per persistence fraction
    curves: Dict[float, List[Tuple[float, float, float]]] = {}

    for kf in k_fracs:
        k = int(round(kf * cw))
        # (recall, precision, margin)
        pr_points: List[Tuple[float, float, float]] = []
        for m in margin_vals:
            cm = compute_confusion3(preds, gts, cw, b0, m, k, gt_persist)
            prec, rec, counts = precision_recall_from_cm(cm, pos_class=2)
            pr_points.append((rec, prec, m))
            rows.append({
                "persistence_frac": kf,
                "persistence_len": k,
                "margin": m,
                "precision": prec,
                "recall": rec,
                **counts,
            })
        # Sort by recall for a sensible curve
        pr_points.sort(key=lambda t: t[0])
        curves[kf] = pr_points

    # Plot
    ckpt_dir = meta["ckpt_dir"]
    out_dir = os.path.join(ckpt_dir, f"pr_curves_b0{b0:.2f}_cw{cw}_gtp{1 if gt_persist else 0}")
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for kf, pts in curves.items():
        if not pts:
            continue
        recalls = [r for (r, p, m) in pts]
        precisions = [p for (r, p, m) in pts]
        margins = [m for (r, p, m) in pts]
        ax.plot(recalls, precisions, marker="o", label=f"k={kf:.2f}*cw")
        # Label each point with its margin value
        for r, p, m in zip(recalls, precisions, margins):
            ax.annotate(f"{m:.2f}", (r, p), textcoords="offset points", xytext=(4, 4), fontsize=8, alpha=0.8)
    ax.set_xlabel("Recall (Pos class)", fontsize=12)
    ax.set_ylabel("Precision (Pos class)", fontsize=12)
    # Ensure x tick labels horizontal and y tick labels vertical
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)
        lbl.set_horizontalalignment('center')
    for lbl in ax.get_yticklabels():
        lbl.set_rotation(90)
        lbl.set_verticalalignment('center')
    ax.set_title(f"PR curves vs margin for b0={b0}, cw={cw}, gtp={gt_persist}")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(title="persistence fraction")
    fig_path = os.path.join(out_dir, "pr_curve_margin_vs_k.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    # Save CSV and JSON
    df_rows = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "pr_points.csv")
    df_rows.to_csv(csv_path, index=False)
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "b0": b0,
            "chunk_window": cw,
            "baseline_margin": base_margin,
            "gt_persistence": gt_persist,
            "margin_values": margin_vals,
            "persistence_fracs": k_fracs,
            "checkpoint": meta["ckpt_path"],
        }, f, indent=2)

    print("Saved:")
    print("  Figure:", fig_path)
    print("  CSV:   ", csv_path)
    print("  Meta:  ", meta_path)


if __name__ == "__main__":
    main()
