#!/usr/bin/env python3
"""
Force/Torque threshold baseline for limit detection.

- Reproduces the exact train/val split by trials used in CNN_model/train_CNN.py
  (same seed and val_split from that CONFIG).
- Learns a single absolute wrench threshold per dimension (Fx,Fy,Fz,Tx,Ty,Tz)
  using only the training trials by maximizing F1 for detecting a limit in the
  corresponding normalized target dimension (|*_norm| >= limit).
- Evaluates on the validation trials and generates confusion matrices similar
  in spirit to eval_CNN.py (binary per-dimension, aggregated appended-dims, and
  overall any-dim limit detection). Also produces optional 3-class per-dimension
  confusion using sign + threshold (Neg/Neu/Pos).

Outputs are saved under: <BASE_CONFIG['out_dir']>/force_baseline_eval

Usage:
  python eval_force_based.py

Optional: tweak EVAL_FB_CONFIG below.
"""
from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Reuse config and utils from training
from CNN_model.train_CNN import (
    CONFIG as BASE_CONFIG,
    zscore_fit,
)


EVAL_FB_CONFIG = {
    # Optional CSV override; if None, uses BASE_CONFIG['csv']
    "csv": None,
    # Limit boundary in normalized label space (|label_norm| >= limit => limit hit)
    "limit": 0.8,
    # Metric to optimize thresholds on training trials: "f1" or "accuracy"
    "opt_metric": "f1",
    # Number of candidate thresholds to sweep per dimension (from quantiles of |wrench| on train)
    "num_threshold_candidates": 512,
    # Whether to z-score normalize wrench using whole-file mu/sd (as in training)
    "normalize_wrench": True,
    # Whether to emit per-dimension 3-class confusion (Neg/Neu/Pos) using sign + threshold
    "emit_3class": True,
    # PR curve sweep on a scale factor applied to learned thresholds (keeps formatting same as eval_CNN)
    "pr_scale_values": [round(x, 3) for x in np.linspace(0.5, 2.0, 31)],
}


def ensure_out_dir() -> str:
    base_out = BASE_CONFIG.get("out_dir", "./checkpoints")
    eval_dir = os.path.join(base_out, "force_baseline_eval")
    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def split_trials_exact(df: pd.DataFrame, trial_col: str, seed: int, val_split: float) -> Tuple[np.ndarray, np.ndarray, List, List]:
    trials_arr = df[trial_col].values
    unique_trials = pd.unique(trials_arr)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_trials))
    n_val = max(1, int(math.ceil(len(unique_trials) * val_split)))
    val_ids = set(unique_trials[perm[:n_val]])
    train_ids = set(unique_trials) - val_ids
    train_mask = np.isin(trials_arr, list(train_ids))
    val_mask = np.isin(trials_arr, list(val_ids))
    return train_mask, val_mask, list(train_ids), list(val_ids)


def pick_threshold(scores_abs: np.ndarray, y_true: np.ndarray, candidates: np.ndarray, metric: str = "f1") -> Tuple[float, Dict[str, float]]:
    """
    Given absolute scores (|wrench|) and binary ground-truth (limit vs not),
    sweep candidate thresholds and pick the one maximizing the chosen metric.
    Returns (best_threshold, metrics_at_best).
    """
    # Guard: if no positives or all positives, handle degenerate cases
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0:
        # No positives => best is to always predict 0; set threshold to +inf
        return float("inf"), {"TP": 0, "FP": 0, "TN": neg, "FN": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 1.0}
    if neg == 0:
        # No negatives => always positive; threshold = 0
        return 0.0, {"TP": pos, "FP": 0, "TN": 0, "FN": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0}

    best_t = candidates[0] if len(candidates) else 0.0
    best = {"f1": -1.0, "accuracy": -1.0}
    best_counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    for t in candidates:
        y_pred = (scores_abs >= t).astype(np.int32)
        TP = int(np.sum((y_true == 1) & (y_pred == 1)))
        FP = int(np.sum((y_true == 0) & (y_pred == 1)))
        TN = int(np.sum((y_true == 0) & (y_pred == 0)))
        FN = int(np.sum((y_true == 1) & (y_pred == 0)))
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        acc = (TP + TN) / max(len(y_true), 1)
        score = f1 if metric == "f1" else acc
        if score > best.get(metric, -1.0) + 1e-12 or (abs(score - best.get(metric, -1.0)) <= 1e-12 and t < best_t):
            best_t = float(t)
            best = {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}
            best_counts = {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

    best.update(best_counts)
    return best_t, best


def confusion_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (TP + TN) / max(len(y_true), 1)
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "specificity": spec, "npv": npv}


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, path: str):
    # Row-normalize (same style as eval_CNN.py heatmaps)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    # Use heatmap for nicer formatting
    annot = np.array([[f"{cm[i,j]}\n{cm_norm[i,j]:.2f}" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
    sns.heatmap(cm_norm, ax=ax, annot=annot, fmt="", cmap="Blues", vmin=0.0, vmax=1.0,
                cbar_kws={"label": "Row-Normalized"})
    ax.set_xticks(np.arange(len(labels))+0.5)
    ax.set_yticks(np.arange(len(labels))+0.5)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    cfg = BASE_CONFIG
    csv_path = EVAL_FB_CONFIG.get("csv") or cfg["csv"]
    limit = float(EVAL_FB_CONFIG.get("limit", 0.8))
    metric = str(EVAL_FB_CONFIG.get("opt_metric", "f1")).lower()
    num_cands = int(EVAL_FB_CONFIG.get("num_threshold_candidates", 512))
    use_norm = bool(EVAL_FB_CONFIG.get("normalize_wrench", True))
    emit_3 = bool(EVAL_FB_CONFIG.get("emit_3class", True))

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = ensure_out_dir()

    # Load CSV
    df = pd.read_csv(csv_path)

    wrench_cols = cfg["wrench_cols"]
    pose_cols = cfg["pose_cols"]
    label_cols = cfg["label_cols"]
    trial_col = cfg["trial_col"]

    for c in wrench_cols + pose_cols + label_cols + [trial_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")

    # Fit normalization on full file as in training (wrench only)
    wrench_all = df[wrench_cols].values.astype(np.float32)
    mu, sd = zscore_fit(wrench_all)
    if use_norm:
        W = (wrench_all - mu) / sd
    else:
        W = wrench_all.copy()

    Y = df[label_cols].values.astype(np.float32)  # normalized targets already

    # Reproduce trial-based split
    train_mask, val_mask, train_ids, val_ids = split_trials_exact(df, trial_col, cfg["seed"], cfg["val_split"])

    # Prepare arrays
    W_train = W[train_mask, :]
    W_val = W[val_mask, :]
    Y_train = Y[train_mask, :]
    Y_val = Y[val_mask, :]

    dim_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]
    # Dynamic labels for 3-class using limit
    three_class_labels = [f"< -{limit:.2f}", f"|x| < {limit:.2f}", f"> {limit:.2f}"]

    # Learn thresholds per dimension
    thresholds: Dict[str, float] = {}
    train_metrics: Dict[str, Dict[str, float]] = {}

    for d, name in enumerate(dim_names):
        scores_abs = np.abs(W_train[:, d])
        y_true = (np.abs(Y_train[:, d]) >= limit).astype(np.int32)

        if len(scores_abs) == 0:
            thresholds[name] = float("inf")
            train_metrics[name] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
            continue

        if np.all(scores_abs == scores_abs[0]):
            # constant => single candidate
            cands = np.array([scores_abs[0]], dtype=np.float32)
        else:
            qs = np.linspace(0.0, 1.0, num=max(2, num_cands), dtype=np.float64)
            cands = np.quantile(scores_abs, qs)
            cands = np.unique(cands)

        t_best, m_best = pick_threshold(scores_abs, y_true, cands, metric=metric)
        thresholds[name] = float(t_best)
        train_metrics[name] = m_best

    # Evaluate on validation
    eval_metrics: Dict[str, Dict[str, float]] = {}
    per_dim_binary_cms: Dict[str, np.ndarray] = {}

    for d, name in enumerate(dim_names):
        y_true = (np.abs(Y_val[:, d]) >= limit).astype(np.int32)
        y_pred = (np.abs(W_val[:, d]) >= thresholds[name]).astype(np.int32)
        eval_metrics[name] = confusion_binary(y_true, y_pred)
        # build 2x2 cm in [0,1] label order
        cm = np.zeros((2, 2), dtype=np.int64)
        # rows true, cols pred
        for yt, yp in zip(y_true, y_pred):
            cm[int(yt), int(yp)] += 1
        per_dim_binary_cms[name] = cm
        plot_confusion_matrix(cm, labels=["NoLimit", "Limit"], title=f"Binary CM - {name} (val)", path=os.path.join(out_dir, f"cm_binary_{name}.png"))

    # NEW: Combined 2x3 per-dimension binary confusion figure (to match eval_CNN formatting)
    sns.set(style="whitegrid")
    fig_bin, axes_bin = plt.subplots(2, 3, figsize=(12, 7))
    axes_bin_f = axes_bin.flatten()
    for i, name in enumerate(dim_names):
        cm = per_dim_binary_cms[name]
        rs = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(rs > 0, cm / rs, 0.0)
        annot = np.array([[f"{cm[0,0]}\n{cm_norm[0,0]:.2f}", f"{cm[0,1]}\n{cm_norm[0,1]:.2f}"],
                          [f"{cm[1,0]}\n{cm_norm[1,0]:.2f}", f"{cm[1,1]}\n{cm_norm[1,1]:.2f}"]])
        axb = axes_bin_f[i]
        sns.heatmap(cm_norm, ax=axb, annot=annot, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
        axb.set_title(name)
        axb.set_xlabel("Predicted")
        axb.set_ylabel("True")
        axb.set_xticklabels(["NoLimit","Limit"]) 
        axb.set_yticklabels(["NoLimit","Limit"], rotation=0)
    fig_bin.suptitle("Per-Dimension Binary Confusion Matrices", fontsize=14)
    fig_bin.tight_layout(rect=[0,0,1,0.95])
    combined_binary_path = os.path.join(out_dir, "combined_dim_binary_confusions.png")
    fig_bin.savefig(combined_binary_path, dpi=160)
    plt.close(fig_bin)

    # Aggregated (appended dims) binary CM
    Y_val_bin = (np.abs(Y_val) >= limit).astype(np.int32).reshape(-1)
    P_val_bin = (np.abs(W_val) >= np.array([thresholds[n] for n in dim_names])[None, :]).astype(np.int32).reshape(-1)

    agg = confusion_binary(Y_val_bin, P_val_bin)
    cm_agg = np.zeros((2, 2), dtype=np.int64)
    for yt, yp in zip(Y_val_bin, P_val_bin):
        cm_agg[int(yt), int(yp)] += 1
    plot_confusion_matrix(cm_agg, labels=["NoLimit", "Limit"], title="Binary CM - Aggregated Appended Dims (val)", path=os.path.join(out_dir, "cm_binary_aggregated_dims.png"))

    # Overall any-dimension limit detection per timestep
    Y_any = (np.abs(Y_val) >= limit).any(axis=1).astype(np.int32)
    P_any = (np.abs(W_val) >= np.array([thresholds[n] for n in dim_names])[None, :]).any(axis=1).astype(np.int32)

    overall = confusion_binary(Y_any, P_any)
    cm_overall = np.zeros((2, 2), dtype=np.int64)
    for yt, yp in zip(Y_any, P_any):
        cm_overall[int(yt), int(yp)] += 1
    plot_confusion_matrix(cm_overall, labels=["NoLimit", "Limit"], title="Binary CM - Overall Any-Dimension (val)", path=os.path.join(out_dir, "cm_binary_overall_any_dim.png"))

    # Optional: 3-class per-dimension confusions using sign + threshold
    # Classes: 0=Neg, 1=Neu, 2=Pos; Ground truth from labels: sign(label), neutral if |label|<limit
    three_class = {}
    three_class_paths = {}
    if emit_3:
        class_names_3 = ["Neg", "Neu", "Pos"]
        for d, name in enumerate(dim_names):
            y_abs = np.abs(Y_val[:, d])
            y_sign = np.sign(Y_val[:, d])
            y_true_3 = np.where(y_abs < limit, 1, np.where(y_sign >= 0, 2, 0)).astype(np.int32)

            w_abs = np.abs(W_val[:, d])
            w_sign = np.sign(W_val[:, d])
            y_pred_3 = np.where(w_abs < thresholds[name], 1, np.where(w_sign >= 0, 2, 0)).astype(np.int32)

            cm3 = np.zeros((3, 3), dtype=np.int64)
            for yt, yp in zip(y_true_3, y_pred_3):
                cm3[int(yt), int(yp)] += 1
            three_class[name] = cm3
            path_cm3 = os.path.join(out_dir, f"cm_3class_{name}.png")
            plot_confusion_matrix(cm3, labels=class_names_3, title=f"3-Class CM - {name} (val)", path=path_cm3)
            three_class_paths[name] = path_cm3

        # NEW: Combined 2x3 figure for 3-class per-dimension confusions (match eval_CNN formatting)
        fig_c3, axes_c3 = plt.subplots(2, 3, figsize=(12, 7))
        axes_c3f = axes_c3.flatten()
        for i, name in enumerate(dim_names):
            cm3 = three_class[name]
            rs3 = cm3.sum(axis=1, keepdims=True)
            cm3_norm = np.where(rs3 > 0, cm3/rs3, 0.0)
            annot = np.array([[f"{cm3[0,0]}\n{cm3_norm[0,0]:.2f}", f"{cm3[0,1]}\n{cm3_norm[0,1]:.2f}", f"{cm3[0,2]}\n{cm3_norm[0,2]:.2f}"],
                              [f"{cm3[1,0]}\n{cm3_norm[1,0]:.2f}", f"{cm3[1,1]}\n{cm3_norm[1,1]:.2f}", f"{cm3[1,2]}\n{cm3_norm[1,2]:.2f}"],
                              [f"{cm3[2,0]}\n{cm3_norm[2,0]:.2f}", f"{cm3[2,1]}\n{cm3_norm[2,1]:.2f}", f"{cm3[2,2]}\n{cm3_norm[2,2]:.2f}"]])
            axc = axes_c3f[i]
            sns.heatmap(cm3_norm, ax=axc, annot=annot, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
            axc.set_title(name)
            axc.set_xlabel("Pred 3-class")
            axc.set_ylabel("True 3-class")
            axc.set_xticklabels(three_class_labels)
            axc.set_yticklabels(three_class_labels, rotation=0)
        fig_c3.suptitle("Per-Dimension 3-Class Confusion Matrices", fontsize=14)
        fig_c3.tight_layout(rect=[0,0,1,0.95])
        combined_dim_3class_confusions_path = os.path.join(out_dir, "combined_dim_3class_confusions.png")
        fig_c3.savefig(combined_dim_3class_confusions_path, dpi=160)
        plt.close(fig_c3)

        # NEW: Combined per-dimension 3-class per-class accuracy bars
        fig_acc, axes_acc = plt.subplots(2, 3, figsize=(12, 7))
        axes_acc_f = axes_acc.flatten()
        for i, name in enumerate(dim_names):
            cm3 = three_class[name]
            den = cm3.sum(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                per_class_acc = np.where(den>0, np.diag(cm3)/den, 0.0)
            axb = axes_acc_f[i]
            sns.barplot(x=three_class_labels, y=per_class_acc, ax=axb, palette="Greens")
            axb.set_ylim(0,1)
            axb.set_title(name)
            for xi, v in enumerate(per_class_acc):
                axb.text(xi, max(0.02, v + 0.02 if v < 0.9 else v - 0.05), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            axb.set_xlabel("Class")
            axb.set_ylabel("Accuracy")
        fig_acc.suptitle("Per-Dimension 3-Class Accuracy", fontsize=14)
        fig_acc.tight_layout(rect=[0,0,1,0.95])
        combined_acc_path = os.path.join(out_dir, "combined_dim_accuracy_bars.png")
        fig_acc.savefig(combined_acc_path, dpi=160)
        plt.close(fig_acc)
    else:
        combined_dim_3class_confusions_path = None
        combined_acc_path = None

    # NEW: Precision-Recall curves per-dimension and overall (format like eval_CNN)
    # We sweep a scale factor s applied to the learned threshold per dim.
    pr_scales = np.array(EVAL_FB_CONFIG.get("pr_scale_values", [round(x,3) for x in np.linspace(0.5, 2.0, 31)]), dtype=float)
    pr_points_per_dim: Dict[str, list] = {}
    pr_paths_per_dim: Dict[str, str] = {}
    for d, name in enumerate(dim_names):
        y_true_bin = (np.abs(Y_val[:, d]) >= limit).astype(np.int32)
        pts = []
        for s in pr_scales:
            t = float(s) * float(thresholds[name])
            y_pred_bin = (np.abs(W_val[:, d]) >= t).astype(np.int32)
            m = confusion_binary(y_true_bin, y_pred_bin)
            pts.append({"scale": float(s), "precision": m["precision"], "recall": m["recall"],
                        "TP": m["TP"], "FP": m["FP"], "TN": m["TN"], "FN": m["FN"]})
        pts_sorted = sorted(pts, key=lambda r: r["recall"]) 
        pr_points_per_dim[name] = pts_sorted
        # Plot
        df_plot = pd.DataFrame(pts_sorted)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.lineplot(data=df_plot, x="recall", y="precision", marker="o", ax=ax)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall (Dim {name})")
        fig.tight_layout()
        pth = os.path.join(out_dir, f"pr_curve_dim_{name}.png")
        fig.savefig(pth, dpi=160)
        plt.close(fig)
        pr_paths_per_dim[name] = pth
        # CSV per dim
        csv_p = os.path.join(out_dir, f"limit_detection_pr_points_dim_{name}.csv")
        with open(csv_p, 'w', newline='') as fcsv:
            writer = __import__('csv').writer(fcsv)
            writer.writerow(["scale","precision","recall","TP","FP","TN","FN"])
            for r in pts_sorted:
                writer.writerow([r['scale'], r['precision'], r['recall'], r['TP'], r['FP'], r['TN'], r['FN']])

    # Combined per-dimension PR curves 2x3 figure
    fig_pr_grid, axes_pr = plt.subplots(2, 3, figsize=(12, 7))
    axes_pr_f = axes_pr.flatten()
    for i, name in enumerate(dim_names):
        axp = axes_pr_f[i]
        pts = pr_points_per_dim.get(name, [])
        if pts:
            dfp = pd.DataFrame(pts)
            sns.lineplot(data=dfp, x="recall", y="precision", marker="o", ax=axp)
            axp.set_xlim(0,1)
            axp.set_ylim(0,1)
            axp.set_title(name)
            axp.set_xlabel("Recall")
            axp.set_ylabel("Precision")
        else:
            axp.set_title(name)
            axp.text(0.5,0.5,"No data", ha="center", va="center")
    fig_pr_grid.suptitle("Per-Dimension Precision-Recall Curves", fontsize=14)
    fig_pr_grid.tight_layout(rect=[0,0,1,0.95])
    combined_pr_grid_path = os.path.join(out_dir, "combined_dim_pr_curves.png")
    fig_pr_grid.savefig(combined_pr_grid_path, dpi=160)
    plt.close(fig_pr_grid)

    # Overall appended PR using a single scale factor across all dims
    pr_points_overall = []
    Wabs = np.abs(W_val)
    Ybin = (np.abs(Y_val) >= limit).astype(np.int32)
    for s in pr_scales:
        thr_vec = np.array([thresholds[n] for n in dim_names], dtype=float) * float(s)
        P_bin = (Wabs >= thr_vec[None, :]).astype(np.int32).reshape(-1)
        G_bin = Ybin.reshape(-1)
        m = confusion_binary(G_bin, P_bin)
        pr_points_overall.append({"scale": float(s), "precision": m["precision"], "recall": m["recall"],
                                  "TP": m["TP"], "FP": m["FP"], "TN": m["TN"], "FN": m["FN"]})
    pr_points_overall = sorted(pr_points_overall, key=lambda r: r["recall"]) 

    df_overall = pd.DataFrame(pr_points_overall)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.lineplot(data=df_overall, x="recall", y="precision", marker="o", ax=ax)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (All Dims Appended)")
    fig.tight_layout()
    pr_curve_overall_path = os.path.join(out_dir, "pr_curve_overall_dims_appended.png")
    fig.savefig(pr_curve_overall_path, dpi=160)
    plt.close(fig)

    # Save PR overall CSV
    pr_points_csv = os.path.join(out_dir, "limit_detection_pr_points_overall.csv")
    with open(pr_points_csv, 'w', newline='') as fcsv:
        writer = __import__('csv').writer(fcsv)
        writer.writerow(["scale","precision","recall","TP","FP","TN","FN"])
        for r in pr_points_overall:
            writer.writerow([r['scale'], r['precision'], r['recall'], r['TP'], r['FP'], r['TN'], r['FN']])

    # Save summary JSON
    summary = {
        "csv": csv_path,
        "train_trials": sorted(list(train_ids)),
        "val_trials": sorted(list(val_ids)),
        "limit": limit,
        "opt_metric": metric,
        "normalize_wrench": use_norm,
        "thresholds": thresholds,
        "train_metrics": train_metrics,
        "eval_metrics_per_dim": eval_metrics,
        "aggregated_binary": agg,
        "overall_any_dim": overall,
        "paths": {
            "per_dim_binary": {k: os.path.join(out_dir, f"cm_binary_{k}.png") for k in dim_names},
            "combined_dim_binary_confusions": combined_binary_path,
            "aggregated_binary": os.path.join(out_dir, "cm_binary_aggregated_dims.png"),
            "overall_any_dim": os.path.join(out_dir, "cm_binary_overall_any_dim.png"),
            "per_dim_3class": three_class_paths if emit_3 else {},
            "combined_dim_3class_confusions": combined_dim_3class_confusions_path,
            "combined_dim_accuracy_bars": combined_acc_path,
            "pr_curve_overall": pr_curve_overall_path,
            "pr_points_overall_csv": pr_points_csv,
            "pr_curve_per_dim": pr_paths_per_dim,
            "pr_points_per_dim_csv": {k: os.path.join(out_dir, f"limit_detection_pr_points_dim_{k}.csv") for k in dim_names},
            "combined_dim_pr_curves": combined_pr_grid_path,
        },
    }
    # Ensure all numpy scalar types are converted to Python native types
    summary["train_trials"] = [int(x) for x in summary["train_trials"]]
    summary["val_trials"] = [int(x) for x in summary["val_trials"]]
    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)
    json_path = os.path.join(out_dir, "force_baseline_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"Saved summary to {json_path}")

    # Minimal console output
    print("--- Thresholds (|wrench|) ---")
    for k in dim_names:
        print(f"{k}: {thresholds[k]:.6f}")
    print("--- Overall Any-Dimension (val) ---")
    print("TP={TP} FP={FP} TN={TN} FN={FN} Prec={precision:.3f} Rec={recall:.3f} Acc={accuracy:.3f}".format(**overall))


if __name__ == "__main__":
    main()
