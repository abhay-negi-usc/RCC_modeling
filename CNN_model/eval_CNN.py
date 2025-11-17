#!/usr/bin/env python3
"""
Evaluate a trained Time-Series CNN (TCN) regression model on the validation set
and reproduce the validation metrics and plots (regression losses, MAE/RMSE per
dimension, classification confusion matrices, overall limit detection metrics,
etc.).

How to use:
  - Edit EVAL_CONFIG below (checkpoint/csv/outputs/device).
  - Run: python eval_CNN.py

If checkpoint is omitted, the script uses CONFIG['out_dir']/best_model_regression.pt
from the training config stored in the repository.

Outputs:
  - Prints metrics to stdout
  - Saves (if requested) a CSV of per-timestep predictions
  - Saves confusion matrices / plots to the checkpoint's out_dir
  - Writes a JSON summary of metrics (regression + classification) to metrics_eval.json

NOTE: This script re-computes the validation trial split using the seed and
val_split stored in the checkpoint's config to ensure identical validation data.
"""

import os
import json
import math
# removed argparse
from typing import Set

import numpy as np
import pandas as pd
import torch

# JSON serialization helper to convert numpy/scalar/set types to pure Python
def make_jsonable(obj):
    import numpy as _np
    if isinstance(obj, dict):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, _np.generic):  # catches np.int64, np.float32, etc.
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj

# Evaluation configuration (edit as needed)
EVAL_CONFIG = {
    # Path to checkpoint (.pt). If None, defaults to BASE_CONFIG['out_dir']/best_model_regression.pt
    # "checkpoint": "/media/rp/Elements1/abhay_ws/RCC_modeling/CNN_model/checkpoints_v5/best_model_regression.pt", # use these for paper results 
    "checkpoint": "/media/rp/Elements1/abhay_ws/RCC_modeling/CNN_model/checkpoints_v6/best_model_regression.pt",
    # Optional CSV override. If None, uses the CSV path from the checkpoint's config
    "csv": None,
    # Optional path to save per-timestep prediction CSV. If None, skip saving predictions
    "save_predictions": None,
    # Force device: "cpu" or "cuda". If None, auto-detect
    "device": None,
    # Optional override for Huber delta used during regression evaluation. If None, use checkpoint config
    "huber_delta": None,
    # Classification margin to use for main metrics/plots. If None, uses checkpoint boundaries as-is
    "margin": [0, 0.05, 0.1, 0.15, 0.2, 0.25],
    # Values to sweep for PR curve. If None, uses 31 values in [0.1, 1.0]
    "pr_margin_values": [0, 0.05, 0.1, 0.15, 0.2, 0.25],
    # Optional custom output paths for PR artifacts
    "pr_curve_path": None,
    "pr_points_csv": None,
    # NEW: chunk-based classification window length (timesteps)
    "chunk_window": 100,
    # NEW: optionally exclude per-timestep evaluation points where any GT dim is within this buffer to any class boundary
    # Units are in normalized target space. Set to None or 0 to disable.
    "exclude_boundary_buffer": None,
    # Mode for exclusion when using the buffer: 'any' => drop a timestep if ANY dim is near a boundary; 'all' => drop only if ALL dims are near.
    "exclude_boundary_mode": "any",

    # NEW — persistence (consecutive) exceedance inside a chunk
    "persistence_min": 10,          # timesteps; set 0/None to disable
    "persistence_margin": 0.1,     # like your margin: shrinks neutral band (0 ⇒ use ±b0)
    "persistence_anydim": False,    # overall label positive if any dimension persists
}

# Reuse utilities / classes / functions from training script without triggering training.
from CNN_model.train_CNN import (
    TimeSeriesCNN,
    CONFIG as BASE_CONFIG,
    ChunkingConfig,
    WrenchPoseChunkDataset,
    zscore_fit,
    evaluate,
    evaluate_classification,
    evaluate_and_save_predictions,
    make_loss_fn,
    compute_trial_segments,
    filter_segments_by_trials,
    map_norm_to_class,
    collapse_to_3class,
)


def load_checkpoint(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in ckpt:
        raise ValueError("Checkpoint missing 'config' key; cannot reconstruct model.")
    ckpt_config = ckpt["config"]
    state_dict = ckpt["model"]
    return ckpt_config, state_dict


def build_val_dataset(df: pd.DataFrame, config: dict, mu: np.ndarray, sd: np.ndarray):
    trial_col = config["trial_col"]
    trials_arr = df[trial_col].values
    unique_trials = pd.unique(trials_arr)
    rng = np.random.default_rng(config["seed"])  # deterministic
    perm = rng.permutation(len(unique_trials))
    n_val_trials = max(1, int(math.ceil(len(unique_trials) * config["val_split"])))
    val_trial_ids: Set = set(unique_trials[perm[:n_val_trials]])

    N = len(df)
    all_segments = compute_trial_segments(trials_arr, 0, N)
    val_segments = filter_segments_by_trials(all_segments, trials_arr, val_trial_ids)

    if not val_segments:
        raise RuntimeError("No validation segments produced; check window/val_split settings.")

    chunk_cfg = ChunkingConfig(window=config["window"], stride=config["stride"], drop_last=True)
    val_ds = WrenchPoseChunkDataset(
        df,
        config["wrench_cols"],
        config["pose_cols"],
        config["label_cols"],
        chunk_cfg,
        mu,
        sd,
        start=0,
        end=N,
        trial_col=trial_col,
        allowed_trials=val_trial_ids,
    )
    return val_ds, val_trial_ids


# Helper: Compute overall limit detection metrics using margin that SHRINKS original neutral region.
# Original neutral region: (-b0, b0). With margin>0: neutral becomes (-b0+margin, b0-margin).
# A detection (positive) occurs if ANY dim is outside the shrunken neutral region.
# If margin >= b0, region invalid.

def compute_limit_detection_metrics(pred_norm: np.ndarray, gt_norm: np.ndarray, margin: float, b0: float):
    # Negative margins are allowed (they expand the neutral region). Only invalid if margin >= b0.
    if margin >= b0:
        raise ValueError(f"margin {margin} >= inner boundary {b0}; neutral region would vanish or invert.")
    neutral_low = -b0 
    neutral_high = b0 
    neutral_low_with_margin = -b0 + margin
    neutral_high_with_margin = b0 - margin
    pred_outside = ((pred_norm <= neutral_low_with_margin) | (pred_norm >= neutral_high_with_margin)).any(axis=1)
    gt_outside = ((gt_norm <= neutral_low) | (gt_norm >= neutral_high)).any(axis=1)
    TP = int(np.sum(pred_outside & gt_outside))
    TN = int(np.sum(~pred_outside & ~gt_outside))
    FP = int(np.sum(pred_outside & ~gt_outside))
    FN = int(np.sum(~pred_outside & gt_outside))
    def sdiv(a,b):
        return float(a/b) if b>0 else 0.0
    precision = sdiv(TP, TP+FP)
    recall = sdiv(TP, TP+FN)
    specificity = sdiv(TN, TN+FP)
    npv = sdiv(TN, TN+FN)
    accuracy = sdiv(TP+TN, TP+TN+FP+FN)
    return {
        "margin": margin,
        "neutral_low": neutral_low,
        "neutral_high": neutral_high,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,  # alias for consistency with training metrics # FIXME 
        "specificity": specificity,
        "npv": npv,
        "accuracy": accuracy,
    }

# Build PR curve over margin offset values (shrinking neutral region)

def build_pr_curve(pred_norm: np.ndarray, gt_norm: np.ndarray, margins: np.ndarray, b0: float):
    pts = [compute_limit_detection_metrics(pred_norm, gt_norm, m, b0) for m in margins]
    return sorted(pts, key=lambda d: d["recall"])  # sort by recall

# Ensure helper exists even if not previously defined (avoid duplicate defs)
try:  # noqa: E722
    build_filter_mask_near_boundaries  # type: ignore[name-defined]
except Exception:  # NameError or others
    def build_filter_mask_near_boundaries(gt_norm: np.ndarray, boundaries, buffer: float, mode: str = "any") -> np.ndarray:
        b0_, b1_ = float(boundaries[0]), float(boundaries[1])
        bvals = np.array([-b1_, -b0_, b0_, b1_], dtype=float)
        dist = np.min(np.abs(gt_norm[..., None] - bvals[None, None, :]), axis=2)  # [N,6]
        near = dist <= float(buffer)
        if str(mode).lower() == "all":
            keep = ~near.all(axis=1)
        else:
            keep = ~near.any(axis=1)
        return keep

def _has_run_of_true(bool_arr: np.ndarray, min_len: int) -> bool:
    """
    Return True if there exists a run of >= min_len consecutive True
    in a 1D boolean array.
    """
    if min_len is None or min_len <= 1:
        return bool(np.any(bool_arr))
    # run-length of True via diff trick
    x = np.concatenate(([False], bool_arr, [False])).astype(np.int8)
    diff = np.diff(x)
    run_starts = np.where(diff == 1)[0]
    run_ends   = np.where(diff == -1)[0]
    if run_starts.size == 0:
        return False
    max_run = int(np.max(run_ends - run_starts))
    return max_run >= int(min_len)


# NEW: A local copy of evaluate_classification logic that supports optional GT-boundary filtering
@torch.no_grad()
def evaluate_classification_filtered(model, loader, device, out_dir: str, boundaries=(0.5, 0.75), *, filter_buffer: float | None = None, filter_mode: str = "any"):
    model.eval()
    preds_list, gts_list = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        preds_list.append(pred.detach().cpu().numpy())
        gts_list.append(yb.detach().cpu().numpy())

    if not preds_list:
        # Return empty defaults
        empty_cm5_path = os.path.join(out_dir, "confusion_matrix_filtered.png")
        return 0.0, [0.0]*6, np.zeros((5,5), dtype=np.int64), empty_cm5_path, 0.0, [0.0]*6, {lbl: {} for lbl in ["X","Y","Z","Roll","Pitch","Yaw"]}, {
            "TP": 0, "FP": 0, "TN": 0, "FN": 0,
            "sensitivity": 0.0, "specificity": 0.0, "precision": 0.0, "npv": 0.0, "accuracy": 0.0,
            "fig_path": os.path.join(out_dir, "overall_limit_confusion_matrix_filtered.png"),
        }

    preds = np.concatenate(preds_list, axis=0)
    gts = np.concatenate(gts_list, axis=0)
    P = preds.reshape(-1, 6)
    G = gts.reshape(-1, 6)

    # Optional GT-boundary filtering
    if filter_buffer is not None and float(filter_buffer) > 0:
        mask_keep = build_filter_mask_near_boundaries(G, boundaries, float(filter_buffer), str(filter_mode).lower())
        if mask_keep.any():
            P_use, G_use = P[mask_keep], G[mask_keep]
        else:
            P_use, G_use = P, G  # fallback if everything filtered
    else:
        P_use, G_use = P, G

    # Per-dimension accuracies + 3-class confusion/fpfn/acc plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    dim_labels_local = ["X","Y","Z","Roll","Pitch","Yaw"]
    class_names_3 = ["Neg","Neu","Pos"]
    # Create dynamic labels with boundaries for 3-class confusion plots
    b0_vis, b1_vis = float(boundaries[0]), float(boundaries[1])
    three_class_labels = [f"< -{b0_vis:.2f}", f"|x| < {b0_vis:.2f}", f"> {b0_vis:.2f}"]

    acc5_per_dim, acc3_per_dim = [], []
    per_dim_plot_paths = {lbl: {} for lbl in dim_labels_local}

    for d, lbl in enumerate(dim_labels_local):
        pred_c5 = map_norm_to_class(P_use[:, d], boundaries)
        gt_c5 = map_norm_to_class(G_use[:, d], boundaries)
        acc5_per_dim.append(float((pred_c5 == gt_c5).mean()) if len(pred_c5) else 0.0)
        pred_c3 = collapse_to_3class(pred_c5)
        gt_c3 = collapse_to_3class(gt_c5)
        acc3_per_dim.append(float((pred_c3 == gt_c3).mean()) if len(pred_c3) else 0.0)

        # 5-class confusion matrix (added)
        cm5_d = np.zeros((5,5), dtype=np.int64)
        for t, p in zip(gt_c5, pred_c5):
            cm5_d[t, p] += 1
        rs5 = cm5_d.sum(axis=1, keepdims=True)
        cm5_norm = np.where(rs5>0, cm5_d/rs5, 0.0)
        annot5 = np.array([[f"{cm5_d[i,j]}\n{cm5_norm[i,j]:.2f}" for j in range(5)] for i in range(5)])
        fig5_d, ax5_d = plt.subplots(figsize=(5,4))
        sns.heatmap(cm5_norm, ax=ax5_d, annot=annot5, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar_kws={"label":"Row-Normalized"})
        ax5_d.set_title(f"5-Class Confusion ({lbl}, filtered)")
        ax5_d.set_xlabel("Predicted")
        ax5_d.set_ylabel("True")
        ax5_d.set_xticklabels(["-2","-1","0","+1","+2"])
        ax5_d.set_yticklabels(["-2","-1","0","+1","+2"], rotation=0)
        fig5_d.tight_layout()
        cm5_d_path = os.path.join(out_dir, f"confusion_matrix_5class_dim_{lbl}_filtered.png")
        fig5_d.savefig(cm5_d_path, dpi=150)
        plt.close(fig5_d)
        per_dim_plot_paths[lbl]["cm5"] = cm5_d_path

        # 3-class confusion matrix
        cm3 = np.zeros((3,3), dtype=np.int64)
        for t, p in zip(gt_c3, pred_c3):
            cm3[t, p] += 1
        row_sums = cm3.sum(axis=1, keepdims=True)
        cm3_norm = np.where(row_sums>0, cm3/row_sums, 0.0)
        annot = np.array([[f"{cm3[0,0]}\n{cm3_norm[0,0]:.2f}", f"{cm3[0,1]}\n{cm3_norm[0,1]:.2f}", f"{cm3[0,2]}\n{cm3_norm[0,2]:.2f}"],
                          [f"{cm3[1,0]}\n{cm3_norm[1,0]:.2f}", f"{cm3[1,1]}\n{cm3_norm[1,1]:.2f}", f"{cm3[1,2]}\n{cm3_norm[1,2]:.2f}"],
                          [f"{cm3[2,0]}\n{cm3_norm[2,0]:.2f}", f"{cm3[2,1]}\n{cm3_norm[2,1]:.2f}", f"{cm3[2,2]}\n{cm3_norm[2,2]:.2f}"]])
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm3_norm, ax=ax_cm, annot=annot, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar_kws={"label": "Row-Normalized"})
        ax_cm.set_title(f"3-Class Confusion ({lbl}, filtered)")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        ax_cm.set_xticklabels(three_class_labels)
        ax_cm.set_yticklabels(three_class_labels, rotation=0)
        fig_cm.tight_layout()
        cm3_path = os.path.join(out_dir, f"confusion_matrix_3class_dim_{lbl}_filtered.png")
        fig_cm.savefig(cm3_path, dpi=150)
        plt.close(fig_cm)
        per_dim_plot_paths[lbl]["cm3"] = cm3_path

        # FP / FN bar chart per class
        fp = np.zeros(3, dtype=np.int64)
        fn = np.zeros(3, dtype=np.int64)
        for k in range(3):
            fp[k] = int(np.sum((pred_c3 == k) & (gt_c3 != k)))
            fn[k] = int(np.sum((gt_c3 == k) & (pred_c3 != k)))
        x = np.arange(3)
        width = 0.35
        fig_b, ax_b = plt.subplots(figsize=(5, 4))
        ax_b.bar(x - width/2, fp, width, label="FP", color="#e57373")
        ax_b.bar(x + width/2, fn, width, label="FN", color="#64b5f6")
        ax_b.set_title(f"3-Class FP/FN ({lbl}, filtered)")
        ax_b.set_xticks(x, three_class_labels)
        ax_b.set_ylabel("Count")
        ax_b.legend()
        fig_b.tight_layout()
        fpfn_path = os.path.join(out_dir, f"fp_fn_bar_3class_dim_{lbl}_filtered.png")
        fig_b.savefig(fpfn_path, dpi=150)
        plt.close(fig_b)
        per_dim_plot_paths[lbl]["fpfn"] = fpfn_path

        # Per-class accuracy bar chart
        per_class_den = cm3.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_acc = np.where(per_class_den>0, np.diag(cm3)/per_class_den, 0.0)
        fig_a, ax_a = plt.subplots(figsize=(5, 4))
        ax_a.bar(three_class_labels, per_class_acc, color="#81c784")
        ax_a.set_ylim(0, 1)
        ax_a.set_title(f"3-Class Per-Class Acc ({lbl}, filtered)")
        ax_a.set_ylabel("Accuracy")
        for xi, v in enumerate(per_class_acc):
            ax_a.text(xi, max(0.02, v + 0.02 if v < 0.9 else v - 0.05), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        fig_a.tight_layout()
        accbar_path = os.path.join(out_dir, f"per_class_acc_3class_dim_{lbl}_filtered.png")
        fig_a.savefig(accbar_path, dpi=150)
        plt.close(fig_a)
        per_dim_plot_paths[lbl]["accbar"] = accbar_path

    mean_acc5 = float(np.mean(acc5_per_dim)) if acc5_per_dim else 0.0
    mean_acc3 = float(np.mean(acc3_per_dim)) if acc3_per_dim else 0.0

    # Aggregated 5-class confusion matrix over all dims
    pred_all = map_norm_to_class(P_use, boundaries).reshape(-1)
    gt_all = map_norm_to_class(G_use, boundaries).reshape(-1)
    num_classes = 5
    cm5 = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(gt_all, pred_all):
        cm5[t, p] += 1

    # Save 5-class confusion image
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm5, cmap="Blues")
    ax.set_title("Validation Confusion Matrix (Aggregated, filtered)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    num_classes = 5
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm5[i, j]), ha="center", va="center", fontsize=8, color="#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm5_path = os.path.join(out_dir, "confusion_matrix_filtered.png")
    fig.savefig(cm5_path, dpi=150)
    plt.close(fig)

    # Overall limit detection (binary) from 5-class mapping: positive if any dim is non-neutral
    pred_c5_matrix = map_norm_to_class(P_use, boundaries)
    gt_c5_matrix = map_norm_to_class(G_use, boundaries)
    gt_pos = (gt_c5_matrix != 2).any(axis=1)
    pred_pos = (pred_c5_matrix != 2).any(axis=1)
    TP = int(np.sum(pred_pos & gt_pos))
    TN = int(np.sum(~pred_pos & ~gt_pos))
    FP = int(np.sum(pred_pos & ~gt_pos))
    FN = int(np.sum(~pred_pos & gt_pos))
    def sdiv(a,b):
        return float(a/b) if b>0 else 0.0
    sensitivity = sdiv(TP, TP+FN)
    specificity = sdiv(TN, TN+FP)
    precision = sdiv(TP, TP+FP)
    npv = sdiv(TN, TN+FN)
    accuracy = sdiv(TP+TN, TP+TN+FP+FN)

    overall_cm = np.array([[TN, FP], [FN, TP]], dtype=np.int64)
    row_sums = overall_cm.sum(axis=1, keepdims=True)
    overall_cm_norm = np.where(row_sums>0, overall_cm/row_sums, 0.0)

    fig2, ax2 = plt.subplots(figsize=(5,4))
    im2 = ax2.imshow(overall_cm_norm, cmap="Purples", vmin=0.0, vmax=1.0)
    ax2.set_title("Overall Limit Detection (Row-Normalized, filtered)")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_xticks([0,1],["Neg","Pos"])
    ax2.set_yticks([0,1],["Neg","Pos"], rotation=0)
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, f"{overall_cm[i,j]}\n{overall_cm_norm[i,j]:.2f}", ha="center", va="center", fontsize=8, color="#222222")
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Row-Normalized")
    fig2.tight_layout()
    overall_fig_path = os.path.join(out_dir, "overall_limit_confusion_matrix_filtered.png")
    fig2.savefig(overall_fig_path, dpi=150)
    plt.close(fig)

    overall_metrics = {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "accuracy": accuracy,
        "fig_path": overall_fig_path,
    }

    return mean_acc5, acc5_per_dim, cm5, cm5_path, mean_acc3, acc3_per_dim, per_dim_plot_paths, overall_metrics


def main():
    # 1. Decide checkpoint path
    config_for_default = BASE_CONFIG  # only for deriving default checkpoint path if not provided
    default_ckpt = os.path.join(config_for_default["out_dir"], "best_model_regression.pt")
    checkpoint_path = EVAL_CONFIG.get("checkpoint") or default_ckpt
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt_config, state_dict = load_checkpoint(checkpoint_path)

    # Allow overriding CSV path
    csv_path = EVAL_CONFIG.get("csv") or ckpt_config["csv"]
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Using data CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Verify required columns exist
    required_cols = ckpt_config["wrench_cols"] + ckpt_config["pose_cols"] + ckpt_config["label_cols"] + [ckpt_config["trial_col"]]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # 2. Fit normalization (wrench only) exactly as in training
    wrench_all = df[ckpt_config["wrench_cols"]].values.astype(np.float32)
    mu, sd = zscore_fit(wrench_all)

    # 3. Build validation dataset (reproduces original split)
    val_ds, val_trial_ids = build_val_dataset(df, ckpt_config, mu, sd)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=ckpt_config["batch_size"], shuffle=False, drop_last=False)

    device_str = EVAL_CONFIG.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Evaluating on device: {device}")

    # 4. Recreate model architecture from checkpoint config
    model = TimeSeriesCNN(
        input_dim=12,
        hidden=ckpt_config["cnn_hidden"],
        layers=ckpt_config["cnn_layers"],
        kernel_size=ckpt_config["cnn_kernel"],
        dropout=ckpt_config["cnn_dropout"],
        dilation_base=ckpt_config["cnn_dilation_base"],
        bidirectional=ckpt_config["cnn_bidirectional"],
        num_tasks=6,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 5. Regression evaluation
    # Override huber delta if requested
    if EVAL_CONFIG.get("huber_delta") is not None:
        ckpt_config["huber_delta"] = EVAL_CONFIG["huber_delta"]
    loss_fn = make_loss_fn(ckpt_config)
    val_loss, mae_per_dim, rmse_per_dim, mae_mean, rmse_mean = evaluate(model, val_loader, device, loss_fn)

    # 6. Classification evaluation (5-class & 3-class + overall limit detection)
    # boundaries = tuple(ckpt_config.get("class_boundaries", [0.5, 0.75]))
    # boundaries = [0.80,0.90] # use this for original paper results
    boundaries = [0.75,0.9] # use this for original paper results
    b0 = float(boundaries[0])  # inner boundary controls neutral region size
    b1 = float(boundaries[1])  # outer boundary for far class thresholds
    b1_h, b1_l = b1, -b1
    b1_h_with_margin = b1 - EVAL_CONFIG.get("persistence_margin", 0.0)
    b1_l_with_margin = -b1 + EVAL_CONFIG.get("persistence_margin", 0.0)
    out_dir = ckpt_config["out_dir"]
    eval_dir = os.path.join(out_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    # Use eval_dir for downstream artifacts
    _buf = EVAL_CONFIG.get("exclude_boundary_buffer")
    _mode = EVAL_CONFIG.get("exclude_boundary_mode", "any")
    margin = EVAL_CONFIG.get("persistence_margin", None)
    if _buf is not None and float(_buf) > 0:
        mean_acc5, acc5_per_dim, cm5, cm5_path, mean_acc3, acc3_per_dim, per_dim_plot_paths, overall_metrics = evaluate_classification_filtered(
            model, val_loader, device, eval_dir, boundaries, filter_buffer=float(_buf), filter_mode=_mode
        )
    else:
        mean_acc5, acc5_per_dim, cm5, cm5_path, mean_acc3, acc3_per_dim, per_dim_plot_paths, overall_metrics = evaluate_classification(
            model, val_loader, device, eval_dir, boundaries, margin
        )

    # Extract raw predictions and GT once for PR sweep (reuse evaluate_and_save_predictions style logic)
    preds_list = []
    gts_list = []
    for xb, yb in val_loader:
        xb = xb.to(device)
        with torch.no_grad():
            pred = model(xb).detach().cpu().numpy()
        gts_list.append(yb.numpy())
        preds_list.append(pred)
    P_all = np.concatenate(preds_list, axis=0).reshape(-1, 6)
    G_all = np.concatenate(gts_list, axis=0).reshape(-1, 6)

    # Optional filtering of timesteps near class boundaries (based on GT values)
    filter_buffer = EVAL_CONFIG.get("exclude_boundary_buffer")
    filter_mode = EVAL_CONFIG.get("exclude_boundary_mode", "any")
    filter_info = {"applied": False, "buffer": None, "mode": filter_mode, "kept": int(P_all.shape[0]), "excluded": 0}
    if filter_buffer is not None and float(filter_buffer) > 0:
        mask_keep = build_filter_mask_near_boundaries(G_all, boundaries, float(filter_buffer), str(filter_mode).lower())
        kept = int(mask_keep.sum())
        total = int(mask_keep.size)
        if kept == 0:
            print(f"Warning: exclude_boundary_buffer={filter_buffer} filtered out all {total} timesteps; disabling filter.")
            P_eval, G_eval = P_all, G_all
        else:
            P_eval, G_eval = P_all[mask_keep], G_all[mask_keep]
            filter_info = {"applied": True, "buffer": float(filter_buffer), "mode": filter_mode, "kept": kept, "excluded": int(total - kept)}
            print(f"Applied boundary buffer filter: kept={kept} excluded={total - kept} (mode={filter_mode}, buffer={float(filter_buffer)})")
    else:
        P_eval, G_eval = P_all, G_all

    # Margin handling for main overall limit detection: override neutral zone if margin provided
    margin_cfg = EVAL_CONFIG.get("margin")
    if isinstance(margin_cfg, (list, tuple)):
        margin_main = float(margin_cfg[0]) if margin_cfg else None
    else:
        margin_main = margin_cfg if margin_cfg is not None else None
    if margin_main is not None:
        overall_metrics = compute_limit_detection_metrics(P_eval, G_eval, float(margin_main), b0)
        # Confusion matrix for selected margin (seaborn)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")
        neutral_low = -b0 
        neutral_high = b0 
        neutral_low_with_margin = -b0 + margin_main
        neutral_high_with_margin = b0 - margin_main
        pred_outside = ((P_eval <= neutral_low_with_margin) | (P_eval >= neutral_high_with_margin)).any(axis=1)
        gt_outside = ((G_eval <= neutral_low) | (G_eval >= neutral_high)).any(axis=1)
        TP = int(np.sum(pred_outside & gt_outside))
        TN = int(np.sum(~pred_outside & ~gt_outside))
        FP = int(np.sum(pred_outside & ~gt_outside))
        FN = int(np.sum(~pred_outside & gt_outside))
        cm2 = np.array([[TN, FP],[FN, TP]], dtype=np.int64)
        row_sums = cm2.sum(axis=1, keepdims=True)
        cm2_norm = np.where(row_sums>0, cm2/row_sums, 0.0)
        fig_overall, ax_overall = plt.subplots(figsize=(5.5,4.5))
        sns.heatmap(cm2_norm, ax=ax_overall, annot=np.array([[f"{cm2[0,0]}\n{cm2_norm[0,0]:.2f}", f"{cm2[0,1]}\n{cm2_norm[0,1]:.2f}"],
                                                             [f"{cm2[1,0]}\n{cm2_norm[1,0]:.2f}", f"{cm2[1,1]}\n{cm2_norm[1,1]:.2f}"]]),
                    fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar_kws={"label": "Row-Normalized"})
        title_suffix = f" (filtered: {filter_info['kept']}/{P_all.shape[0]})" if filter_info["applied"] else ""
        ax_overall.set_title(f"Limit Detection (margin={margin_main:.3f}, neutral=({neutral_low:.3f},{neutral_high:.3f}))" + title_suffix)
        ax_overall.set_xlabel("Predicted nominal-range exceedance")
        ax_overall.set_ylabel("True nominal-range exceedance")
        ax_overall.set_xticklabels(["Neg","Pos"]) 
        ax_overall.set_yticklabels(["Neg","Pos"], rotation=0)
        metrics_str = (f"Precision: {overall_metrics['precision']:.3f}\nRecall: {overall_metrics['recall']:.3f}\n"
                       f"Specificity: {overall_metrics['specificity']:.3f}\nAcc: {overall_metrics['accuracy']:.3f}")
        ax_overall.text(2.05, 0.5, metrics_str, transform=ax_overall.transAxes, va="center", ha="left", fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
        fig_overall.tight_layout()
        overall_fig_path_margin = os.path.join(eval_dir, f"overall_limit_confusion_matrix_margin_{margin_main:.3f}.png")
        fig_overall.savefig(overall_fig_path_margin, dpi=150)
        plt.close(fig_overall)
        overall_metrics["fig_path"] = overall_fig_path_margin
    else:
        margin_main = None

    # Build PR curves across margin offsets (per-dimension and overall by appending all dimensions)
    pr_margins_cfg = EVAL_CONFIG.get("pr_margin_values")
    if pr_margins_cfg is None:
        pr_margins = np.linspace(0.0, b0 - 1e-3, 31)
    else:
        pr_margins = np.array(pr_margins_cfg, dtype=float)
    pr_margins = pr_margins[pr_margins < b0]
    pr_points = build_pr_curve(P_eval, G_eval, pr_margins, b0)

    # ---- PR margin grid (define & normalize once) ----
    pr_margins_cfg = EVAL_CONFIG.get("pr_margin_values")
    if pr_margins_cfg is None:
        pr_margins = np.linspace(0.0, b0 - 1e-3, 31, dtype=float)
    else:
        pr_margins = np.asarray(pr_margins_cfg, dtype=float)

    # Robust normalization to 1-D float array
    pr_margins = np.asarray(pr_margins, dtype=float)
    pr_margins = np.ravel(pr_margins)                 # flatten (handles [[...]] or (N,1))
    pr_margins = pr_margins[np.isfinite(pr_margins)]  # keep finite values only
    pr_margins = pr_margins[(pr_margins >= 0) & (pr_margins < b0)]  # valid range


    # Confusion matrix for each margin (overall any-dim detection)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    margin_cm_dir = os.path.join(eval_dir, "confusion_matrices_by_margin")
    os.makedirs(margin_cm_dir, exist_ok=True)
    cm_paths_by_margin = {}
    combined_dim_confusion_paths_by_margin = {}
    # NEW: per-margin 3-class combined confusion and per-dim accuracy bar paths
    combined_dim_3class_cm_paths_by_margin = {}
    combined_dim_accbar_paths_by_margin = {}
    # NEW: per-margin 5-class combined per-dimension and aggregated paths
    combined_dim_5class_cm_paths_by_margin = {}
    aggregated_5class_cm_paths_by_margin = {}
    dim_labels_margin = ["X","Y","Z","Roll","Pitch","Yaw"]
    for m in pr_margins:
        nl, nh = -b0, b0 
        nl_with_margin, nh_with_margin = -b0 + float(m), b0 - float(m)
        pred_outside_any = ((P_eval <= nl_with_margin) | (P_eval >= nh_with_margin)).any(axis=1)
        gt_outside_any = ((G_eval <= nl) | (G_eval >= nh)).any(axis=1)
        TP = int(np.sum(pred_outside_any & gt_outside_any))
        TN = int(np.sum(~pred_outside_any & ~gt_outside_any))
        FP = int(np.sum(pred_outside_any & ~gt_outside_any))
        FN = int(np.sum(~pred_outside_any & gt_outside_any))
        cm2 = np.array([[TN, FP],[FN, TP]], dtype=np.int64)
        row_sums = cm2.sum(axis=1, keepdims=True)
        cm2_norm = np.where(row_sums>0, cm2/row_sums, 0.0)
        fig, ax = plt.subplots(figsize=(5.5,4.5))
        annot = np.array([[f"{cm2[0,0]}\n{cm2_norm[0,0]:.2f}", f"{cm2[0,1]}\n{cm2_norm[0,1]:.2f}"],
                           [f"{cm2[1,0]}\n{cm2_norm[1,0]:.2f}", f"{cm2[1,1]}\n{cm2_norm[1,1]:.2f}"]])
        sns.heatmap(cm2_norm, ax=ax, annot=annot, fmt="", cmap="Blues", vmin=0.0, vmax=1.0,
                    cbar_kws={"label": "Row-Normalized"})
        title_suffix = f" (filtered: {filter_info['kept']}/{P_all.shape[0]})" if filter_info["applied"] else ""
        ax.set_title(f"Overall Limit Detection Confusion (margin={float(m):.3f})" + title_suffix)
        ax.set_xlabel("Predicted nominal-range exceedance")
        ax.set_ylabel("True nominal-range exceedance")
        ax.set_xticklabels(["Neg","Pos"]) 
        ax.set_yticklabels(["Neg","Pos"], rotation=0)
        fig.tight_layout()
        out_path = os.path.join(margin_cm_dir, f"overall_confusion_margin_{float(m):.3f}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        cm_paths_by_margin[float(m)] = out_path

        # Per-dimension binary confusion combined figure
        fig_b, axes_b = plt.subplots(2, 3, figsize=(12, 7))
        axes_bf = axes_b.flatten()
        for d, lbl in enumerate(dim_labels_margin):
            pred_out_d = (P_eval[:, d] <= nl_with_margin) | (P_eval[:, d] >= nh_with_margin)
            gt_out_d = (G_eval[:, d] <= nl) | (G_eval[:, d] >= nh)
            TPd = int(np.sum(pred_out_d & gt_out_d))
            TNd = int(np.sum(~pred_out_d & ~gt_out_d))
            FPd = int(np.sum(pred_out_d & ~gt_out_d))
            FNd = int(np.sum(~pred_out_d & gt_out_d))
            cm_d = np.array([[TNd, FPd],[FNd, TPd]], dtype=np.int64)
            rsd = cm_d.sum(axis=1, keepdims=True)
            cm_d_norm = np.where(rsd>0, cm_d/rsd, 0.0)
            annot_d = np.array([[f"{cm_d[0,0]}\n{cm_d_norm[0,0]:.2f}", f"{cm_d[0,1]}\n{cm_d_norm[0,1]:.2f}"],
                                [f"{cm_d[1,0]}\n{cm_d_norm[1,0]:.2f}", f"{cm_d[1,1]}\n{cm_d_norm[1,1]:.2f}"]])
            axd = axes_bf[d]
            sns.heatmap(cm_d_norm, ax=axd, annot=annot_d, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
            axd.set_title(f"{lbl}")
            axd.set_xlabel("Predicted nominal-range exceedance")
            axd.set_ylabel("True nominal-range exceedance")
            axd.set_xticklabels([f"<= -{b0:.2f}", f">= {b0:.2f}"]) 
            axd.set_yticklabels([f"<= -{b0:.2f}", f">= {b0:.2f}"], rotation=0)
        fig_b.suptitle(f"Chunk Per-Dim Confusions (m={float(m):.3f})" + title_suffix, fontsize=14)
        fig_b.tight_layout(rect=[0,0,1,0.95])
        out_b = os.path.join(margin_cm_dir, f"combined_dim_confusions_margin_{float(m):.3f}.png")
        fig_b.savefig(out_b, dpi=160)
        plt.close(fig_b)
        combined_dim_confusion_paths_by_margin[float(m)] = out_b

        # NEW: Combined 2x3 per-dimension 3-class confusion matrices using margin-based neutral band
        class_names_3 = ["Neg","Neu","Pos"]
        fig_cm3, axes_cm3 = plt.subplots(2, 3, figsize=(12, 7))
        axes_cm3_flat = axes_cm3.flatten()
        for d, lbl in enumerate(dim_labels_margin):
            pred_vals = P_eval[:, d]
            gt_vals = G_eval[:, d]
            pred_c = np.where(pred_vals <= nl_with_margin, 0, np.where(pred_vals >= nh_with_margin, 2, 1))
            gt_c = np.where(gt_vals <= nl, 0, np.where(gt_vals >= nh, 2, 1))
            cm3 = np.zeros((3,3), dtype=np.int64)
            for t, p in zip(gt_c, pred_c):
                cm3[t, p] += 1
            rs3 = cm3.sum(axis=1, keepdims=True)
            cm3_norm = np.where(rs3>0, cm3/rs3, 0.0)
            annot = np.array([[f"{cm3[0,0]}\n{cm3_norm[0,0]:.2f}", f"{cm3[0,1]}\n{cm3_norm[0,1]:.2f}", f"{cm3[0,2]}\n{cm3_norm[0,2]:.2f}"],
                              [f"{cm3[1,0]}\n{cm3_norm[1,0]:.2f}", f"{cm3[1,1]}\n{cm3_norm[1,1]:.2f}", f"{cm3[1,2]}\n{cm3_norm[1,2]:.2f}"],
                              [f"{cm3[2,0]}\n{cm3_norm[2,0]:.2f}", f"{cm3[2,1]}\n{cm3_norm[2,1]:.2f}", f"{cm3[2,2]}\n{cm3_norm[2,2]:.2f}"]])
            axc = axes_cm3_flat[d]
            sns.heatmap(cm3_norm, ax=axc, annot=annot, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
            axc.set_title(f"{lbl}")
            axc.set_xlabel("Pred 3-class")
            axc.set_ylabel("True 3-class")
            axc.set_xticklabels(class_names_3)
            axc.set_yticklabels(class_names_3, rotation=0)
        fig_cm3.suptitle(f"Per-Dimension 3-Class Confusions (margin={float(m):.3f})", fontsize=14)
        fig_cm3.tight_layout(rect=[0,0,1,0.95])
        combined_cm3_path = os.path.join(margin_cm_dir, f"combined_dim_3class_confusions_margin_{float(m):.3f}.png")
        fig_cm3.savefig(combined_cm3_path, dpi=160)
        plt.close(fig_cm3)
        combined_dim_3class_cm_paths_by_margin[float(m)] = combined_cm3_path

        # NEW: Combined 2x3 per-dimension 5-class confusion matrices using margin-based neutral band
        class_names_5 = ["-2","-1","0","+1","+2"]
        fig_cm5, axes_cm5 = plt.subplots(2, 3, figsize=(12, 7))
        axes_cm5_flat = axes_cm5.flatten()
        for d, lbl in enumerate(dim_labels_margin):
            pred_vals = P_eval[:, d]
            gt_vals = G_eval[:, d]
            # Map to 5 classes with neutral band shrunk by margin m
            # Far thresholds use b1; near thresholds use (b0 - m)
            nl, nh = -b0, b0 
            nl_with_margin, nh_with_margin = -b0 + float(m), b0 - float(m)
            pred_c5 = np.where(pred_vals <= b1_l_with_margin, 0,
                        np.where(pred_vals <= nl_with_margin, 1,
                        np.where(pred_vals <  nh_with_margin, 2,
                        np.where(pred_vals <  b1_h_with_margin, 3, 4))))
            gt_c5 = np.where(gt_vals <= b1_l, 0,
                     np.where(gt_vals <= nl, 1,
                     np.where(gt_vals <  nh, 2,
                     np.where(gt_vals <  b1_h, 3, 4))))
            cm5 = np.zeros((5,5), dtype=np.int64)
            for t, p in zip(gt_c5, pred_c5):
                cm5[t, p] += 1
            rs5 = cm5.sum(axis=1, keepdims=True)
            cm5_norm = np.where(rs5>0, cm5/rs5, 0.0)
            annot5 = np.array([[f"{cm5[0,0]}\n{cm5_norm[0,0]:.2f}", f"{cm5[0,1]}\n{cm5_norm[0,1]:.2f}", f"{cm5[0,2]}\n{cm5_norm[0,2]:.2f}", f"{cm5[0,3]}\n{cm5_norm[0,3]:.2f}", f"{cm5[0,4]}\n{cm5_norm[0,4]:.2f}"],
                               [f"{cm5[1,0]}\n{cm5_norm[1,0]:.2f}", f"{cm5[1,1]}\n{cm5_norm[1,1]:.2f}", f"{cm5[1,2]}\n{cm5_norm[1,2]:.2f}", f"{cm5[1,3]}\n{cm5_norm[1,3]:.2f}", f"{cm5[1,4]}\n{cm5_norm[1,4]:.2f}"],
                               [f"{cm5[2,0]}\n{cm5_norm[2,0]:.2f}", f"{cm5[2,1]}\n{cm5_norm[2,1]:.2f}", f"{cm5[2,2]}\n{cm5_norm[2,2]:.2f}", f"{cm5[2,3]}\n{cm5_norm[2,3]:.2f}", f"{cm5[2,4]}\n{cm5_norm[2,4]:.2f}"],
                               [f"{cm5[3,0]}\n{cm5_norm[3,0]:.2f}", f"{cm5[3,1]}\n{cm5_norm[3,1]:.2f}", f"{cm5[3,2]}\n{cm5_norm[3,2]:.2f}", f"{cm5[3,3]}\n{cm5_norm[3,3]:.2f}", f"{cm5[3,4]}\n{cm5_norm[3,4]:.2f}"],
                               [f"{cm5[4,0]}\n{cm5_norm[4,0]:.2f}", f"{cm5[4,1]}\n{cm5_norm[4,1]:.2f}", f"{cm5[4,2]}\n{cm5_norm[4,2]:.2f}", f"{cm5[4,3]}\n{cm5_norm[4,3]:.2f}", f"{cm5[4,4]}\n{cm5_norm[4,4]:.2f}"]])
            ax5 = axes_cm5_flat[d]
            sns.heatmap(cm5_norm, ax=ax5, annot=annot5, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
            ax5.set_title(f"{lbl}")
            ax5.set_xlabel("Pred 5-class")
            ax5.set_ylabel("True 5-class")
            ax5.set_xticklabels(class_names_5)
            ax5.set_yticklabels(class_names_5, rotation=0)
        fig_cm5.suptitle(f"Per-Dimension 5-Class Confusions (margin={float(m):.3f})", fontsize=14)
        fig_cm5.tight_layout(rect=[0,0,1,0.95])
        combined_cm5_path = os.path.join(margin_cm_dir, f"combined_dim_5class_confusions_margin_{float(m):.3f}.png")
        fig_cm5.savefig(combined_cm5_path, dpi=160)
        plt.close(fig_cm5)
        combined_dim_5class_cm_paths_by_margin[float(m)] = combined_cm5_path

        # NEW: Aggregated 5-class confusion across all dimensions using margin-based neutral band
        pred_c5_all = np.where(P_eval <= b1_l_with_margin, 0,
                        np.where(P_eval <= nl, 1,
                        np.where(P_eval <  nh, 2,
                        np.where(P_eval <  b1_h_with_margin, 3, 4))))
        gt_c5_all = np.where(G_eval <= b1_l, 0,
                      np.where(G_eval <= nl, 1,
                      np.where(G_eval <  nh, 2,
                      np.where(G_eval <  b1_h, 3, 4))))
        P5 = pred_c5_all.reshape(-1)
        G5 = gt_c5_all.reshape(-1)
        cm5_agg = np.zeros((5,5), dtype=np.int64)
        for t, p in zip(G5, P5):
            cm5_agg[t, p] += 1

        # Save 5-class confusion image
        fig5a, ax5a = plt.subplots(figsize=(6, 5))
        im5a = ax5a.imshow(cm5_agg, cmap="Blues")
        ax5a.set_title("Validation Confusion Matrix (Aggregated, filtered)")
        ax5a.set_xlabel("Predicted")
        ax5a.set_ylabel("True")
        num_classes = 5
        ax5a.set_xticks(range(num_classes))
        ax5a.set_yticks(range(num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                ax5a.text(j, i, str(cm5_agg[i, j]), ha="center", va="center", fontsize=8, color="#222222")
        fig5a.colorbar(im5a, ax=ax5a, fraction=0.046, pad=0.04)
        fig5a.tight_layout()
        agg5_path = os.path.join(margin_cm_dir, f"aggregated_5class_confusion_margin_{float(m):.3f}.png")
        fig5a.savefig(agg5_path, dpi=150)
        plt.close(fig5a)
        aggregated_5class_cm_paths_by_margin[float(m)] = agg5_path

        # NEW: Combined 2x3 per-dimension 3-class per-class accuracy bars for this margin
        fig_acc_m, axes_acc_m = plt.subplots(2, 3, figsize=(12, 7))
        axes_acc_m_flat = axes_acc_m.flatten()
        for d, lbl in enumerate(dim_labels_margin):
            pred_vals = P_eval[:, d]
            gt_vals = G_eval[:, d]
            pred_c = np.where(pred_vals <= nl_with_margin, 0, np.where(pred_vals >= nh_with_margin, 2, 1))
            gt_c = np.where(gt_vals <= nl, 0, np.where(gt_vals >= nh, 2, 1))
            cm3 = np.zeros((3,3), dtype=np.int64)
            for t, p in zip(gt_c, pred_c):
                cm3[t, p] += 1
            den = cm3.sum(axis=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                per_class_acc = np.where(den>0, np.diag(cm3)/den, 0.0)
            axb = axes_acc_m_flat[d]
            # Fixed: use class_names_3 (already defined) instead of undefined class_names_3c
            sns.barplot(x=class_names_3, y=per_class_acc, ax=axb, palette="Greens")
            axb.set_ylim(0,1)
            axb.set_title(f"{lbl}")
            for xi, v in enumerate(per_class_acc):
                axb.text(xi, max(0.02, v + 0.02 if v < 0.9 else v - 0.05), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            axb.set_xlabel("Class")
            axb.set_ylabel("Accuracy")
        fig_acc_m.suptitle(f"Per-Dimension 3-Class Accuracy (margin={float(m):.3f})", fontsize=14)
        fig_acc_m.tight_layout(rect=[0,0,1,0.95])
        combined_acc_m_path = os.path.join(margin_cm_dir, f"combined_dim_accuracy_bars_margin_{float(m):.3f}.png")
        fig_acc_m.savefig(combined_acc_m_path, dpi=160)
        plt.close(fig_acc_m)
        combined_dim_accbar_paths_by_margin[float(m)] = combined_acc_m_path
        
    # Labels for 6 dimensions
    dim_labels = ["X","Y","Z","Roll","Pitch","Yaw"]

    # Build per-dimension PR points for summary JSON
    pr_points_per_dim = {}
    for d, lbl in enumerate(dim_labels):
        pts = []
        for m in pr_margins:
            # element-wise per-dimension decision
            nl, nh = -b0, b0 
            nl_with_margin, nh_with_margin = -b0 + m, b0 - m
            pred_out = (P_eval[:, d] <= nl_with_margin) | (P_eval[:, d] >= nh_with_margin)
            gt_out = (G_eval[:, d] <= nl) | (G_eval[:, d] >= nh)
            TP = int(np.sum(pred_out & gt_out))
            TN = int(np.sum(~pred_out & ~gt_out))
            FP = int(np.sum(pred_out & ~gt_out))
            FN = int(np.sum(~pred_out & gt_out))
            prec = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
            rec = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
            pts.append({"margin": float(m), "precision": prec, "recall": rec, "TP": TP, "FP": FP, "TN": TN, "FN": FN})
        pr_points_per_dim[lbl] = sorted(pts, key=lambda r: r["recall"]) 

    # Overall by appending dimensions (treat each dimension-timestep as an example)
    pr_points_overall = []
    P_flat = P_eval.reshape(-1)
    G_flat = G_eval.reshape(-1)
    for m in pr_margins:
        nl, nh = -b0, b0 
        nl_with_margin, nh_with_margin = -b0 + m, b0 - m
        pred_out = (P_flat <= nl_with_margin) | (P_flat >= nh_with_margin)
        gt_out = (G_flat <= nl) | (G_flat >= nh)
        TP = int(np.sum(pred_out & gt_out))
        TN = int(np.sum(~pred_out & ~gt_out))
        FP = int(np.sum(pred_out & ~gt_out))
        FN = int(np.sum(~pred_out & gt_out))
        prec = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
        rec = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
        pr_points_overall.append({"margin": float(m), "precision": prec, "recall": rec, "TP": TP, "FP": FP, "TN": TN, "FN": FN})
    pr_points_overall = sorted(pr_points_overall, key=lambda r: r["recall"]) 

    # Plot PR curves with seaborn (axes 0..1)
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")

    # Per-dim plots
    pr_paths_per_dim = {}
    for lbl in dim_labels:
        pts = pr_points_per_dim[lbl]
        fig, ax = plt.subplots(figsize=(6,5))
        df_plot = pd.DataFrame(pts)
        sns.lineplot(data=df_plot, x="recall", y="precision", marker="o", ax=ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        title_suffix = f" (filtered: {filter_info['kept']}/{P_all.shape[0]})" if filter_info["applied"] else ""
        ax.set_title(f"Precision-Recall (Dim {lbl})" + title_suffix)
        # annotate margins sparsely
        if len(df_plot) > 0:
            for idx in np.linspace(0, len(df_plot)-1, min(6, len(df_plot))).astype(int):
                r = df_plot.iloc[idx]
                ax.annotate(f"m={r['margin']:.2f}", (r['recall'], r['precision']), textcoords="offset points", xytext=(6,-10), fontsize=8)
        fig.tight_layout()
        path = os.path.join(eval_dir, f"pr_curve_dim_{lbl}.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        pr_paths_per_dim[lbl] = path

    # Overall appended plot
    fig, ax = plt.subplots(figsize=(6,5))
    df_overall = pd.DataFrame(pr_points_overall)
    sns.lineplot(data=df_overall, x="recall", y="precision", marker="o", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    title_suffix = f" (filtered: {filter_info['kept']}/{P_all.shape[0]})" if filter_info["applied"] else ""
    ax.set_title("Precision-Recall (All Dims Appended)" + title_suffix)
    if len(df_overall) > 0:
        for idx in np.linspace(0, len(df_overall)-1, min(8, len(df_overall))).astype(int):
            r = df_overall.iloc[idx]
            ax.annotate(f"m={r['margin']:.2f}", (r['recall'], r['precision']), textcoords="offset points", xytext=(6,-10), fontsize=8)
    fig.tight_layout()
    pr_curve_overall_path = os.path.join(eval_dir, "pr_curve_overall_dims_appended.png")
    fig.savefig(pr_curve_overall_path, dpi=160)
    plt.close(fig)

    # ---- PR margin grid (define ONCE, before any usage) ----
    pr_margins_cfg = EVAL_CONFIG.get("pr_margin_values")
    if pr_margins_cfg is None:
        pr_margins = np.linspace(0.0, b0 - 1e-3, 31, dtype=float)
    else:
        pr_margins = np.asarray(pr_margins_cfg, dtype=float)
    # keep only valid margins (< b0)
    pr_margins = pr_margins[pr_margins < b0]


    # Save PR points CSV (overall appended)
    pr_points_csv = EVAL_CONFIG.get("pr_points_csv") or os.path.join(eval_dir, "limit_detection_pr_points_overall.csv")
    import csv
    with open(pr_points_csv, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["margin","precision","recall","TP","FP","TN","FN"])
        for pt in pr_points_overall:
            writer.writerow([pt['margin'], pt['precision'], pt['recall'], pt['TP'], pt['FP'], pt['TN'], pt['FN']])

    # Also save per-dim PR points CSVs
    pr_points_csv_per_dim = {}
    for lbl in dim_labels:
        csv_path_dim = os.path.join(eval_dir, f"limit_detection_pr_points_dim_{lbl}.csv")
        with open(csv_path_dim, 'w', newline='') as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["margin","precision","recall","TP","FP","TN","FN"])
            for pt in pr_points_per_dim[lbl]:
                writer.writerow([pt['margin'], pt['precision'], pt['recall'], pt['TP'], pt['FP'], pt['TN'], pt['FN']])
        pr_points_csv_per_dim[lbl] = csv_path_dim

    # Combined per-dimension PR curves 2x3 figure (seaborn)
    import seaborn as sns
    fig_pr_grid, axes_pr = plt.subplots(2, 3, figsize=(12, 7))
    axes_pr_flat = axes_pr.flatten()
    for d, lbl in enumerate(dim_labels):
        axp = axes_pr_flat[d]
        pts = pr_points_per_dim[lbl]
        if pts:
            dfp = pd.DataFrame(pts)
            sns.lineplot(data=dfp, x="recall", y="precision", marker="o", ax=axp)
            axp.set_xlim(0,1)
            axp.set_ylim(0,1)
            axp.set_title(lbl)
            axp.set_xlabel("Recall")
            axp.set_ylabel("Precision")
        else:
            axp.set_title(lbl)
            axp.text(0.5,0.5,"No data", ha="center", va="center")
    fig_pr_grid.suptitle("Per-Dimension Precision-Recall Curves", fontsize=14)
    fig_pr_grid.tight_layout(rect=[0,0,1,0.95])
    combined_pr_grid_path = os.path.join(eval_dir, "combined_dim_pr_curves.png")
    fig_pr_grid.savefig(combined_pr_grid_path, dpi=160)
    plt.close(fig_pr_grid)

    # Combined per-dimension accuracy bar plots 2x3 figure (3-class per-class accuracy)
    pred_c5_all = map_norm_to_class(P_eval, boundaries)  # shape [N,6]
    gt_c5_all = map_norm_to_class(G_eval, boundaries)
    pred_c3_all = collapse_to_3class(pred_c5_all)
    gt_c3_all = collapse_to_3class(gt_c5_all)
    class_names_3 = ["Neg","Neu","Pos"]
    fig_acc, axes_acc = plt.subplots(2, 3, figsize=(12, 7))
    axes_acc_flat = axes_acc.flatten()
    for d, lbl in enumerate(dim_labels):
        axb = axes_acc_flat[d]
        pred_dim = pred_c3_all[:, d]
        gt_dim = gt_c3_all[:, d]
        cm3 = np.zeros((3,3), dtype=np.int64)
        for t, p in zip(gt_dim, pred_dim):
            cm3[t,p] += 1
        den = cm3.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_acc = np.where(den>0, np.diag(cm3)/den, 0.0)
        sns.barplot(x=class_names_3, y=per_class_acc, ax=axb, palette="Greens")
        axb.set_ylim(0,1)
        axb.set_title(f"{lbl}")
        for xi, v in enumerate(per_class_acc):
            axb.text(xi, max(0.02, v + 0.02 if v < 0.9 else v - 0.05), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        axb.set_xlabel("Class")
        axb.set_ylabel("Accuracy")
    fig_acc.suptitle("Per-Dimension 3-Class Accuracy", fontsize=14)
    fig_acc.tight_layout(rect=[0,0,1,0.95])
    combined_acc_path = os.path.join(eval_dir, "combined_dim_accuracy_bars.png")
    fig_acc.savefig(combined_acc_path, dpi=160)
    plt.close(fig)

    # Combined 2x3 figure for 3-class confusion matrices per dimension
    fig_cm3, axes_cm3 = plt.subplots(2, 3, figsize=(12, 7))
    axes_cm3_flat = axes_cm3.flatten()
    for d, lbl in enumerate(dim_labels):
        axc = axes_cm3_flat[d]
        pred_dim = pred_c3_all[:, d]
        gt_dim = gt_c3_all[:, d]
        cm3 = np.zeros((3,3), dtype=np.int64)
        for t, p in zip(gt_dim, pred_dim):
            cm3[t,p] += 1
        # Row-normalize for color values
        row_sums = cm3.sum(axis=1, keepdims=True)
        cm3_norm = np.where(row_sums>0, cm3/row_sums, 0.0)
        annot = np.array([[f"{cm3[0,0]}\n{cm3_norm[0,0]:.2f}", f"{cm3[0,1]}\n{cm3_norm[0,1]:.2f}", f"{cm3[0,2]}\n{cm3_norm[0,2]:.2f}"],
                          [f"{cm3[1,0]}\n{cm3_norm[1,0]:.2f}", f"{cm3[1,1]}\n{cm3_norm[1,1]:.2f}", f"{cm3[1,2]}\n{cm3_norm[1,2]:.2f}"],
                          [f"{cm3[2,0]}\n{cm3_norm[2,0]:.2f}", f"{cm3[2,1]}\n{cm3_norm[2,1]:.2f}", f"{cm3[2,2]}\n{cm3_norm[2,2]:.2f}"]])
        sns.heatmap(cm3_norm, ax=axc, annot=annot, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
        axc.set_title(f"{lbl}")
        axc.set_xlabel("Pred 3-class")
        axc.set_ylabel("True 3-class")
        axc.set_xticklabels(class_names_3)
        axc.set_yticklabels(class_names_3, rotation=0)
    fig_cm3.suptitle("Per-Dimension 3-Class Confusion Matrices", fontsize=14)
    fig_cm3.tight_layout(rect=[0,0,1,0.95])
    combined_dim_3class_confusions_path = os.path.join(eval_dir, "combined_dim_3class_confusions.png")
    fig_cm3.savefig(combined_dim_3class_confusions_path, dpi=160)
    plt.close(fig_cm3)

    # Combined 2x3 figure for 5-class confusion matrices per dimension
    fig_cm5, axes_cm5 = plt.subplots(2, 3, figsize=(12, 7))
    axes_cm5_flat = axes_cm5.flatten()
    for d, lbl in enumerate(dim_labels):
        ax5 = axes_cm5_flat[d]
        pred_dim = pred_c5_all[:, d]
        gt_dim = gt_c5_all[:, d]
        cm5 = np.zeros((5,5), dtype=np.int64)
        for t, p in zip(gt_dim, pred_dim):
            cm5[t,p] += 1
        # Row-normalize for color values
        row_sums = cm5.sum(axis=1, keepdims=True)
        cm5_norm = np.where(row_sums>0, cm5/row_sums, 0.0)
        annot5 = np.array([[f"{cm5[0,0]}\n{cm5_norm[0,0]:.2f}", f"{cm5[0,1]}\n{cm5_norm[0,1]:.2f}", f"{cm5[0,2]}\n{cm5_norm[0,2]:.2f}", f"{cm5[0,3]}\n{cm5_norm[0,3]:.2f}", f"{cm5[0,4]}\n{cm5_norm[0,4]:.2f}"],
                           [f"{cm5[1,0]}\n{cm5_norm[1,0]:.2f}", f"{cm5[1,1]}\n{cm5_norm[1,1]:.2f}", f"{cm5[1,2]}\n{cm5_norm[1,2]:.2f}", f"{cm5[1,3]}\n{cm5_norm[1,3]:.2f}", f"{cm5[1,4]}\n{cm5_norm[1,4]:.2f}"],
                           [f"{cm5[2,0]}\n{cm5_norm[2,0]:.2f}", f"{cm5[2,1]}\n{cm5_norm[2,1]:.2f}", f"{cm5[2,2]}\n{cm5_norm[2,2]:.2f}", f"{cm5[2,3]}\n{cm5_norm[2,3]:.2f}", f"{cm5[2,4]}\n{cm5_norm[2,4]:.2f}"],
                           [f"{cm5[3,0]}\n{cm5_norm[3,0]:.2f}", f"{cm5[3,1]}\n{cm5_norm[3,1]:.2f}", f"{cm5[3,2]}\n{cm5_norm[3,2]:.2f}", f"{cm5[3,3]}\n{cm5_norm[3,3]:.2f}", f"{cm5[3,4]}\n{cm5_norm[3,4]:.2f}"],
                           [f"{cm5[4,0]}\n{cm5_norm[4,0]:.2f}", f"{cm5[4,1]}\n{cm5_norm[4,1]:.2f}", f"{cm5[4,2]}\n{cm5_norm[4,2]:.2f}", f"{cm5[4,3]}\n{cm5_norm[4,3]:.2f}", f"{cm5[4,4]}\n{cm5_norm[4,4]:.2f}"]])
        sns.heatmap(cm5_norm, ax=ax5, annot=annot5, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
        ax5.set_title(f"{lbl}")
        ax5.set_xlabel("Pred 5-class")
        ax5.set_ylabel("True 5-class")
        ax5.set_xticklabels(class_names_5)
        ax5.set_yticklabels(class_names_5, rotation=0)
    fig_cm5.suptitle("Per-Dimension 5-Class Confusion Matrices", fontsize=14)
    fig_cm5.tight_layout(rect=[0,0,1,0.95])
    combined_dim_5class_confusions_path = os.path.join(eval_dir, "combined_dim_5class_confusions.png")
    fig_cm5.savefig(combined_dim_5class_confusions_path, dpi=160)
    plt.close(fig)

    # 6b. Chunk-based classification (windowed) evaluation
    chunk_window = int(EVAL_CONFIG.get("chunk_window", 10))
    chunk_eval_dir = os.path.join(eval_dir, f"chunk_classification_w{chunk_window}")
    os.makedirs(chunk_eval_dir, exist_ok=True)

    # Helper to map a scalar to 5-class using boundaries
    def map_scalar_to_5class(x: float, bounds):
        b0_, b1_ = float(bounds[0]), float(bounds[1])
        if x <= -b1_:
            return 0  # far negative
        elif x <= -b0_:
            return 1  # near negative
        elif x < b0_:
            return 2  # neutral
        elif x < b1_:
            return 3  # near positive
        else:
            return 4  # far positive

    # Build chunked labels per dimension
    T = P_all.shape[0]
    n_chunks = T // chunk_window
    chunk_metrics = {
        "window": chunk_window,
        "n_chunks": int(n_chunks),
        "note": "Per-dimension labels are taken from the timestep with maximum absolute normalized value within each chunk.",
        "filter_info": filter_info,
    }
    chunk_mean_acc5 = 0.0
    chunk_mean_acc3 = 0.0
    chunk_acc5_per_dim = []
    chunk_acc3_per_dim = []

    if n_chunks > 0:
        # Truncate remainder
        P_use = P_all[: n_chunks * chunk_window, :]
        G_use = G_all[: n_chunks * chunk_window, :]

        # Collect confusion matrices and accuracies
        class_names_3c = ["Neg", "Neu", "Pos"]
        class_names_5c = ["-2", "-1", "0", "+1", "+2"]

        # Per-dimension 3-class confusion for a combined 2x3 figure
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")
        fig_chunk_cm3, axes_chunk_cm3 = plt.subplots(2, 3, figsize=(12, 7))
        axes_chunk_cm3_flat = axes_chunk_cm3.flatten()

        # Per-dimension 3-class accuracy bars
        fig_chunk_acc, axes_chunk_acc = plt.subplots(2, 3, figsize=(12, 7))
        axes_chunk_acc_flat = axes_chunk_acc.flatten()

        # Optional: save 5-class per-dimension confusion matrices as separate files
        cm5_paths_per_dim_chunk = {}

        # NEW: store peak values per chunk for margin-based evaluation
        gt_peak_vals = np.zeros((n_chunks, 6), dtype=np.float32)
        pr_peak_vals = np.zeros((n_chunks, 6), dtype=np.float32)

        for d, lbl in enumerate(dim_labels):
            gt_labels5 = []
            pred_labels5 = []
            for c in range(n_chunks):
                s = c * chunk_window
                e = s + chunk_window
                gt_segment = G_use[s:e, d]
                pr_segment = P_use[s:e, d]
                k_gt = int(np.argmax(np.abs(gt_segment)))
                k_pr = int(np.argmax(np.abs(pr_segment)))
                gt_val = float(gt_segment[k_gt])
                pr_val = float(pr_segment[k_pr])
                # record peak values for margin-based evaluation
                gt_peak_vals[c, d] = gt_val
                pr_peak_vals[c, d] = pr_val
                gt_labels5.append(map_scalar_to_5class(gt_val, boundaries))
                pred_labels5.append(map_scalar_to_5class(pr_val, boundaries))

            gt_labels5 = np.array(gt_labels5, dtype=np.int64)
            pred_labels5 = np.array(pred_labels5, dtype=np.int64)






            # 5-class confusion matrix image (per dim)
            cm5_d = np.zeros((5,5), dtype=np.int64)
            for t, p in zip(gt_labels5, pred_labels5):
                cm5_d[t, p] += 1
            # Save per-dim 5-class cm
            fig5, ax5 = plt.subplots(figsize=(5,4))
            rs5 = cm5_d.sum(axis=1, keepdims=True)
            cm5_norm = np.where(rs5>0, cm5_d/rs5, 0.0)
            annot5 = np.array([[f"{cm5_d[i,j]}\n{cm5_norm[i,j]:.2f}" for j in range(5)] for i in range(5)])
            sns.heatmap(cm5_norm, ax=ax5, annot=annot5, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar_kws={"label":"Row-Normalized"})
            ax5.set_title(f"{lbl} - Chunk 5-class")
            ax5.set_xlabel("Pred")
            ax5.set_ylabel("True")
            ax5.set_xticklabels(class_names_5c)
            ax5.set_yticklabels(class_names_5c, rotation=0)
            fig5.tight_layout()
            path5 = os.path.join(chunk_eval_dir, f"cm5_dim_{lbl}.png")
            fig5.savefig(path5, dpi=160)
            plt.close(fig5)
            cm5_paths_per_dim_chunk[lbl] = path5

            # 3-class collapsed
            map3 = np.array([0, 0, 1, 2, 2], dtype=np.int64)
            gt_labels3 = map3[gt_labels5]
            pred_labels3 = map3[pred_labels5]

            # Accuracies
            acc5 = float((gt_labels5 == pred_labels5).mean()) if len(gt_labels5) > 0 else 0.0
            acc3 = float((gt_labels3 == pred_labels3).mean()) if len(gt_labels3) > 0 else 0.0
            chunk_acc5_per_dim.append(acc5)
            chunk_acc3_per_dim.append(acc3)

            # 3-class confusion for combined figure
            cm3_d = np.zeros((3,3), dtype=np.int64)
            for t, p in zip(gt_labels3, pred_labels3):
                cm3_d[t, p] += 1
            rs3 = cm3_d.sum(axis=1, keepdims=True)
            cm3_norm = np.where(rs3>0, cm3_d/rs3, 0.0)
            annot = np.array([[f"{cm3_d[0,0]}\n{cm3_norm[0,0]:.2f}", f"{cm3_d[0,1]}\n{cm3_norm[0,1]:.2f}", f"{cm3_d[0,2]}\n{cm3_norm[0,2]:.2f}"],
                              [f"{cm3_d[1,0]}\n{cm3_norm[1,0]:.2f}", f"{cm3_d[1,1]}\n{cm3_norm[1,1]:.2f}", f"{cm3_d[1,2]}\n{cm3_norm[1,2]:.2f}"],
                              [f"{cm3_d[2,0]}\n{cm3_norm[2,0]:.2f}", f"{cm3_d[2,1]}\n{cm3_norm[2,1]:.2f}", f"{cm3_d[2,2]}\n{cm3_norm[2,2]:.2f}"]])
            axc = axes_chunk_cm3_flat[d]
            sns.heatmap(cm3_norm, ax=axc, annot=annot, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
            axc.set_title(f"{lbl}")
            axc.set_xlabel("Pred 3-class")
            axc.set_ylabel("True 3-class")
            axc.set_xticklabels(class_names_3)
            axc.set_yticklabels(class_names_3, rotation=0)
        fig_chunk_cm3.suptitle(f"Chunk 3-Class Confusions (window={chunk_window})", fontsize=14)
        fig_chunk_cm3.tight_layout(rect=[0,0,1,0.95])
        combined_dim_3class_confusions_chunk_path = os.path.join(chunk_eval_dir, "combined_dim_3class_confusions_chunks.png")
        fig_chunk_cm3.savefig(combined_dim_3class_confusions_chunk_path, dpi=160)
        plt.close(fig_chunk_cm3)

        fig_chunk_acc.suptitle(f"Chunk 3-Class Accuracy (window={chunk_window})", fontsize=14)
        fig_chunk_acc.tight_layout(rect=[0,0,1,0.95])
        combined_dim_accuracy_bars_chunk_path = os.path.join(chunk_eval_dir, "combined_dim_accuracy_bars_chunks.png")
        fig_chunk_acc.savefig(combined_dim_accuracy_bars_chunk_path, dpi=160)
        plt.close(fig_chunk_acc)

        # Aggregate mean accuracies (pre-filter)
        chunk_mean_acc5 = float(np.mean(chunk_acc5_per_dim)) if len(chunk_acc5_per_dim) else 0.0
        chunk_mean_acc3 = float(np.mean(chunk_acc3_per_dim)) if len(chunk_acc3_per_dim) else 0.0

        # APPLY CHUNK-LEVEL BOUNDARY FILTER (peak-based) IF ACTIVE
        chunk_filter_info = {"applied": False, "buffer": None, "mode": filter_info.get("mode"), "kept": int(n_chunks), "excluded": 0}
        if filter_info.get("applied") and filter_info.get("buffer") is not None and float(filter_info.get("buffer")) > 0:
            chunk_mask_keep = build_filter_mask_near_boundaries(gt_peak_vals, boundaries, float(filter_info["buffer"]), str(filter_info.get("mode", "any")))
            kept_chunks = int(chunk_mask_keep.sum())
            if kept_chunks > 0 and kept_chunks < n_chunks:
                # Filter peak arrays for subsequent margin sweeps & overall metrics
                gt_peak_vals = gt_peak_vals[chunk_mask_keep]
                pr_peak_vals = pr_peak_vals[chunk_mask_keep]
                chunk_filter_info = {"applied": True, "buffer": float(filter_info["buffer"]), "mode": filter_info.get("mode"), "kept": kept_chunks, "excluded": int(n_chunks - kept_chunks)}
            elif kept_chunks == 0:
                print(f"Warning: chunk boundary filter removed all {n_chunks} chunks; skipping chunk-level filtering.")
            else:  # kept all
                chunk_filter_info = {"applied": True, "buffer": float(filter_info["buffer"]), "mode": filter_info.get("mode"), "kept": kept_chunks, "excluded": 0}
        chunk_metrics["chunk_filter_info"] = chunk_filter_info

        # NEW: Margin-based evaluation on chunk peaks (similar to per-timestep margin sweep)
        chunk_margin_dir = os.path.join(chunk_eval_dir, "confusion_matrices_by_margin")
        os.makedirs(chunk_margin_dir, exist_ok=True)
        chunk_cm_paths_by_margin = {}
        chunk_combined_dim_confusion_paths_by_margin = {}
        chunk_combined_dim_3class_cm_paths_by_margin = {}
        chunk_combined_dim_accbar_paths_by_margin = {}
        # NEW: per-margin 5-class combined per-dimension and aggregated paths (chunks)
        chunk_combined_dim_5class_cm_paths_by_margin = {}
        chunk_aggregated_5class_cm_paths_by_margin = {}

        # Use filtered gt_peak_vals/pr_peak_vals for margin sweeps and metrics
        for m in pr_margins:
            nl, nh = -b0, b0 
            nl_with_margin, nh_with_margin = -b0 + float(m), b0 - float(m)
            # Overall any-dim exceedance at chunk level
            pred_out_any = ((pr_peak_vals <= nl_with_margin) | (pr_peak_vals >= nh_with_margin)).any(axis=1)
            gt_out_any = ((gt_peak_vals <= nl) | (gt_peak_vals >= nh)).any(axis=1)
            TP = int(np.sum(pred_out_any & gt_out_any))
            TN = int(np.sum(~pred_out_any & ~gt_out_any))
            FP = int(np.sum(pred_out_any & ~gt_out_any))
            FN = int(np.sum(~pred_out_any & gt_out_any))
            cm2 = np.array([[TN, FP],[FN, TP]], dtype=np.int64)
            rs = cm2.sum(axis=1, keepdims=True)
            cm2_norm = np.where(rs>0, cm2/rs, 0.0)
            fig_o, ax_o = plt.subplots(figsize=(5.5,4.5))
            annot_o = np.array([[f"{cm2[0,0]}\n{cm2_norm[0,0]:.2f}", f"{cm2[0,1]}\n{cm2_norm[0,1]:.2f}"],
                                [f"{cm2[1,0]}\n{cm2_norm[1,0]:.2f}", f"{cm2[1,1]}\n{cm2_norm[1,1]:.2f}"]])
            sns.heatmap(cm2_norm, ax=ax_o, annot=annot_o, fmt="", cmap="Blues", vmin=0.0, vmax=1.0,
                        cbar_kws={"label": "Row-Normalized"})
            title_suffix_chunk = f" (chunk-filter kept {chunk_filter_info['kept']}/{n_chunks})" if chunk_filter_info["applied"] else ""
            ax_o.set_title(f"Chunk Overall Limit Confusion (m={float(m):.3f})" + title_suffix_chunk)
            ax_o.set_xlabel("Predicted nominal-range exceedance")
            ax_o.set_ylabel("True nominal-range exceedance")
            ax_o.set_xticklabels(["Neg","Pos"])
            ax_o.set_yticklabels(["Neg","Pos"], rotation=0)
            fig_o.tight_layout()
            out_o = os.path.join(chunk_margin_dir, f"overall_confusion_margin_{float(m):.3f}.png")
            fig_o.savefig(out_o, dpi=150)
            plt.close(fig_o)
            chunk_cm_paths_by_margin[float(m)] = out_o

            # Per-dimension binary confusion combined figure
            fig_b, axes_b = plt.subplots(2, 3, figsize=(12, 7))
            axes_bf = axes_b.flatten()
            for d, lbl in enumerate(dim_labels):
                pred_out_d = (pr_peak_vals[:, d] <= nl_with_margin) | (pr_peak_vals[:, d] >= nh_with_margin)
                gt_out_d = (gt_peak_vals[:, d] <= nl) | (gt_peak_vals[:, d] >= nh)
                TPd = int(np.sum(pred_out_d & gt_out_d))
                TNd = int(np.sum(~pred_out_d & ~gt_out_d))
                FPd = int(np.sum(pred_out_d & ~gt_out_d))
                FNd = int(np.sum(~pred_out_d & gt_out_d))
                cm_d = np.array([[TNd, FPd],[FNd, TPd]], dtype=np.int64)
                rsd = cm_d.sum(axis=1, keepdims=True)
                cm_d_norm = np.where(rsd>0, cm_d/rsd, 0.0)
                annot_d = np.array([[f"{cm_d[0,0]}\n{cm_d_norm[0,0]:.2f}", f"{cm_d[0,1]}\n{cm_d_norm[0,1]:.2f}"],
                                    [f"{cm_d[1,0]}\n{cm_d_norm[1,0]:.2f}", f"{cm_d[1,1]}\n{cm_d_norm[1,1]:.2f}"]])
                axd = axes_bf[d]
                sns.heatmap(cm_d_norm, ax=axd, annot=annot_d, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
                axd.set_title(f"{lbl}")
                axd.set_xlabel("Predicted nominal-range exceedance")
                axd.set_ylabel("True nominal-range exceedance")
                axd.set_xticklabels([f"<= -{b0:.2f}", f">= {b0:.2f}"]) 
                axd.set_yticklabels([f"<= -{b0:.2f}", f">= {b0:.2f}"], rotation=0)
            fig_b.suptitle(f"Chunk Per-Dim Confusions (m={float(m):.3f})" + title_suffix_chunk, fontsize=14)
            fig_b.tight_layout(rect=[0,0,1,0.95])
            out_b = os.path.join(chunk_margin_dir, f"combined_dim_confusions_margin_{float(m):.3f}.png")
            fig_b.savefig(out_b, dpi=160)
            plt.close(fig_b)
            chunk_combined_dim_confusion_paths_by_margin[float(m)] = out_b

            # Per-dimension 3-class confusion for combined figure with margin band
            fig_c3, axes_c3 = plt.subplots(2, 3, figsize=(12, 7))
            axes_c3f = axes_c3.flatten()
            for d, lbl in enumerate(dim_labels_margin):
                pred_vals = pr_peak_vals[:, d]
                gt_vals = gt_peak_vals[:, d]
                pred_c = np.where(pred_vals <= nl_with_margin, 0, np.where(pred_vals >= nh_with_margin, 2, 1))
                gt_c = np.where(gt_vals <= nl, 0, np.where(gt_vals >= nh, 2, 1))
                cm3_d = np.zeros((3,3), dtype=np.int64)
                for t, p in zip(gt_c, pred_c):
                    cm3_d[t, p] += 1
                rs3 = cm3_d.sum(axis=1, keepdims=True)
                cm3_norm = np.where(rs3>0, cm3_d/rs3, 0.0)
                annot = np.array([[f"{cm3_d[0,0]}\n{cm3_norm[0,0]:.2f}", f"{cm3_d[0,1]}\n{cm3_norm[0,1]:.2f}", f"{cm3_d[0,2]}\n{cm3_norm[0,2]:.2f}"],
                                  [f"{cm3_d[1,0]}\n{cm3_norm[1,0]:.2f}", f"{cm3_d[1,1]}\n{cm3_norm[1,1]:.2f}", f"{cm3_d[1,2]}\n{cm3_norm[1,2]:.2f}"],
                                  [f"{cm3_d[2,0]}\n{cm3_norm[2,0]:.2f}", f"{cm3_d[2,1]}\n{cm3_norm[2,1]:.2f}", f"{cm3_d[2,2]}\n{cm3_norm[2,2]:.2f}"]])
                axc = axes_c3f[d]
                sns.heatmap(cm3_norm, ax=axc, annot=annot, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
                axc.set_title(f"{lbl}")
                # axc.set_xlabel("Pred 3-class")
                # axc.set_ylabel("True 3-class")
                axc.set_xlabel("Inferred Class")
                axc.set_ylabel("True Class")
                axc.set_xticklabels(class_names_3)
                axc.set_yticklabels(class_names_3, rotation=0)
            # fig_c3.suptitle(f"Chunk Per-Dim 3-Class Confusions (m={float(m):.3f})" + title_suffix_chunk, fontsize=14)
            fig_c3.suptitle(f"Classification Confusion Matrices Per Dimension", fontsize=14)
            fig_c3.tight_layout(rect=[0,0,1,0.95])
            out_c3 = os.path.join(chunk_margin_dir, f"combined_dim_3class_confusions_margin_{float(m):.3f}.png")
            fig_c3.savefig(out_c3, dpi=300)
            plt.close(fig_c3)
            chunk_combined_dim_3class_cm_paths_by_margin[float(m)] = out_c3

            # NEW: Per-dimension 5-class confusion combined figure with margin band
            class_names_5c = ["-2","-1","0","+1","+2"]
            fig_c5, axes_c5 = plt.subplots(2, 3, figsize=(12, 7))
            axes_c5f = axes_c5.flatten()
            nl, nh = -b0, b0 
            nl_with_margin, nh_with_margin = -b0 + float(m), b0 - float(m)
            for d, lbl in enumerate(dim_labels_margin):
                pred_vals = pr_peak_vals[:, d]
                gt_vals = gt_peak_vals[:, d]
                pred_c5 = np.where(pred_vals <= b1_l_with_margin, 0,
                            np.where(pred_vals <= nl_with_margin, 1,
                            np.where(pred_vals <  nh_with_margin, 2,
                            np.where(pred_vals <  b1_h_with_margin, 3, 4))))
                gt_c5 = np.where(gt_vals <= b1_l, 0,
                          np.where(gt_vals <= nl, 1,
                          np.where(gt_vals <  nh, 2,
                          np.where(gt_vals <  b1_h, 3, 4))))
                cm5_d = np.zeros((5,5), dtype=np.int64)
                for t, p in zip(gt_c5, pred_c5):
                    cm5_d[t, p] += 1
                rs5d = cm5_d.sum(axis=1, keepdims=True)
                cm5d_norm = np.where(rs5d>0, cm5_d/rs5d, 0.0)
                annot5d = np.array([[f"{cm5_d[i,j]}\n{cm5d_norm[i,j]:.2f}" for j in range(5)] for i in range(5)])
                ax5d = axes_c5f[d]
                sns.heatmap(cm5d_norm, ax=ax5d, annot=annot5d, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
                ax5d.set_title(f"{lbl}")
                ax5d.set_xlabel("Pred 5-class")
                ax5d.set_ylabel("True 5-class")
                ax5d.set_xticklabels(class_names_5c)
                ax5d.set_yticklabels(class_names_5c, rotation=0)
            fig_c5.suptitle(f"Chunk Per-Dim 5-Class Confusions (m={float(m):.3f})" + title_suffix_chunk, fontsize=14)
            fig_c5.tight_layout(rect=[0,0,1,0.95])
            out_c5 = os.path.join(chunk_margin_dir, f"combined_dim_5class_confusions_margin_{float(m):.3f}.png")
            fig_c5.savefig(out_c5, dpi=160)
            plt.close(fig_c5)
            chunk_combined_dim_5class_cm_paths_by_margin[float(m)] = out_c5

            # NEW: Aggregated 5-class confusion across dims for chunks
            pred_c5_all = np.where(pr_peak_vals <= b1_l_with_margin, 0,
                            np.where(pr_peak_vals <= nl, 1,
                            np.where(pr_peak_vals <  nh, 2,
                            np.where(pr_peak_vals <  b1_h_with_margin, 3, 4))))
            gt_c5_all = np.where(gt_peak_vals <= -b1, 0,
                      np.where(gt_peak_vals <= nl, 1,
                      np.where(gt_peak_vals <  nh, 2,
                      np.where(gt_peak_vals <  b1, 3, 4))))
            P5c = pred_c5_all.reshape(-1)
            G5c = gt_c5_all.reshape(-1)
            cm5c = np.zeros((5,5), dtype=np.int64)
            for t, p in zip(G5c, P5c):
                cm5c[t, p] += 1
            rs5c = cm5c.sum(axis=1, keepdims=True)
            cm5c_norm = np.where(rs5c>0, cm5c/rs5c, 0.0)
            annot5c = np.array([[f"{cm5c[i,j]}\n{cm5c_norm[i,j]:.2f}" for j in range(5)] for i in range(5)])
            fig5c, ax5c = plt.subplots(figsize=(6,5))
            sns.heatmap(cm5c_norm, ax=ax5c, annot=annot5c, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar_kws={"label":"Row-Normalized"})
            ax5c.set_title(f"Chunk Aggregated 5-Class Confusion (m={float(m):.3f})")
            ax5c.set_xlabel("Predicted")
            ax5c.set_ylabel("True")
            ax5c.set_xticklabels(class_names_5c)
            ax5c.set_yticklabels(class_names_5c, rotation=0)
            fig5c.tight_layout()
            out_agg5 = os.path.join(chunk_margin_dir, f"aggregated_5class_confusion_margin_{float(m):.3f}.png")
            fig5c.savefig(out_agg5, dpi=150)
            plt.close(fig5c)
            chunk_aggregated_5class_cm_paths_by_margin[float(m)] = out_agg5

        # --- NEW: margin-based overall metrics & per-chunk 3-class predictions CSV ---
        # Collect metrics for each margin (overall any-dim exceedance at chunk level)
        chunk_overall_metrics_by_margin = [compute_limit_detection_metrics(pr_peak_vals, gt_peak_vals, float(m), b0) for m in pr_margins]
        # If a main margin was specified earlier, compute its metrics & confusion matrix path (reusing generated figure if exists)
        if margin_main is not None and margin_main in chunk_cm_paths_by_margin:
            chunk_overall_limit_detection_margin_main = compute_limit_detection_metrics(pr_peak_vals, gt_peak_vals, float(margin_main), b0)
            chunk_overall_limit_detection_margin_main["fig_path"] = chunk_cm_paths_by_margin[float(margin_main)]
        else:
            chunk_overall_limit_detection_margin_main = None
        # Also save per-chunk 3-class predictions for margin_main (if provided)
        chunk_margin_predictions_csv = None
        if margin_main is not None:
            nl_main, nh_main = -b0, b0
            nl_main_with_margin, nh_main_with_margin = -b0 + float(margin_main), b0 - float(margin_main)
            # 3-class mapping using margin_main neutral band
            gt_c_margin = np.where(gt_peak_vals <= nl_main_with_margin, 0, np.where(gt_peak_vals >= nh_main_with_margin, 2, 1))  # shape [kept_chunks,6]
            pr_margins = np.where(pr_peak_vals <= nl_main, 0, np.where(pr_peak_vals >= nh_main, 2, 1))
            # Save CSV: chunk_index plus GT_/Pred_ per dim (indices refer to filtered ordering)
            chunk_margin_predictions_csv = os.path.join(chunk_eval_dir, f"chunk_margin_{margin_main:.3f}_predictions.csv")
            import csv as _csv
            with open(chunk_margin_predictions_csv, 'w', newline='') as fcsv_margin:
                writer = _csv.writer(fcsv_margin)
                header = ["chunk_filtered_idx"] + [f"gt_{d}" for d in dim_labels] + [f"pred_{d}" for d in dim_labels]
                writer.writerow(header)
                for ci in range(gt_peak_vals.shape[0]):
                    writer.writerow([ci] + gt_c_margin[ci].tolist() + pr_margins[ci].tolist())

        
        # 6c. Chunk-based PERSISTENCE evaluation (consecutive exceedance inside chunk)
        K = EVAL_CONFIG.get("persistence_min", None)
        if K is not None and int(K) >= 1 and n_chunks > 0:
            K = int(K)
            p_margin = float(EVAL_CONFIG.get("persistence_margin", 0.0))
            # neutral band becomes (-b0 + p_margin, b0 - p_margin)
            nl_p, nh_p = -b0, b0 
            nl_p_with_margin, nh_p_with_margin = -b0 + p_margin, b0 - p_margin
            anydim_mode = bool(EVAL_CONFIG.get("persistence_anydim", True))

            chunk_persist_dir = os.path.join(
                eval_dir,
                f"chunk_persistence_w{chunk_window}_k{K}_m{p_margin:.3f}{'_anydim' if anydim_mode else '_alldims'}"
            )
            os.makedirs(chunk_persist_dir, exist_ok=True)

            # Use the same truncated arrays as the chunk evaluation
            P_use = P_all[: n_chunks * chunk_window, :]
            G_use = G_all[: n_chunks * chunk_window, :]

            # Reshape to [n_chunks, chunk_window, 6]
            P_ch = P_use.reshape(n_chunks, chunk_window, 6)
            G_ch = G_use.reshape(n_chunks, chunk_window, 6)

            # Per-dimension persistence decisions for each chunk
            # pred/gt exceedance (per timestep)
            pred_ex = (P_ch <= nl_p_with_margin) | (P_ch >= nh_p_with_margin)           # [C,W,6]
            gt_ex   = (G_ch <= nl_p) | (G_ch >= nh_p)           # [C,W,6]

            # Reduce inside chunk using “persistence of length ≥ K”
            pred_persist = np.zeros((n_chunks, 6), dtype=bool)
            gt_persist   = np.zeros((n_chunks, 6), dtype=bool)
            for c in range(n_chunks):
                for d in range(6):
                    pred_persist[c, d] = _has_run_of_true(pred_ex[c, :, d], K)
                    gt_persist[c, d]   = _has_run_of_true(gt_ex[c, :, d],   K)

            # ---------- Per-dimension persistence confusion (binary) ----------
            import seaborn as sns
            sns.set(style="whitegrid")

            dim_labels = ["X","Y","Z","Roll","Pitch","Yaw"]
            fig_pd, axes_pd = plt.subplots(2, 3, figsize=(12, 7))
            axes_pd_f = axes_pd.flatten()
            per_dim_cm_paths = {}

            for d, lbl in enumerate(dim_labels):
                y_true = gt_persist[:, d]
                y_pred = pred_persist[:, d]
                # cm rows = true [Neg, Pos], cols = pred [Neg, Pos]
                TN = int(np.sum(~y_true & ~y_pred))
                FP = int(np.sum(~y_true &  y_pred))
                FN = int(np.sum( y_true & ~y_pred))
                TP = int(np.sum( y_true &  y_pred))
                cm = np.array([[TN, FP],[FN, TP]], dtype=np.int64)
                rs = cm.sum(axis=1, keepdims=True)
                cm_norm = np.where(rs>0, cm/rs, 0.0)

                ax = axes_pd_f[d]
                annot = np.array([[f"{cm[0,0]}\n{cm_norm[0,0]:.2f}", f"{cm[0,1]}\n{cm_norm[0,1]:.2f}"],
                                [f"{cm[1,0]}\n{cm_norm[1,0]:.2f}", f"{cm[1,1]}\n{cm_norm[1,1]:.2f}"]])
                sns.heatmap(cm_norm, ax=ax, annot=annot, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
                ax.set_title(lbl)
                ax.set_xlabel("Predicted (persist)")
                ax.set_ylabel("True (persist)")
                ax.set_xticklabels(["Neg","Pos"])
                ax.set_yticklabels(["Neg","Pos"], rotation=0)

                # save individual cm too
                fig_i, ax_i = plt.subplots(figsize=(5.5,4.5))
                sns.heatmap(cm_norm, ax=ax_i, annot=annot, fmt="", cmap="Blues", vmin=0.0, vmax=1.0,
                            cbar_kws={"label":"Row-Normalized"})
                ax_i.set_title(f"Persistence Confusion ({lbl})  K={K}, m={p_margin:.3f}")
                ax_i.set_xlabel("Predicted (persist)")
                ax_i.set_ylabel("True (persist)")
                ax_i.set_xticklabels(["Neg","Pos"])
                ax_i.set_yticklabels(["Neg","Pos"], rotation=0)
                fig_i.tight_layout()
                path_i = os.path.join(chunk_persist_dir, f"persist_confusion_dim_{lbl}.png")
                fig_i.savefig(path_i, dpi=160)
                plt.close(fig_i)
                per_dim_cm_paths[lbl] = path_i

            fig_pd.suptitle(f"Per-Dimension Persistence Confusions  (W={chunk_window}, K={K}, m={p_margin:.3f})",
                            fontsize=14)
            fig_pd.tight_layout(rect=[0,0,1,0.95])
            combined_pd_path = os.path.join(chunk_persist_dir, "combined_dim_persistence_confusions.png")
            fig_pd.savefig(combined_pd_path, dpi=160)
            plt.close(fig_pd)

            # ---------- Overall (any-dimension) persistence confusion ----------
            if anydim_mode:
                y_true_all = gt_persist.any(axis=1)   # any dim persists in GT
                y_pred_all = pred_persist.any(axis=1) # any dim persists in Pred
            else:
                # “all dims must persist” option
                y_true_all = gt_persist.all(axis=1)
                y_pred_all = pred_persist.all(axis=1)

            TN = int(np.sum(~y_true_all & ~y_pred_all))
            FP = int(np.sum(~y_true_all &  y_pred_all))
            FN = int(np.sum( y_true_all & ~y_pred_all))
            TP = int(np.sum( y_true_all &  y_pred_all))

            def _sdiv(a,b): return float(a/b) if b>0 else 0.0
            sensitivity = _sdiv(TP, TP+FN)
            specificity = _sdiv(TN, TN+FP)
            precision  = _sdiv(TP, TP+FP)
            npv        = _sdiv(TN, TN+FN)
            accuracy   = _sdiv(TP+TN, TP+TN+FP+FN)

            cm_overall = np.array([[TN, FP],[FN, TP]], dtype=np.int64)
            rs = cm_overall.sum(axis=1, keepdims=True)
            cm_overall_norm = np.where(rs>0, cm_overall/rs, 0.0)

            fig_o, ax_o = plt.subplots(figsize=(5.5,4.5))
            annot_o = np.array([[f"{cm_overall[0,0]}\n{cm_overall_norm[0,0]:.2f}", f"{cm_overall[0,1]}\n{cm_overall_norm[0,1]:.2f}"],
                                [f"{cm_overall[1,0]}\n{cm_overall_norm[1,0]:.2f}", f"{cm_overall[1,1]}\n{cm_overall_norm[1,1]:.2f}"]])
            sns.heatmap(cm_overall_norm, ax=ax_o, annot=annot_o, fmt="", cmap="Purples", vmin=0.0, vmax=1.0,
                        cbar_kws={"label":"Row-Normalized"})
            ax_o.set_title(f"Overall Persistence (W={chunk_window}, K={K}, m={p_margin:.3f}, "
                        f"{'ANY' if anydim_mode else 'ALL'}-dim)")
            ax_o.set_xlabel("Predicted (persist)")
            ax_o.set_ylabel("True (persist)")
            ax_o.set_xticklabels(["Neg","Pos"])
            ax_o.set_yticklabels(["Neg","Pos"], rotation=0)
            fig_o.tight_layout()
            overall_persist_fig = os.path.join(chunk_persist_dir, "overall_persistence_confusion.png")
            fig_o.savefig(overall_persist_fig, dpi=160)
            plt.close(fig_o)

            # Dump a small JSON alongside other eval outputs
            persist_summary = {
                "window": int(chunk_window),
                "K_min_consecutive": int(K),
                "margin": float(p_margin),
                "anydim_mode": bool(anydim_mode),
                "overall": {
                    "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                    "sensitivity": sensitivity, "specificity": specificity,
                    "precision": precision, "npv": npv, "accuracy": accuracy,
                    "fig_path": overall_persist_fig,
                },
                "per_dim_cm_paths": per_dim_cm_paths,
                "combined_dim_fig": combined_pd_path,
                "note": "A chunk is labeled positive iff the threshold is exceeded for >= K consecutive timesteps inside the chunk.",
            }
            with open(os.path.join(chunk_persist_dir, "persistence_metrics.json"), "w") as fjson:
                json.dump(make_jsonable(persist_summary), fjson, indent=2)

            # ====== Persistence-based 5-class & 3-class labels + confusion matrices ======
            # We use inner boundary b0 (with persistence_margin) for "near" and outer boundary b1 (no margin) for "far".
            # Labeling rule per chunk/dim:
            #   0 far-  : run(K) over x <= -b1
            #   1 near- : run(K) over x <  -b0 + m   (and NOT far-)
            #   4 far+  : run(K) over x >= +b1
            #   3 near+ : run(K) over x >  +b0 - m   (and NOT far+)
            #   2 neutral: none of the above persist
            b0 = float(boundaries[0])
            b1 = float(boundaries[1])
            m  = float(p_margin)

            # Masks per sign and band for the whole chunk tensor [C,W,6]
            # Near (inner, sign-aware, margin applied)
            gt_near_neg = (G_ch <  (-b0))
            gt_near_pos = (G_ch >  ( b0))
            pr_near_neg = (P_ch <  (-b0 + m))
            pr_near_pos = (P_ch >  ( b0 - m))

            # Far (outer, sign-aware, no margin)
            gt_far_neg  = (G_ch <= (-b1))
            gt_far_pos  = (G_ch >= ( b1))
            pr_far_neg  = (P_ch <= (-b1 + m))
            pr_far_pos  = (P_ch >= ( b1 - m))

            # Build per-chunk, per-dim 5-class labels with persistence (K-run)
            def _label_5class_from_masks(near_neg_row, far_neg_row, near_pos_row, far_pos_row, K):
                # Priority: far- | near- | far+ | near+ | neutral
                if _has_run_of_true(far_neg_row, K):  # 0
                    return 0
                if _has_run_of_true(near_neg_row, K): # 1
                    return 1
                if _has_run_of_true(far_pos_row, K):  # 4
                    return 4
                if _has_run_of_true(near_pos_row, K): # 3
                    return 3
                return 2

            # Compute labels arrays [n_chunks, 6] for GT and Pred
            gt_labels5_persist = np.zeros((n_chunks, 6), dtype=np.int64)
            pr_labels5_persist = np.zeros((n_chunks, 6), dtype=np.int64)
            for c in range(n_chunks):
                for d in range(6):
                    gt_labels5_persist[c, d] = _label_5class_from_masks(
                        gt_near_neg[c, :, d], gt_far_neg[c, :, d],
                        gt_near_pos[c, :, d], gt_far_pos[c, :, d], K
                    )
                    pr_labels5_persist[c, d] = _label_5class_from_masks(
                        pr_near_neg[c, :, d], pr_far_neg[c, :, d],
                        pr_near_pos[c, :, d], pr_far_pos[c, :, d], K
                    )

            # ---- Per-dimension 5-class confusion matrices (persistence) ----
            class_names_5p = ["-2","-1","0","+1","+2"]
            fig_cm5p, axes_cm5p = plt.subplots(2, 3, figsize=(12, 7))
            axes_cm5p_f = axes_cm5p.flatten()
            cm5p_paths_per_dim = {}

            for d, lbl in enumerate(dim_labels):
                y_true5 = gt_labels5_persist[:, d]
                y_pred5 = pr_labels5_persist[:, d]
                cm5 = np.zeros((5,5), dtype=np.int64)
                for t, p in zip(y_true5, y_pred5):
                    cm5[t, p] += 1
                rs5 = cm5.sum(axis=1, keepdims=True)
                cm5n = np.where(rs5 > 0, cm5/rs5, 0.0)

                # combined grid cell
                ax = axes_cm5p_f[d]
                annot5 = np.array([[f"{cm5[i,j]}\n{cm5n[i,j]:.2f}" for j in range(5)] for i in range(5)])
                sns.heatmap(cm5n, ax=ax, annot=annot5, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
                ax.set_title(lbl)
                ax.set_xlabel("Pred 5-class (persist)")
                ax.set_ylabel("True 5-class (persist)")
                ax.set_xticklabels(class_names_5p)
                ax.set_yticklabels(class_names_5p, rotation=0)

                # save per-dim figure too
                fig_i, ax_i = plt.subplots(figsize=(5,4))
                sns.heatmap(cm5n, ax=ax_i, annot=annot5, fmt="", cmap="Blues", vmin=0.0, vmax=1.0,
                            cbar_kws={"label":"Row-Normalized"})
                ax_i.set_title(f"{lbl} - Persistence 5-class (K={K}, m={m:.3f})")
                ax_i.set_xlabel("Pred 5-class (persist)")
                ax_i.set_ylabel("True 5-class (persist)")
                ax_i.set_xticklabels(class_names_5p)
                ax_i.set_yticklabels(class_names_5p, rotation=0)
                fig_i.tight_layout()
                path5 = os.path.join(chunk_persist_dir, f"persist_cm5_dim_{lbl}.png")
                fig_i.savefig(path5, dpi=160)
                plt.close(fig_i)
                cm5p_paths_per_dim[lbl] = path5

            fig_cm5p.suptitle(f"Per-Dimension 5-Class (Persistence)  W={chunk_window}, K={K}, m={m:.3f}", fontsize=14)
            fig_cm5p.tight_layout(rect=[0,0,1,0.95])
            combined_dim_5class_persist_path = os.path.join(chunk_persist_dir, "combined_dim_5class_persistence_confusions.png")
            fig_cm5p.savefig(combined_dim_5class_persist_path, dpi=160)
            plt.close(fig_cm5p)

            # ---- Collapse to 3-class and produce per-dimension 3-class confusions ----
            # Map: [-2,-1,0,+1,+2] -> [Neg,Neu,Pos] via [-2,-1]->Neg, 0->Neu, [+1,+2]->Pos
            map3 = np.array([0, 0, 1, 2, 2], dtype=np.int64)
            class_names_3p = ["Neg","Neu","Pos"]

            fig_cm3p, axes_cm3p = plt.subplots(2, 3, figsize=(12, 7))
            axes_cm3p_f = axes_cm3p.flatten()
            cm3p_paths_per_dim = {}

            for d, lbl in enumerate(dim_labels):
                y_true3 = map3[gt_labels5_persist[:, d]]
                y_pred3 = map3[pr_labels5_persist[:, d]]

                cm3 = np.zeros((3,3), dtype=np.int64)
                for t, p in zip(y_true3, y_pred3):
                    cm3[t, p] += 1
                rs3 = cm3.sum(axis=1, keepdims=True)
                cm3n = np.where(rs3 > 0, cm3/rs3, 0.0)

                ax = axes_cm3p_f[d]
                annot3 = np.array([[f"{cm3[i,j]}\n{cm3n[i,j]:.2f}" for j in range(3)] for i in range(3)])
                sns.heatmap(cm3n, ax=ax, annot=annot3, fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
                ax.set_title(lbl)
                ax.set_xlabel("Pred 3-class (persist)")
                ax.set_ylabel("True 3-class (persist)")
                ax.set_xticklabels(class_names_3p)
                ax.set_yticklabels(class_names_3p, rotation=0)

                # save per-dim figure too
                fig_i, ax_i = plt.subplots(figsize=(5,4))
                sns.heatmap(cm3n, ax=ax_i, annot=annot3, fmt="", cmap="Blues", vmin=0.0, vmax=1.0,
                            cbar_kws={"label":"Row-Normalized"})
                ax_i.set_title(f"{lbl} - Persistence 3-class (K={K}, m={m:.3f})")
                ax_i.set_xlabel("Pred 3-class (persist)")
                ax_i.set_ylabel("True 3-class (persist)")
                ax_i.set_xticklabels(class_names_3p)
                ax_i.set_yticklabels(class_names_3p, rotation=0)
                fig_i.tight_layout()
                path3 = os.path.join(chunk_persist_dir, f"persist_cm3_dim_{lbl}.png")
                fig_i.savefig(path3, dpi=160)
                plt.close(fig_i)
                cm3p_paths_per_dim[lbl] = path3

            fig_cm3p.suptitle(f"Per-Dimension 3-Class (Persistence)  W={chunk_window}, K={K}, m={m:.3f}", fontsize=14)
            fig_cm3p.tight_layout(rect=[0,0,1,0.95])
            combined_dim_3class_persist_path = os.path.join(chunk_persist_dir, "combined_dim_3class_persistence_confusions.png")
            fig_cm3p.savefig(combined_dim_3class_persist_path, dpi=160)
            plt.close(fig_cm3p)

            # Attach paths to JSON summary (already created later in this block)
            # If you keep the JSON dump below, just extend 'persist_summary' before writing:
            # (Place these assignments right before json.dump(...))
            # persist_summary["per_dim_cm5_paths"] = cm5p_paths_per_dim
            # persist_summary["combined_dim_5class_fig"] = combined_dim_5class_persist_path
            # persist_summary["per_dim_cm3_paths"] = cm3p_paths_per_dim
            # persist_summary["combined_dim_3class_fig"] = combined_dim_3class_persist_path

            # ====== Aggregated (combined) 5-class & 3-class confusions for PERSISTENCE ======
            # Flatten across all dimensions to mirror the original "aggregated" matrices.
            # 5-class first
            G5p = gt_labels5_persist.reshape(-1)
            P5p = pr_labels5_persist.reshape(-1)
            cm5p_agg = np.zeros((5, 5), dtype=np.int64)
            for t, p in zip(G5p, P5p):
                cm5p_agg[t, p] += 1

            fig5p_agg, ax5p_agg = plt.subplots(figsize=(6, 5))
            im5p = ax5p_agg.imshow(cm5p_agg, cmap="Blues")
            ax5p_agg.set_title(f"Persistence Confusion (Aggregated 5-class)  W={chunk_window}, K={K}, m={m:.3f}")
            ax5p_agg.set_xlabel("Predicted")
            ax5p_agg.set_ylabel("True")
            ax5p_agg.set_xticks(range(5)); ax5p_agg.set_yticks(range(5))
            ax5p_agg.set_xticklabels(["-2","-1","0","+1","+2"])
            ax5p_agg.set_yticklabels(["-2","-1","0","+1","+2"])
            for i in range(5):
                for j in range(5):
                    ax5p_agg.text(j, i, str(cm5p_agg[i, j]), ha="center", va="center", fontsize=8, color="#222222")
            fig5p_agg.colorbar(im5p, ax=ax5p_agg, fraction=0.046, pad=0.04)
            fig5p_agg.tight_layout()
            aggregated_5class_persist_path = os.path.join(
                chunk_persist_dir, "aggregated_5class_persistence_confusion.png"
            )
            fig5p_agg.savefig(aggregated_5class_persist_path, dpi=150)
            plt.close(fig5p_agg)

            # 3-class: collapse 5-class via [-2,-1]->Neg(0), 0->Neu(1), [+1,+2]->Pos(2)
            map3 = np.array([0, 0, 1, 2, 2], dtype=np.int64)
            G3p = map3[G5p]
            P3p = map3[P5p]
            cm3p_agg = np.zeros((3, 3), dtype=np.int64)
            for t, p in zip(G3p, P3p):
                cm3p_agg[t, p] += 1

            fig3p_agg, ax3p_agg = plt.subplots(figsize=(6, 5))
            im3p = ax3p_agg.imshow(cm3p_agg, cmap="Blues")
            ax3p_agg.set_title(f"Persistence Confusion (Aggregated 3-class)  W={chunk_window}, K={K}, m={m:.3f}")
            ax3p_agg.set_xlabel("Predicted")
            ax3p_agg.set_ylabel("True")
            ax3p_agg.set_xticks(range(3)); ax3p_agg.set_yticks(range(3))
            class_names_3p = ["Neg","Neu","Pos"]
            ax3p_agg.set_xticklabels(class_names_3p)
            ax3p_agg.set_yticklabels(class_names_3p)
            for i in range(3):
                for j in range(3):
                    ax3p_agg.text(j, i, str(cm3p_agg[i, j]), ha="center", va="center", fontsize=8, color="#222222")
            fig3p_agg.colorbar(im3p, ax=ax3p_agg, fraction=0.046, pad=0.04)
            fig3p_agg.tight_layout()
            aggregated_3class_persist_path = os.path.join(
                chunk_persist_dir, "aggregated_3class_persistence_confusion.png"
            )
            fig3p_agg.savefig(aggregated_3class_persist_path, dpi=150)
            plt.close(fig3p_agg)

            # ---- Add these to the JSON summary before json.dump(...) ----
            # (Place the assignments below right before you write persist_summary to disk.)
            # persist_summary["aggregated_5class_fig"] = aggregated_5class_persist_path
            # persist_summary["aggregated_3class_fig"] = aggregated_3class_persist_path



        # Save chunk metrics JSON (including margin sweep)
        chunk_metrics.update({
            "mean_acc_5class": chunk_mean_acc5,
            "acc_5class_per_dim": {d: v for d, v in zip(dim_labels, chunk_acc5_per_dim)},
            "mean_acc_3class": chunk_mean_acc3,
            "acc_3class_per_dim": {d: v for d, v in zip(dim_labels, chunk_acc3_per_dim)},
            "cm5_paths_per_dim": cm5_paths_per_dim_chunk,
            "combined_dim_3class_confusions_path": combined_dim_3class_confusions_chunk_path,
            "combined_dim_accuracy_bars_path": combined_dim_accuracy_bars_chunk_path,
            # margin sweep summary (paths only)
            "by_margin": {
                "dir": chunk_margin_dir,
                # "margins": [float(m) for m in pr_margins],
                "margins": pr_margins.astype(float).ravel().tolist(), 
                "overall_confusion_paths_by_margin": chunk_cm_paths_by_margin,
                "combined_dim_confusions_by_margin": chunk_combined_dim_confusion_paths_by_margin,
                "combined_dim_3class_confusions_by_margin": chunk_combined_dim_3class_cm_paths_by_margin,
                "combined_dim_5class_confusions_by_margin": chunk_combined_dim_5class_cm_paths_by_margin,
                "aggregated_5class_confusions_by_margin": chunk_aggregated_5class_cm_paths_by_margin,
                "combined_dim_accuracy_bars_by_margin": chunk_combined_dim_accbar_paths_by_margin,
                "margin_semantics": "neutral region=(-b0+margin, b0-margin) with b0=boundaries[0]"
            },
            # NEW: overall limit detection metrics for this chunk evaluation
            "overall_limit_detection_margin_main": chunk_overall_limit_detection_margin_main,
            "overall_limit_detection_metrics_by_margin": chunk_overall_metrics_by_margin,
            "chunk_margin_predictions_csv": chunk_margin_predictions_csv,
            "chunk_filter_info": chunk_filter_info,
        })
        with open(os.path.join(chunk_eval_dir, "metrics_chunk.json"), "w") as fch:
            json.dump(make_jsonable(chunk_metrics), fch, indent=2)
    else:
        # No chunks available
        with open(os.path.join(chunk_eval_dir, "metrics_chunk.json"), "w") as fch:
            json.dump(make_jsonable(chunk_metrics), fch, indent=2)
        combined_dim_3class_confusions_chunk_path = None
        combined_dim_accuracy_bars_chunk_path = None

    #  8. Save metrics summary JSON
    metrics_summary = {
        "checkpoint": checkpoint_path,
        "csv": csv_path,
        "val_trials": sorted(list(val_trial_ids)),
        "regression": {
            "val_loss": val_loss,
            "mae_mean": mae_mean,
            "rmse_mean": rmse_mean,
            "mae_per_dim": {d: v for d, v in zip(["X","Y","Z","Roll","Pitch","Yaw"], mae_per_dim)},
            "rmse_per_dim": {d: v for d, v in zip(["X","Y","Z","Roll","Pitch","Yaw"], rmse_per_dim)},
        },
        "classification": {
            "boundaries": boundaries,
            "inner_boundary_b0": b0,
            "mean_acc_5class": mean_acc5,
            "acc_5class_per_dim": {d: v for d, v in zip(["X","Y","Z","Roll","Pitch","Yaw"], acc5_per_dim)},
            "mean_acc_3class": mean_acc3,
            "acc_3class_per_dim": {d: v for d, v in zip(["X","Y","Z","Roll","Pitch","Yaw"], acc3_per_dim)},
            "overall_limit_detection": overall_metrics,
            "confusion_matrix_5class_path": cm5_path,
            "per_dim_plots": per_dim_plot_paths,
            "pr_curve_overall_path": pr_curve_overall_path,
            "pr_points_overall_csv": pr_points_csv,
            "pr_curve_per_dim_paths": pr_paths_per_dim,
            "pr_points_per_dim_csv": pr_points_csv_per_dim,
            "combined_dim_accuracy_bars_path": combined_acc_path,
            "combined_dim_pr_curves_path": combined_pr_grid_path,
            "combined_dim_3class_confusions_path": combined_dim_3class_confusions_path,
            "combined_dim_3class_confusions_by_margin": combined_dim_3class_cm_paths_by_margin,
            "combined_dim_5class_confusions_by_margin": combined_dim_5class_cm_paths_by_margin,
            "aggregated_5class_confusions_by_margin": aggregated_5class_cm_paths_by_margin,
            "combined_dim_accuracy_bars_by_margin": combined_dim_accbar_paths_by_margin,
            # NEW: record filtering info for per-timestep evaluation
            "filter_info": filter_info,
        },
        # NEW: chunk-based classification summary (paths only; full details in chunk subdir JSON)
        "classification_chunk": {
            "chunk_window": chunk_window,
            "chunk_eval_dir": chunk_eval_dir,
            "combined_dim_3class_confusions_path": combined_dim_3class_confusions_chunk_path,
            "combined_dim_accuracy_bars_path": combined_dim_accuracy_bars_chunk_path,
            # margin sweep summary (paths only)
            "confusion_matrices_by_margin_dir": os.path.join(chunk_eval_dir, "confusion_matrices_by_margin"),
        },
    }
    json_path = os.path.join(eval_dir, "metrics_eval.json")
    with open(json_path, "w") as f:
        json.dump(make_jsonable(metrics_summary), f, indent=2)
    print(f"Saved metrics summary to {json_path}")

    # 9. Print concise summary
    print("--- Regression Metrics ---")
    print(f"val_loss={val_loss:.6f} mae_mean={mae_mean:.6f} rmse_mean={rmse_mean:.6f}")
    print("MAE per dim: " + ", ".join(f"{d}:{v:.4f}" for d, v in zip(["X","Y","Z","Roll","Pitch","Yaw"], mae_per_dim)))
    print("RMSE per dim: " + ", ".join(f"{d}:{v:.4f}" for d, v in zip(["X","Y","Z","Roll","Pitch","Yaw"], rmse_per_dim)))

    # NEW: Output and print a table of normalized MAE and RMSE per dimension (plus mean)
    dims_table = ["X","Y","Z","Roll","Pitch","Yaw"]
    df_err = pd.DataFrame({
        "Dim": dims_table,
        "MAE_norm": [float(x) for x in mae_per_dim],
        "RMSE_norm": [float(x) for x in rmse_per_dim],
    })
    df_err.loc[len(df_err)] = ["MEAN", float(mae_mean), float(rmse_mean)]
    # Save as CSV in eval directory
    err_table_csv = os.path.join(eval_dir, "regression_errors_normalized.csv")
    try:
        df_err.to_csv(err_table_csv, index=False)
    except Exception:
        pass
    print("--- Regression Errors (normalized) ---")
    print(df_err.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    if os.path.isfile(err_table_csv):
        print(f"Saved normalized regression error table CSV: {err_table_csv}")

    print("--- Classification (5-class) ---")
    print(f"mean_acc_5class={mean_acc5:.4f} per_dim=" + ", ".join(f"{d}:{v:.4f}" for d, v in zip(["X","Y","Z","Roll","Pitch","Yaw"], acc5_per_dim)))
    print("--- Classification (3-class) ---")
    print(f"mean_acc_3class={mean_acc3:.4f} per_dim=" + ", ".join(f"{d}:{v:.4f}" for d, v in zip(["X","Y","Z","Roll","Pitch","Yaw"], acc3_per_dim)))
    if filter_info["applied"]:
        print(f"Applied GT-boundary buffer filter: buffer={filter_info['buffer']} mode={filter_info['mode']} kept={filter_info['kept']} excluded={filter_info['excluded']}")
    print("--- Overall Limit Detection ---")
    print(
        "TP={TP} FP={FP} TN={TN} FN={FN} Sens={sensitivity:.3f} Spec={specificity:.3f} "
        "Prec={precision:.3f} NPV={npv:.3f} Acc={accuracy:.3f}".format(**overall_metrics)
    )
    print(f"5-class confusion matrix image: {cm5_path}")
    print(f"Overall limit detection confusion matrix image: {overall_metrics['fig_path']}")
    print(f"PR curve (overall dims appended) saved: {pr_curve_overall_path}")
    print(f"PR points CSV (overall) saved: {pr_points_csv}")
    print(f"Confusion matrices by margin saved in: {margin_cm_dir} ({len(cm_paths_by_margin)} files)")
    print(f"Combined per-dim confusion figures by margin saved in: {margin_cm_dir} ({len(combined_dim_confusion_paths_by_margin)} files)")
    print(f"Combined per-dim 3-class confusion figures by margin saved in: {margin_cm_dir} ({len(combined_dim_3class_cm_paths_by_margin)} files)")
    print(f"Combined per-dim 5-class confusion figures by margin saved in: {margin_cm_dir} ({len(combined_dim_5class_cm_paths_by_margin)} files)")
    print(f"Aggregated 5-class confusion figures by margin saved in: {margin_cm_dir} ({len(aggregated_5class_cm_paths_by_margin)} files)")
    print(f"Combined per-dim 3-class accuracy bar figures by margin saved in: {margin_cm_dir} ({len(combined_dim_accbar_paths_by_margin)} files)")
    print(f"Combined per-dim PR curves figure: {combined_pr_grid_path}")
    print(f"Combined per-dim 3-class confusion figure: {combined_dim_3class_confusions_path}")
    for lbl in dim_labels:
        print(f"PR curve (dim {lbl}) saved: {pr_paths_per_dim[lbl]}")
        print(f"PR points CSV (dim {lbl}) saved: {pr_points_csv_per_dim[lbl]}")
    predictions_path = EVAL_CONFIG.get("save_predictions")
    if predictions_path:
        print(f"Predictions saved to: {predictions_path}")
    if margin_main is not None:
        print(f"Margin main: {margin_main:.4f} => neutral region (-{b0:.3f}+{margin_main:.3f}, {b0:.3f}-{margin_main:.3f}) = ({-b0+margin_main:.3f},{b0-margin_main:.3f})")

    # Print chunk-based classification summary
    print("--- Chunk Classification ---")
    print(f"window={chunk_window} subdir={chunk_eval_dir}")
    if n_chunks > 0:
        print(f"chunk mean_acc_5class={chunk_mean_acc5:.4f} per_dim=" + ", ".join(f"{d}:{v:.4f}" for d, v in zip(dim_labels, chunk_acc5_per_dim)))
        print(f"chunk mean_acc_3class={chunk_mean_acc3:.4f} per_dim=" + ", ".join(f"{d}:{v:.4f}" for d, v in zip(dim_labels, chunk_acc3_per_dim)))
        print(f"chunk combined 3-class confusions: {combined_dim_3class_confusions_chunk_path}")
        print(f"chunk combined 3-class accuracy bars: {combined_dim_accuracy_bars_chunk_path}")
        # NEW: margin sweep prints for chunk-based eval
        print(f"chunk confusion matrices by margin saved in: {os.path.join(chunk_eval_dir, 'confusion_matrices_by_margin')}")
    else:
        print("No chunks available for the given window.")

if __name__ == "__main__":
    main()