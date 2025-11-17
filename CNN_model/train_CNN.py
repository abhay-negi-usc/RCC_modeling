#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a **Time-Series CNN (Temporal ConvNet/TCN)** to REGRESS EACH TIMESTEP of a
wrench+relative-pose time-series chunk into a 6D continuous vector
(X_norm, Y_norm, Z_norm, A_norm, B_norm, C_norm) — i.e., output a value for
every timestep and each of the 6 dimensions.

This file is a drop-in replacement for the original Transformer script: it keeps
identical data loading, chunking, normalization, logging, checkpoints, and
metrics — only the model is changed to a TCN built from dilated causal/bidirectional
conv1d residual blocks.

Usage:
  python train_regression_tscnn.py

Requires: torch, pandas, numpy, wandb
"""

# ----------------------------
# Configuration
# ----------------------------
# CNN architecture hyperparameters (how they affect the model):
# - cnn_hidden: number of feature channels inside the TCN. Larger -> higher capacity, more VRAM/compute.
# - cnn_layers: number of residual blocks stacked. With dilation, this controls the receptive field (RF).
#   RF approx (for dilation_base > 1): 1 + (kernel-1) * (dilation_base^layers - 1) / (dilation_base - 1).
#   Example: kernel=7, layers=6, dilation_base=2 -> RF ≈ 1 + 6 * (2^6 - 1) = 1 + 6*63 = 379 timesteps.
# - cnn_kernel: temporal kernel size per conv. Must be odd here to preserve SAME length with symmetric padding.
#   Bigger kernels widen local context but increase parameters; combine with dilation to grow RF efficiently.
# - cnn_dropout: dropout applied inside each residual block for regularization (typical 0.0–0.3).
# - cnn_dilation_base: growth factor for dilations across layers (1 disables dilation; 2 doubles each layer: 1,2,4,...).
# - cnn_bidirectional: when True, uses symmetric (SAME) padding so each timestep can see past and future context
#   (non-causal). A purely causal variant would use left-only padding; this implementation uses SAME padding.
CONFIG = {
    # "csv": "./data/RCC_combined_14_processed.csv",
    "csv": "./data/RCC_kuka_ati_15_trials.csv",
    # "wrench_cols": ["FX", "FY", "FZ", "TX", "TY", "TZ"],
    "wrench_cols": ["Fx_ati", "Fy_ati", "Fz_ati", "Tx_ati", "Ty_ati", "Tz_ati"],
    "pose_cols": ["X", "Y", "Z", "A", "B", "C"],
    "label_cols": ["X_norm", "Y_norm", "Z_norm", "A_norm", "B_norm", "C_norm"],
    "trial_col": "trial",  # must exist in CSV; used to prevent cross-trial chunks
    "window": 16384,
    "stride": 16384,
    "batch_size": 64,
    "epochs": 1_000_000,
    "lr": 1e-3,
    "val_split": 0.2,
    "seed": 42,
    # NEW: enable per-epoch resampling of random training dataset
    "resample_train_each_epoch": True,

    # CNN architecture (replaces Transformer params)
    "cnn_hidden": 36,            # base channels per residual block
    "cnn_layers": 8,              # number of residual blocks (controls depth and receptive field)
    "cnn_kernel": 11,              # odd kernel for SAME-length convolution
    "cnn_dropout": 0.1,           # dropout inside blocks for regularization
    "cnn_dilation_base": 16,       # dilations grow as base^layer: 1,2,4,... to expand receptive field
    "cnn_bidirectional": True,    # SAME padding (non-causal); uses future context in addition to past

    # Random-chunk training
    # Number of randomly sampled, fixed-length train chunks per epoch. If None, defaults to the
    # number of stride-based windows that fit within each trial in the training split (for similar epoch size).
    "train_samples_per_epoch": 1_000,

    # "out_dir": "/media/rp/Elements1/abhay_ws/RCC_modeling/CNN_model/checkpoints",
    "out_dir": "./CNN_model/checkpoints_v8",
    "save_every": 100,
    "wandb_project": "rcc_regression_cnn",
    "wandb_name": None,
    # "continue_from": "/media/rp/Elements1/abhay_ws/RCC_modeling/CNN_model/checkpoints_v2/best_model_regression (copy).pt",        # "best", "latest", path, or None
    "continue_from": None,        # "best", "latest", path, or None

    # Regression loss
    "loss": "huber",      # "huber" or "mse"
    "huber_delta": 1.0,

    # Classification-from-regression evaluation
    "classify_every": 10,
    "class_boundaries": [0.5, 0.75],
}

import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import wandb
import itertools  # ADDED
import json       # ADDED

# ----------------------------
# Utilities
# ----------------------------

def zscore_fit(x: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return mu, sd


def zscore_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


def compute_relative_pose(pose_chunk: np.ndarray) -> np.ndarray:
    """Convert absolute pose to relative pose per chunk: first pose becomes zero reference."""
    if len(pose_chunk) == 0:
        return pose_chunk
    reference_pose = pose_chunk[0:1, :]
    relative_pose = pose_chunk - reference_pose
    return relative_pose.astype(np.float32)


def compute_trial_segments(trials: np.ndarray, start: int, end: int) -> List[Tuple[int, int]]:
    """Return contiguous [s,e) segments between start and end indices where `trial` value is constant."""
    segments: List[Tuple[int, int]] = []
    i = max(0, start)
    end = min(len(trials), end)
    if i >= end:
        return segments
    while i < end:
        tval = trials[i]
        j = i + 1
        while j < end and trials[j] == tval:
            j += 1
        segments.append((i, j))
        i = j
    return segments


def filter_segments_by_trials(segments: List[Tuple[int, int]], trials: np.ndarray, allowed: Optional[Set]) -> List[Tuple[int, int]]:
    """Keep only segments whose trial id (at start index) is in `allowed` if provided."""
    if not allowed:
        return segments
    return [(s, e) for (s, e) in segments if trials[s] in allowed]


def count_stride_windows_in_segments(segments: List[Tuple[int, int]], window: int, stride: int) -> int:
    total = 0
    for s, e in segments:
        seg_len = e - s
        if seg_len >= window:
            total += max(0, (seg_len - window) // stride + 1)
    return total

# ----------------------------
# Dataset
# ----------------------------

@dataclass
class ChunkingConfig:
    window: int
    stride: int
    drop_last: bool = True


class WrenchPoseChunkDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        wrench_cols: List[str],
        pose_cols: List[str],
        label_cols: List[str],
        cfg: ChunkingConfig,
        wrench_norm_mu: np.ndarray,
        wrench_norm_sd: np.ndarray,
        start: int,
        end: int,
        trial_col: str,
        allowed_trials: Optional[Set] = None,
    ):
        assert len(wrench_cols) == 6, "Need exactly 6 wrench columns."
        assert len(pose_cols) == 6, "Need exactly 6 pose columns."
        assert len(label_cols) == 6, "Need exactly 6 label columns."
        self.wrench = df[wrench_cols].values.astype(np.float32)
        self.pose = df[pose_cols].values.astype(np.float32)
        self.labels = df[label_cols].values.astype(np.float32)
        self.trials = df[trial_col].values
        # Normalize wrench only
        self.wrench = zscore_apply(self.wrench, wrench_norm_mu, wrench_norm_sd).astype(np.float32)

        self.cfg = cfg
        self.start = max(0, start)
        self.end = min(len(df), end)
        all_segments = compute_trial_segments(self.trials, self.start, self.end)
        self.segments = filter_segments_by_trials(all_segments, self.trials, allowed_trials)
        self.indices = self._make_indices()

    def _make_indices(self) -> List[int]:
        # Build stride-based window starts within each trial segment; do not cross trial boundaries
        idxs: List[int] = []
        for s, e in self.segments:
            if e - s >= self.cfg.window:
                idxs.extend(range(s, e - self.cfg.window + 1, self.cfg.stride))
        return idxs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s = self.indices[i]
        e = s + self.cfg.window
        wrench_chunk = self.wrench[s:e, :]              # (T, 6)
        pose_chunk = self.pose[s:e, :]                  # (T, 6) absolute
        rel_pose_chunk = compute_relative_pose(pose_chunk)  # (T, 6)
        x = np.concatenate([wrench_chunk, rel_pose_chunk], axis=1).astype(np.float32)  # (T, 12)
        targets_Tx6 = self.labels[s:e, :].astype(np.float32)  # (T, 6)
        return torch.from_numpy(x), torch.from_numpy(targets_Tx6)


class RandomWrenchPoseChunkDataset(Dataset):
    """Randomly sample fixed-length chunks that never cross trial boundaries.

    Length of the dataset controls how many random samples per epoch are drawn. Each __getitem__
    independently samples a trial segment, then a random valid start within that segment.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        wrench_cols: List[str],
        pose_cols: List[str],
        label_cols: List[str],
        window: int,
        wrench_norm_mu: np.ndarray,
        wrench_norm_sd: np.ndarray,
        start: int,
        end: int,
        trial_col: str,
        samples_per_epoch: int,
        rng: np.random.Generator | None = None,
        allowed_trials: Optional[Set] = None,
    ):
        assert len(wrench_cols) == 6 and len(pose_cols) == 6 and len(label_cols) == 6
        self.wrench = df[wrench_cols].values.astype(np.float32)
        self.pose = df[pose_cols].values.astype(np.float32)
        self.labels = df[label_cols].values.astype(np.float32)
        self.trials = df[trial_col].values
        self.wrench = zscore_apply(self.wrench, wrench_norm_mu, wrench_norm_sd).astype(np.float32)

        self.window = window
        self.start = max(0, start)
        self.end = min(len(df), end)
        all_segments = compute_trial_segments(self.trials, self.start, self.end)
        all_segments = filter_segments_by_trials(all_segments, self.trials, allowed_trials)
        # Keep only segments that can fit at least one window
        self.valid_segments = [(s, e) for (s, e) in all_segments if (e - s) >= self.window]
        self.starts_per_segment = np.array([(e - s) - self.window + 1 for (s, e) in self.valid_segments], dtype=np.int64)
        self.total_starts = int(self.starts_per_segment.sum())
        if self.total_starts <= 0:
            self.samples_per_epoch = 0
        else:
            self.samples_per_epoch = int(samples_per_epoch)
        self.probs = (self.starts_per_segment / self.total_starts) if self.total_starts > 0 else None
        self.rng = rng if rng is not None else np.random.default_rng()

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):  # idx is ignored; we draw randomly per call
        # Sample a segment proportional to number of valid starts inside it
        seg_idx = int(self.rng.choice(len(self.valid_segments), p=self.probs))
        s, e = self.valid_segments[seg_idx]
        max_offset = (e - s) - self.window + 1
        start_idx = int(s + self.rng.integers(0, max_offset))
        end_idx = start_idx + self.window

        wrench_chunk = self.wrench[start_idx:end_idx, :]
        pose_chunk = self.pose[start_idx:end_idx, :]
        rel_pose_chunk = compute_relative_pose(pose_chunk)
        x = np.concatenate([wrench_chunk, rel_pose_chunk], axis=1).astype(np.float32)
        targets_Tx6 = self.labels[start_idx:end_idx, :].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(targets_Tx6)

# ----------------------------
# Time-Series CNN (TCN)
# ----------------------------

class ResidualTCNBlock(nn.Module):
    # Each block applies two dilated Conv1d layers with GELU, dropout, and a residual connection.
    # Padding is symmetric (SAME) so the temporal length is preserved; with dilation>1 this yields
    # a non-causal receptive field that includes both left and right context around each timestep.
    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1, dropout=0.1, bidirectional=True):
        super().__init__()
        # SAME-length padding for conv1d: pad = floor((k-1)*d/2)
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.act1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.act2 = nn.GELU()
        self.do2 = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(out_ch)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.bidirectional = bidirectional

    def forward(self, x):  # x: [B, C, T]
        y = self.conv1(x)
        y = self.act1(y)
        y = self.do1(y)
        y = self.conv2(y)
        y = self.act2(y)
        y = self.do2(y)
        y = self.norm(y)
        return y + self.down(x)


class TimeSeriesCNN(nn.Module):
    def __init__(self, input_dim=12, hidden=128, layers=6, kernel_size=7, dropout=0.1, dilation_base=2, bidirectional=True, num_tasks=6):
        super().__init__()
        chans = [hidden] * layers
        dilations = [dilation_base ** i for i in range(layers)]

        self.in_proj = nn.Conv1d(input_dim, hidden, kernel_size=1)

        blocks = []
        in_ch = hidden
        for i, (ch, d) in enumerate(zip(chans, dilations)):
            blocks.append(ResidualTCNBlock(in_ch, ch, kernel_size=kernel_size, dilation=d, dropout=dropout, bidirectional=bidirectional))
            in_ch = ch
        self.tcn = nn.Sequential(*blocks)

        self.head_norm = nn.LayerNorm(hidden)
        self.head = nn.Conv1d(hidden, num_tasks, kernel_size=1)

    def forward(self, x):  # x: [B, T, 12]
        # Conv1d expects [B, C, T]
        x = x.transpose(1, 2)  # [B, 12, T]
        h = self.in_proj(x)    # [B, H, T]
        h = self.tcn(h)        # [B, H, T]
        # LayerNorm over channel: move to [B, T, H]
        h_ln = h.transpose(1, 2)  # [B, T, H]
        h_ln = self.head_norm(h_ln)
        h = h_ln.transpose(1, 2)  # [B, H, T]
        y = self.head(h)          # [B, 6, T]
        y = y.transpose(1, 2)     # [B, T, 6]
        return y

# ----------------------------
# Training / Evaluation (Regression)
# ----------------------------

def make_loss_fn(cfg):
    if cfg.get("loss", "huber").lower() == "huber":
        if hasattr(nn, "HuberLoss"):
            return nn.HuberLoss(delta=cfg.get("huber_delta", 1.0), reduction="mean")
        return nn.SmoothL1Loss(beta=cfg.get("huber_delta", 1.0), reduction="mean")
    else:
        return nn.MSELoss(reduction="mean")


def train_one_epoch(model, loader, opt, device, loss_fn):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total = 0.0
    n = 0
    mae_sum = torch.zeros(6, device=device)
    mse_sum = torch.zeros(6, device=device)
    count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        total += float(loss.item())
        n += 1
        diff = preds - yb
        B, T, D = diff.shape
        mae_sum += diff.abs().sum(dim=(0, 1))
        mse_sum += (diff ** 2).sum(dim=(0, 1))
        count += B * T

    avg_loss = total / max(n, 1)
    mae_per_dim = (mae_sum / max(count, 1)).detach().cpu().tolist()
    rmse_per_dim = (torch.sqrt(mse_sum / max(count, 1))).detach().cpu().tolist()
    mae_mean = float(np.mean(mae_per_dim))
    rmse_mean = float(np.mean(rmse_per_dim))
    return avg_loss, mae_per_dim, rmse_per_dim, mae_mean, rmse_mean


@torch.no_grad()
def evaluate_and_save_predictions(model, loader, device, output_path):
    model.eval()
    loss_fn = nn.MSELoss(reduction="mean")
    total = 0.0
    n = 0

    all_predictions = []
    all_ground_truth = []
    all_chunk_ids = []
    all_t_ids = []
    chunk_offset = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        B, T, _ = yb.shape
        preds = model(xb)
        loss = loss_fn(preds, yb)

        all_predictions.append(preds.detach().cpu().numpy().reshape(-1, 6))
        all_ground_truth.append(yb.detach().cpu().numpy().reshape(-1, 6))
        chunk_ids = np.repeat(np.arange(chunk_offset, chunk_offset + B), T)
        t_ids = np.tile(np.arange(T), B)
        all_chunk_ids.append(chunk_ids)
        all_t_ids.append(t_ids)
        chunk_offset += B

        total += float(loss.item())
        n += 1

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)
    all_chunk_ids = np.concatenate(all_chunk_ids, axis=0)
    all_t_ids = np.concatenate(all_t_ids, axis=0)

    df_results = pd.DataFrame({
        "chunk": all_chunk_ids,
        "t": all_t_ids,
    })
    dim_names = ["X", "Y", "Z", "A", "B", "C"]
    for i, dim_name in enumerate(dim_names):
        df_results[f"gt_{dim_name}_norm"] = all_ground_truth[:, i]
        df_results[f"pred_{dim_name}_norm"] = all_predictions[:, i]

    df_results.to_csv(output_path, index=False)
    print(f"Saved validation predictions to {output_path}")

    avg_loss = total / max(n, 1)
    return avg_loss

# ----------------------------
# Classification from regression utilities
# ----------------------------

def map_norm_to_class(values: np.ndarray, boundaries=(0.5, 0.75)) -> np.ndarray:
    v = values
    classes = np.full(v.shape, 2, dtype=np.int64)
    b0, b1 = boundaries
    classes[v <= -b1] = 0
    classes[(v > -b1) & (v <= -b0)] = 1
    classes[(v >= b0) & (v < b1)] = 3
    classes[v >= b1] = 4
    return classes

# New helper to collapse 5-class labels to 3-class
def collapse_to_3class(c5: np.ndarray) -> np.ndarray:
    return np.where(c5 == 2, 1, np.where(np.isin(c5, [3, 4]), 2, 0))


@torch.no_grad()
def evaluate_classification(model, loader, device, out_dir: str, boundaries=(0.5, 0.75), margin=0.0):
    boundaries_with_margin = (boundaries[0] - margin, boundaries[1] - margin)
    model.eval()
    preds_list = []
    gts_list = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        preds_list.append(pred.detach().cpu().numpy())
        gts_list.append(yb.detach().cpu().numpy())

    if not preds_list:
        empty_cm5_path = os.path.join(out_dir, "confusion_matrix.png")
        # Added: empty overall detection metrics structure
        empty_overall_fig_path = os.path.join(out_dir, "overall_limit_confusion_matrix.png")
        return 0.0, [0.0]*6, np.zeros((5, 5), dtype=np.int64), empty_cm5_path, 0.0, [0.0]*6, {}, {
            "TP": 0, "FP": 0, "TN": 0, "FN": 0,
            "sensitivity": 0.0, "specificity": 0.0, "precision": 0.0, "npv": 0.0, "accuracy": 0.0,
            "fig_path": empty_overall_fig_path
        }

    preds = np.concatenate(preds_list, axis=0)
    gts = np.concatenate(gts_list, axis=0)

    P = preds.reshape(-1, 6)
    G = gts.reshape(-1, 6)

    acc_per_dim = []
    acc3_per_dim = []
    dim_labels = ["X", "Y", "Z", "A", "B", "C"]
    per_dim_plot_paths = {lbl: {} for lbl in dim_labels}

    for d, lbl in enumerate(dim_labels):
        pred_c5 = map_norm_to_class(P[:, d], boundaries_with_margin)
        gt_c5 = map_norm_to_class(G[:, d], boundaries)
        acc = float((pred_c5 == gt_c5).mean())
        acc_per_dim.append(acc)
        pred_c3 = collapse_to_3class(pred_c5)
        gt_c3 = collapse_to_3class(gt_c5)
        acc3 = float((pred_c3 == gt_c3).mean())
        acc3_per_dim.append(acc3)
        num_c3 = 3
        cm3 = np.zeros((num_c3, num_c3), dtype=np.int64)
        for t, p in zip(gt_c3, pred_c3):
            cm3[t, p] += 1

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 3-class confusion matrix heatmap
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        im = ax_cm.imshow(cm3, cmap="Blues")
        ax_cm.set_title(f"3-Class Confusion ({lbl})")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        ax_cm.set_xticks(range(num_c3), ["Neg", "Neu", "Pos"])
        ax_cm.set_yticks(range(num_c3), ["Neg", "Neu", "Pos"])
        for i in range(num_c3):
            for j in range(num_c3):
                ax_cm.text(j, i, cm3[i, j], ha="center", va="center", color="black", fontsize=8)
        fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
        fig_cm.tight_layout()
        cm3_path = os.path.join(out_dir, f"confusion_matrix_3class_dim_{lbl}.png")
        fig_cm.savefig(cm3_path, dpi=150)
        plt.close(fig_cm)
        per_dim_plot_paths[lbl]["cm3"] = cm3_path

        # FP / FN bar chart
        fp = np.zeros(num_c3, dtype=np.int64)
        fn = np.zeros(num_c3, dtype=np.int64)
        for k in range(num_c3):
            tp_k = cm3[k, k]
            fn[k] = cm3[k, :].sum() - tp_k
            fp[k] = cm3[:, k].sum() - tp_k
        x = np.arange(num_c3)
        width = 0.35
        fig_b, ax_b = plt.subplots(figsize=(5, 4))
        ax_b.bar(x - width/2, fp, width, label="FP", color="#e57373")
        ax_b.bar(x + width/2, fn, width, label="FN", color="#64b5f6")
        ax_b.set_title(f"3-Class FP/FN ({lbl})")
        ax_b.set_xticks(x, ["Neg", "Neu", "Pos"])
        ax_b.set_ylabel("Count")
        ax_b.legend()
        fig_b.tight_layout()
        fpfn_path = os.path.join(out_dir, f"fp_fn_bar_3class_dim_{lbl}.png")
        fig_b.savefig(fpfn_path, dpi=150)
        plt.close(fig_b)
        per_dim_plot_paths[lbl]["fpfn"] = fpfn_path

        # Per-class accuracy bar chart
        per_class_den = cm3.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_acc = np.where(per_class_den > 0, np.diag(cm3) / per_class_den, 0.0)
        fig_a, ax_a = plt.subplots(figsize=(5, 4))
        ax_a.bar(["Neg", "Neu", "Pos"], per_class_acc, color="#81c784")
        ax_a.set_ylim(0, 1)
        ax_a.set_title(f"3-Class Per-Class Acc ({lbl})")
        ax_a.set_ylabel("Accuracy")
        fig_a.tight_layout()
        accbar_path = os.path.join(out_dir, f"per_class_acc_3class_dim_{lbl}.png")
        fig_a.savefig(accbar_path, dpi=150)
        plt.close(fig_a)
        per_dim_plot_paths[lbl]["accbar"] = accbar_path

    mean_acc = float(np.mean(acc_per_dim))
    mean_acc3 = float(np.mean(acc3_per_dim))

    # Aggregated 5-class confusion matrix (unchanged logic)
    pred_all = map_norm_to_class(P, boundaries_with_margin).reshape(-1)
    gt_all = map_norm_to_class(G, boundaries).reshape(-1)
    num_classes = 5
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(gt_all, pred_all):
        cm[t, p] += 1

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Validation Confusion Matrix (Aggregated)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ----------------------------
    # NEW: Overall limit detection confusion matrix & metrics
    # Positive iff ANY dimension is non-neutral (class != 2). Negative iff ALL are neutral.
    # ----------------------------
    pred_c5_matrix = map_norm_to_class(P, boundaries_with_margin)  # shape [N,6]
    gt_c5_matrix = map_norm_to_class(G, boundaries)
    gt_pos = (gt_c5_matrix != 2).any(axis=1)
    pred_pos = (pred_c5_matrix != 2).any(axis=1)

    TP = int(np.sum(pred_pos & gt_pos))
    TN = int(np.sum(~pred_pos & ~gt_pos))
    FP = int(np.sum(pred_pos & ~gt_pos))
    FN = int(np.sum(~pred_pos & gt_pos))

    def safe_div(a, b):
        return float(a / b) if b > 0 else 0.0

    sensitivity = safe_div(TP, TP + FN)        # recall
    specificity = safe_div(TN, TN + FP)
    precision = safe_div(TP, TP + FP)          # PPV
    npv = safe_div(TN, TN + FN)                # NPV
    accuracy = safe_div(TP + TN, TP + TN + FP + FN)

    overall_cm = np.array([[TN, FP], [FN, TP]], dtype=np.int64)  # rows=true (Neg,Pos) cols=pred (Neg,Pos)

    # Normalize confusion matrix by true class (row) sums
    row_sums = overall_cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        overall_cm_norm = np.where(row_sums > 0, overall_cm / row_sums, 0.0)

    fig_overall, ax_overall = plt.subplots(figsize=(5,4))
    im2 = ax_overall.imshow(overall_cm_norm, cmap="Purples", vmin=0.0, vmax=1.0)
    ax_overall.set_title("Overall Limit Detection Confusion Matrix (Row-Normalized)")
    ax_overall.set_xlabel("Predicted")
    ax_overall.set_ylabel("True")
    ax_overall.set_xticks([0,1],["Neg","Pos"])
    ax_overall.set_yticks([0,1],["Neg","Pos"])
    for i in range(2):
        for j in range(2):
            ax_overall.text(j, i, f"{overall_cm[i, j]}\n{overall_cm_norm[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)
    fig_overall.colorbar(im2, ax=ax_overall, fraction=0.046, pad=0.04, label="Row-Normalized Value")
    # Add metrics text box
    metrics_str = (f"Sensitivity: {sensitivity:.3f}\nSpecificity: {specificity:.3f}\n"\
                   f"Precision: {precision:.3f}\nNPV: {npv:.3f}\nAccuracy: {accuracy:.3f}")
    ax_overall.text(1.05, 0.5, metrics_str, transform=ax_overall.transAxes, va="center", ha="left",
                    fontsize=9, bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    fig_overall.tight_layout()
    overall_fig_path = os.path.join(out_dir, "overall_limit_confusion_matrix.png")
    fig_overall.savefig(overall_fig_path, dpi=150)
    plt.close(fig_overall)

    overall_metrics = {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "accuracy": accuracy,
        "fig_path": overall_fig_path,
    }

    return mean_acc, acc_per_dim, cm, fig_path, mean_acc3, acc3_per_dim, per_dim_plot_paths, overall_metrics
# ----------------------------
# Checkpoint utilities
# ----------------------------

def find_checkpoint(out_dir: str, checkpoint_type: str) -> str:
    if checkpoint_type == "best":
        best_path = os.path.join(out_dir, "best_model_regression.pt")
        return best_path if os.path.exists(best_path) else None
    elif checkpoint_type == "latest":
        checkpoint_files = []
        for file in os.listdir(out_dir):
            if file.startswith("regression_checkpoint_epoch_") and file.endswith(".pt"):
                try:
                    epoch_num = int(file.split("_")[-1].split(".")[0])
                    checkpoint_files.append((epoch_num, os.path.join(out_dir, file)))
                except ValueError:
                    continue
        if checkpoint_files:
            latest_epoch, latest_path = max(checkpoint_files, key=lambda x: x[0])
            return latest_path
        return None
    elif os.path.isfile(checkpoint_type):
        return checkpoint_type
    else:
        return None


def load_checkpoint(checkpoint_path: str, model, optimizer=None):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"]) 
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_val = checkpoint.get("best_val", float("inf"))
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"]) 
    print(f"Resuming from epoch {start_epoch}, best val loss: {best_val:.4f}")
    return start_epoch, best_val

# ----------------------------
# Main
# ----------------------------

def main():
    config = CONFIG
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    os.makedirs(config["out_dir"], exist_ok=True)

    wandb_name = config["wandb_name"] or f"rcc_tscnn_regression_{config['window']}w_{config['stride']}s"
    wandb.init(project=config["wandb_project"], name=wandb_name, config=config)

    # Load CSV
    df = pd.read_csv(config["csv"]) 
    for c in config["wrench_cols"] + config["pose_cols"] + config["label_cols"]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV. Available: {list(df.columns)}")
    if config.get("trial_col") not in df.columns:
        raise ValueError(f"Trial column '{config.get('trial_col')}' not found in CSV. Columns: {list(df.columns)}")

    # Fit normalization on the whole file (wrench channels only)
    wrench_all = df[config["wrench_cols"]].values.astype(np.float32)
    mu, sd = zscore_fit(wrench_all)

    # Trial-aware split by trials, not by contiguous rows
    trial_col = config["trial_col"]
    trials_arr = df[trial_col].values
    unique_trials = pd.unique(trials_arr)  # preserves order of appearance

    # ------------------------------
    # NEW: Trial subset selection enforcing:
    # 1. Validation consists of whole trials only.
    # 2. Validation ratio within 10% relative of target (val_split).
    # 3. Train/Val per-dimension MIN and MAX are approximately equal (within tolerance) when normalized by global range.
    # ------------------------------
    target_val_ratio = float(config["val_split"])
    ratio_tol = 0.10  # 10%
    minmax_tol = 0.05  # 5%

    # Precompute per-trial indices
    trial_to_indices: dict[str, np.ndarray] = {}
    for t in unique_trials:
        trial_to_indices[str(t)] = np.where(trials_arr == t)[0]

    total_len = len(df)

    # Precompute label arrays for efficiency
    label_values = df[config["label_cols"]].values.astype(np.float32)
    # Global stats used to normalize min/max differences (stable even if values near 0)
    global_mins = label_values.min(axis=0)
    global_maxs = label_values.max(axis=0)
    global_ranges = np.maximum(global_maxs - global_mins, 1e-8)

    def compute_ranges(indices: np.ndarray):
        subset = label_values[indices]
        mins = subset.min(axis=0)
        maxs = subset.max(axis=0)
        ranges = maxs - mins
        return mins, maxs, ranges

    best_subset = None
    best_ratio_diff = float("inf")

    trial_list = [str(t) for t in unique_trials]
    n_trials = len(trial_list)

    # Enumerate all subsets (except empty/full) – feasible (e.g. 2^15=32768)
    for r in range(1, n_trials):
        for combo in itertools.combinations(trial_list, r):
            val_indices = np.concatenate([trial_to_indices[t] for t in combo])
            val_len = len(val_indices)
            val_ratio = val_len / total_len
            # Check ratio tolerance first
            if abs(val_ratio - target_val_ratio) / max(target_val_ratio, 1e-8) > ratio_tol:
                continue
            train_trials = [t for t in trial_list if t not in combo]
            train_indices = np.concatenate([trial_to_indices[t] for t in train_trials])
            # Compute min/max
            train_mins, train_maxs, _ = compute_ranges(train_indices)
            val_mins, val_maxs, _ = compute_ranges(val_indices)
            # Relative min/max difference normalized by global range
            rel_min_diff = np.abs(train_mins - val_mins) / global_ranges
            rel_max_diff = np.abs(train_maxs - val_maxs) / global_ranges
            if np.any(rel_min_diff > minmax_tol) or np.any(rel_max_diff > minmax_tol):
                continue
            # Candidate acceptable
            ratio_diff = abs(val_ratio - target_val_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_subset = {
                    "val_trials": set(combo),
                    "train_trials": set(train_trials),
                    "val_indices": val_indices,
                    "train_indices": train_indices,
                    "val_ratio": val_ratio,
                    "train_mins": train_mins.tolist(),
                    "train_maxs": train_maxs.tolist(),
                    "val_mins": val_mins.tolist(),
                    "val_maxs": val_maxs.tolist(),
                    "rel_min_diff": rel_min_diff.tolist(),
                    "rel_max_diff": rel_max_diff.tolist(),
                }
        if best_ratio_diff == 0:
            break

    if best_subset is None:
        # Fallback: ratio only (ignore min/max), choose deterministic permutation like original method.
        rng = np.random.default_rng(config["seed"])  # deterministic
        perm = rng.permutation(len(unique_trials))
        n_val_trials = max(1, int(math.ceil(len(unique_trials) * target_val_ratio)))
        val_trial_ids = set(unique_trials[perm[:n_val_trials]])
        train_trial_ids = set(unique_trials) - val_trial_ids
        print("WARNING: Could not find trial subset satisfying ratio + min/max constraints; using ratio-only split.")
        # Compute min/max and relative diffs for reporting
        train_indices_fb = np.concatenate([trial_to_indices[str(t)] for t in train_trial_ids])
        val_indices_fb = np.concatenate([trial_to_indices[str(t)] for t in val_trial_ids])
        train_mins, train_maxs, _ = compute_ranges(train_indices_fb)
        val_mins, val_maxs, _ = compute_ranges(val_indices_fb)
        rel_min_diff_fb = np.abs(train_mins - val_mins) / global_ranges
        rel_max_diff_fb = np.abs(train_maxs - val_maxs) / global_ranges
        split_meta = {
            "target_val_ratio": target_val_ratio,
            "actual_val_ratio": float(len(val_indices_fb) / total_len),
            "val_trials": sorted(list(val_trial_ids)),
            "train_trials": sorted(list(train_trial_ids)),
            "label_minmax": {
                "global": {col: {"min": float(global_mins[i]), "max": float(global_maxs[i]), "range": float(global_ranges[i])}
                            for i, col in enumerate(config["label_cols"])},
                "train": {col: {"min": float(train_mins[i]), "max": float(train_maxs[i])}
                           for i, col in enumerate(config["label_cols"])},
                "val": {col: {"min": float(val_mins[i]), "max": float(val_maxs[i])}
                         for i, col in enumerate(config["label_cols"])},
                "relative_min_diff": {col: float(rel_min_diff_fb[i]) for i, col in enumerate(config["label_cols"])},
                "relative_max_diff": {col: float(rel_max_diff_fb[i]) for i, col in enumerate(config["label_cols"])},
            },
            "constraints_met": False
        }
    else:
        val_trial_ids = set(best_subset["val_trials"])
        train_trial_ids = set(best_subset["train_trials"])
        split_meta = {
            "target_val_ratio": target_val_ratio,
            "actual_val_ratio": best_subset["val_ratio"],
            "val_trials": sorted(list(val_trial_ids)),
            "train_trials": sorted(list(train_trial_ids)),
            "label_minmax": {
                "global": {col: {"min": float(global_mins[i]), "max": float(global_maxs[i]), "range": float(global_ranges[i])}
                            for i, col in enumerate(config["label_cols"])},
                "train": {col: {"min": float(best_subset["train_mins"][i]), "max": float(best_subset["train_maxs"][i])}
                           for i, col in enumerate(config["label_cols"])},
                "val": {col: {"min": float(best_subset["val_mins"][i]), "max": float(best_subset["val_maxs"][i])}
                         for i, col in enumerate(config["label_cols"])},
                "relative_min_diff": {col: float(best_subset["rel_min_diff"][i]) for i, col in enumerate(config["label_cols"])},
                "relative_max_diff": {col: float(best_subset["rel_max_diff"][i]) for i, col in enumerate(config["label_cols"])},
            },
            "constraints_met": True
        }

    # Persist split metadata for reproducibility (used by eval script)
    split_path = os.path.join(config["out_dir"], "data_split.json")
    try:
        with open(split_path, "w") as f:
            json.dump(split_meta, f, indent=2)
        print(f"Saved data split metadata to {split_path}")
        # NEW: write trial-level split labels CSV
        trial_split_csv = os.path.join(config["out_dir"], "trial_split_labels.csv")
        import csv as _csv
        with open(trial_split_csv, "w", newline="") as fcsv:
            writer = _csv.writer(fcsv)
            writer.writerow(["trial", "split"])  # header
            for t in sorted(list(train_trial_ids)):
                writer.writerow([t, "train"])
            for t in sorted(list(val_trial_ids)):
                writer.writerow([t, "val"])
        print(f"Saved trial split labels to {trial_split_csv}")
    except Exception as e:
        print(f"Failed to save data split metadata or trial labels: {e}")

    print(f"Selected val trials ({len(val_trial_ids)}): {sorted(list(val_trial_ids))}")
    print(f"Selected train trials ({len(train_trial_ids)}): {sorted(list(train_trial_ids))}")
    print(f"Actual val ratio: {split_meta['actual_val_ratio']:.4f} (target={target_val_ratio:.4f}) | constraints_met={split_meta['constraints_met']}")
    for i, col in enumerate(config["label_cols"]):
        tmin = split_meta["label_minmax"]["train"][col]["min"]
        tmax = split_meta["label_minmax"]["train"][col]["max"]
        vmin = split_meta["label_minmax"]["val"][col]["min"]
        vmax = split_meta["label_minmax"]["val"][col]["max"]
        rmin = split_meta["label_minmax"]["relative_min_diff"][col]
        rmax = split_meta["label_minmax"]["relative_max_diff"][col]
        print(f"{col}: train[min,max]=({tmin:.4f},{tmax:.4f}) | val[min,max]=({vmin:.4f},{vmax:.4f}) | rel_min_diff={rmin:.4f} rel_max_diff={rmax:.4f}")

    # Optionally log to wandb once
    wandb.log({
        "data_split_actual_val_ratio": split_meta['actual_val_ratio'],
        **{f"data_split_rel_min_diff_{col}": split_meta['label_minmax']['relative_min_diff'][col] for col in config['label_cols']},
        **{f"data_split_rel_max_diff_{col}": split_meta['label_minmax']['relative_max_diff'][col] for col in config['label_cols']},
        "data_split_constraints_met": split_meta['constraints_met']
    })

    # NEW: build segments now that trials are split
    N = len(df)
    all_segments = compute_trial_segments(trials_arr, 0, N)
    train_segments = filter_segments_by_trials(all_segments, trials_arr, train_trial_ids)
    val_segments = filter_segments_by_trials(all_segments, trials_arr, val_trial_ids)

    cfg = ChunkingConfig(window=config["window"], stride=config["stride"], drop_last=True)

    # Determine number of train samples per epoch (default to stride-based count within train trials)
    default_train_samples = count_stride_windows_in_segments(train_segments, window=config["window"], stride=config["stride"])
    samples_per_epoch = config.get("train_samples_per_epoch") or default_train_samples

    # Datasets
    def build_random_train_ds(epoch_seed_offset: int = 0):
        # Different RNG seed each epoch => different random sampling distribution of chunk starts.
        return RandomWrenchPoseChunkDataset(
            df,
            config["wrench_cols"],
            config["pose_cols"],
            config["label_cols"],
            window=config["window"],
            wrench_norm_mu=mu,
            wrench_norm_sd=sd,
            start=0,
            end=N,
            trial_col=trial_col,
            samples_per_epoch=int(samples_per_epoch),
            rng=np.random.default_rng(config["seed"] + epoch_seed_offset),
            allowed_trials=train_trial_ids,
        )

    train_ds = build_random_train_ds(epoch_seed_offset=0)
    val_ds = WrenchPoseChunkDataset(
        df,
        config["wrench_cols"],
        config["pose_cols"],
        config["label_cols"],
        cfg,
        mu,
        sd,
        start=0,
        end=N,
        trial_col=trial_col,
        allowed_trials=val_trial_ids,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"Insufficient data after chunking. Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}. Try reducing window or val_split."
        )

    # For random dataset, shuffling isn't necessary; each __getitem__ draws a fresh random chunk.
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesCNN(
        input_dim=12,
        hidden=config["cnn_hidden"],
        layers=config["cnn_layers"],
        kernel_size=config["cnn_kernel"],
        dropout=config["cnn_dropout"],
        dilation_base=config["cnn_dilation_base"],
        bidirectional=config["cnn_bidirectional"],
        num_tasks=6,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    loss_fn = make_loss_fn(config)

    # Load checkpoint if specified
    start_epoch = 1
    best_val = float("inf")
    if config["continue_from"]:
        checkpoint_path = find_checkpoint(config["out_dir"], config["continue_from"])
        if checkpoint_path:
            start_epoch, best_val = load_checkpoint(checkpoint_path, model, opt)
        else:
            print("Starting training from scratch")

    print(
        f"Train trials: {len(train_trial_ids)} | Val trials: {len(val_trial_ids)} | "
        f"Train chunks/epoch: {len(train_ds)} | Val chunks: {len(val_ds)} | Device: {device}"
    )
    print(f"Input features: 12 (6 wrench + 6 relative pose) | window={config['window']} (fixed), random-chunk training enabled")

    for epoch in range(start_epoch, config["epochs"] + 1):
        # Optionally rebuild (resample) random training dataset each epoch with fresh RNG seed.
        if config.get("resample_train_each_epoch", False) and epoch > start_epoch:
            train_ds = build_random_train_ds(epoch_seed_offset=epoch)
            train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False, drop_last=True)
        tr_loss = train_one_epoch(model, train_loader, opt, device, loss_fn)
        vl_loss, mae_per_dim, rmse_per_dim, mae_mean, rmse_mean = evaluate(model, val_loader, device, loss_fn)

        wandb.log({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": vl_loss,
            "mae_mean": mae_mean,
            "rmse_mean": rmse_mean,
            **{f"mae_{label}": val for label, val in zip(["X","Y","Z","A","B","C"], mae_per_dim)},
            **{f"rmse_{label}": val for label, val in zip(["X","Y","Z","A","B","C"], rmse_per_dim)},
        })

        if (epoch % int(config.get("classify_every", 5))) == 0:
            mean_acc, acc_per_dim, cm, fig_path, mean_acc3, acc3_per_dim, per_dim_plot_paths, overall_metrics = evaluate_classification(
                model, val_loader, device, config["out_dir"], tuple(config.get("class_boundaries", [0.5, 0.75]))
            )
            print(f"Val classification mean accuracy: {mean_acc:.4f} | per-dim={[f'{a:.4f}' for a in acc_per_dim]} | 3-class_mean_acc={mean_acc3:.4f}")
            print(f"Overall limit detection: TP={overall_metrics['TP']} FP={overall_metrics['FP']} TN={overall_metrics['TN']} FN={overall_metrics['FN']} | "
                  f"Sens={overall_metrics['sensitivity']:.3f} Spec={overall_metrics['specificity']:.3f} Prec={overall_metrics['precision']:.3f} NPV={overall_metrics['npv']:.3f} Acc={overall_metrics['accuracy']:.3f}")
            log_payload = {
                "val_class_mean_acc": mean_acc,
                **{f"val_class_acc_{label}": val for label, val in zip(["X","Y","Z","A","B","C"], acc_per_dim)},
                "confusion_matrix_img": wandb.Image(fig_path),
                "classification_accuracy_3class": mean_acc3,
                **{f"val_class_acc_3class_{label}": val for label, val in zip(["X","Y","Z","A","B","C"], acc3_per_dim)},
                # Overall limit detection metrics
                "overall_limit_confusion_matrix": wandb.Image(overall_metrics["fig_path"]),
                "overall_limit_TP": overall_metrics["TP"],
                "overall_limit_FP": overall_metrics["FP"],
                "overall_limit_TN": overall_metrics["TN"],
                "overall_limit_FN": overall_metrics["FN"],
                "overall_limit_sensitivity": overall_metrics["sensitivity"],
                "overall_limit_specificity": overall_metrics["specificity"],
                "overall_limit_precision": overall_metrics["precision"],
                "overall_limit_NPV": overall_metrics["npv"],
                "overall_limit_accuracy": overall_metrics["accuracy"],
            }
            for lbl, paths in per_dim_plot_paths.items():
                if "cm3" in paths:
                    log_payload[f"val_3class_confusion_{lbl}"] = wandb.Image(paths["cm3"])
                if "fpfn" in paths:
                    log_payload[f"val_3class_fpfn_{lbl}"] = wandb.Image(paths["fpfn"])
                if "accbar" in paths:
                    log_payload[f"val_3class_accbar_{lbl}"] = wandb.Image(paths["accbar"])
            wandb.log(log_payload)

        is_best = vl_loss < best_val
        if is_best:
            best_val = vl_loss
            best_path = os.path.join(config["out_dir"], "best_model_regression.pt")
            torch.save({"model": model.state_dict(), "config": config}, best_path)

        if epoch % config["save_every"] == 0:
            checkpoint_path = os.path.join(config["out_dir"], f"regression_checkpoint_epoch_{epoch}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "config": config,
                "best_val": best_val
            }, checkpoint_path)

            predictions_path = os.path.join(config["out_dir"], f"val_predictions_regression_epoch_{epoch}.csv")
            evaluate_and_save_predictions(model, val_loader, device, predictions_path)

        print(
            f"Epoch {epoch:02d} | train_loss={tr_loss:.6f} | val_loss={vl_loss:.6f} | "
            f"mae_dim={[f'{a:.4f}' for a in mae_per_dim]} | rmse_dim={[f'{a:.4f}' for a in rmse_per_dim]} | "
            f"mae_mean={mae_mean:.4f} rmse_mean={rmse_mean:.4f} | {'** saved **' if is_best else ''}"
        )


if __name__ == "__main__":
    main()