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
    "window": 8192,
    "stride": 8192,
    "batch_size": 64,
    "epochs": 1_000_000,
    "lr": 1e-4,
    "val_split": 0.2,
    "seed": 42,

    # CNN architecture (replaces Transformer params)
    "cnn_hidden": 64,            # base channels per residual block
    "cnn_layers": 16,              # number of residual blocks (controls depth and receptive field)
    "cnn_kernel": 11,              # odd kernel for SAME-length convolution
    "cnn_dropout": 0.1,           # dropout inside blocks for regularization
    "cnn_dilation_base": 2,       # dilations grow as base^layer: 1,2,4,... to expand receptive field
    "cnn_bidirectional": False,    # SAME padding (non-causal); uses future context in addition to past

    # Random-chunk training
    # Number of randomly sampled, fixed-length train chunks per epoch. If None, defaults to the
    # number of stride-based windows that fit within each trial in the training split (for similar epoch size).
    "train_samples_per_epoch": 100_000,

    "out_dir": "/home/nom4d/RCC_modeling/CNN_model/checkpoints/",
    "save_every": 100,
    "wandb_project": "rcc_regression_cnn",
    "wandb_name": None,
    "continue_from": "best",        # "best", "latest", path, or None

    # Regression loss
    "loss": "huber",      # "huber" or "mse"
    "huber_delta": 1.0,

    # Classification-from-regression evaluation
    "classify_every": 5,
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


@torch.no_grad()
def evaluate_classification(model, loader, device, out_dir: str, boundaries=(0.5, 0.75)):
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
        import numpy as _np
        return 0.0, [0.0]*6, _np.zeros((5,5), dtype=_np.int64), os.path.join(out_dir, "confusion_matrix.png")

    preds = np.concatenate(preds_list, axis=0)
    gts = np.concatenate(gts_list, axis=0)

    P = preds.reshape(-1, 6)
    G = gts.reshape(-1, 6)

    acc_per_dim = []
    for d in range(6):
        pred_c = map_norm_to_class(P[:, d], boundaries)
        gt_c = map_norm_to_class(G[:, d], boundaries)
        acc = float((pred_c == gt_c).mean())
        acc_per_dim.append(acc)
    mean_acc = float(np.mean(acc_per_dim))

    pred_all = map_norm_to_class(P, boundaries).reshape(-1)
    gt_all = map_norm_to_class(G, boundaries).reshape(-1)
    num_classes = 5
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(gt_all, pred_all):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1

    import matplotlib
    matplotlib.use("Agg")
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
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)

    return mean_acc, acc_per_dim, cm, fig_path

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
    rng = np.random.default_rng(config["seed"])  # deterministic
    perm = rng.permutation(len(unique_trials))
    n_val_trials = max(1, int(math.ceil(len(unique_trials) * config["val_split"])))
    val_trial_ids = set(unique_trials[perm[:n_val_trials]])
    train_trial_ids = set(unique_trials) - val_trial_ids

    # Build segments once, then filter per split
    N = len(df)
    all_segments = compute_trial_segments(trials_arr, 0, N)
    train_segments = filter_segments_by_trials(all_segments, trials_arr, train_trial_ids)
    val_segments = filter_segments_by_trials(all_segments, trials_arr, val_trial_ids)

    cfg = ChunkingConfig(window=config["window"], stride=config["stride"], drop_last=True)

    # Determine number of train samples per epoch (default to stride-based count within train trials)
    default_train_samples = count_stride_windows_in_segments(train_segments, window=config["window"], stride=config["stride"])
    samples_per_epoch = config.get("train_samples_per_epoch") or default_train_samples

    # Datasets
    train_ds = RandomWrenchPoseChunkDataset(
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
        rng=np.random.default_rng(config["seed"]),
        allowed_trials=train_trial_ids,
    )
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
            mean_acc, acc_per_dim, cm, fig_path = evaluate_classification(
                model, val_loader, device, config["out_dir"], tuple(config.get("class_boundaries", [0.5, 0.75]))
            )
            print(f"Val classification mean accuracy: {mean_acc:.4f} | per-dim={[f'{a:.4f}' for a in acc_per_dim]}")
            wandb.log({
                "val_class_mean_acc": mean_acc,
                **{f"val_class_acc_{label}": val for label, val in zip(["X","Y","Z","A","B","C"], acc_per_dim)},
                "confusion_matrix_img": wandb.Image(fig_path)
            })

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