#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a **Recurrent Neural Network (RNN)** (GRU/LSTM) to REGRESS EACH TIMESTEP of a
wrench+relative-pose time-series chunk into a 6D continuous vector
(X_norm, Y_norm, Z_norm, A_norm, B_norm, C_norm) — i.e., output a value for
every timestep and each of the 6 dimensions.

This is a drop-in replacement for the Transformer script. It preserves your data
loading, chunking, normalization, logging, checkpoints, and metrics — only the
model is swapped to an RNN with optional bidirectionality.

Usage:
  python train_regression_rnn.py

Requires: torch, pandas, numpy, wandb
"""

# ----------------------------
# Configuration
# ----------------------------
CONFIG = {
    "csv": "./data/RCC_combined_14_processed.csv",
    "wrench_cols": ["FX", "FY", "FZ", "TX", "TY", "TZ"],
    "pose_cols": ["X", "Y", "Z", "A", "B", "C"],
    "label_cols": ["X_norm", "Y_norm", "Z_norm", "A_norm", "B_norm", "C_norm"],
    "window": 512,
    "stride": 256,
    "batch_size": 64,
    "epochs": 1_000_000,
    "lr": 1e-4,
    "val_split": 0.2,
    "seed": 42,

    # RNN architecture
    "rnn_type": "gru",          # "gru" or "lstm"
    "rnn_hidden": 256,
    "rnn_layers": 3,
    "rnn_bidirectional": True,   # bidirectional for offline accuracy
    "rnn_dropout": 0.1,          # dropout between RNN layers
    "pre_cnn_kernel": 5,         # optional short 1D conv frontend (set to 0 to disable)

    "out_dir": "./regression_rnn_model/checkpoints",
    "save_every": 100,
    "wandb_project": "rcc_regression_rnn",
    "wandb_name": None,
    "continue_from": None,        # "best", "latest", path, or None

    # Regression loss
    "loss": "huber",      # "huber" or "mse"
    "huber_delta": 1.0,

    # Classification-from-regression evaluation (optional)
    "classify_every": 5,
    "class_boundaries": [0.5, 0.75],
}

import math
import os
from dataclasses import dataclass
from typing import List, Tuple

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
    ):
        assert len(wrench_cols) == 6, "Need exactly 6 wrench columns."
        assert len(pose_cols) == 6, "Need exactly 6 pose columns."
        assert len(label_cols) == 6, "Need exactly 6 label columns."
        self.wrench = df[wrench_cols].values.astype(np.float32)
        self.pose = df[pose_cols].values.astype(np.float32)
        self.labels = df[label_cols].values.astype(np.float32)
        # Normalize wrench only (labels are already normalized in [-1,1])
        self.wrench = zscore_apply(self.wrench, wrench_norm_mu, wrench_norm_sd).astype(np.float32)

        self.cfg = cfg
        self.start = max(0, start)
        self.end = min(len(df), end)
        self.indices = self._make_indices()

    def _make_indices(self) -> List[int]:
        T = self.end - self.start
        if T < self.cfg.window:
            return []
        idxs = list(range(self.start, self.end - self.cfg.window + 1, self.cfg.stride))
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

# ----------------------------
# Model: RNN (GRU/LSTM)
# ----------------------------

class PreConv(nn.Module):
    """Optional short-term conv front-end to capture local motifs before the RNN."""
    def __init__(self, in_dim: int, kernel: int, hidden: int):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Conv1d(in_dim, hidden, kernel_size=kernel, padding=pad)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm1d(hidden)
    def forward(self, x):  # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]
        y = self.conv(x)
        y = self.act(y)
        y = self.norm(y)
        return y.transpose(1, 2)  # [B, T, H]


class TimeSeriesRNN(nn.Module):
    def __init__(self, input_dim=12, hidden=256, layers=3, bidirectional=True, dropout=0.1, rnn_type="gru", pre_cnn_kernel=5, out_dim=6):
        super().__init__()
        self.use_pre = pre_cnn_kernel and pre_cnn_kernel > 0
        pre_hidden = hidden if self.use_pre else input_dim
        self.pre = PreConv(input_dim, pre_cnn_kernel, hidden) if self.use_pre else nn.Identity()

        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type.lower()]
        self.rnn = rnn_cls(input_size=pre_hidden, hidden_size=hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
        rnn_out = hidden * (2 if bidirectional else 1)
        self.head_norm = nn.LayerNorm(rnn_out)
        self.head = nn.Linear(rnn_out, out_dim)

    def forward(self, x):  # x: [B, T, 12]
        h = self.pre(x)            # [B, T, pre_hidden]
        y, _ = self.rnn(h)         # [B, T, rnn_out]
        y = self.head_norm(y)
        y = self.head(y)           # [B, T, 6]
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
# Classification from regression utilities (optional)
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

    wandb_name = config["wandb_name"] or f"rcc_rnn_regression_{config['window']}w_{config['stride']}s"
    wandb.init(project=config["wandb_project"], name=wandb_name, config=config)

    # Load CSV
    df = pd.read_csv(config["csv"]) 
    for c in config["wrench_cols"] + config["pose_cols"] + config["label_cols"]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV. Available: {list(df.columns)}")

    # Fit normalization on the whole file (wrench channels only)
    wrench_all = df[config["wrench_cols"]].values.astype(np.float32)
    mu, sd = zscore_fit(wrench_all)

    # Time-preserving split indices
    N = len(df)
    val_len = int(math.floor(N * config["val_split"]))
    train_end = max(0, N - val_len)
    train_start = 0
    val_start = train_end
    val_end = N

    cfg = ChunkingConfig(window=config["window"], stride=config["stride"], drop_last=True)

    # Datasets
    train_ds = WrenchPoseChunkDataset(
        df, config["wrench_cols"], config["pose_cols"], config["label_cols"], cfg, mu, sd, start=train_start, end=train_end
    )
    val_ds = WrenchPoseChunkDataset(
        df, config["wrench_cols"], config["pose_cols"], config["label_cols"], cfg, mu, sd, start=val_start, end=val_end
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            f"Insufficient data after chunking. Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}. Try reducing window or val_split."
        )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesRNN(
        input_dim=12,
        hidden=config["rnn_hidden"],
        layers=config["rnn_layers"],
        bidirectional=config["rnn_bidirectional"],
        dropout=config["rnn_dropout"],
        rnn_type=config["rnn_type"],
        pre_cnn_kernel=config.get("pre_cnn_kernel", 5),
        out_dim=6,
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

    print(f"Train chunks: {len(train_ds)} | Val chunks: {len(val_ds)} | Device: {device}")
    print(f"Input features: 12 (6 wrench + 6 relative pose)")

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
