#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a Transformer to REGRESS EACH TIMESTEP of a wrench+true-pose time-series chunk into a 6D continuous vector
(values are in columns: X_norm, Y_norm, Z_norm, A_norm, B_norm, C_norm) — i.e., output a value for
every timestep and each of the 6 dimensions.

Assumptions / Notes:
- CSV must contain 6 wrench columns, 6 absolute pose columns, and 6 per-timestep regression columns.
- Wrench columns: FX,FY,FZ,TX,TY,TZ
- Pose columns: X,Y,Z,A,B,C (absolute, used as-is; no per-chunk relative conversion)
- Target columns: X_norm,Y_norm,Z_norm,A_norm,B_norm,C_norm
- Normalization: per-file z-score for each wrench channel (fit on the whole file). Pose is not normalized, used as true/absolute.
- Split: first (1 - val-split) for train, last (val-split) for val, preserving time order.

Usage:
Configure the config dict below and run: python train_regression_true_pose.py

Requires: torch, pandas, numpy
"""
# Configuration
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
    "d_model": 128,
    "nhead": 4,
    "layers": 2,
    "ffn": 256,
    "dropout": 0.1,
    "use_cls": False,  # Per-timestep regression uses sequence outputs directly
    "seed": 42,
    # Save/load checkpoints, predictions, and artifacts in a separate directory for true-pose runs
    "out_dir": "./regression_transformer_model/checkpoints_true_pose",
    "save_every": 100,  # Save checkpoint every N epochs
    "wandb_project": "rcc_regression_transformer",
    "wandb_name": None,  # Auto-generated if None
    "continue_from": None,  # "best", "latest", or path to checkpoint file, or None to start fresh

    # Regression loss
    "loss": "huber",      # "huber" or "mse"
    "huber_delta": 1.0,

    # Classification-from-regression evaluation
    "classify_every": 5,           # run classification eval every N epochs
    "class_boundaries": [0.5, 0.75],  # must match raw_data_processing.py
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
    """Return mean, std for z-score normalization along axis=0."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return mu, sd

def zscore_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd

# Using true (absolute) pose directly; no relative conversion per chunk is applied in this script.

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
        """
        Use rows in [start, end) to create chunks.
        Returns per-timestep REGRESSION targets for each chunk.
        Input features are concatenated: [normalized wrench (6), true pose (6)] -> 12 dims per timestep.
        """
        assert len(wrench_cols) == 6, "Need exactly 6 wrench columns."
        assert len(pose_cols) == 6, "Need exactly 6 pose columns."
        assert len(label_cols) == 6, "Need exactly 6 label columns."
        self.wrench = df[wrench_cols].values.astype(np.float32)
        self.pose = df[pose_cols].values.astype(np.float32)
        self.labels = df[label_cols].values.astype(np.float32)
        # Normalize wrench only
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
        if not self.cfg.drop_last and (len(idxs) == 0 or idxs[-1] != self.end - self.cfg.window):
            pass
        return idxs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s = self.indices[i]
        e = s + self.cfg.window
        wrench_chunk = self.wrench[s:e, :]              # (T, 6)
        pose_chunk = self.pose[s:e, :]                  # (T, 6) absolute (true pose)
        # Use true (absolute) pose directly
        x = np.concatenate([wrench_chunk, pose_chunk], axis=1).astype(np.float32)  # (T, 12)
        targets_Tx6 = self.labels[s:e, :].astype(np.float32)  # (T, 6)
        return torch.from_numpy(x), torch.from_numpy(targets_Tx6)

# ----------------------------
# Model
# ----------------------------

class WrenchTransformer(nn.Module):
    def __init__(
        self,
        input_dim=6,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        num_tasks=6,     # 6 dimensions (X,Y,Z,A,B,C)
        use_cls=True,
        max_len=10000
    ):
        super().__init__()
        self.use_cls = use_cls
        self.input_proj = nn.Linear(input_dim, d_model)

        # positional encoding (learned)
        self.pos_embed = nn.Embedding(max_len + (1 if use_cls else 0), d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None

        # 6 separate regression heads -> output 1 value per timestep per task
        self.heads = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        ) for _ in range(num_tasks)])

        self._reset_parameters()

    def _reset_parameters(self):
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x):
        """
        x: (B, T, input_dim=12)
        returns: predictions tensor (B, T, 6)
        """
        B, T, _ = x.shape
        h = self.input_proj(x)  # (B, T, d_model)

        # positions
        device = x.device
        pos_ids = torch.arange(T, device=device).unsqueeze(0)  # (1,T)
        pos = self.pos_embed(pos_ids)  # (1, T, d_model)

        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,d_model)
            pos_cls = self.pos_embed(torch.tensor([T], device=device)).unsqueeze(0).expand(B, 1, -1)
            enc = self.encoder(torch.cat([cls + pos_cls, h + pos], dim=1))  # (B, T+1, d_model)
            feats = enc[:, 1:, :]  # (B, T, d_model)
        else:
            enc = self.encoder(h + pos)
            feats = enc  # (B, T, d_model)

        # per-timestep regression outputs for each task, concatenate to (B, T, 6)
        outs = [head(feats) for head in self.heads]  # each (B, T, 1)
        y = torch.cat(outs, dim=-1)  # (B, T, 6)
        return y

# ----------------------------
# Training / Evaluation (Regression)
# ----------------------------

def make_loss_fn(cfg):
    if cfg.get("loss", "huber").lower() == "huber":
        # Prefer Huber for robustness
        if hasattr(nn, "HuberLoss"):
            return nn.HuberLoss(delta=cfg.get("huber_delta", 1.0), reduction="mean")
        # Fallback for older torch
        return nn.SmoothL1Loss(beta=cfg.get("huber_delta", 1.0), reduction="mean")
    else:
        return nn.MSELoss(reduction="mean")


def train_one_epoch(model, loader, opt, device, loss_fn):
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)            # (B, T, 6)
        yb = yb.to(device)            # (B, T, 6) float

        preds = model(xb)             # (B, T, 6)
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
    # accumulate for metrics
    mae_sum = torch.zeros(6, device=device)
    mse_sum = torch.zeros(6, device=device)
    count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        preds = model(xb)  # (B, T, 6)
        loss = loss_fn(preds, yb)
        total += float(loss.item())
        n += 1

        diff = preds - yb
        B, T, D = diff.shape
        mae_sum += diff.abs().sum(dim=(0, 1))  # (6)
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
    """
    Evaluate model and save per-timestep predictions to CSV file.
    CSV columns: chunk, t, gt_*_norm, pred_*_norm
    """
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

        # Store predictions and ground truth for this batch (flatten B,T)
        all_predictions.append(preds.detach().cpu().numpy().reshape(-1, 6))
        all_ground_truth.append(yb.detach().cpu().numpy().reshape(-1, 6))

        # chunk and timestep ids
        chunk_ids = np.repeat(np.arange(chunk_offset, chunk_offset + B), T)
        t_ids = np.tile(np.arange(T), B)
        all_chunk_ids.append(chunk_ids)
        all_t_ids.append(t_ids)
        chunk_offset += B

        total += float(loss.item())
        n += 1

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)  # (total_samples, 6)
    all_ground_truth = np.concatenate(all_ground_truth, axis=0)  # (total_samples, 6)
    all_chunk_ids = np.concatenate(all_chunk_ids, axis=0)
    all_t_ids = np.concatenate(all_t_ids, axis=0)

    # Create DataFrame and save to CSV
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
    """Vectorized mapping of normalized values to 5 bins with sign, matching raw_data_processing.py.
    boundaries: (b0, b1) corresponds to 0.5 and 0.75 in raw_data_processing.py
    Returns an array of same shape (int64) with classes in {0,1,2,3,4}.
    """
    v = values
    classes = np.full(v.shape, 2, dtype=np.int64)
    b0, b1 = boundaries
    # Negative side
    classes[v <= -b1] = 0
    classes[(v > -b1) & (v <= -b0)] = 1
    # Positive side
    classes[(v >= b0) & (v < b1)] = 3
    classes[v >= b1] = 4
    return classes

@torch.no_grad()
def evaluate_classification(model, loader, device, out_dir: str, boundaries=(0.5, 0.75)):
    """Use regression outputs on val loader to derive classes and compute metrics and confusion matrix.
    Saves a confusion matrix image to out_dir/confusion_matrix.png (overwrites each time).
    Returns (mean_acc, acc_per_dim(list of 6), cm (5x5 np.array), fig_path)
    """
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
        return 0.0, [0.0]*6, np.zeros((5,5), dtype=np.int64), os.path.join(out_dir, "confusion_matrix.png")

    preds = np.concatenate(preds_list, axis=0)  # (B, T, 6)
    gts = np.concatenate(gts_list, axis=0)      # (B, T, 6)

    P = preds.reshape(-1, 6)
    G = gts.reshape(-1, 6)

    acc_per_dim = []
    for d in range(6):
        pred_c = map_norm_to_class(P[:, d], boundaries)
        gt_c = map_norm_to_class(G[:, d], boundaries)
        acc = float((pred_c == gt_c).mean())
        acc_per_dim.append(acc)
    mean_acc = float(np.mean(acc_per_dim))

    # Confusion matrix aggregated across dims
    pred_all = map_norm_to_class(P, boundaries).reshape(-1)
    gt_all = map_norm_to_class(G, boundaries).reshape(-1)
    num_classes = 5
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # rows=true, cols=pred
    for t, p in zip(gt_all, pred_all):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1

    # Plot and save (overwrite)
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
    """Find checkpoint file based on type: 'best', 'latest', or specific path (regression naming)."""
    if checkpoint_type == "best":
        best_path = os.path.join(out_dir, "best_model_regression.pt")
        if os.path.exists(best_path):
            return best_path
        else:
            print(f"Best model not found at {best_path}")
            return None
    elif checkpoint_type == "latest":
        # Find the latest checkpoint by epoch number
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
        else:
            print(f"No checkpoint files found in {out_dir}")
            return None
    elif os.path.isfile(checkpoint_type):
        return checkpoint_type
    else:
        print(f"Checkpoint not found: {checkpoint_type}")
        return None

def load_checkpoint(checkpoint_path: str, model, optimizer=None):
    """Load checkpoint and return epoch, best_val info."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model"])

    start_epoch = 1
    best_val = float("inf")

    if "epoch" in checkpoint:
        start_epoch = checkpoint["epoch"] + 1
    if "best_val" in checkpoint:
        best_val = checkpoint["best_val"]
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Loaded optimizer state")

    print(f"Resuming from epoch {start_epoch}, best val loss: {best_val:.4f}")
    return start_epoch, best_val

# ----------------------------
# Main
# ----------------------------

def main():
    config = CONFIG
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"]) 

    # Create output directory
    os.makedirs(config["out_dir"], exist_ok=True)

    # Initialize wandb
    wandb_name = config["wandb_name"] or f"rcc_transformer_regression_truepose_{config['window']}w_{config['stride']}s"
    wandb.init(
        project=config["wandb_project"],
        name=wandb_name,
        config=config
    )

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
            f"Insufficient data after chunking. "
            f"Train chunks: {len(train_ds)}, Val chunks: {len(val_ds)}. "
            f"Try reducing window or val_split."
        )

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WrenchTransformer(
        input_dim=12, d_model=config["d_model"], nhead=config["nhead"], num_layers=config["layers"],
        dim_feedforward=config["ffn"], dropout=config["dropout"], num_tasks=6,
        use_cls=config["use_cls"], max_len=config["window"] + 1
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
    print(f"Input features: 12 (6 wrench + 6 true pose)")

    for epoch in range(start_epoch, config["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, device, loss_fn)

        vl_loss, mae_per_dim, rmse_per_dim, mae_mean, rmse_mean = evaluate(model, val_loader, device, loss_fn)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": vl_loss,
            "mae_mean": mae_mean,
            "rmse_mean": rmse_mean,
            **{f"mae_{label}": val for label, val in zip(["X","Y","Z","A","B","C"], mae_per_dim)},
            **{f"rmse_{label}": val for label, val in zip(["X","Y","Z","A","B","C"], rmse_per_dim)},
        })

        # Periodic classification-from-regression on validation set
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

        # Periodic checkpoint saving and validation predictions
        if epoch % config["save_every"] == 0:
            checkpoint_path = os.path.join(config["out_dir"], f"regression_checkpoint_epoch_{epoch}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "config": config,
                "best_val": best_val
            }, checkpoint_path)

            # Save validation predictions
            predictions_path = os.path.join(config["out_dir"], f"val_predictions_regression_epoch_{epoch}.csv")
            evaluate_and_save_predictions(model, val_loader, device, predictions_path)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.6f} | val_loss={vl_loss:.6f} | "
            f"mae_dim={[f'{a:.4f}' for a in mae_per_dim]} | rmse_dim={[f'{a:.4f}' for a in rmse_per_dim]} | "
            f"mae_mean={mae_mean:.4f} rmse_mean={rmse_mean:.4f} | "
            f"{'** saved **' if is_best else ''}"
        )

if __name__ == "__main__":
    main()
