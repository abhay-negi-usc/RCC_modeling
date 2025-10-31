#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a Transformer to classify EACH TIMESTEP of a wrench time-series chunk into a 6D label vector
(classes 0..4 per dimension) — i.e., output a label for every timestep and each of the 6 dimensions.

Assumptions / Notes:
- CSV must contain 6 wrench columns and 6 per-timestep label columns.
- Wrench columns: FX,FY,FZ,TX,TY,TZ
- Label columns: X_class,Y_class,Z_class,A_class,B_class,C_class
- Normalization: per-file z-score for each wrench channel (fit on the whole file).
- Split: first (1 - val-split) for train, last (val-split) for val, preserving time order.

Usage:
Configure the config dict below and run: python train_classify_each_timestep.py

Requires: torch, pandas, numpy
"""
# Configuration
CONFIG = {
    "csv": "./data/RCC_combined_14_processed.csv",
    "wrench_cols": ["FX", "FY", "FZ", "TX", "TY", "TZ"],
    "label_cols": ["X_class", "Y_class", "Z_class", "A_class", "B_class", "C_class"],
    "window": 256,
    "stride": 64,
    "batch_size": 4096,
    "epochs": 1_000_000,
    "lr": 1e-3,
    "val_split": 0.2,
    "d_model": 64,
    "nhead": 8,
    "layers": 8,
    "ffn": 128,
    "dropout": 0.0,
    "use_cls": False,  # Per-timestep classification uses sequence outputs directly
    "seed": 42,
    "out_dir": "./transformer_model/checkpoints",
    "save_every": 10,  # Save checkpoint every N epochs
    "wandb_project": "rcc_transformer",
    "wandb_name": None,  # Auto-generated if None
    "continue_from": None, # "best",  # "best", "latest", or path to checkpoint file, or None to start fresh
    
    # Class imbalance handling
    "use_class_weights": False,  # Use inverse frequency weighting
    "use_focal_loss": True,    # Use focal loss instead of weighted CE
    "focal_alpha": 1.0,         # Focal loss alpha parameter
    "focal_gamma": 5.0,         # Focal loss gamma parameter (higher = more focus on hard examples)
    "label_smoothing": 0.0,     # Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

# ----------------------------
# Dataset
# ----------------------------

@dataclass
class ChunkingConfig:
    window: int
    stride: int
    drop_last: bool = True

class WrenchChunkDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        wrench_cols: List[str],
        label_cols: List[str],
        cfg: ChunkingConfig,
        norm_mu: np.ndarray,
        norm_sd: np.ndarray,
        start: int,
        end: int,
    ):
        """
        Use rows in [start, end) to create chunks.
        Returns per-timestep labels for each chunk.
        """
        assert len(wrench_cols) == 6, "Need exactly 6 wrench columns."
        assert len(label_cols) == 6, "Need exactly 6 label columns."
        self.wrench = df[wrench_cols].values.astype(np.float32)
        self.labels = df[label_cols].values.astype(np.int64)
        self.wrench = zscore_apply(self.wrench, norm_mu, norm_sd).astype(np.float32)

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
            # include last partial if requested (not typical for Transformers)
            pass
        return idxs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        s = self.indices[i]
        e = s + self.cfg.window
        x = self.wrench[s:e, :]                    # (T, 6)
        labels_Tx6 = self.labels[s:e, :]           # (T, 6)
        return torch.from_numpy(x), torch.from_numpy(labels_Tx6)

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
        num_classes=5,   # classes 0..4
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

        # 6 separate classification heads (applied per timestep)
        self.heads = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        ) for _ in range(num_tasks)])

        self._reset_parameters()

    def _reset_parameters(self):
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def forward(self, x):
        """
        x: (B, T, 6)
        returns: list of 6 logits tensors, each (B, T, num_classes)
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
            feats = enc[:, 1:, :]  # drop CLS, keep per-timestep features -> (B, T, d_model)
        else:
            enc = self.encoder(h + pos)
            feats = enc  # (B, T, d_model)

        # per-timestep logits for each task
        return [head(feats) for head in self.heads]  # each: (B, T, num_classes)

# ----------------------------
# Training / Evaluation
# ----------------------------

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def compute_class_weights(dataset, num_classes=5, num_dims=6):
    """Compute class weights for each dimension across ALL TIMESTEPS to handle class imbalance."""
    class_counts = np.zeros((num_dims, num_classes), dtype=np.float64)
    
    for i in range(len(dataset)):
        _, labels = dataset[i]  # labels: (T, 6)
        for d in range(num_dims):
            vals, cnts = np.unique(labels[:, d].numpy(), return_counts=True)
            for v, c in zip(vals, cnts):
                if 0 <= int(v) < num_classes:
                    class_counts[d, int(v)] += int(c)
    
    # Compute weights inversely proportional to frequency
    weights = np.zeros((num_dims, num_classes), dtype=np.float64)
    for d in range(num_dims):
        total_samples = class_counts[d].sum()
        for c in range(num_classes):
            if class_counts[d, c] > 0:
                weights[d, c] = total_samples / (num_classes * class_counts[d, c])
            else:
                weights[d, c] = 1.0  # Default weight for unseen classes
    
    return torch.tensor(weights, dtype=torch.float32)

def train_one_epoch(model, loader, opt, device, class_weights=None, use_focal=False, focal_alpha=1, focal_gamma=2):
    model.train()
    total = 0.0
    n = 0
    
    if use_focal:
        focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    for xb, yb in loader:
        xb = xb.to(device)            # (B, T, 6)
        yb = yb.to(device)            # (B, T, 6) int in {0..4}

        logits_list = model(xb)       # list of 6 tensors (B, T, 5)
        loss = 0.0
        for d in range(6):
            logits = logits_list[d].reshape(-1, logits_list[d].size(-1))  # (B*T, 5)
            targets = yb[:, :, d].reshape(-1)                              # (B*T)
            if use_focal:
                loss += focal_loss(logits, targets)
            elif class_weights is not None:
                loss += nn.functional.cross_entropy(logits, targets, 
                                                   weight=class_weights[d].to(device))
            else:
                loss += nn.functional.cross_entropy(logits, targets)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        total += float(loss.item())
        n += 1
    return total / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    correct = np.zeros(6, dtype=np.int64)
    count = np.zeros(6, dtype=np.int64)
    for xb, yb in loader:
        xb = xb.to(device)            # (B, T, 6)
        yb = yb.to(device)            # (B, T, 6)
        logits_list = model(xb)       # each (B, T, 5)
        loss = 0.0
        for d in range(6):
            logits = logits_list[d]
            targets = yb[:, :, d]
            loss += nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            pred = logits.argmax(dim=-1)  # (B, T)
            correct[d] += (pred == targets).sum().item()
            count[d] += targets.numel()
        total += float(loss.item())
        n += 1
    avg_loss = total / max(n, 1)
    per_dim_acc = (correct / np.maximum(count, 1)).tolist()
    mean_acc = float(np.mean(per_dim_acc))
    return avg_loss, per_dim_acc, mean_acc

@torch.no_grad()
def evaluate_and_save_predictions(model, loader, device, output_path):
    """
    Evaluate model and save per-timestep predictions to CSV file.
    Returns same metrics as evaluate() but also saves predictions.
    CSV columns: chunk, t, gt_X..gt_C, pred_X..pred_C
    """
    model.eval()
    total = 0.0
    n = 0
    correct = np.zeros(6, dtype=np.int64)
    count = np.zeros(6, dtype=np.int64)
    
    all_predictions = []
    all_ground_truth = []
    all_chunk_ids = []
    all_t_ids = []

    chunk_offset = 0
    
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        B, T, _ = yb.shape
        logits_list = model(xb)
        loss = 0.0
        batch_preds = []
        
        for d in range(6):
            logits = logits_list[d]  # (B, T, 5)
            targets = yb[:, :, d]    # (B, T)
            loss += nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            pred = logits.argmax(dim=-1)  # (B, T)
            batch_preds.append(pred.cpu().numpy())
            correct[d] += (pred == targets).sum().item()
            count[d] += targets.numel()
        
        # Store predictions and ground truth for this batch (flatten B,T)
        batch_preds = np.stack(batch_preds, axis=-1)           # (B, T, 6)
        all_predictions.append(batch_preds.reshape(-1, 6))     # (B*T, 6)
        all_ground_truth.append(yb.cpu().numpy().reshape(-1, 6))

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
        df_results[f"gt_{dim_name}"] = all_ground_truth[:, i]
        df_results[f"pred_{dim_name}"] = all_predictions[:, i]
    
    df_results.to_csv(output_path, index=False)
    print(f"Saved validation predictions to {output_path}")
    
    avg_loss = total / max(n, 1)
    per_dim_acc = (correct / np.maximum(count, 1)).tolist()
    mean_acc = float(np.mean(per_dim_acc))
    return avg_loss, per_dim_acc, mean_acc

# Reuse confusion matrix plotting with flattened timestep samples

def save_confusion_matrices(all_ground_truth, all_predictions, output_dir, epoch):
    """
    Create and save confusion matrix plots for each dimension.
    
    Args:
        all_ground_truth: numpy array of shape (n_samples, 6) where n_samples=chunks*T
        all_predictions: numpy array of shape (n_samples, 6)
        output_dir: directory to save the confusion matrix image
        epoch: current epoch number for filename
    """
    dim_names = ["X", "Y", "Z", "A", "B", "C"]
    class_names = ["0", "1", "2", "3", "4"]
    
    # Create a 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Confusion Matrices - Epoch {epoch}', fontsize=16)
    
    for i, dim_name in enumerate(dim_names):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Compute confusion matrix for this dimension
        cm = confusion_matrix(all_ground_truth[:, i], all_predictions[:, i], 
                            labels=[0, 1, 2, 3, 4])
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title(f'Dimension {dim_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.tight_layout()
    
    # Save the figure (overwrite each time)
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrices_timestep.png")
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"Saved confusion matrices to {confusion_matrix_path}")

@torch.no_grad()
def evaluate_save_predictions_and_confusion(model, loader, device, output_path, output_dir, epoch):
    """
    Combined function that evaluates, saves per-timestep predictions CSV, and creates confusion matrices.
    """
    model.eval()
    total = 0.0
    n = 0
    correct = np.zeros(6, dtype=np.int64)
    count = np.zeros(6, dtype=np.int64)
    
    all_predictions = []
    all_ground_truth = []
    all_chunk_ids = []
    all_t_ids = []

    chunk_offset = 0
    
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        B, T, _ = yb.shape
        logits_list = model(xb)
        loss = 0.0
        batch_preds = []
        
        for d in range(6):
            logits = logits_list[d]
            targets = yb[:, :, d]
            loss += nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            pred = logits.argmax(dim=-1)
            batch_preds.append(pred.cpu().numpy())
            correct[d] += (pred == targets).sum().item()
            count[d] += targets.numel()
        
        # Store predictions and ground truth for this batch (flatten B,T)
        batch_preds = np.stack(batch_preds, axis=-1)           # (B, T, 6)
        all_predictions.append(batch_preds.reshape(-1, 6))     # (B*T, 6)
        all_ground_truth.append(yb.cpu().numpy().reshape(-1, 6))

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
        df_results[f"gt_{dim_name}"] = all_ground_truth[:, i]
        df_results[f"pred_{dim_name}"] = all_predictions[:, i]
    
    df_results.to_csv(output_path, index=False)
    print(f"Saved validation predictions to {output_path}")
    
    # Create and save confusion matrices
    save_confusion_matrices(all_ground_truth, all_predictions, output_dir, epoch)
    
    avg_loss = total / max(n, 1)
    per_dim_acc = (correct / np.maximum(count, 1)).tolist()
    mean_acc = float(np.mean(per_dim_acc))
    return avg_loss, per_dim_acc, mean_acc

# ----------------------------
# Checkpoint utilities
# ----------------------------

def find_checkpoint(out_dir: str, checkpoint_type: str) -> str:
    """Find checkpoint file based on type: 'best', 'latest', or specific path (timestep naming)."""
    if checkpoint_type == "best":
        best_path = os.path.join(out_dir, "best_model_timestep.pt")
        if os.path.exists(best_path):
            return best_path
        else:
            print(f"Best model not found at {best_path}")
            return None
    elif checkpoint_type == "latest":
        # Find the latest checkpoint by epoch number
        checkpoint_files = []
        for file in os.listdir(out_dir):
            if file.startswith("timestep_checkpoint_epoch_") and file.endswith(".pt"):
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
    wandb_name = config["wandb_name"] or f"rcc_transformer_timestep_{config['window']}w_{config['stride']}s"
    wandb.init(
        project=config["wandb_project"],
        name=wandb_name,
        config=config
    )

    # Load CSV
    df = pd.read_csv(config["csv"])
    for c in config["wrench_cols"] + config["label_cols"]:
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
    train_ds = WrenchChunkDataset(
        df, config["wrench_cols"], config["label_cols"], cfg, mu, sd, start=train_start, end=train_end
    )
    val_ds = WrenchChunkDataset(
        df, config["wrench_cols"], config["label_cols"], cfg, mu, sd, start=val_start, end=val_end
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
        input_dim=6, d_model=config["d_model"], nhead=config["nhead"], num_layers=config["layers"],
        dim_feedforward=config["ffn"], dropout=config["dropout"], num_classes=5, num_tasks=6,
        use_cls=config["use_cls"], max_len=config["window"] + 1
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    # Compute class weights if enabled
    class_weights = None
    if config["use_class_weights"] and not config["use_focal_loss"]:
        print("Computing class weights for imbalanced data (per-timestep)...")
        class_weights = compute_class_weights(train_ds)
        print("Class weights computed:")
        dim_names = ["X", "Y", "Z", "A", "B", "C"]
        for d, dim_name in enumerate(dim_names):
            weights_str = ", ".join([f"{w:.3f}" for w in class_weights[d]])
            print(f"  {dim_name}: [{weights_str}]")

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
    print(f"Class imbalance handling: weights={config['use_class_weights']}, focal={config['use_focal_loss']}")
    
    for epoch in range(start_epoch, config["epochs"] + 1):
        tr_loss = train_one_epoch(
            model, train_loader, opt, device, 
            class_weights=class_weights,
            use_focal=config["use_focal_loss"],
            focal_alpha=config["focal_alpha"],
            focal_gamma=config["focal_gamma"]
        )
        
        # Regular evaluation for logging
        vl_loss, per_dim_acc, mean_acc = evaluate(model, val_loader, device)
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": vl_loss,
            "mean_acc": mean_acc,
            **{f"acc_{label}": acc for label, acc in zip(["X", "Y", "Z", "A", "B", "C"], per_dim_acc)}
        })
        
        is_best = vl_loss < best_val
        if is_best:
            best_val = vl_loss
            best_path = os.path.join(config["out_dir"], "best_model_timestep.pt")
            torch.save({"model": model.state_dict(), "config": config}, best_path)
        
        # Periodic checkpoint saving and validation predictions
        if epoch % config["save_every"] == 0:
            checkpoint_path = os.path.join(config["out_dir"], f"timestep_checkpoint_epoch_{epoch}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "config": config,
                "best_val": best_val
            }, checkpoint_path)
            
            # Save validation predictions and confusion matrices
            predictions_path = os.path.join(config["out_dir"], f"val_predictions_timestep_epoch_{epoch}.csv")
            evaluate_save_predictions_and_confusion(model, val_loader, device, predictions_path, config["out_dir"], epoch)
        
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.4f} | val_loss={vl_loss:.4f} | "
            f"acc_dim={['%.3f'%a for a in per_dim_acc]} | mean_acc={mean_acc:.3f} | "
            f"{'** saved **' if is_best else ''}"
        )

if __name__ == "__main__":
    main()
