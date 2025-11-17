#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eval script for Time-Series CNN trained by train_CNN.py.

- Loads checkpoint from CONFIG['out_dir'] (best/latest/path).
- Recreates the validation split EXACTLY from data_split.json (or trial_split_labels.csv).
- Builds the identical validation dataset & chunking.
- Reports per-dimension MAE and RMSE (timestep-wise, normalized label space).
- Performs 5-class and 3-class classification per dimension over windows of length 'chunk_window':
    • GT class per window: determined by dominant extremum (max |value| in window, signed).
    • Prediction class per window: uses 'margin' to shrink thresholds toward 0, and
      requires 'persistence_length' consecutive timesteps crossing a class threshold before
      assigning a farther-from-zero class; otherwise downgrades to nearer class.
    • Optional: apply the same persistence logic (no margin) to GT when CONFIG['gt_persistence']=True.

- Outputs two seaborn figures (2×3 subplots):
    • 5-class confusion matrices per dimension
    • 3-class confusion matrices per dimension
  Each cell shows "count\n(row-normalized fraction)" and is colored by the fraction.
  Figures saved under <checkpoint_dir>/<subfolder_named_by_hparams>/

Author: you :)
"""

import os, json, math
from typing import List, Tuple, Optional, Set, Dict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json, os

# ----------------------------
# Configuration (no argparse)
# ----------------------------
CONFIG = {
    # === Data (must match training) ===
    "csv": "./data/RCC_kuka_ati_15_trials.csv",
    "wrench_cols": ["Fx_ati", "Fy_ati", "Fz_ati", "Tx_ati", "Ty_ati", "Tz_ati"],
    "pose_cols":   ["X", "Y", "Z", "A", "B", "C"],
    "label_cols":  ["X_norm", "Y_norm", "Z_norm", "A_norm", "B_norm", "C_norm"],
    "trial_col": "trial",

    # === Validation chunking (must match training) ===
    "window": 16384,
    "stride": 16384,
    "batch_size": 64,

    # === Where training wrote checkpoints & split files ===
    "out_dir": "./CNN_model/checkpoints_v8/",
    # "checkpoint": "best" | "latest" | "/path/to/file.pt"
    "checkpoint": "best",

    # === Classification parameters ===
    # Thresholds (normalized label space): b2 is implicitly 1.0
    "b0": 0.7,
    "b1": 0.8,
    "chunk_window": 10,
    "margin": 0.2,               # applied to PRED thresholds only
    "persistence_length": 0,     # consecutive timesteps needed to claim a farther-from-zero class
    "gt_persistence": False,     # if True, apply persistence to GT labels (no margin)

    # === Misc ===
    "seed": 42,
}

# ----------------------------
# Utilities (mirrors train_CNN.py behavior)
# ----------------------------
def zscore_fit(x: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return mu, sd

def zscore_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd

def compute_relative_pose(pose_chunk: np.ndarray) -> np.ndarray:
    if len(pose_chunk) == 0:
        return pose_chunk
    reference_pose = pose_chunk[0:1, :]
    return (pose_chunk - reference_pose).astype(np.float32)

def compute_trial_segments(trials: np.ndarray, start: int, end: int) -> List[Tuple[int,int]]:
    segs = []
    i = max(0, start); end = min(len(trials), end)
    if i >= end: return segs
    while i < end:
        tval = trials[i]
        j = i + 1
        while j < end and trials[j] == tval:
            j += 1
        segs.append((i, j))
        i = j
    return segs

def filter_segments_by_trials(segments: List[Tuple[int,int]], trials: np.ndarray, allowed: Optional[Set]) -> List[Tuple[int,int]]:
    if not allowed: return segments
    return [(s,e) for (s,e) in segments if trials[s] in allowed]

class ChunkingConfig:
    def __init__(self, window: int, stride: int, drop_last: bool = True):
        self.window = window
        self.stride = stride
        self.drop_last = drop_last

class WrenchPoseChunkDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, wrench_cols: List[str], pose_cols: List[str], label_cols: List[str],
        cfg: ChunkingConfig, wrench_norm_mu: np.ndarray, wrench_norm_sd: np.ndarray,
        start: int, end: int, trial_col: str, allowed_trials: Optional[Set]
    ):
        self.wrench = df[wrench_cols].values.astype(np.float32)
        self.pose   = df[pose_cols].values.astype(np.float32)
        self.labels = df[label_cols].values.astype(np.float32)
        # self.trials = df[trial_col].values
        # Normalize to strings for robust matching
        self.trials = df[trial_col].astype(str).values
        allowed_str = set(map(str, allowed_trials)) if allowed_trials is not None else None
        
        self.wrench = zscore_apply(self.wrench, wrench_norm_mu, wrench_norm_sd).astype(np.float32)

        self.cfg = cfg
        self.start = max(0, start); self.end = min(len(df), end)
        # all_segments = compute_trial_segments(self.trials, self.start, self.end)
        # self.segments = filter_segments_by_trials(all_segments, self.trials, allowed_trials)
        # self.indices = self._make_indices()
        all_segments = compute_trial_segments(self.trials, self.start, self.end)
        self.segments = filter_segments_by_trials(all_segments, self.trials, allowed_str)
        self.indices = self._make_indices()

        # --- Diagnostics ---
        if allowed_str is not None:
            seg_lens = [e - s for (s, e) in self.segments]
            print(f"[VAL DATASET] allowed_trials={sorted(list(allowed_str))[:8]}{'...' if len(allowed_str)>8 else ''}")
            print(f"[VAL DATASET] window={self.cfg.window} stride={self.cfg.stride} "
                  f"segments={len(self.segments)} min_seg={min(seg_lens) if seg_lens else 0} "
                  f"max_seg={max(seg_lens) if seg_lens else 0}")
            print(f"[VAL DATASET] produced {len(self.indices)} chunk starts")

    def _make_indices(self) -> List[int]:
        idxs = []
        for s,e in self.segments:
            if e - s >= self.cfg.window:
                idxs.extend(range(s, e - self.cfg.window + 1, self.cfg.stride))
        return idxs

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        s = self.indices[i]; e = s + self.cfg.window
        wrench_chunk = self.wrench[s:e, :]
        pose_chunk   = self.pose[s:e, :]
        rel_pose_chunk = compute_relative_pose(pose_chunk)
        x = np.concatenate([wrench_chunk, rel_pose_chunk], axis=1).astype(np.float32)  # (T,12)
        y = self.labels[s:e, :].astype(np.float32)  # (T,6)
        return torch.from_numpy(x), torch.from_numpy(y)

# NEW: Dataset using fixed window start indices
class FixedStartIndexChunkDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        wrench_cols: List[str],
        pose_cols: List[str],
        label_cols: List[str],
        start_indices: np.ndarray,
        window: int,
        wrench_norm_mu: np.ndarray,
        wrench_norm_sd: np.ndarray,
    ):
        self.wrench = zscore_apply(df[wrench_cols].values.astype(np.float32), wrench_norm_mu, wrench_norm_sd).astype(np.float32)
        self.pose   = df[pose_cols].values.astype(np.float32)
        self.labels = df[label_cols].values.astype(np.float32)
        self.starts = np.array(start_indices, dtype=np.int64)
        self.window = int(window)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i]); e = s + self.window
        wrench_chunk = self.wrench[s:e, :]
        pose_chunk   = self.pose[s:e, :]
        rel_pose_chunk = compute_relative_pose(pose_chunk)
        x = np.concatenate([wrench_chunk, rel_pose_chunk], axis=1).astype(np.float32)
        y = self.labels[s:e, :].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

# ----------------------------
# Model (matches train_CNN.py)
# ----------------------------
class ResidualTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=7, dilation=1, dropout=0.1, bidirectional=True):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.act1  = nn.GELU()
        self.do1   = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.act2  = nn.GELU()
        self.do2   = nn.Dropout(dropout)
        self.norm  = nn.BatchNorm1d(out_ch)
        self.down  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        y = self.conv1(x); y = self.act1(y); y = self.do1(y)
        y = self.conv2(y); y = self.act2(y); y = self.do2(y)
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
        for ch, d in zip(chans, dilations):
            blocks.append(ResidualTCNBlock(in_ch, ch, kernel_size=kernel_size, dilation=d, dropout=dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*blocks)
        self.head_norm = nn.LayerNorm(hidden)
        self.head = nn.Conv1d(hidden, num_tasks, kernel_size=1)

    def forward(self, x):  # x: [B,T,12]
        x = x.transpose(1,2)      # [B,12,T]
        h = self.in_proj(x)       # [B,H,T]
        h = self.tcn(h)           # [B,H,T]
        h_ln = h.transpose(1,2)   # [B,T,H]
        h_ln = self.head_norm(h_ln)
        h = h_ln.transpose(1,2)   # [B,H,T]
        y = self.head(h)          # [B,6,T]
        return y.transpose(1,2)   # [B,T,6]

# ----------------------------
# Checkpoints & split
# ----------------------------
def find_checkpoint(out_dir: str, checkpoint_type: str) -> Optional[str]:
    if checkpoint_type == "best":
        p = os.path.join(out_dir, "best_model_regression.pt")
        return p if os.path.exists(p) else None
    if checkpoint_type == "latest":
        files = []
        for f in os.listdir(out_dir):
            if f.startswith("regression_checkpoint_epoch_") and f.endswith(".pt"):
                try:
                    ep = int(f.split("_")[-1].split(".")[0]); files.append((ep, os.path.join(out_dir,f)))
                except Exception: pass
        if files:
            files.sort(key=lambda x: x[0])
            return files[-1][1]
        return None
    if os.path.isfile(checkpoint_type):
        return checkpoint_type
    return None
def load_split_trials(out_dir: str):
    """
    Returns (val_trials_set, train_trials_set) with trial IDs as strings.
    Looks for:
      - data_split.json  (keys: train_trials/val_trials OR train/val)
      - trial_split_labels.csv (columns: trial, split)
      - checkpoint 'config' dict (train_trials/val_trials)
    """
    import os, re, json, pandas as pd, ast
    def _read_json_relaxed(path: str):
        import re, json, ast
        raw = open(path, "r", encoding="utf-8").read()
        if raw.strip() == "":
            raise ValueError("Empty file")
        no_comments = re.sub(r"//.*?$|#.*?$", "", raw, flags=re.MULTILINE)
        no_trailing = re.sub(r",(\s*[\]}])", r"\1", no_comments)
        try:
            return json.loads(no_trailing)
        except json.JSONDecodeError:
            return ast.literal_eval(no_trailing)

    # 1) JSON
    dsp = os.path.join(out_dir, "data_split.json")
    if os.path.exists(dsp):
        obj = _read_json_relaxed(dsp)
        if isinstance(obj, dict):
            if "train_trials" in obj or "val_trials" in obj:
                train = obj.get("train_trials", [])
                val   = obj.get("val_trials", [])
            elif "train" in obj or "val" in obj:
                train = obj.get("train", [])
                val   = obj.get("val", [])
            elif "split" in obj and isinstance(obj["split"], dict):
                train = obj["split"].get("train", [])
                val   = obj["split"].get("val", [])
            else:
                raise ValueError(f"Unrecognized keys in {dsp}: {list(obj.keys())}")
            # RETURN (val, train) — in string form
            return set(map(str, val)), set(map(str, train))

    # 2) CSV
    csvp = os.path.join(out_dir, "trial_split_labels.csv")
    if os.path.exists(csvp):
        df = pd.read_csv(csvp)
        tcol = next((c for c in ["trial","trial_id","trial_name"] if c in df.columns), None)
        scol = next((c for c in ["split","set","partition"] if c in df.columns), None)
        if tcol is None or scol is None:
            raise RuntimeError(f"{csvp} missing 'trial' and/or 'split' columns.")
        train = df[df[scol].astype(str).str.lower().eq("train")][tcol].astype(str).tolist()
        val   = df[df[scol].astype(str).str.lower().eq("val")][tcol].astype(str).tolist()
        return set(val), set(train)

    # 3) Checkpoint config
    ckpt_path = None
    best = os.path.join(out_dir, "best_model_regression.pt")
    if os.path.exists(best):
        ckpt_path = best
    else:
        cands = [f for f in os.listdir(out_dir) if f.startswith("regression_checkpoint_epoch_") and f.endswith(".pt")]
        if cands:
            import re
            cands.sort(key=lambda s: int(re.findall(r"(\d+)\.pt$", s)[0]), reverse=True)
            ckpt_path = os.path.join(out_dir, cands[0])
    if ckpt_path:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg  = ckpt.get("config", {})
        train = cfg.get("train_trials")
        val   = cfg.get("val_trials")
        if train is not None and val is not None:
            return set(map(str, val)), set(map(str, train))

    raise FileNotFoundError(
        f"Could not find a valid split in {out_dir} (data_split.json / trial_split_labels.csv / checkpoint config)."
    )

def load_window_split(out_dir: str) -> Optional[np.ndarray]:
    """Return array of validation window start indices if window split exists, else None.
    Looks for window_split.json (val_starts) or window_split_labels.csv (rows with split=='val')."""
    js = os.path.join(out_dir, "window_split.json")
    if os.path.exists(js):
        with open(js, "r") as f:
            meta = json.load(f)
        if isinstance(meta, dict) and "val_starts" in meta:
            arr = np.array(meta["val_starts"], dtype=np.int64)
            return arr
    csvp = os.path.join(out_dir, "window_split_labels.csv")
    if os.path.exists(csvp):
        df = pd.read_csv(csvp)
        if "split" in df.columns:
            if "start" not in df.columns:
                raise RuntimeError(f"{csvp} missing 'start' column.")
            val = df[df["split"].astype(str).str.lower().eq("val")]["start"].astype(int).values
            return val.astype(np.int64)
    return None

# ----------------------------
# Evaluation helpers
# ----------------------------
@torch.no_grad()
def predict_all(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        pr = model(xb)
        preds.append(pr.detach().cpu().numpy())
        gts.append(yb.detach().cpu().numpy())
    if not preds:
        return np.zeros((0,0,0)), np.zeros((0,0,0))
    return np.concatenate(preds, axis=0), np.concatenate(gts, axis=0)  # [B,T,6]

def mae_rmse_per_dim(pred: np.ndarray, gt: np.ndarray) -> Tuple[List[float], List[float]]:
    diff = pred - gt
    mae = np.mean(np.abs(diff), axis=(0,1))
    rmse = np.sqrt(np.mean(diff**2, axis=(0,1)))
    return mae.tolist(), rmse.tolist()

# --- Window classification with persistence ---
def has_persistent_run(x: np.ndarray, thr: float, k: int, positive: bool) -> bool:
    """Return True if there exists a run of length>=k where x>=thr (positive) or x<=-thr (negative)."""
    if k <= 1:
        return np.any(x >= thr) if positive else np.any(x <= -thr)
    if positive:
        mask = (x >= thr).astype(np.int32)
    else:
        mask = (x <= -thr).astype(np.int32)
    # check consecutive run length >= k
    run = 0
    for v in mask:
        run = run + 1 if v == 1 else 0
        if run >= k: return True
    return False

def dominant_value(v: np.ndarray) -> float:
    """Return the signed value (max abs) within window preserving its sign."""
    imax = np.argmax(np.abs(v))
    return float(v[imax])

def classify_window_five_class(gt_vals: np.ndarray, pr_vals: np.ndarray,
                               b0: float, b1: float,
                               margin: float, k: int,
                               apply_gt_persistence: bool = False) -> Tuple[int, int]:
    """
    Returns (gt_class, pred_class) for one dim within a window.
    5-class mapping indices:
      0: [-1, -b1], 1: (-b1, -b0], 2: (-b0, +b0), 3: [ +b0, +b1 ), 4: [ +b1, +1 ]
    If apply_gt_persistence is True, GT class uses persistence checks (no margin);
    otherwise GT uses dominant extremum threshold mapping.
    """
    b2 = 1.0
    # ---- GT class ----
    if apply_gt_persistence:
        # Prefer outer class if persistent at b1, else inner class at b0, else neutral
        pos_b1 = has_persistent_run(gt_vals, b1, k, positive=True)
        neg_b1 = has_persistent_run(gt_vals, b1, k, positive=False)
        if pos_b1 or neg_b1:
            if pos_b1 and not neg_b1:
                gt_c = 4
            elif neg_b1 and not pos_b1:
                gt_c = 0
            else:
                pos_max = float(np.max(gt_vals))
                neg_min = float(np.min(gt_vals))
                gt_c = 4 if abs(pos_max) >= abs(neg_min) else 0
        else:
            pos_b0 = has_persistent_run(gt_vals, b0, k, positive=True)
            neg_b0 = has_persistent_run(gt_vals, b0, k, positive=False)
            if pos_b0 and not neg_b0:
                gt_c = 3
            elif neg_b0 and not pos_b0:
                gt_c = 1
            elif pos_b0 and neg_b0:
                pos_max = float(np.max(gt_vals))
                neg_min = float(np.min(gt_vals))
                gt_c = 3 if abs(pos_max) >= abs(neg_min) else 1
            else:
                gt_c = 2
    else:
        v = dominant_value(gt_vals)
        def to_c5(x):
            if x <= -b1: return 0
            if x <= -b0: return 1
            if x <  b0:  return 2
            if x <  b1:  return 3
            return 4
        gt_c = to_c5(v)

    # ---- Prediction class (margin + persistence) ----
    m_b0 = max(0.0, b0 - margin)
    m_b1 = max(0.0, b1 - margin)
    # Candidate side is sign of dominant predicted extremum
    vp = dominant_value(pr_vals)
    pos_side = (vp >= 0.0)

    # try outer class (4 or 0) with m_b1
    if pos_side:
        if has_persistent_run(pr_vals, m_b1, k, positive=True):
            pred_c = 4
        elif has_persistent_run(pr_vals, m_b0, k, positive=True):
            pred_c = 3
        elif np.max(np.abs(pr_vals)) < m_b0:
            pred_c = 2
        else:
            # cross sporadically but not persistent -> nearest neutral
            pred_c = 2
    else:
        if has_persistent_run(pr_vals, m_b1, k, positive=False):
            pred_c = 0
        elif has_persistent_run(pr_vals, m_b0, k, positive=False):
            pred_c = 1
        elif np.max(np.abs(pr_vals)) < m_b0:
            pred_c = 2
        else:
            pred_c = 2

    return gt_c, pred_c

def classify_window_three_class(gt_vals: np.ndarray, pr_vals: np.ndarray,
                                b0: float, margin: float, k: int,
                                apply_gt_persistence: bool = False) -> Tuple[int, int]:
    """
    3-class indices based on b0:
      0: [-1, -b0]
      1: (-b0, +b0)
      2: [+b0, +1]
    If apply_gt_persistence is True, GT class uses persistence checks (no margin);
    otherwise GT uses dominant extremum threshold mapping.
    """
    # ---- GT class ----
    if apply_gt_persistence:
        pos_persist = has_persistent_run(gt_vals, b0, k, positive=True)
        neg_persist = has_persistent_run(gt_vals, b0, k, positive=False)
        if pos_persist and not neg_persist:
            gt_c = 2
        elif neg_persist and not pos_persist:
            gt_c = 0
        elif pos_persist and neg_persist:
            pos_max = float(np.max(gt_vals))
            neg_min = float(np.min(gt_vals))
            gt_c = 2 if abs(pos_max) >= abs(neg_min) else 0
        else:
            gt_c = 1
    else:
        v = dominant_value(gt_vals)
        def to_c3(x):
            if x <= -b0: return 0
            if x <  b0:  return 1
            return 2
        gt_c = to_c3(v)

    # ---- Prediction class (margin + persistence) ----
    m_b0 = max(0.0, b0 - margin)
    vp = dominant_value(pr_vals)
    pos_side = (vp >= 0.0)
    if pos_side:
        if has_persistent_run(pr_vals, m_b0, k, positive=True):
            pred_c = 2
        elif np.max(np.abs(pr_vals)) < m_b0:
            pred_c = 1
        else:
            pred_c = 1
    else:
        if has_persistent_run(pr_vals, m_b0, k, positive=False):
            pred_c = 0
        elif np.max(np.abs(pr_vals)) < m_b0:
            pred_c = 1
        else:
            pred_c = 1
    return gt_c, pred_c

def build_confusion(num_classes: int, gt: np.ndarray, pr: np.ndarray) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for g, p in zip(gt, pr):
        cm[g, p] += 1
    return cm

# Increase annotation font size for publication
ANNOT_FONT_SIZE = 18  # increased for paper-quality
TEXT_WHITE_THRESHOLD = 0.6  # fraction above which to use white text for readability
RANGE_LABEL_FONT_SIZE = 12  # reduced font size for range labels

def annotate_cm(ax, cm: np.ndarray):
    """Annotate counts and row-normalized fractions centered in each heatmap cell.
    Use white text if the cell's normalized fraction exceeds TEXT_WHITE_THRESHOLD.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        row = cm.sum(axis=1, keepdims=True)
        frac = np.where(row > 0, cm / row, 0.0)
    n_rows, n_cols = cm.shape
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    for i in range(n_rows):
        for j in range(n_cols):
            val = frac[i,j]
            color = 'white' if val >= TEXT_WHITE_THRESHOLD else 'black'
            ax.text(j + 0.5, i + 0.5, f"{cm[i,j]}\n{val:.2f}",
                    ha="center", va="center", fontsize=ANNOT_FONT_SIZE, color=color)

# ----------------------------
# Main
# ----------------------------
def main():
    cfg = CONFIG
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    # Locate checkpoint & its directory
    ckpt_path = find_checkpoint(cfg["out_dir"], cfg["checkpoint"])
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found via '{cfg['checkpoint']}' under {cfg['out_dir']}")
    ckpt_dir = os.path.dirname(ckpt_path)

    # Load training config from checkpoint to instantiate model exactly as trained
    ckpt = torch.load(ckpt_path, map_location="cpu")
    train_conf = ckpt.get("config", {})
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
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load CSV & compute wrench normalization (as in training: fit on the whole file)
    df = pd.read_csv(cfg["csv"])
    for c in cfg["wrench_cols"] + cfg["pose_cols"] + cfg["label_cols"]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in CSV.")
    wrench_all = df[cfg["wrench_cols"]].values.astype(np.float32)
    mu, sd = zscore_fit(wrench_all)

    # Recreate validation split trials OR windows
    val_trials, train_trials = set(), set()
    val_window_starts = load_window_split(cfg["out_dir"])

    # Build validation dataset
    dcfg = ChunkingConfig(window=train_conf.get("window", cfg["window"]),
                          stride=train_conf.get("stride", cfg["stride"]),
                          drop_last=True)

    if val_window_starts is not None and val_window_starts.size > 0:
        # Window-based split
        val_ds = FixedStartIndexChunkDataset(
            df, cfg["wrench_cols"], cfg["pose_cols"], cfg["label_cols"],
            start_indices=val_window_starts, window=dcfg.window, wrench_norm_mu=mu, wrench_norm_sd=sd
        )
    else:
        # Trial-based (legacy) split
        val_trials, train_trials = load_split_trials(cfg["out_dir"])  # sets of strings
        trials_arr = df[cfg["trial_col"]].values
        N = len(df)
        all_segments = compute_trial_segments(trials_arr, 0, N)
        _val_segments = filter_segments_by_trials(all_segments, trials_arr, val_trials)
        val_ds = WrenchPoseChunkDataset(
            df, cfg["wrench_cols"], cfg["pose_cols"], cfg["label_cols"],
            dcfg, mu, sd, 0, N, cfg["trial_col"], allowed_trials=val_trials
        )

    if len(val_ds) == 0:
        raise RuntimeError("No validation chunks; check window/stride or split.")

    val_loader = DataLoader(val_ds, batch_size=train_conf.get("batch_size", cfg["batch_size"]),
                            shuffle=False, drop_last=False)

    # Predict
    preds, gts = predict_all(model, val_loader, device)  # [B,T,6] each
    # Per-dim MAE/RMSE
    mae, rmse = mae_rmse_per_dim(preds, gts)
    dim_names = ["X","Y","Z","A","B","C"]
    print("Per-dimension MAE:", {d: round(m,6) for d,m in zip(dim_names, mae)})
    print("Per-dimension RMSE:", {d: round(r,6) for d,r in zip(dim_names, rmse)})

    # --- Build windowed labels/preds for classification ---
    # Flatten batches and process per dimension with windows of 'chunk_window'
    _, T, D = preds.shape
    P = preds.reshape(-1, T, D)  # already [B,T,6]
    G = gts.reshape(-1, T, D)

    cw   = cfg["chunk_window"]
    b0   = cfg["b0"]; b1 = cfg["b1"]
    marg = cfg["margin"]; k = cfg["persistence_length"]
    gtp  = 1 if cfg.get("gt_persistence", False) else 0

    # Subfolder for figures
    subdir = f"class_eval_b0{b0:.2f}_b1{b1:.2f}_cw{cw}_m{marg:.2f}_pl{k}_gtp{gtp}"
    out_dir = os.path.join(ckpt_dir, subdir)
    os.makedirs(out_dir, exist_ok=True)

    # For each dimension, build arrays of window classes (GT & Pred)
    gt5_all, pr5_all = {}, {}
    gt3_all, pr3_all = {}, {}
    for di, dn in enumerate(dim_names):
        gt5_list, pr5_list = [], []
        gt3_list, pr3_list = [], []
        for b in range(P.shape[0]):
            series_gt = G[b, :, di]
            series_pr = P[b, :, di]
            # iterate non-overlapping windows
            nwin = series_gt.shape[0] // cw
            for w in range(nwin):
                s = w*cw; e = s+cw
                g5, p5 = classify_window_five_class(series_gt[s:e], series_pr[s:e], b0, b1, marg, k, cfg.get("gt_persistence", False))
                g3, p3 = classify_window_three_class(series_gt[s:e], series_pr[s:e], b0, marg, k, cfg.get("gt_persistence", False))
                gt5_list.append(g5); pr5_list.append(p5)
                gt3_list.append(g3); pr3_list.append(p3)
        gt5_all[dn] = np.array(gt5_list, dtype=np.int64)
        pr5_all[dn] = np.array(pr5_list, dtype=np.int64)
        gt3_all[dn] = np.array(gt3_list, dtype=np.int64)
        pr3_all[dn] = np.array(pr3_list, dtype=np.int64)

    # --- Plot confusion matrices (2×3) for 5-class ---
    display_names_map = {"A": "Roll", "B": "Pitch", "C": "Yaw"}
    import matplotlib.gridspec as gridspec
    fig5 = plt.figure(figsize=(18, 10))
    gs5 = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.25, hspace=0.35)
    axes5 = [[fig5.add_subplot(gs5[r, c]) for c in range(3)] for r in range(2)]
    cbar_ax5 = fig5.add_subplot(gs5[:, 3])
    mappable5 = None
    for idx, dn in enumerate(dim_names):
        r = idx // 3; c = idx % 3
        ax = axes5[r][c]
        cm5 = build_confusion(5, gt5_all[dn], pr5_all[dn])
        with np.errstate(divide='ignore', invalid='ignore'):
            rn = cm5.sum(axis=1, keepdims=True)
            frac = np.where(rn > 0, cm5 / rn, 0.0)
        disp_dn = display_names_map.get(dn, dn)
        hmap = sns.heatmap(frac, ax=ax, annot=False, cmap="Blues", vmin=0.0, vmax=1.0, cbar=False, square=True)
        if mappable5 is None:
            mappable5 = hmap
        ax.set_title(disp_dn, fontsize=20)
        ax.set_xlabel("Pred", fontsize=20); ax.set_ylabel("True", fontsize=20)
        ax.set_xticks(np.arange(5) + 0.5); ax.set_yticks(np.arange(5) + 0.5)
        ax.set_xticklabels(["N2","N1","Neu","P1","P2"], rotation=0, fontsize=16)
        ax.set_yticklabels(["N2","N1","Neu","P1","P2"], rotation=90, ha='center', fontsize=16)
        annotate_cm(ax, cm5)
    # Shared colorbar
    if mappable5 is not None:
        cb5 = fig5.colorbar(mappable5.collections[0], cax=cbar_ax5)
        cb5.ax.tick_params(labelsize=14)
        cb5.set_label("Row-normalized fraction", fontsize=16)
    # Add legend/explanation text
    # fig5.text(0.5, 0.015, "Cell text: count and row-normalized fraction; color encodes row-normalized fraction.", ha='center', fontsize=14)
    p5_path = os.path.join(out_dir, "confusion_5class_2x3.png")
    fig5.tight_layout(rect=[0,0.02,0.97,1])
    fig5.savefig(p5_path, dpi=300)
    plt.close(fig5)

    # --- Plot confusion matrices (2×3) for 3-class ---
    fig3 = plt.figure(figsize=(18, 10))
    gs3 = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.25, hspace=0.6)
    axes3 = [[fig3.add_subplot(gs3[r, c]) for c in range(3)] for r in range(2)]
    cbar_ax3 = fig3.add_subplot(gs3[:, 3])
    # Prepare range labels using b0 with two decimals
    b0_label_neg = f"[-1,-{b0:.2f}]"
    b0_label_mid = f"(-{b0:.2f},{b0:.2f})"
    b0_label_pos = f"[{b0:.2f},1]"
    range_labels = [b0_label_neg, b0_label_mid, b0_label_pos]
    mappable3 = None
    for idx, dn in enumerate(dim_names):
        r = idx // 3; c = idx % 3
        ax = axes3[r][c]
        cm3 = build_confusion(3, gt3_all[dn], pr3_all[dn])
        with np.errstate(divide='ignore', invalid='ignore'):
            rn = cm3.sum(axis=1, keepdims=True)
            frac = np.where(rn > 0, cm3 / rn, 0.0)
        disp_dn = display_names_map.get(dn, dn)
        hmap = sns.heatmap(frac, ax=ax, annot=False, cmap="Blues", vmin=0.0, vmax=1.0, cbar=False, square=True)
        if mappable3 is None:
            mappable3 = hmap
        ax.set_title(disp_dn, fontsize=20)
        ax.set_xlabel("Pred", fontsize=20); ax.set_ylabel("True", fontsize=20)
        ax.set_xticks(np.arange(3) + 0.5); ax.set_yticks(np.arange(3) + 0.5)
        ax.set_xticklabels(range_labels, rotation=0, ha='center', fontsize=RANGE_LABEL_FONT_SIZE)
        ax.set_yticklabels(range_labels, rotation=90, ha='center', fontsize=RANGE_LABEL_FONT_SIZE)
        annotate_cm(ax, cm3)
    if mappable3 is not None:
        cb3 = fig3.colorbar(mappable3.collections[0], cax=cbar_ax3)
        cb3.ax.tick_params(labelsize=14)
        cb3.set_label("Row-normalized fraction", fontsize=16)
    # Add legend/explanation text
    fig3.text(0.5, 0.015, "Cell text: count and row-normalized fraction; color encodes row-normalized fraction.", ha='center', fontsize=14)
    p3_path = os.path.join(out_dir, "confusion_3class_2x3.png")
    fig3.tight_layout(rect=[0,0.02,0.97,1])
    fig3.savefig(p3_path, dpi=300)
    plt.close(fig3)

    # --- Write a small metrics summary ---
    summary = {
        "mae": {d: float(m) for d,m in zip(dim_names, mae)},
        "rmse": {d: float(r) for d,r in zip(dim_names, rmse)},
        "fig_5class": p5_path,
        "fig_3class": p3_path,
        "params": {"b0": b0, "b1": b1, "chunk_window": cw, "margin": marg, "persistence_length": k, "gt_persistence": bool(cfg.get("gt_persistence", False))},
        "checkpoint": ckpt_path,
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved:")
    print("  5-class figs:", p5_path)
    print("  3-class figs:", p3_path)
    print("  Summary:     ", os.path.join(out_dir, "summary.json"))

if __name__ == "__main__":
    main()
