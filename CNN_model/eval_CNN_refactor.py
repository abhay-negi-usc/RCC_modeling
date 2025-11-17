#!/usr/bin/env python3
"""
Refactored evaluation script with modular/reusable abstractions.
This version collapses repeated logic for:
  - checkpoint/data loading
  - model restore and prediction
  - class mapping (3/5-class, margins, persistence)
  - confusion matrix building (binary, 3-class, 5-class; per-dim + aggregated)
  - precision–recall point generation across margins (per-dim + overall)
  - plotting helpers (single matrix, 2x3 grids, PR curves, accuracy bars)
  - boundary filtering near class thresholds

Keep your existing EVAL_CONFIG at the bottom if you want to drop-in replace.
"""

from __future__ import annotations
import os, json, math, csv
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

# ---------- JSON helper ----------

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
    if isinstance(obj, _np.generic):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj

# ---------- Boundaries + margin semantics ----------

@dataclass(frozen=True)
class Boundaries:
    b0: float  # inner (neutral) half-width
    b1: float  # outer half-width for far classes

    @property
    def neutral(self) -> Tuple[float,float]:
        return (-self.b0, self.b0)

    def with_margin(self, margin: float) -> Tuple[float,float]:
        if margin >= self.b0:
            raise ValueError(f"margin {margin} >= b0 {self.b0}")
        return (-self.b0 + margin, self.b0 - margin)

@dataclass(frozen=True)
class Persistence:
    min_len: Optional[int] = None
    margin: float = 0.0           # shrink neutral band for persistence criterion
    any_dim: bool = False         # if True: any dimension persisting => positive

# ---------- Checkpoint + dataset restore ----------

from CNN_model.train_CNN import (
    TimeSeriesCNN,
    CONFIG as BASE_CONFIG,
    ChunkingConfig,
    WrenchPoseChunkDataset,
    zscore_fit,
    evaluate as eval_regression,
    make_loss_fn,
    compute_trial_segments,
    filter_segments_by_trials,
)


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if "config" not in ckpt or "model" not in ckpt:
        raise ValueError("Checkpoint missing 'config' or 'model'.")
    return ckpt["config"], ckpt["model"]


def build_val_dataset(df: pd.DataFrame, cfg: dict):
    trial_col = cfg["trial_col"]
    trials_arr = df[trial_col].values
    unique_trials = pd.unique(trials_arr)
    rng = np.random.default_rng(cfg["seed"])  # deterministic
    perm = rng.permutation(len(unique_trials))
    n_val = max(1, int(math.ceil(len(unique_trials) * cfg["val_split"])))
    val_ids = set(unique_trials[perm[:n_val]])

    N = len(df)
    segs = compute_trial_segments(trials_arr, 0, N)
    _ = filter_segments_by_trials(segs, trials_arr, val_ids)  # existence check

    chunk_cfg = ChunkingConfig(window=cfg["window"], stride=cfg["stride"], drop_last=True)

    wrench_all = df[cfg["wrench_cols"]].values.astype(np.float32)
    mu, sd = zscore_fit(wrench_all)

    val_ds = WrenchPoseChunkDataset(
        df,
        cfg["wrench_cols"], cfg["pose_cols"], cfg["label_cols"],
        chunk_cfg, mu, sd, start=0, end=N, trial_col=trial_col, allowed_trials=val_ids
    )
    return val_ds, val_ids, mu, sd


# ---------- Prediction helpers ----------

@torch.no_grad()
def predict_all(model: torch.nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        pred = model(xb)
        preds.append(pred.detach().cpu().numpy())
        gts.append(yb.detach().cpu().numpy())
    if not preds:
        return np.zeros((0,6), dtype=float), np.zeros((0,6), dtype=float)
    P = np.concatenate(preds, axis=0).reshape(-1,6)
    G = np.concatenate(gts,   axis=0).reshape(-1,6)
    return P, G


# ---------- Filtering timesteps near boundaries ----------

def boundary_filter_mask(gt_norm: np.ndarray, boundaries: Boundaries, buffer: float, mode: str = "any") -> np.ndarray:
    b0, b1 = boundaries.b0, boundaries.b1
    bvals = np.array([-b1, -b0, b0, b1], dtype=float)
    dist = np.min(np.abs(gt_norm[..., None] - bvals[None, None, :]), axis=2)  # [N,6]
    near = dist <= float(buffer)
    mode = str(mode).lower()
    return (~near.all(axis=1)) if mode == "all" else (~near.any(axis=1))


# ---------- Class mapping (vectorized) ----------

class ClassMapper:
    def __init__(self, boundaries: Boundaries):
        self.b = boundaries

    def map3(self, x: np.ndarray, margin: float = 0.0) -> np.ndarray:
        lo, hi = self.b.with_margin(margin)
        return np.where(x <= lo, 0, np.where(x >= hi, 2, 1))

    def map5(self, x: np.ndarray, margin: float = 0.0) -> np.ndarray:
        b0, b1 = self.b.b0, self.b.b1
        lo, hi = self.b.with_margin(margin)
        return np.where(x <= -b1, 0,
                np.where(x <= lo, 1,
                np.where(x <  hi, 2,
                np.where(x <   b1, 3, 4))))


# ---------- Confusion + metrics builders ----------

def confusion_binary(pred_pos: np.ndarray, gt_pos: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    pred_pos = pred_pos.astype(bool); gt_pos = gt_pos.astype(bool)
    TP = int(np.sum(pred_pos & gt_pos))
    TN = int(np.sum(~pred_pos & ~gt_pos))
    FP = int(np.sum(pred_pos & ~gt_pos))
    FN = int(np.sum(~pred_pos &  gt_pos))
    cm = np.array([[TN, FP], [FN, TP]], dtype=np.int64)
    def sdiv(a,b):
        return float(a/b) if b>0 else 0.0
    metrics = {
        "precision": sdiv(TP, TP+FP),
        "recall":    sdiv(TP, TP+FN),
        "specificity": sdiv(TN, TN+FP),
        "npv": sdiv(TN, TN+FN),
        "accuracy": sdiv(TP+TN, TP+TN+FP+FN),
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
    }
    return cm, metrics


def confusion_k(gt_cls: np.ndarray, pred_cls: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k,k), dtype=np.int64)
    for t, p in zip(gt_cls.reshape(-1), pred_cls.reshape(-1)):
        cm[int(t), int(p)] += 1
    return cm


# ---------- Persistence over a 1D boolean series ----------

def has_run_of_true(bool_arr: np.ndarray, min_len: Optional[int]) -> bool:
    if not min_len or min_len <= 1:
        return bool(np.any(bool_arr))
    x = np.concatenate(([False], bool_arr, [False])).astype(np.int8)
    diff = np.diff(x)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    if starts.size == 0:
        return False
    return int(np.max(ends - starts)) >= int(min_len)


def persistence_label_chunk(x_chunk: np.ndarray, mapper: ClassMapper, pers: Persistence) -> bool:
    """
    x_chunk: (T,6) normalized values for a single chunk window.
    Returns True if persistence exceedance detected according to 'pers'.
    """
    lo_m, hi_m = mapper.b.with_margin(pers.margin)
    # exceedance per-dim over time
    exceed = (x_chunk <= lo_m) | (x_chunk >= hi_m)  # (T,6)
    if pers.any_dim:
        series = exceed.any(axis=1)                 # (T,)
        return has_run_of_true(series, pers.min_len)
    else:
        # require persistence on each dim independently, then OR across dims
        return bool(np.any([has_run_of_true(exceed[:, d], pers.min_len) for d in range(exceed.shape[1])]))


# ---------- Plotting helpers ----------

# NOTE: Using seaborn here for parity with original script. You can swap to MPL-only
# if you prefer fewer deps.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


def plot_cm(cm: np.ndarray, title: str, labels_x: List[str], labels_y: List[str], out: str,
            normalize: str = "row", cmap: str = "Blues", annotate_counts: bool = True):
    cm = cm.astype(float)
    if normalize == "row":
        rs = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(rs>0, cm/rs, 0.0)
    elif normalize == "all":
        s = cm.sum()
        cm_norm = cm / s if s>0 else cm
    else:
        cm_norm = cm
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm_norm, ax=ax, annot=False, cmap=cmap, vmin=0.0, vmax=1.0,
                cbar_kws={"label": "Row-Normalized" if normalize=="row" else "Value"})
    if annotate_counts:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j+0.5, i+0.5, f"{int(cm[i,j])}\n{cm_norm[i,j]:.2f}",
                        ha="center", va="center", fontsize=8, color="#222")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels_x))+0.5, labels_x)
    ax.set_yticks(np.arange(len(labels_y))+0.5, labels_y, rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_grid(fn_make_ax, titles: List[str], out: str, suptitle: str, nrows: int = 2, ncols: int = 3):
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 7))
    axes_f = axes.flatten()
    for i, t in enumerate(titles):
        fn_make_ax(axes_f[i], t, i)
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_pr_curve(points: List[Dict[str,float]], title: str, out: str):
    if not points:
        return
    dfp = pd.DataFrame(points)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.lineplot(data=dfp, x="recall", y="precision", marker="o", ax=ax)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title)
    # annotate a few margins
    if len(dfp) > 0:
        for idx in np.linspace(0, len(dfp)-1, min(8, len(dfp))).astype(int):
            r = dfp.iloc[idx]
            ax.annotate(f"m={r['margin']:.2f}", (r['recall'], r['precision']),
                        textcoords="offset points", xytext=(6,-10), fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ---------- PR generation ----------

def pr_points_anydim(P: np.ndarray, G: np.ndarray, b: Boundaries, margins: Iterable[float]) -> List[Dict[str,float]]:
    pts = []
    for m in margins:
        lo, hi = b.with_margin(float(m))
        pred = ((P <= lo) | (P >= hi)).any(axis=1)
        gt   = ((G <= -b.b0) | (G >= b.b0)).any(axis=1)
        cm, metrics = confusion_binary(pred, gt)
        pts.append({"margin": float(m), "precision": metrics["precision"], "recall": metrics["recall"],
                    "TP": int(cm[1,1]), "FP": int(cm[0,1]), "TN": int(cm[0,0]), "FN": int(cm[1,0])})
    return sorted(pts, key=lambda d: d["recall"])  # monotone for nicer plots


def pr_points_per_dim(P: np.ndarray, G: np.ndarray, b: Boundaries, margins: Iterable[float], labels: List[str]) -> Dict[str, List[Dict[str,float]]]:
    out: Dict[str, List[Dict[str,float]]] = {}
    for d, lbl in enumerate(labels):
        pts = []
        P_d, G_d = P[:, d], G[:, d]
        for m in margins:
            lo, hi = b.with_margin(float(m))
            pred = (P_d <= lo) | (P_d >= hi)
            gt   = (G_d <= -b.b0) | (G_d >= b.b0)
            cm, metrics = confusion_binary(pred, gt)
            pts.append({"margin": float(m), "precision": metrics["precision"], "recall": metrics["recall"],
                        "TP": int(cm[1,1]), "FP": int(cm[0,1]), "TN": int(cm[0,0]), "FN": int(cm[1,0])})
        out[lbl] = sorted(pts, key=lambda d: d["recall"])
    return out


# ---------- Main eval pipeline (example usage) ----------

EVAL_CONFIG = {
    "checkpoint": None,                 # if None, use BASE_CONFIG['out_dir']/best_model_regression.pt
    "csv": None,                        # if None, use ckpt config CSV
    "device": None,
    "huber_delta": None,

    # margins + PR
    "margin": [0.0, 0.05, 0.10],
    "pr_margin_values": [0.0, 0.05, 0.10, 0.15, 0.2],

    # persistence settings
    "chunk_window": 100,
    "persistence_min": 10,
    "persistence_margin": 0.10,
    "persistence_anydim": False,

    # optional exclusion buffer around boundaries (GT-based)
    "exclude_boundary_buffer": None,
    "exclude_boundary_mode": "any",

    # plotting outputs
    "out_subdir": "eval_results_modular",
}


def main():
    # 1) resolve checkpoint + data
    default_ckpt = os.path.join(BASE_CONFIG["out_dir"], "best_model_regression.pt")
    ckpt_path = EVAL_CONFIG.get("checkpoint") or default_ckpt
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt_cfg, state = load_checkpoint(ckpt_path)

    csv_path = EVAL_CONFIG.get("csv") or ckpt_cfg["csv"]
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    print(f"Using CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = ckpt_cfg["wrench_cols"] + ckpt_cfg["pose_cols"] + ckpt_cfg["label_cols"] + [ckpt_cfg["trial_col"]]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    # 2) dataset + loader (reproduces original split)
    val_ds, val_ids, mu, sd = build_val_dataset(df, ckpt_cfg)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=ckpt_cfg["batch_size"], shuffle=False, drop_last=False)

    # 3) device + model
    device = torch.device(EVAL_CONFIG.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = TimeSeriesCNN(
        input_dim=12,
        hidden=ckpt_cfg["cnn_hidden"],
        layers=ckpt_cfg["cnn_layers"],
        kernel_size=ckpt_cfg["cnn_kernel"],
        dropout=ckpt_cfg["cnn_dropout"],
        dilation_base=ckpt_cfg["cnn_dilation_base"],
        bidirectional=ckpt_cfg["cnn_bidirectional"],
        num_tasks=6,
    ).to(device)
    model.load_state_dict(state)

    # 4) regression
    if EVAL_CONFIG.get("huber_delta") is not None:
        ckpt_cfg["huber_delta"] = float(EVAL_CONFIG["huber_delta"])
    loss_fn = make_loss_fn(ckpt_cfg)
    val_loss, mae_d, rmse_d, mae_mean, rmse_mean = eval_regression(model, loader, device, loss_fn)
    print(f"Regression — loss: {val_loss:.4f}, MAE: {mae_mean:.4f}, RMSE: {rmse_mean:.4f}")

    # 5) predictions (raw normalized target space)
    P_all, G_all = predict_all(model, loader, device)

    # 6) optional GT-boundary filtering
    filter_info = {"applied": False, "kept": int(P_all.shape[0])}
    b = Boundaries(b0=0.75, b1=0.90)  # keep consistent with paper results
    buf = EVAL_CONFIG.get("exclude_boundary_buffer")
    if buf is not None and float(buf) > 0:
        mask = boundary_filter_mask(G_all, b, float(buf), EVAL_CONFIG.get("exclude_boundary_mode", "any"))
        if mask.any():
            P_use, G_use = P_all[mask], G_all[mask]
            filter_info = {"applied": True, "kept": int(mask.sum())}
        else:
            P_use, G_use = P_all, G_all
    else:
        P_use, G_use = P_all, G_all

    # 7) per-dim + overall class mapping and confusions
    mapper = ClassMapper(b)
    out_dir = os.path.join(ckpt_cfg["out_dir"], EVAL_CONFIG.get("out_subdir", "eval_results_modular"))
    os.makedirs(out_dir, exist_ok=True)

    # 7a) overall binary confusion for a selected margin
    margins = EVAL_CONFIG.get("margin") or [0.0]
    m0 = float(margins[0])
    lo_m, hi_m = b.with_margin(m0)
    pred_pos = ((P_use <= lo_m) | (P_use >= hi_m)).any(axis=1)
    gt_pos   = ((G_use <= -b.b0) | (G_use >= b.b0)).any(axis=1)
    cm_bin, met_bin = confusion_binary(pred_pos, gt_pos)
    plot_cm(cm_bin, title=f"Overall Limit Detection (margin={m0:.2f})",
            labels_x=["Neg","Pos"], labels_y=["Neg","Pos"], out=os.path.join(out_dir, f"overall_limit_cm_m{m0:.2f}.png"))

    # 7b) per-dim 3-class + 5-class grids for that margin
    dim_labels = ["X","Y","Z","Roll","Pitch","Yaw"]

    def make_ax_cm3(ax, title_lbl, d):
        gt_c = mapper.map3(G_use[:, d], margin=0.0)
        pd_c = mapper.map3(P_use[:, d], margin=m0)
        cm3   = confusion_k(gt_c, pd_c, 3)
        rs = cm3.sum(axis=1, keepdims=True)
        cmn = np.where(rs>0, cm3/rs, 0.0)
        sns.heatmap(cmn, ax=ax, annot=np.array([[f"{cm3[0,0]}\n{cmn[0,0]:.2f}", f"{cm3[0,1]}\n{cmn[0,1]:.2f}", f"{cm3[0,2]}\n{cmn[0,2]:.2f}"],
                                                [f"{cm3[1,0]}\n{cmn[1,0]:.2f}", f"{cm3[1,1]}\n{cmn[1,1]:.2f}", f"{cm3[1,2]}\n{cmn[1,2]:.2f}"],
                                                [f"{cm3[2,0]}\n{cmn[2,0]:.2f}", f"{cm3[2,1]}\n{cmn[2,1]:.2f}", f"{cm3[2,2]}\n{cmn[2,2]:.2f}"]]),
                    fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
        ax.set_title(title_lbl); ax.set_xlabel("Pred 3-class"); ax.set_ylabel("True 3-class")
        ax.set_xticklabels(["Neg","Neu","Pos"]); ax.set_yticklabels(["Neg","Neu","Pos"], rotation=0)

    plot_grid(make_ax_cm3, dim_labels, os.path.join(out_dir, f"grid_cm3_m{m0:.2f}.png"),
              suptitle=f"Per-Dimension 3-Class Confusions (m={m0:.2f})")

    def make_ax_cm5(ax, title_lbl, d):
        gt_c = mapper.map5(G_use[:, d], margin=0.0)
        pd_c = mapper.map5(P_use[:, d], margin=m0)
        cm5   = confusion_k(gt_c, pd_c, 5)
        rs = cm5.sum(axis=1, keepdims=True)
        cmn = np.where(rs>0, cm5/rs, 0.0)
        annot = [[f"{cm5[i,j]}\n{cmn[i,j]:.2f}" for j in range(5)] for i in range(5)]
        sns.heatmap(cmn, ax=ax, annot=np.array(annot), fmt="", cmap="Blues", vmin=0.0, vmax=1.0, cbar=False)
        ax.set_title(title_lbl); ax.set_xlabel("Pred 5-class"); ax.set_ylabel("True 5-class")
        ax.set_xticklabels(["-2","-1","0","+1","+2"]); ax.set_yticklabels(["-2","-1","0","+1","+2"], rotation=0)

    plot_grid(make_ax_cm5, dim_labels, os.path.join(out_dir, f"grid_cm5_m{m0:.2f}.png"),
              suptitle=f"Per-Dimension 5-Class Confusions (m={m0:.2f})")

    # 8) PR curves
    margins_valid = np.asarray(EVAL_CONFIG.get("pr_margin_values", [0.0, 0.05, 0.1]), dtype=float)
    margins_valid = margins_valid[(margins_valid >= 0) & (margins_valid < b.b0)]

    pr_overall = pr_points_anydim(P_use, G_use, b, margins_valid)
    plot_pr_curve(pr_overall, title="Precision–Recall (All dims appended)", out=os.path.join(out_dir, "pr_overall.png"))
    with open(os.path.join(out_dir, "pr_overall_points.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["margin","precision","recall","TP","FP","TN","FN"])
        for r in pr_overall: w.writerow([r[k] for k in ["margin","precision","recall","TP","FP","TN","FN"]])

    pr_per_dim = pr_points_per_dim(P_use, G_use, b, margins_valid, dim_labels)
    for lbl, pts in pr_per_dim.items():
        plot_pr_curve(pts, title=f"Precision–Recall (Dim {lbl})", out=os.path.join(out_dir, f"pr_dim_{lbl}.png"))
        with open(os.path.join(out_dir, f"pr_dim_{lbl}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["margin","precision","recall","TP","FP","TN","FN"])
            for r in pts: w.writerow([r[k] for k in ["margin","precision","recall","TP","FP","TN","FN"]])

    # 9) Optional: persistence over chunk windows (example pipeline)
    pers = Persistence(min_len=EVAL_CONFIG.get("persistence_min"), margin=float(EVAL_CONFIG.get("persistence_margin", 0.0)), any_dim=bool(EVAL_CONFIG.get("persistence_anydim", False)))
    # If you want to evaluate persistence-based labels, iterate each chunk in the loader and apply persistence_label_chunk on model outputs.
    # Keeping minimal here to illustrate abstraction rather than re-implement the entire chunking eval.

    # 10) Save summary JSON
    summary = {
        "regression": {"loss": float(val_loss), "mae_mean": float(mae_mean), "rmse_mean": float(rmse_mean),
                        "mae_per_dim": make_jsonable(mae_d), "rmse_per_dim": make_jsonable(rmse_d)},
        "filter": filter_info,
        "binary_margin_m0": {"margin": m0, **make_jsonable(met_bin)},
    }
    with open(os.path.join(out_dir, "metrics_eval.json"), "w") as f:
        json.dump(make_jsonable(summary), f, indent=2)
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
