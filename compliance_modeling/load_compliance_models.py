#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate previously saved compliance models from `compliance_modeling.py` by computing
R² per dimension (Fx,Fy,Fz,Tx,Ty,Tz) and the mean R² for each model.

Models supported (matching save format from the fitter):
  1) Linear (no bias):           w ≈ K x                       -> <stem>.csv
  2) Affine (with bias):         w ≈ K x + c                   -> <stem>_affine_K.csv, <stem>_affine_bias.csv
  3) Quadratic (poly2):          w ≈ K x + 1/2 xᵀ H x + c      -> <stem>_poly2_K.csv, <stem>_poly2_bias.csv, <stem>_poly2_H.npz
  4) Piecewise-Affine (K regs):  w ≈ K_k x + c_k (hard gating) -> <stem>_pwl_info.npz + per-region files
  5) Piecewise-Quadratic:        w ≈ K_k x + 1/2 xᵀ H_k x + c_k -> <stem>_pwq_info.npz + per-region files

This script:
  - Loads the dataset using the same column naming and preprocessing choices.
  - Optionally converts angles from degrees to radians and demeans (to match training).
  - For each model found on disk, scans integer lags in [-lag_scan_eval, +lag_scan_eval],
    applies the fixed saved parameters, and reports the best (highest mean R²) result.
  - Saves a compact CSV and JSON summary and prints a pretty table.

Outputs:
  <out_dir>/compliance_eval_r2/r2_summary.csv
  <out_dir>/compliance_eval_r2/r2_summary.json
"""

import os
import json
import glob
import numpy as np
import pandas as pd

# ----------------------- CONFIG -----------------------
CONFIG = {
    # --- Data & preprocessing (match your fitter) ---
    "csv": "./data/RCC_kuka_ati_15_trials.csv",
    "wrench_cols": ["Fx_ati", "Fy_ati", "Fz_ati", "Tx_ati", "Ty_ati", "Tz_ati"],
    "pose_cols":   ["X", "Y", "Z", "C", "B", "A"],
    "angles_deg": True,    # convert pose angles (C,B,A) deg→rad
    "demean": True,        # subtract mean from both x and w

    # --- Where models were saved (same 'stem' as in compliance_modeling.py) ---
    # If your fitter wrote to CONFIG['out'] = "./compliance_modeling/results/stiffness_matrix.csv",
    # then stem = "./compliance_modeling/results/stiffness_matrix"
    "stem": "./compliance_modeling/results/stiffness_matrix",

    # --- Evaluation ---
    "lag_scan_eval": 10,   # evaluate fixed params across lags in [-L, +L]
    "out_dir": "./checkpoints",  # where to drop summary CSV/JSON
}

DIM_NAMES = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


# ----------------------- Utilities -----------------------
def ensure_out_dir(base_out: str) -> str:
    out = os.path.join(base_out, "compliance_eval_r2")
    os.makedirs(out, exist_ok=True)
    return out

def to_radians(X: np.ndarray, angle_indices=(3,4,5)) -> np.ndarray:
    X = X.copy()
    X[:, angle_indices] = np.deg2rad(X[:, angle_indices])
    return X

def apply_lag(U: np.ndarray, Y: np.ndarray, lag: int):
    if lag == 0:
        return U, Y
    if lag > 0:
        return U[:-lag, :], Y[lag:, :]
    L = -lag
    return U[L:, :], Y[:-L, :]

def r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return R² for each column, robust to zero-variance columns."""
    r2 = np.zeros(y_true.shape[1], dtype=float)
    for d in range(y_true.shape[1]):
        yt = y_true[:, d]
        yp = y_pred[:, d]
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]; yp = yp[mask]
        if yt.size == 0:
            r2[d] = 0.0
            continue
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2[d] = 0.0 if ss_tot <= 1e-12 else (1.0 - ss_res / ss_tot)
    return r2

def load_csv(cfg):
    df = pd.read_csv(cfg["csv"])
    X = df[cfg["pose_cols"]].to_numpy(float)
    W = df[cfg["wrench_cols"]].to_numpy(float)
    if cfg.get("angles_deg", False):
        X = to_radians(X, (3,4,5))
    if cfg.get("demean", True):
        X = X - X.mean(axis=0, keepdims=True)
        W = W - W.mean(axis=0, keepdims=True)
    # drop rows with NaNs in either
    m = np.isfinite(X).all(axis=1) & np.isfinite(W).all(axis=1)
    return X[m], W[m]

# ----------------------- Model loaders & predictors -----------------------
def load_linear(stem: str):
    path = f"{stem}.csv"
    if not os.path.isfile(path): return None
    K = pd.read_csv(path, index_col=0).to_numpy(float)  # (6,6)
    return {"K": K}

def pred_linear(K: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ K.T

def load_affine(stem: str):
    pk = f"{stem}_affine_K.csv"
    pb = f"{stem}_affine_bias.csv"
    if not (os.path.isfile(pk) and os.path.isfile(pb)): return None
    K = pd.read_csv(pk, index_col=0).to_numpy(float)
    b = pd.read_csv(pb, index_col=0).to_numpy(float).reshape(-1)  # saved as column vector per wrench dim
    return {"K": K, "b": b}

def pred_affine(K: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return X @ K.T + b.reshape(1, -1)

def load_poly2(stem: str):
    pk = f"{stem}_poly2_K.csv"
    pb = f"{stem}_poly2_bias.csv"
    pH = f"{stem}_poly2_H.npz"
    if not (os.path.isfile(pk) and os.path.isfile(pb) and os.path.isfile(pH)): return None
    K = pd.read_csv(pk, index_col=0).to_numpy(float)
    b = pd.read_csv(pb, index_col=0).to_numpy(float).reshape(-1)
    H = dict(np.load(pH, allow_pickle=True))["H"]  # (6,6,6)
    return {"K": K, "b": b, "H": H}

def pred_poly2(K: np.ndarray, b: np.ndarray, H: np.ndarray, X: np.ndarray) -> np.ndarray:
    # w = X @ K^T + 0.5 * quad + b
    lin = X @ K.T
    # quad term: for each output t, w_t += 0.5 * x^T H[t] x
    # efficient batch using einsum
    quad = 0.5 * np.einsum("nd,tde,ne->nt", X, H, X)  # (N,6)
    return lin + quad + b.reshape(1, -1)

def _load_region_param_csv(path_csv: str) -> np.ndarray:
    return pd.read_csv(path_csv, index_col=0).to_numpy(float)

def load_pwl(stem: str):
    pinfo = f"{stem}_pwl_info.npz"
    if not os.path.isfile(pinfo): return None
    info = dict(np.load(pinfo, allow_pickle=True))
    Kreg = int(info.get("K", 0))
    centroids = info.get("centroids")
    scales = info.get("scales")
    gating_space = str(info.get("gating_space", "pose"))
    # Collect region parameter files that follow the fitter's naming
    regions = []
    for r in range(Kreg):
        pk = f"{stem}_pwl_region{r}_K.csv"
        pb = f"{stem}_pwl_region{r}_bias.csv"
        if os.path.isfile(pk) and os.path.isfile(pb):
            regions.append({"K": _load_region_param_csv(pk), "b": _load_region_param_csv(pb).reshape(-1)})
        else:
            regions.append(None)  # missing region => will fallback later
    return {"K": Kreg, "centroids": centroids, "scales": scales,
            "gating_space": gating_space, "regions": regions}

def pred_pwl(model, X: np.ndarray, W_fallback: np.ndarray = None):
    """
    Hard gating by nearest centroid (z-scored by 'scales').
    For missing region params, fall back to global affine prediction if provided (W_fallback),
    otherwise zeros.
    """
    C = np.asarray(model["centroids"])
    S = np.asarray(model["scales"])
    regs = model["regions"]
    # distance in normalized coords
    d2 = ((X[:, None, :] - C[None, :, :]) / (S[None, :, :] + 1e-8)) ** 2
    idx = np.argmin(d2.sum(axis=2), axis=1)  # (N,)

    Yhat = np.zeros((X.shape[0], 6), dtype=float)
    for k in range(C.shape[0]):
        mask = (idx == k)
        if not np.any(mask): continue
        if regs[k] is None:
            if W_fallback is not None:
                Yhat[mask] = W_fallback[mask]
            else:
                Yhat[mask] = 0.0
        else:
            Kk = regs[k]["K"]; bk = regs[k]["b"]
            Yhat[mask] = X[mask] @ Kk.T + bk.reshape(1, -1)
    return Yhat

def load_pwq(stem: str):
    pinfo = f"{stem}_pwq_info.npz"
    if not os.path.isfile(pinfo): return None
    info = dict(np.load(pinfo, allow_pickle=True))
    Kreg = int(info.get("K", 0))
    centroids = info.get("centroids")
    scales = info.get("scales")
    gating_space = str(info.get("gating_space", "pose"))
    regions = []
    for r in range(Kreg):
        pk = f"{stem}_pwq_region{r}_K.csv"
        pb = f"{stem}_pwq_region{r}_bias.csv"
        pH = f"{stem}_pwq_region{r}_H.npz"
        if os.path.isfile(pk) and os.path.isfile(pb) and os.path.isfile(pH):
            regions.append({
                "K": _load_region_param_csv(pk),
                "b": _load_region_param_csv(pb).reshape(-1),
                "H": dict(np.load(pH, allow_pickle=True))["H"],
            })
        else:
            regions.append(None)
    return {"K": Kreg, "centroids": centroids, "scales": scales,
            "gating_space": gating_space, "regions": regions}

def pred_pwq(model, X: np.ndarray, W_fallback: np.ndarray = None):
    C = np.asarray(model["centroids"])
    S = np.asarray(model["scales"])
    regs = model["regions"]
    d2 = ((X[:, None, :] - C[None, :, :]) / (S[None, :, :] + 1e-8)) ** 2
    idx = np.argmin(d2.sum(axis=2), axis=1)

    Yhat = np.zeros((X.shape[0], 6), dtype=float)
    for k in range(C.shape[0]):
        mask = (idx == k)
        if not np.any(mask): continue
        if regs[k] is None:
            if W_fallback is not None:
                Yhat[mask] = W_fallback[mask]
            else:
                Yhat[mask] = 0.0
        else:
            Kk = regs[k]["K"]; bk = regs[k]["b"]; Hk = regs[k]["H"]
            lin = X[mask] @ Kk.T
            quad = 0.5 * np.einsum("nd,tde,ne->nt", X[mask], Hk, X[mask])
            Yhat[mask] = lin + quad + bk.reshape(1, -1)
    return Yhat

# ----------------------- Evaluation core -----------------------
def eval_with_lag_scan(predict_fn, params, X, W, lag_scan):
    best = {"lag": 0, "r2": None, "mean_r2": -1.0}
    for lag in range(-lag_scan, lag_scan + 1):
        Xl, Wl = apply_lag(X, W, lag)
        Yhat = predict_fn(params, Xl)
        r2 = r2_per_dim(Wl, Yhat)
        m = float(np.mean(r2))
        if m > best["mean_r2"]:
            best = {"lag": lag, "r2": r2, "mean_r2": m}
    return best

def eval_with_lag_scan_pair(predict_fn, params, fallback_pred, X, W, lag_scan):
    """For piecewise models that may need a global fallback prediction."""
    best = {"lag": 0, "r2": None, "mean_r2": -1.0}
    for lag in range(-lag_scan, lag_scan + 1):
        Xl, Wl = apply_lag(X, W, lag)
        Fb = None if fallback_pred is None else fallback_pred(Xl)
        Yhat = predict_fn(params, Xl, Fb)
        r2 = r2_per_dim(Wl, Yhat)
        m = float(np.mean(r2))
        if m > best["mean_r2"]:
            best = {"lag": lag, "r2": r2, "mean_r2": m}
    return best

# ----------------------- Main -----------------------
def main():
    cfg = CONFIG
    out_dir = ensure_out_dir(cfg["out_dir"])
    csv_path = cfg["csv"]
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    X, W = load_csv(cfg)
    stem = cfg["stem"]
    L = int(cfg.get("lag_scan_eval", 10))

    rows = []

    # --- Linear ---
    lin = load_linear(stem)
    if lin is not None:
        res = eval_with_lag_scan(lambda p, Xl: pred_linear(p["K"], Xl), lin, X, W, L)
        row = {"model": "Linear", "mean_R2": res["mean_r2"], "lag": res["lag"]}
        for i, d in enumerate(DIM_NAMES): row[f"R2_{d}"] = res["r2"][i]
        rows.append(row)

    # --- Affine ---
    aff = load_affine(stem)
    if aff is not None:
        res = eval_with_lag_scan(lambda p, Xl: pred_affine(p["K"], p["b"], Xl), aff, X, W, L)
        row = {"model": "Affine", "mean_R2": res["mean_r2"], "lag": res["lag"]}
        for i, d in enumerate(DIM_NAMES): row[f"R2_{d}"] = res["r2"][i]
        rows.append(row)
        # handy fallback predictor for piecewise models
        def aff_pred(Xl):
            return pred_affine(aff["K"], aff["b"], Xl)
    else:
        aff_pred = None

    # --- Poly2 (Quadratic) ---
    poly = load_poly2(stem)
    if poly is not None:
        res = eval_with_lag_scan(lambda p, Xl: pred_poly2(p["K"], p["b"], p["H"], Xl), poly, X, W, L)
        row = {"model": "Poly2", "mean_R2": res["mean_r2"], "lag": res["lag"]}
        for i, d in enumerate(DIM_NAMES): row[f"R2_{d}"] = res["r2"][i]
        rows.append(row)

    # --- Piecewise-Affine ---
    pwl = load_pwl(stem)
    if pwl is not None:
        res = eval_with_lag_scan_pair(lambda p, Xl, Fb: pred_pwl(p, Xl, Fb), pwl, aff_pred, X, W, L)
        row = {"model": f"PWL(K={pwl['K']})", "mean_R2": res["mean_r2"], "lag": res["lag"]}
        for i, d in enumerate(DIM_NAMES): row[f"R2_{d}"] = res["r2"][i]
        rows.append(row)

    # --- Piecewise-Quadratic ---
    pwq = load_pwq(stem)
    if pwq is not None:
        res = eval_with_lag_scan_pair(lambda p, Xl, Fb: pred_pwq(p, Xl, Fb), pwq, aff_pred, X, W, L)
        row = {"model": f"PWQ(K={pwq['K']})", "mean_R2": res["mean_r2"], "lag": res["lag"]}
        for i, d in enumerate(DIM_NAMES): row[f"R2_{d}"] = res["r2"][i]
        rows.append(row)

    # --- Summarize ---
    if not rows:
        print("[WARN] No saved models found for the provided stem. Nothing to evaluate.")
        return

    table = pd.DataFrame(rows)
    # Column order
    col_order = ["model", "lag", "mean_R2"] + [f"R2_{d}" for d in DIM_NAMES]
    table = table[col_order]

    # Save
    csv_out = os.path.join(out_dir, "r2_summary.csv")
    json_out = os.path.join(out_dir, "r2_summary.json")
    table.to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump({"results": rows}, f, indent=2)

    # Print nicely
    print("\n=== Compliance Models: R² Summary (best lag per model) ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(table.to_string(index=False, justify="center", float_format=lambda x: f"{x:0.4f}"))
    print(f"\nSaved CSV:  {csv_out}")
    print(f"Saved JSON: {json_out}")


if __name__ == "__main__":
    main()
