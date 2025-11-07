#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compliance modeling with hysteresis (Prandtl–Ishlinskii play operator basis).

Model:
    x_t ≈ C·w_t + sum_{j=1..6} sum_{m=1..M} H[j,m] · play_r[j,m]( w_t[j] ) + b

Where:
    - w_t ∈ R^6 is wrench at time t
    - x_t ∈ R^6 is deflection/pose at time t (small-angle: rx,ry,rz)
    - play_r(·) is the discrete 'play' operator with threshold r (rate independent)
    - C ∈ R^{6×6}, H[j,m] ∈ R^{6} and b ∈ R^{6}

Features:
    [ W | PLAY(Fx) | ... | PLAY(Tz) | (optional branch/rate) | 1 ]
All robustly fit with Huber IRWLS + ridge. A lag scan aligns W→X.

Keeps CONFIG dict (no argparse).
"""

import os
import json
import numpy as np
import pandas as pd

CONFIG = {
    # ---- IO ----
    "csv": "./data/RCC_kuka_ati_15_trials.csv",
    "wrench_cols": ["Fx_ati", "Fy_ati", "Fz_ati", "Tx_ati", "Ty_ati", "Tz_ati"],
    "pose_cols":    ["X", "Y", "Z", "C", "B", "A"],
    "angles_deg": True,          # convert pose angles (last 3) from deg→rad
    "demean": True,              # mean-center W and X (recommended)
    "out_dir": "./compliance_hysteresis",  # output folder

    # ---- Alignment & Robustness ----
    "lag_scan": 10,              # search lags in [-lag_scan, +lag_scan]
    "ridge": 1e-3,               # L2 regularization
    "huber_delta": 1.0,          # 0 to disable Huber IRWLS (pure ridge)
    "max_irwls_iters": 100,

    # ---- Linear post-processing (baseline C only) ----
    "symmetrize_linear_C": False,
    "spd_project_linear_C": True,

    # ---- Hysteresis Basis (PI) ----
    "hyst_enable": True,         # turn on play-operator features
    "hyst_per_input": 8,         # number M of hysterons per input channel
    "hyst_radius_min_frac": 0.05,# radii span [min_frac, max_frac] * (range of that input)
    "hyst_radius_max_frac": 0.80,
    "hyst_radius_mode": "quantile",  # "linear" or "quantile" spacing across |Δu| magnitude
    "hyst_normalize": True,      # scale each play feature to unit std for conditioning

    # ---- Optional extra dynamics (cheap hysteresis proxies) ----
    "use_branch_features": True, # separate loading vs unloading via sign(d/dt)·w
    "use_rate_features": True,   # include finite difference dw/dt as extra linear terms
    "rate_window": 1,            # dt steps for dw/dt (1 = adjacent difference)

    # ---- Repro ----
    "seed": 0,
}

# ============================================================
# Utilities
# ============================================================

def to_radians(X, angle_indices):
    X = X.copy()
    X[:, angle_indices] = np.deg2rad(X[:, angle_indices])
    return X

def standardize_missing(W, X):
    mask = np.isfinite(W).all(axis=1) & np.isfinite(X).all(axis=1)
    return W[mask], X[mask]

def apply_lag(W, X, lag):
    if lag == 0:
        return W, X
    if lag > 0:
        return W[:-lag, :], X[lag:, :]
    else:
        L = -lag
        return W[L:, :], X[:-L, :]

def huber_weights(residuals, delta):
    # residuals: (N,6)
    r = np.sqrt((residuals**2).sum(axis=1) + 1e-12)
    w = np.ones_like(r)
    big = r > delta
    w[big] = delta / r[big]
    return np.clip(w, 1e-6, None)

def weighted_ridge(Phi, X, w=None, ridge=0.0, ridge_mask=None):
    """
    Solve A = argmin ||diag(sqrt(w))(X - Phi A)||_F^2 + ridge*||A||_F^2
    Phi: (N,D), X: (N,6), A: (D,6)
    ridge_mask: length-D weights (0 to avoid penalizing bias column).
    """
    if w is not None:
        sw = np.sqrt(w).reshape(-1, 1)
        Pw = Phi * sw
        Xw = X * sw
    else:
        Pw, Xw = Phi, X

    G = Pw.T @ Pw
    if ridge > 0:
        if ridge_mask is None:
            G += ridge * np.eye(Phi.shape[1])
        else:
            G += ridge * np.diag(ridge_mask)
    B = Pw.T @ Xw
    A = np.linalg.solve(G, B)
    return A

def r2_multitarget(Phi, X, A):
    pred = Phi @ A
    ss_res = ((X - pred)**2).sum(axis=0)
    ss_tot = ((X - X.mean(axis=0, keepdims=True))**2).sum(axis=0) + 1e-12
    return (1.0 - ss_res/ss_tot), pred

def project_spd(M, eps=1e-9):
    Ms = 0.5*(M + M.T)
    vals, vecs = np.linalg.eigh(Ms)
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(vals) @ vecs.T

# ============================================================
# Hysteresis (Prandtl–Ishlinskii) Play Operator
# ============================================================

def play_operator(u, r):
    """
    Discrete play operator (rate-independent):
        y_t = y_{t-1} + clip(u_t - y_{t-1}, -r, r)
    Ensures |u_t - y_t| <= r at all t.
    u: (N,) numpy array, r>0 threshold
    returns y: (N,)
    """
    y = np.zeros_like(u)
    if len(u) == 0:
        return y
    y[0] = np.clip(u[0], -r, r)
    for t in range(1, len(u)):
        y[t] = y[t-1] + np.clip(u[t] - y[t-1], -r, r)
    return y

def make_hysteresis_features(W, cfg):
    """
    Build play-operator features per input channel with multiple radii.
    Returns:
        Hfeat: (N, 6*M) stacked as [Fx_r1..rM | Fy_r1..rM | ... | Tz_r1..rM]
        radii: list of length 6, each an array of radii used for that input
        col_slices: list of 6 slices showing columns for each input's play features
        scales: per-column std used for normalization if hyst_normalize=True
    """
    if not cfg["hyst_enable"]:
        N = W.shape[0]
        return np.zeros((N,0)), [np.array([]) for _ in range(6)], [], np.array([])

    N, D = W.shape
    M = int(cfg["hyst_per_input"])
    rminf = float(cfg["hyst_radius_min_frac"])
    rmaxf = float(cfg["hyst_radius_max_frac"])
    mode = cfg.get("hyst_radius_mode", "quantile")
    normalize = bool(cfg.get("hyst_normalize", True))

    features = []
    radii_all = []
    col_slices = []
    start = 0
    rng = np.random.default_rng(cfg.get("seed", 0))

    for j in range(D):  # per input channel
        u = W[:, j]
        urange = float(np.max(u) - np.min(u) + 1e-12)
        rmin = rminf * urange
        rmax = rmaxf * urange
        if mode == "linear":
            radii = np.linspace(rmin, rmax, num=M)
        else:
            # use quantiles of absolute increments to space thresholds
            du = np.abs(np.diff(u, prepend=u[0]))
            q = np.linspace(0.1, 0.9, num=M)
            qv = np.quantile(du, q) + 1e-12
            scale = (rmax - rmin)/(np.max(qv) - np.min(qv) + 1e-12)
            radii = rmin + scale*(qv - np.min(qv))
        radii = np.clip(radii, 1e-9, None)
        radii_all.append(radii)

        cols = []
        for r in radii:
            y = play_operator(u, r)
            cols.append(y[:, None])
        cols = np.hstack(cols) if cols else np.zeros((N,0))
        features.append(cols)
        col_slices.append(slice(start, start + cols.shape[1]))
        start += cols.shape[1]

    Hfeat = np.hstack(features) if features else np.zeros((N,0))

    # Normalize play features column-wise for conditioning (keeps bias separate)
    if normalize and Hfeat.shape[1] > 0:
        std = Hfeat.std(axis=0, ddof=0)
        std = np.where(std < 1e-8, 1.0, std)
        Hfeat = Hfeat / std
        scales = std
    else:
        scales = np.ones(Hfeat.shape[1], dtype=float)

    return Hfeat, radii_all, col_slices, scales

def make_branch_and_rate(W, cfg):
    """
    Optional extra features:
      - branch: sign(dw/dt) * w  (loading vs unloading separation)
      - rate:   dw/dt (finite differences)
    Returns (BranchFeat, RateFeat). Either may have 0 columns.
    """
    N, D = W.shape
    dw = np.diff(W, axis=0, prepend=W[[0], :])
    sgn = np.sign(dw)
    branch = sgn * W if cfg.get("use_branch_features", True) else np.zeros((N,0))
    rate = dw if cfg.get("use_rate_features", True) else np.zeros((N,0))

    # simple multi-step derivative if requested
    k = int(cfg.get("rate_window", 1))
    if rate.shape[1] > 0 and k > 1:
        for j in range(D):
            rate[:, j] = np.concatenate([
                np.zeros(k),
                W[k:, j] - W[:-k, j]
            ])
    return branch, rate

# ============================================================
# Feature Builder (affine + hysteresis + optionals)
# ============================================================

def build_features(W, cfg):
    """
    Returns:
        Phi: (N, D_total)
        meta: dict with keys describing layout for saving & inference
    Order:
        [ W(6)
          | HYST(6*M)
          | BRANCH(6?) | RATE(6?)
          | BIAS(1) ]
    """
    N, D = W.shape
    parts = []
    layout = {}

    # 1) Linear terms
    parts.append(W)
    layout["lin_slice"] = slice(0, D)
    col = D

    # 2) Hysteresis terms
    Hfeat, radii, hyst_slices, hyst_scales = make_hysteresis_features(W, cfg)
    if Hfeat.shape[1] > 0:
        parts.append(Hfeat)
        layout["hyst_slice"] = slice(col, col + Hfeat.shape[1])
        layout["hyst_per_input_slices"] = hyst_slices
        layout["hyst_radii"] = radii
        layout["hyst_scales"] = hyst_scales.tolist()
        col += Hfeat.shape[1]
    else:
        layout["hyst_slice"] = slice(col, col)
        layout["hyst_per_input_slices"] = []
        layout["hyst_radii"] = [ [] for _ in range(D) ]
        layout["hyst_scales"] = []

    # 3) Branch & Rate
    Br, Rt = make_branch_and_rate(W, cfg)
    if Br.shape[1] > 0:
        parts.append(Br)
        layout["branch_slice"] = slice(col, col + Br.shape[1])
        col += Br.shape[1]
    else:
        layout["branch_slice"] = slice(col, col)

    if Rt.shape[1] > 0:
        parts.append(Rt)
        layout["rate_slice"] = slice(col, col + Rt.shape[1])
        col += Rt.shape[1]
    else:
        layout["rate_slice"] = slice(col, col)

    # 4) Bias
    bias = np.ones((N, 1))
    parts.append(bias)
    layout["bias_index"] = col
    col += 1

    Phi = np.hstack(parts)
    ridge_mask = np.ones(Phi.shape[1], dtype=float)
    ridge_mask[layout["bias_index"]] = 0.0  # do not penalize bias
    return Phi, ridge_mask, layout

# ============================================================
# Fitting with lag-scan (Huber IRWLS + ridge)
# ============================================================

def fit_with_lag_scan(W, X, cfg):
    best = (-np.inf, 0, None, None, None, None)  # score, lag, A, r2, layout, Phi_at_best
    for lag in range(-cfg['lag_scan'], cfg['lag_scan'] + 1):
        Wl, Xl = apply_lag(W, X, lag)
        if len(Wl) < 10:
            continue
        Phi, ridge_mask, layout = build_features(Wl, cfg)
        # IRWLS
        if cfg['huber_delta'] > 0:
            A = weighted_ridge(Phi, Xl, None, cfg['ridge'], ridge_mask)
            for _ in range(cfg['max_irwls_iters']):
                res = Xl - Phi @ A
                w = huber_weights(res, cfg['huber_delta'])
                A_new = weighted_ridge(Phi, Xl, w, cfg['ridge'], ridge_mask)
                if np.linalg.norm(A_new - A) / (np.linalg.norm(A) + 1e-12) < 1e-5:
                    A = A_new
                    break
                A = A_new
        else:
            A = weighted_ridge(Phi, Xl, None, cfg['ridge'], ridge_mask)

        r2, _ = r2_multitarget(Phi, Xl, A)
        score = float(np.mean(r2))
        if score > best[0]:
            best = (score, lag, A, r2, layout, Phi)
    return best  # lag, A, r2, layout, etc.

# ============================================================
# Main
# ============================================================

def main():
    cfg = CONFIG
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # Load data
    df = pd.read_csv(cfg["csv"])
    W = df[cfg["wrench_cols"]].to_numpy(float)  # (N,6)
    X = df[cfg["pose_cols"]].to_numpy(float)    # (N,6)

    # Convert angles to radians if specified
    if cfg["angles_deg"]:
        X = to_radians(X, [3,4,5])

    # Drop NaNs
    W, X = standardize_missing(W, X)

    # Demean (common for PI identification to improve conditioning)
    W_mean = W.mean(axis=0, keepdims=True)
    X_mean = X.mean(axis=0, keepdims=True)
    if cfg["demean"]:
        Wc = W - W_mean
        Xc = X - X_mean
    else:
        Wc, Xc = W, X

    # Fit with lag-scan
    score, best_lag, A, r2, layout, Phi_best = fit_with_lag_scan(Wc, Xc, cfg)
    if A is None:
        raise RuntimeError("Fitting failed: not enough data or ill-conditioning.")

    # Extract components
    lin_slice   = layout["lin_slice"]
    hyst_slice  = layout["hyst_slice"]
    bias_index  = layout["bias_index"]
    branch_slice= layout["branch_slice"]
    rate_slice  = layout["rate_slice"]

    C_lin = A[lin_slice, :].T             # (6,6)
    b     = A[bias_index, :].reshape(6)   # (6,)
    H_mat = A[hyst_slice, :]              # (6*M, 6) (may be empty)
    BrW   = A[branch_slice, :]            # (6,6) if enabled else (0,6)
    RtW   = A[rate_slice, :]              # (6,6) if enabled else (0,6)

    # Optional post-processing of linear C only
    if cfg["symmetrize_linear_C"]:
        C_lin = 0.5*(C_lin + C_lin.T)
    if cfg["spd_project_linear_C"]:
        C_lin = project_spd(C_lin)

    # Report
    print(f"[HYST] Best lag: {best_lag}  mean R2: {np.mean(r2):.4f}")
    print(f"[HYST] R2 per dimension: {np.round(r2,4).tolist()}")
    print("[HYST] Linear baseline C:\n",
          np.array2string(C_lin, formatter={'float_kind':lambda x: f'{x: .6e}'}))
    print("[HYST] Bias b:\n",
          np.array2string(b, formatter={'float_kind':lambda x: f'{x: .6e}'}))
    if H_mat.shape[0] > 0:
        print(f"[HYST] Hysteresis feature count: {H_mat.shape[0]}")

    # Save
    stem = os.path.join(cfg["out_dir"], "compliance_hysteresis")

    pd.DataFrame(C_lin,
        index=['x','y','z','rx','ry','rz'],
        columns=['Fx','Fy','Fz','Tx','Ty','Tz']).to_csv(f"{stem}_C_linear.csv", float_format="%.10e")
    pd.DataFrame(b[None,:],
        index=['bias'], columns=['x','y','z','rx','ry','rz']).T.to_csv(f"{stem}_bias.csv", float_format="%.10e")

    # Pack hysteresis blocks per-input for readability
    hyst_blocks = []
    if H_mat.shape[0] > 0:
        per_input_slices = layout["hyst_per_input_slices"]
        for j, sl in enumerate(per_input_slices):
            Hj = H_mat[sl, :]  # shape (M_j, 6)
            hyst_blocks.append(Hj)
            out_j = f"{stem}_H_input{j}.csv"
            pd.DataFrame(Hj, columns=['x','y','z','rx','ry','rz']).to_csv(out_j, float_format="%.10e")
        print(f"[HYST] Saved {len(hyst_blocks)} per-input hysteresis weight CSVs.")

    # Save metadata (JSON) and full coefficient matrix (NPZ)
    meta = dict(
        cfg=cfg,
        best_lag=int(best_lag),
        r2=r2.tolist(),
        layout={
            "lin_slice": [lin_slice.start, lin_slice.stop],
            "hyst_slice": [hyst_slice.start, hyst_slice.stop],
            "branch_slice": [branch_slice.start, branch_slice.stop],
            "rate_slice": [rate_slice.start, rate_slice.stop],
            "bias_index": int(bias_index),
            "hyst_radii": [[float(v) for v in vs] for vs in layout["hyst_radii"]],
            "hyst_scales": layout["hyst_scales"]
        },
        means={
            "W_mean": W_mean.flatten().tolist(),
            "X_mean": X_mean.flatten().tolist()
        }
    )
    with open(f"{stem}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    np.savez(f"{stem}_coef.npz",
             A=A,                   # full coefficient matrix (features × 6)
             C_lin=C_lin,           # 6×6
             b=b,                   # 6
             H_mat=H_mat,           # (6*M)×6 (maybe empty)
             BrW=BrW, RtW=RtW,      # optional blocks
             r2=r2,
             best_lag=best_lag)

    print(f"[HYST] Saved model to: {cfg['out_dir']}")

# ============================================================
# Inference helper
# ============================================================

def predict_from_files(W, out_dir, center=True):
    """
    Run inference with a previously saved model.
    Args:
        W: (N,6) wrench array
        out_dir: directory containing *_coef.npz and *_meta.json
        center: if True, subtract W_mean and add X_mean to outputs (match training)
    Returns:
        Xhat: (N,6) predicted deflection
    """
    # Load
    stem = os.path.join(out_dir, "compliance_hysteresis")
    coef = np.load(f"{stem}_coef.npz", allow_pickle=True)
    with open(f"{stem}_meta.json", "r") as f:
        meta = json.load(f)

    A = coef["A"]                 # (D,6)
    best_lag = int(coef["best_lag"])
    W_mean = np.array(meta["means"]["W_mean"]).reshape(1,6)
    X_mean = np.array(meta["means"]["X_mean"]).reshape(1,6)
    layout = meta["layout"]
    cfg = meta["cfg"]

    # Center
    Wc = W - W_mean if (center and cfg["demean"]) else W

    # Build features with same recipe & parameters
    # Reconstruct hysteresis features using stored radii & scales
    N, D = Wc.shape
    parts = []
    # Linear
    parts.append(Wc)
    # Hysteresis
    hyst_radii = [np.array(rs, dtype=float) for rs in layout["hyst_radii"]]
    hyst_scales = np.array(layout["hyst_scales"], dtype=float)
    Hcols = []
    for j in range(D):
        u = Wc[:, j]
        for r in hyst_radii[j]:
            y = play_operator(u, r)
            Hcols.append(y[:, None])
    if Hcols:
        Hfeat = np.hstack(Hcols)
        if cfg.get("hyst_normalize", True) and hyst_scales.size == Hfeat.shape[1]:
            Hfeat = Hfeat / hyst_scales
        parts.append(Hfeat)
    # Branch & Rate
    if cfg.get("use_branch_features", True):
        dw = np.diff(Wc, axis=0, prepend=Wc[[0], :])
        parts.append(np.sign(dw) * Wc)
    if cfg.get("use_rate_features", True):
        k = int(cfg.get("rate_window", 1))
        if k <= 1:
            parts.append(np.diff(Wc, axis=0, prepend=Wc[[0], :]))
        else:
            Rt = np.concatenate([np.zeros((k, D)), Wc[k:, :] - Wc[:-k, :]], axis=0)
            parts.append(Rt)
    # Bias
    parts.append(np.ones((N,1)))
    Phi = np.hstack(parts)

    # Apply lag alignment used in training
    if best_lag != 0:
        if best_lag > 0:
            Phi = Phi[:-best_lag, :]
            pad = np.repeat(Phi[[-1], :], repeats=best_lag, axis=0)
            Phi = np.vstack([Phi, pad])
        else:
            L = -best_lag
            Phi = Phi[L:, :]
            pad = np.repeat(Phi[[0], :], repeats=L, axis=0)
            Phi = np.vstack([pad, Phi])

    Xhat = Phi @ A
    if center and cfg["demean"]:
        Xhat = Xhat + X_mean
    return Xhat

# ============================================================

if __name__ == "__main__":
    main()
