#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrench modeling from pose→wrench time series.

We fit mappings from 6D pose x = [X,Y,Z,C,B,A] to 6D wrench w = [Fx,Fy,Fz,Tx,Ty,Tz].
Angles are converted to radians if configured.

(1) Linear (no bias):       w ≈ K x
(2) Affine (with bias):     w ≈ K x + c
(3) Quadratic (poly2):      w ≈ K x + 1/2 * x^T H x + c
(4) Piecewise-affine (K regions): for k=1..K, w ≈ K_k x + c_k
    Regions formed in input space (pose) by default, or wrench if configured.

Lag-scan searches alignment between input and output. Robust IRWLS optional.
"""

import numpy as np
import pandas as pd
import os

# -------------------- Hyperparameters and I/O configuration --------------------
CONFIG = {
    # Input data
    "csv": "./data/RCC_kuka_ati_15_trials.csv",   # Path to CSV with pose & wrench time series
    "wrench_cols": ["Fx_ati", "Fy_ati", "Fz_ati", "Tx_ati", "Ty_ati", "Tz_ati"],  # Output columns (order matters)
    "pose_cols": ["X", "Y", "Z", "C", "B", "A"],  # Input columns (order matters)
    "angles_deg": True,       # Convert pose angle columns (indices 3,4,5) from degrees to radians
    "demean": True,           # Subtract mean from both x and w before modeling
    "trial_col": "trial",    # Column name indicating trial index for noise filtering / downsampling

    # Optional trial-wise moving-mean downsampling (noise filtering)
    "downsample_enable": False,  # If True, perform trial-wise block moving-mean downsampling
    "downsample_factor": 10,     # Factor f: average non-overlapping windows of length f (per trial)

    # Model fitting
    "lag_scan": 10,           # Search lag in samples over [-lag_scan, +lag_scan] aligning x to w
    "ridge": 1e-3,            # L2 regularization strength (0 disables ridge)
    "huber_delta": 1.0,       # Huber delta for robust IRWLS; <=0 disables robust weighting
    "max_irwls_iters": 100,   # Max IRWLS iterations when huber_delta > 0

    # Post-processing for learned linear maps K
    "symmetrize": False,      # If True, symmetrize K as 0.5*(K + K^T)
    "spd_project": False,      # If True, project K to nearest SPD via eigenvalue clipping

    # Output (also used as stem for additional files)
    "out": "./compliance_modeling/results/stiffness_matrix.csv",  # Output CSV for basic linear K

    # Piecewise-affine (K-region) model options
    "pwl_enable": True,           # Enable piecewise-affine modeling stage
    "pwl_K": 50,                 # Number of clusters (regions)
    "pwl_soft_gating": False,     # If True, soft RBF gating; else hard assignments
    "pwl_rbf_gamma": 1.0,         # RBF gamma for soft gating; larger -> sharper weights
    "pwl_min_pts": 100,            # Minimum samples required to train a region; else fallback to global affine
    "pwl_region_lag_scan": False, # If True, run lag-scan per region; else reuse global lag
    "pwl_gating_space": "pose",  # "pose" (default, input) or "wrench" (output) for region formation

    # Piecewise-quadratic (K-region) model options
    "pwq_enable": True,          # Enable piecewise-quadratic modeling stage
    "pwq_region_lag_scan": False, # If True, run lag-scan per region for quadratic models

    # Filtering on inputs (pose)
    # Keep samples whose selected pose dims lie within the inner band:
    #   frac*min_neg <= value <= frac*max_pos
    # Works best after demeaning (so mean≈0). Fraction in [0,1].
    "filter_enable": True,
    "filter_pose_dims": ["X", "Y", "Z", "A", "B", "C"],
    "filter_logic": "all",       # Combine per-dim masks: "any" (OR) or "all" (AND)
    "filter_fraction": 0.8,
}

# -------------------- Utils --------------------

def to_radians(X, angle_indices):
    X = X.copy()
    X[:, angle_indices] = np.deg2rad(X[:, angle_indices])
    return X


def trialwise_moving_mean_downsample(df, pose_cols, wrench_cols, trial_col='trial', factor=10):
    """Downsample pose & wrench by non-overlapping moving-mean windows within each trial.

    For each trial of length N, we use floor(N/factor) full windows and drop the tail to avoid
    mixing with other trials. Trials shorter than 'factor' are kept unchanged.
    Returns (X_ds, W_ds) or None if 'trial_col' is missing.
    """
    if trial_col not in df.columns:
        print(f"[DOWNSAMPLE] Column '{trial_col}' not found; skipping downsampling.")
        return None

    X_parts, W_parts = [], []
    total_before = 0
    total_after = 0

    for t, g in df.groupby(trial_col, sort=False):
        Xg = g[pose_cols].to_numpy(float)
        Wg = g[wrench_cols].to_numpy(float)
        n = len(g)
        total_before += n
        if n < factor:
            # Keep as-is if not enough points for a single full window
            X_parts.append(Xg)
            W_parts.append(Wg)
            total_after += n
            print(f"[DOWNSAMPLE] trial={t}: {n} < factor {factor}; kept original samples.")
            continue
        trim = n - (n % factor)
        if trim < factor:
            X_parts.append(Xg)
            W_parts.append(Wg)
            total_after += n
            print(f"[DOWNSAMPLE] trial={t}: insufficient full windows; kept original samples.")
            continue
        X_trim = Xg[:trim]
        W_trim = Wg[:trim]
        X_ds = X_trim.reshape(-1, factor, X_trim.shape[1]).mean(axis=1)
        W_ds = W_trim.reshape(-1, factor, W_trim.shape[1]).mean(axis=1)
        X_parts.append(X_ds)
        W_parts.append(W_ds)
        total_after += len(X_ds)
        print(f"[DOWNSAMPLE] trial={t}: {n} -> {len(X_ds)} (factor={factor})")

    print(f"[DOWNSAMPLE] Total samples: {total_before} -> {total_after}")
    return np.vstack(X_parts), np.vstack(W_parts)


def standardize_missing(U, Y):
    mask = np.isfinite(U).all(axis=1) & np.isfinite(Y).all(axis=1)
    return U[mask], Y[mask]

def apply_lag(U, Y, lag):
    if lag == 0:
        return U, Y
    if lag > 0:
        return U[:-lag, :], Y[lag:, :]
    else:
        L = -lag
        return U[L:, :], Y[:-L, :]

def weighted_ridge(Phi, Y, w=None, ridge=0.0, ridge_mask=None):
    if w is not None:
        sw = np.sqrt(w).reshape(-1, 1)
        Pw = Phi * sw
        Yw = Y * sw
    else:
        Pw, Yw = Phi, Y
    PtP = Pw.T @ Pw
    if ridge > 0:
        if ridge_mask is None:
            PtP += ridge * np.eye(Phi.shape[1])
        else:
            PtP += ridge * np.diag(ridge_mask)
    RHS = Pw.T @ Yw
    A = np.linalg.solve(PtP, RHS)
    return A

def huber_weights(residuals, delta):
    r = np.sqrt((residuals**2).sum(axis=1) + 1e-12)
    w = np.ones_like(r)
    large = r > delta
    w[large] = delta / r[large]
    return np.clip(w, 1e-6, None)

def r2_multitarget(Phi, Y, A):
    pred = Phi @ A
    ss_res = ((Y - pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return r2, pred

def project_spd(M, eps=1e-9):
    Ms = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(Ms)
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(vals) @ vecs.T

# -------------------- Feature builders (input→features) --------------------

def build_features_linear(U, with_bias=False):
    if with_bias:
        return np.hstack([U, np.ones((U.shape[0], 1))]), np.r_[np.ones(U.shape[1]), 0.0]
    else:
        return U, np.ones(U.shape[1])

def build_features_poly2(U, with_bias=True):
    N, D = U.shape
    parts = [U]
    lin_idx = slice(0, D)
    quad_pairs, quad_cols = [], []
    for i in range(D):
        for j in range(i, D):
            quad_pairs.append((i, j))
            quad_cols.append((U[:, i] * U[:, j])[:, None])
    if quad_cols:
        parts.append(np.hstack(quad_cols))
    bias_idx = None
    if with_bias:
        bias_idx = D + len(quad_pairs)
        parts.append(np.ones((N, 1)))
    Phi = np.hstack(parts)
    ridge_mask = np.ones(Phi.shape[1])
    if with_bias:
        ridge_mask[bias_idx] = 0.0
    mapping = {"lin_idx": lin_idx, "quad_pairs": quad_pairs, "bias_idx": bias_idx}
    return Phi, ridge_mask, mapping

# -------------------- Generic lag-scan fitter --------------------

def fit_with_lag_scan(Phi_builder, U, Y, cfg):
    best = (-np.inf, 0, None, None, None)  # score, lag, A, r2, mapping
    for lag in range(-cfg['lag_scan'], cfg['lag_scan'] + 1):
        Ul, Yl = apply_lag(U, Y, lag)
        if len(Ul) < 10:
            continue
        out = Phi_builder(Ul)
        if len(out) == 2:
            Phi, ridge_mask = out
            mapping = None
        else:
            Phi, ridge_mask, mapping = out
        if cfg['huber_delta'] > 0:
            A = weighted_ridge(Phi, Yl, None, cfg['ridge'], ridge_mask)
            for _ in range(cfg['max_irwls_iters']):
                res = Yl - Phi @ A
                w = huber_weights(res, cfg['huber_delta'])
                A_new = weighted_ridge(Phi, Yl, w, cfg['ridge'], ridge_mask)
                if np.linalg.norm(A_new - A) / (np.linalg.norm(A) + 1e-12) < 1e-5:
                    A = A_new
                    break
                A = A_new
        else:
            A = weighted_ridge(Phi, Yl, None, cfg['ridge'], ridge_mask)
        r2, _ = r2_multitarget(Phi, Yl, A)
        score = float(np.mean(r2))
        if score > best[0]:
            best = (score, lag, A, r2, mapping)
    return best

# -------------------- K-means (simple, no external deps) --------------------

def kmeans(U, K, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    N, D = U.shape
    idx = rng.choice(N, size=K, replace=False)
    C = U[idx].copy()
    assign = np.zeros(N, dtype=int)
    for _ in range(iters):
        d2 = ((U[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)  # (N,K)
        new_assign = np.argmin(d2, axis=1)
        if np.all(new_assign == assign):
            break
        assign = new_assign
        for k in range(K):
            mask = assign == k
            if np.any(mask):
                C[k] = U[mask].mean(axis=0)
            else:
                C[k] = U[rng.integers(0, N)]
    S = np.zeros_like(C)
    global_std = U.std(axis=0) + 1e-8
    for k in range(K):
        mask = assign == k
        if np.sum(mask) >= 2:
            S[k] = U[mask].std(axis=0) + 1e-8
        else:
            S[k] = global_std
    return C, S, assign

# -------------------- Piecewise (K-region) affine modeling --------------------

def fit_piecewise_affine(U, Y, cfg, global_affine=None):
    """Piecewise-affine mapping with local biases per region for w ≈ K_k x + c_k.

    Clusters in input space (pose) by default, or in output space if configured.

    Returns dict with:
      'K', 'centroids', 'scales', 'regions', 'overall_r2', 'assign', 'gating_space',
      'missing_regions_count', 'missing_regions_idx'
    Each region: 'idx', 'lag', 'K', 'c', 'r2', 'missing'.
    """
    gating_space = cfg.get('pwl_gating_space', 'pose').lower()
    Kreg = int(cfg['pwl_K'])

    # Use global affine’s best lag if available; otherwise 0
    global_lag = int(global_affine.get('lag', 0)) if global_affine else 0

    # Align once for clustering
    Uc, Yc = apply_lag(U, Y, global_lag)

    # Choose features for clustering
    clustering_features = Uc if gating_space == 'pose' else Yc

    Ctr, Scales, hard_assign = kmeans(clustering_features, Kreg, iters=50, seed=0)

    regions = []
    missing_regions_idx = []
    min_pts = int(cfg['pwl_min_pts'])

    for k in range(Kreg):
        idx = np.where(hard_assign == k)[0]
        is_missing = len(idx) < min_pts
        if is_missing:
            if global_affine is None:
                continue
            Kk = global_affine['K']
            ck = global_affine['c']
            lag_k = global_lag
            r2_k = np.zeros(6)
            missing_regions_idx.append(k)
        else:
            if cfg.get('pwl_region_lag_scan', False):
                Ui = U[idx]
                Yi = Y[idx]
                score, lag_k, A, r2_k, _ = fit_with_lag_scan(
                    lambda U_: build_features_linear(U_, with_bias=True), Ui, Yi, cfg
                )
                Kk = A[:6, :].T
                ck = A[6, :].reshape(6)
            else:
                Ui, Yi = apply_lag(U[idx], Y[idx], global_lag)
                Phi, ridge_mask = build_features_linear(Ui, with_bias=True)
                if cfg['huber_delta'] > 0:
                    A = weighted_ridge(Phi, Yi, None, cfg['ridge'], ridge_mask)
                    for _ in range(cfg['max_irwls_iters']):
                        res = Yi - Phi @ A
                        w = huber_weights(res, cfg['huber_delta'])
                        A_new = weighted_ridge(Phi, Yi, w, cfg['ridge'], ridge_mask)
                        if np.linalg.norm(A_new - A) / (np.linalg.norm(A) + 1e-12) < 1e-5:
                            A = A_new
                            break
                        A = A_new
                else:
                    A = weighted_ridge(Phi, Yi, None, cfg['ridge'], ridge_mask)
                r2_k, _ = r2_multitarget(Phi, Yi, A)
                Kk = A[:6, :].T
                ck = A[6, :].reshape(6)
                lag_k = global_lag

        if cfg['symmetrize']:
            Kk = 0.5 * (Kk + Kk.T)
        if cfg['spd_project']:
            Kk = project_spd(Kk)

        regions.append({
            'idx': idx,
            'lag': lag_k,
            'K': Kk,
            'c': ck,
            'r2': r2_k,
            'missing': is_missing,
        })

    # Overall R2 on aligned data using chosen gating
    if cfg.get('pwl_soft_gating', False):
        gamma = float(cfg.get('pwl_rbf_gamma', 1.0))
        normed = [(clustering_features - Ctr[k]) / Scales[k] for k in range(Kreg)]
        d2 = np.stack([np.sum(n**2, axis=1) for n in normed], axis=1)
        Wsoft = np.exp(-gamma * d2) + 1e-12
        Wsoft /= Wsoft.sum(axis=1, keepdims=True)
        preds = np.zeros_like(Yc)
        for k, reg in enumerate(regions):
            Kk, ck = reg['K'], reg['c']
            preds += Wsoft[:, [k]] * (Uc @ Kk.T + ck)
        ss_res = ((Yc - preds) ** 2).sum(axis=0)
        ss_tot = ((Yc - Yc.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
        overall_r2 = 1.0 - ss_res / ss_tot
    else:
        preds = np.zeros_like(Yc)
        if not regions:
            overall_r2 = np.zeros(6)
        else:
            region_models = {k: (reg['K'], reg['c']) for k, reg in enumerate(regions)}
            for n in range(Uc.shape[0]):
                k = int(hard_assign[n])
                if k not in region_models:
                    # Fallback to global affine if region missing
                    Kk, ck = global_affine['K'], global_affine['c']
                else:
                    Kk, ck = region_models[k]
                preds[n] = Uc[n] @ Kk.T + ck
            ss_res = ((Yc - preds) ** 2).sum(axis=0)
            ss_tot = ((Yc - Yc.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
            overall_r2 = 1.0 - ss_res / ss_tot

    return {
        'K': Kreg,
        'centroids': Ctr,
        'scales': Scales,
        'regions': regions,
        'overall_r2': overall_r2,
        'assign': hard_assign,
        'gating_space': gating_space,
        'missing_regions_count': len(missing_regions_idx),
        'missing_regions_idx': np.array(missing_regions_idx, dtype=int),
    }

# -------------------- Piecewise (K-region) quadratic modeling --------------------

def fit_piecewise_quadratic(U, Y, cfg, global_quad=None):
    """Piecewise-quadratic mapping with local biases per region for w ≈ K_k x + 1/2 x^T H_k x + c_k.

    Regions are formed via K-means on pose (input) by default, or wrench (output) if cfg['pwl_gating_space']=='wrench'.

    global_quad: optional dict with keys {'K','H','c','lag'} to use as fallback for underfilled regions.

    Returns dict with:
      'K': number of regions, 'centroids', 'scales', 'regions', 'overall_r2', 'assign',
      'gating_space', 'missing_regions_count', 'missing_regions_idx'.
    Each region has: 'idx','lag','K','H','c','r2','missing'.
    """
    gating_space = cfg.get('pwl_gating_space', 'pose').lower()
    Kreg = int(cfg['pwl_K'])
    global_lag = int(global_quad.get('lag', 0)) if global_quad else 0

    # Align once for clustering
    Uc, Yc = apply_lag(U, Y, global_lag)
    clustering_features = Uc if gating_space == 'pose' else Yc

    Ctr, Scales, hard_assign = kmeans(clustering_features, Kreg, iters=50, seed=0)

    regions = []
    missing_regions_idx = []
    min_pts = int(cfg['pwl_min_pts'])

    for k in range(Kreg):
        idx = np.where(hard_assign == k)[0]
        is_missing = len(idx) < min_pts
        if is_missing:
            if global_quad is None:
                continue
            Kk = global_quad['K']
            Hk = global_quad['H']
            ck = global_quad['c']
            lag_k = global_lag
            r2_k = np.zeros(6)
            missing_regions_idx.append(k)
        else:
            if cfg.get('pwq_region_lag_scan', False):
                Ui = U[idx]
                Yi = Y[idx]
                score, lag_k, A, r2_k, mapping = fit_with_lag_scan(
                    lambda U_: build_features_poly2(U_, with_bias=True), Ui, Yi, cfg
                )
                lin_idx = mapping['lin_idx']; quad_pairs = mapping['quad_pairs']; bias_idx = mapping['bias_idx']
                A_lin = A[lin_idx, :]
                A_quad = A[len(range(lin_idx.start, lin_idx.stop)) : len(range(lin_idx.start, lin_idx.stop)) + len(quad_pairs), :]
                ck = A[bias_idx, :].reshape(6)
                Kk = A_lin.T
                Hk = np.zeros((6, 6, 6))
                for q_idx, (i, j) in enumerate(quad_pairs):
                    for t in range(6):
                        coeff = A_quad[q_idx, t]
                        if i == j:
                            Hk[t, i, i] += 2.0 * coeff
                        else:
                            Hk[t, i, j] += coeff
                            Hk[t, j, i] += coeff
            else:
                Ui, Yi = apply_lag(U[idx], Y[idx], global_lag)
                Phi, ridge_mask, mapping = build_features_poly2(Ui, with_bias=True)
                if cfg['huber_delta'] > 0:
                    A = weighted_ridge(Phi, Yi, None, cfg['ridge'], ridge_mask)
                    for _ in range(cfg['max_irwls_iters']):
                        res = Yi - Phi @ A
                        w = huber_weights(res, cfg['huber_delta'])
                        A_new = weighted_ridge(Phi, Yi, w, cfg['ridge'], ridge_mask)
                        if np.linalg.norm(A_new - A) / (np.linalg.norm(A) + 1e-12) < 1e-5:
                            A = A_new
                            break
                        A = A_new
                else:
                    A = weighted_ridge(Phi, Yi, None, cfg['ridge'], ridge_mask)
                r2_k, _ = r2_multitarget(Phi, Yi, A)
                lin_idx = mapping['lin_idx']; quad_pairs = mapping['quad_pairs']; bias_idx = mapping['bias_idx']
                A_lin = A[lin_idx, :]
                A_quad = A[len(range(lin_idx.start, lin_idx.stop)) : len(range(lin_idx.start, lin_idx.stop)) + len(quad_pairs), :]
                ck = A[bias_idx, :].reshape(6)
                Kk = A_lin.T
                Hk = np.zeros((6, 6, 6))
                for q_idx, (i, j) in enumerate(quad_pairs):
                    for t in range(6):
                        coeff = A_quad[q_idx, t]
                        if i == j:
                            Hk[t, i, i] += 2.0 * coeff
                        else:
                            Hk[t, i, j] += coeff
                            Hk[t, j, i] += coeff
                lag_k = global_lag

        if cfg['symmetrize']:
            Kk = 0.5 * (Kk + Kk.T)
        if cfg['spd_project']:
            Kk = project_spd(Kk)

        regions.append({
            'idx': idx,
            'lag': lag_k,
            'K': Kk,
            'H': Hk,
            'c': ck,
            'r2': r2_k,
            'missing': is_missing,
        })

    # Evaluation on aligned data
    if cfg.get('pwl_soft_gating', False):
        gamma = float(cfg.get('pwl_rbf_gamma', 1.0))
        normed = [(clustering_features - Ctr[k]) / Scales[k] for k in range(Kreg)]
        d2 = np.stack([np.sum(n**2, axis=1) for n in normed], axis=1)
        Wsoft = np.exp(-gamma * d2) + 1e-12
        Wsoft /= Wsoft.sum(axis=1, keepdims=True)
        preds = np.zeros_like(Yc)
        for k, reg in enumerate(regions):
            Kk, ck, Hk = reg['K'], reg['c'], reg['H']
            # compute quadratic term per sample efficiently
            quad = np.einsum('nd,kde,ne->nk', Uc, Hk, Uc) * 0.5  # (N,6)
            preds += Wsoft[:, [k]] * (Uc @ Kk.T + quad + ck)
        ss_res = ((Yc - preds) ** 2).sum(axis=0)
        ss_tot = ((Yc - Yc.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
        overall_r2 = 1.0 - ss_res / ss_tot
    else:
        preds = np.zeros_like(Yc)
        if not regions:
            overall_r2 = np.zeros(6)
        else:
            # build region map, fallback to global quad if missing
            for n in range(Uc.shape[0]):
                k = int(hard_assign[n])
                if regions[k]['missing'] and global_quad is not None:
                    Kk, ck, Hk = global_quad['K'], global_quad['c'], global_quad['H']
                else:
                    Kk, ck, Hk = regions[k]['K'], regions[k]['c'], regions[k]['H']
                quad = 0.5 * (Uc[n][None, :] @ Hk @ Uc[n][:, None]).reshape(-1)  # (6,)
                preds[n] = Uc[n] @ Kk.T + quad + ck
            ss_res = ((Yc - preds) ** 2).sum(axis=0)
            ss_tot = ((Yc - Yc.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
            overall_r2 = 1.0 - ss_res / ss_tot

    return {
        'K': Kreg,
        'centroids': Ctr,
        'scales': Scales,
        'regions': regions,
        'overall_r2': overall_r2,
        'assign': hard_assign,
        'gating_space': gating_space,
        'missing_regions_count': len(missing_regions_idx),
        'missing_regions_idx': np.array(missing_regions_idx, dtype=int),
    }

# -------------------- Main pipeline --------------------

def main():
    cfg = CONFIG
    os.makedirs(os.path.dirname(cfg['out']), exist_ok=True)

    df = pd.read_csv(cfg['csv'])
    W = df[cfg['wrench_cols']].to_numpy(float)  # targets (N,6)
    X = df[cfg['pose_cols']].to_numpy(float)    # inputs  (N,6)

    # Optional trial-wise moving mean downsampling before any angle conversion (average degrees, then convert)
    if cfg.get('downsample_enable', False):
        factor = int(cfg.get('downsample_factor', 10))
        trial_col = cfg.get('trial_col', 'trial')
        res = trialwise_moving_mean_downsample(df, cfg['pose_cols'], cfg['wrench_cols'], trial_col=trial_col, factor=factor)
        if res is not None:
            X, W = res
        else:
            print("[DOWNSAMPLE] Disabled (missing trial column).")

    if cfg['angles_deg']:
        X = to_radians(X, [3, 4, 5])

    X, W = standardize_missing(X, W)
    if cfg['demean']:
        X -= X.mean(0, keepdims=True)
        W -= W.mean(0, keepdims=True)

    # ---------- Optional filtering on pose (input) ----------
    if cfg.get("filter_enable", False):
        dims = cfg.get("filter_pose_dims")
        if not dims:
            legacy_dim = cfg.get("filter_pose_dim")
            dims = [legacy_dim] if legacy_dim else []
        frac = float(cfg.get("filter_fraction", 0.5))
        logic = cfg.get("filter_logic", "any").lower()
        valid_dims, dim_masks = [], []
        for dim_name in dims:
            if dim_name not in cfg['pose_cols']:
                print(f"[FILTER] Dimension {dim_name} not in pose_cols; skipping this dim.")
                continue
            dim_idx = cfg['pose_cols'].index(dim_name)
            vals = X[:, dim_idx]
            max_pos = np.max(vals)
            max_neg = np.min(vals)
            if max_pos <= 0 or max_neg >= 0:
                print(f"[FILTER] Pose dimension {dim_name} not suitable (no positive/negative span); skipping.")
                continue
            pos_thresh = frac * max_pos
            neg_thresh = frac * max_neg
            mask = (vals <= pos_thresh) & (vals >= neg_thresh)
            dim_masks.append(mask)
            valid_dims.append(dim_name)
            print(f"[FILTER] {dim_name}: within-band keep {int(mask.sum())}/{len(mask)}")
        if dim_masks:
            combined_mask = np.logical_and.reduce(dim_masks) if logic == "all" else np.logical_or.reduce(dim_masks)
            kept = int(combined_mask.sum()); total = len(combined_mask)
            if kept < 10:
                print(f"[FILTER] Combined mask keeps only {kept} samples (<10); skipping filter.")
            else:
                X = X[combined_mask]
                W = W[combined_mask]
                logic_str = "ALL" if logic == "all" else "ANY"
                print(f"[FILTER] dims {valid_dims} ({logic_str} logic): kept {kept}/{total} within-band samples (fraction={frac})")
        else:
            print("[FILTER] No valid dimensions for filtering; skipping.")

    # ---------- (1) Linear (no bias): w ≈ K x ----------
    score_b, lag_b, A_b, r2_b, _ = fit_with_lag_scan(
        lambda U_: build_features_linear(U_, with_bias=False), X, W, cfg
    )
    K_basic = A_b.T
    if cfg['symmetrize']:
        K_basic = 0.5 * (K_basic + K_basic.T)
    if cfg['spd_project']:
        K_basic = project_spd(K_basic)
    out_basic = cfg['out']
    pd.DataFrame(K_basic,
        index=['Fx','Fy','Fz','Tx','Ty','Tz'],
        columns=['X','Y','Z','C','B','A']).to_csv(out_basic, float_format='%.10e')
    print(f"[LINEAR] Best lag: {lag_b}  mean R2: {np.mean(r2_b):.4f}")

    # ---------- (2) Affine: w ≈ K x + c ----------
    score_a, lag_a, A_a, r2_a, _ = fit_with_lag_scan(
        lambda U_: build_features_linear(U_, with_bias=True), X, W, cfg
    )
    K_aff = A_a[:6, :].T
    c_aff = A_a[6, :].reshape(6)
    if cfg['symmetrize']:
        K_aff = 0.5 * (K_aff + K_aff.T)
    if cfg['spd_project']:
        K_aff = project_spd(K_aff)
    stem = out_basic[:-4] if out_basic.lower().endswith('.csv') else out_basic
    out_aff_K = f"{stem}_affine_K.csv"
    out_aff_c = f"{stem}_affine_bias.csv"
    pd.DataFrame(K_aff,
        index=['Fx','Fy','Fz','Tx','Ty','Tz'],
        columns=['X','Y','Z','C','B','A']).to_csv(out_aff_K, float_format='%.10e')
    pd.DataFrame(c_aff[None, :], index=['bias'], columns=['Fx','Fy','Fz','Tx','Ty','Tz']).T.to_csv(out_aff_c, float_format='%.10e')
    print(f"[AFFINE] Best lag: {lag_a}  mean R2: {np.mean(r2_a):.4f}")

    # ---------- (3) Quadratic (poly2): w ≈ K x + 1/2 x^T H x + c ----------
    score_p, lag_p, A_p, r2_p, mapping = fit_with_lag_scan(
        lambda U_: build_features_poly2(U_, with_bias=True), X, W, cfg
    )
    lin_idx = mapping["lin_idx"]; quad_pairs = mapping["quad_pairs"]; bias_idx = mapping["bias_idx"]
    A_lin = A_p[lin_idx, :]
    A_quad = A_p[len(range(lin_idx.start, lin_idx.stop)) : len(range(lin_idx.start, lin_idx.stop)) + len(quad_pairs), :]
    c_poly = A_p[bias_idx, :].reshape(6)
    K_poly = A_lin.T
    H = np.zeros((6, 6, 6))
    for q_idx, (i, j) in enumerate(quad_pairs):
        for k in range(6):
            coeff = A_quad[q_idx, k]
            if i == j:
                H[k, i, i] += 2.0 * coeff
            else:
                H[k, i, j] += coeff
                H[k, j, i] += coeff
    if cfg['symmetrize']:
        K_poly = 0.5 * (K_poly + K_poly.T)
    if cfg['spd_project']:
        K_poly = project_spd(K_poly)
    out_poly_K = f"{stem}_poly2_K.csv"
    out_poly_c = f"{stem}_poly2_bias.csv"
    out_poly_H = f"{stem}_poly2_H.npz"
    pd.DataFrame(K_poly,
        index=['Fx','Fy','Fz','Tx','Ty','Tz'],
        columns=['X','Y','Z','C','B','A']).to_csv(out_poly_K, float_format='%.10e')
    pd.DataFrame(c_poly[None, :], index=['bias'], columns=['Fx','Fy','Fz','Tx','Ty','Tz']).T.to_csv(out_poly_c, float_format='%.10e')
    np.savez(out_poly_H, H=H)
    print(f"[POLY2] Best lag: {lag_p}  mean R2: {np.mean(r2_p):.4f}")

    # ---------- (4) Piecewise-affine (K regions) ----------
    if cfg.get("pwl_enable", True):
        global_affine = {"K": K_aff, "c": c_aff, "lag": lag_a}
        pwl = fit_piecewise_affine(X, W, cfg, global_affine=global_affine)
        out_pwl_info = f"{stem}_pwl_info.npz"
        out_pwl_csvs = []
        np.savez(out_pwl_info,
                 centroids=pwl['centroids'],
                 scales=pwl['scales'],
                 assign=pwl['assign'],
                 K=pwl['K'],
                 overall_r2=pwl['overall_r2'],
                 gating_space=pwl.get('gating_space', 'pose'),
                 missing_regions_count=pwl.get('missing_regions_count', 0),
                 missing_regions_idx=pwl.get('missing_regions_idx', np.array([], dtype=int)))
        for r_idx, reg in enumerate(pwl['regions']):
            out_Kk = f"{stem}_pwl_region{r_idx}_K.csv"
            out_ck = f"{stem}_pwl_region{r_idx}_bias.csv"
            pd.DataFrame(reg['K'],
                index=['Fx','Fy','Fz','Tx','Ty','Tz'],
                columns=['X','Y','Z','C','B','A']).to_csv(out_Kk, float_format='%.10e')
            pd.DataFrame(reg['c'][None, :], index=['bias'], columns=['Fx','Fy','Fz','Tx','Ty','Tz']).T.to_csv(out_ck, float_format='%.10e')
            out_pwl_csvs += [out_Kk, out_ck]
        print(f"[PWL] K={pwl['K']}  overall mean R2: {np.mean(pwl['overall_r2']):.4f}")
        print(f"[PWL] Missing regions (|idx| < {cfg['pwl_min_pts']}): {pwl['missing_regions_count']}")
        print(f"[PWL] Saved gating + assignments -> {out_pwl_info}")
        if out_pwl_csvs:
            print("[PWL] Saved region models (K + local bias)")
        else:
            print("[PWL] No region models saved (likely insufficient points per region).")

    # ---------- (5) Piecewise-quadratic (K regions) ----------
    if cfg.get("pwq_enable", False):
        global_quad = {"K": K_poly, "H": H, "c": c_poly, "lag": lag_p}
        pwq = fit_piecewise_quadratic(X, W, cfg, global_quad=global_quad)
        out_pwq_info = f"{stem}_pwq_info.npz"
        out_pwq_csvs = []
        np.savez(out_pwq_info,
                 centroids=pwq['centroids'],
                 scales=pwq['scales'],
                 assign=pwq['assign'],
                 K=pwq['K'],
                 overall_r2=pwq['overall_r2'],
                 gating_space=pwq.get('gating_space', 'pose'),
                 missing_regions_count=pwq.get('missing_regions_count', 0),
                 missing_regions_idx=pwq.get('missing_regions_idx', np.array([], dtype=int)))
        for r_idx, reg in enumerate(pwq['regions']):
            out_Kk = f"{stem}_pwq_region{r_idx}_K.csv"
            out_ck = f"{stem}_pwq_region{r_idx}_bias.csv"
            out_Hk = f"{stem}_pwq_region{r_idx}_H.npz"
            pd.DataFrame(reg['K'],
                index=['Fx','Fy','Fz','Tx','Ty','Tz'],
                columns=['X','Y','Z','C','B','A']).to_csv(out_Kk, float_format='%.10e')
            pd.DataFrame(reg['c'][None, :], index=['bias'], columns=['Fx','Fy','Fz','Tx','Ty','Tz']).T.to_csv(out_ck, float_format='%.10e')
            np.savez(out_Hk, H=reg['H'])
            out_pwq_csvs += [out_Kk, out_ck, out_Hk]
        print(f"[PWQ] K={pwq['K']}  overall mean R2: {np.mean(pwq['overall_r2']):.4f}")
        print(f"[PWQ] Missing regions (|idx| < {cfg['pwl_min_pts']}): {pwq['missing_regions_count']}")
        print(f"[PWQ] Saved gating + assignments -> {out_pwq_info}")
        if out_pwq_csvs:
            print("[PWQ] Saved region models (K, H + local bias)")
        else:
            print("[PWQ] No region models saved (likely insufficient points per region).")

if __name__ == '__main__':
    main()
