#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compliance modeling from wrench→pose time series.

(1) Linear (no bias):     x ≈ C w
(2) Affine (with bias):   x ≈ C w + b
(3) Quadratic (poly2):    x ≈ C w + 1/2 * w^T Q w + b
(4) Piecewise-linear (K regions): for k=1..K, x ≈ C_k w + b_k
    Regions are formed in wrench space using K-means (hard or soft gating).

Keeps CONFIG dict (no argparse).
"""

import numpy as np
import pandas as pd
import os

CONFIG = {
    # "csv": "./data/RCC_combined_14_processed.csv",
    "csv": "./data/RCC_kuka_ati_15_trials.csv",
    "wrench_cols": ["Fx_ati", "Fy_ati", "Fz_ati", "Tx_ati", "Ty_ati", "Tz_ati"],
    "pose_cols": ["X", "Y", "Z", "C", "B", "A"],
    "angles_deg": True,
    "demean": True,
    "lag_scan": 10,
    "ridge": 1e-3,
    "huber_delta": 1.0,
    "max_irwls_iters": 100,
    "symmetrize": False,   # applies to C only
    "spd_project": True,   # applies to C only
    "out": "./compliance_modeling/compliance_matrix.csv",

    # ---------------- Piecewise options (new) ----------------
    "pwl_enable": True,           # set False to skip
    "pwl_K": 10,                   # number of regions
    "pwl_soft_gating": False,     # False: hard K-means; True: soft RBF weights over K-means centroids
    "pwl_rbf_gamma": 1.0,         # only used if soft_gating=True (weight = exp(-gamma * ||w - mu_k||^2))
    "pwl_min_pts": 200,           # minimum samples to train a region; fallback to global affine if fewer
    "pwl_region_lag_scan": True,  # run lag-scan per region (slower, but more accurate)
}

# ---------- Utils (unchanged from your previous version) ----------

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

def weighted_ridge(W, X, w=None, ridge=0.0, ridge_mask=None):
    if w is not None:
        sw = np.sqrt(w).reshape(-1, 1)
        Ww = W * sw
        Xw = X * sw
    else:
        Ww, Xw = W, X
    WTW = Ww.T @ Ww
    if ridge > 0:
        if ridge_mask is None:
            WTW += ridge * np.eye(W.shape[1])
        else:
            WTW += ridge * np.diag(ridge_mask)
    RHS = Ww.T @ Xw
    A = np.linalg.solve(WTW, RHS)
    return A

def huber_weights(residuals, delta):
    r = np.sqrt((residuals**2).sum(axis=1) + 1e-12)
    w = np.ones_like(r)
    large = r > delta
    w[large] = delta / r[large]
    return np.clip(w, 1e-6, None)

def r2_multitarget(W, X, A):
    pred = W @ A
    ss_res = ((X - pred) ** 2).sum(axis=0)
    ss_tot = ((X - X.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return r2, pred

def project_spd(M, eps=1e-9):
    Ms = 0.5 * (M + M.T)
    vals, vecs = np.linalg.eigh(Ms)
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(vals) @ vecs.T

# ---------- Feature builders (linear/affine and poly2) ----------

def build_features_linear(W, with_bias=False):
    if with_bias:
        return np.hstack([W, np.ones((W.shape[0], 1))]), np.r_[np.ones(W.shape[1]), 0.0]
    else:
        return W, np.ones(W.shape[1])

def build_features_poly2(W, with_bias=True):
    N, D = W.shape
    parts = [W]
    lin_idx = slice(0, D)
    quad_pairs, quad_cols = [], []
    for i in range(D):
        for j in range(i, D):
            quad_pairs.append((i, j))
            quad_cols.append((W[:, i] * W[:, j])[:, None])
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

# ---------- Generic lag-scan fitter ----------

def fit_with_lag_scan(Phi_builder, W, X, cfg):
    best = (-np.inf, 0, None, None, None)  # score, lag, A, r2, mapping
    for lag in range(-cfg['lag_scan'], cfg['lag_scan'] + 1):
        Wl, Xl = apply_lag(W, X, lag)
        if len(Wl) < 10:
            continue
        out = Phi_builder(Wl)
        if len(out) == 2:
            Phi, ridge_mask = out
            mapping = None
        else:
            Phi, ridge_mask, mapping = out
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
            best = (score, lag, A, r2, mapping)
    return best

# ---------- K-means (simple, no external deps) ----------

def kmeans(W, K, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    N, D = W.shape
    # init: sample K points
    idx = rng.choice(N, size=K, replace=False)
    C = W[idx].copy()
    assign = np.zeros(N, dtype=int)
    for _ in range(iters):
        # distances and assignments
        d2 = ((W[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)  # (N,K)
        new_assign = np.argmin(d2, axis=1)
        if np.all(new_assign == assign):
            break
        assign = new_assign
        # update centroids
        for k in range(K):
            mask = assign == k
            if np.any(mask):
                C[k] = W[mask].mean(axis=0)
            else:
                # re-init an empty cluster
                C[k] = W[rng.integers(0, N)]
    # cluster scales (per-dim std; fallback to global)
    S = np.zeros_like(C)
    global_std = W.std(axis=0) + 1e-8
    for k in range(K):
        mask = assign == k
        if np.sum(mask) >= 2:
            S[k] = W[mask].std(axis=0) + 1e-8
        else:
            S[k] = global_std
    return C, S, assign

# ---------- Piecewise (K-region) affine modeling ----------

def fit_piecewise_affine(W, X, cfg, global_affine=None):
    """
    Returns dictionary with:
      'K', 'centroids', 'scales', 'regions': list of dicts with keys
        - 'idx': indices used
        - 'lag': best lag for region (if pwl_region_lag_scan)
        - 'C': (6,6)
        - 'b': (6,)
        - 'r2': (6,)
      'overall_r2': (6,)
      'assign': hard assignments used for training
    """
    N, D = W.shape
    K = int(cfg['pwl_K'])
    # Use global affine’s best lag alignment if available; otherwise 0
    global_lag = 0
    if global_affine is not None and "lag" in global_affine:
        global_lag = int(global_affine["lag"])

    # Align once for clustering
    Wc, Xc = apply_lag(W, X, global_lag)

    # K-means over aligned wrench
    Ctr, Scales, hard_assign = kmeans(Wc, K, iters=50, seed=0)

    regions = []
    preds_sum = np.zeros_like(Xc)
    counts = np.zeros(N - max(global_lag, 0), dtype=float)  # for soft gating blend

    for k in range(K):
        idx = np.where(hard_assign == k)[0]
        if len(idx) < cfg['pwl_min_pts']:
            # too small; fallback to global affine if provided
            if global_affine is None:
                continue
            Ck = global_affine["C"]
            bk = global_affine["b"]
            lag_k = global_lag
            r2_k = np.zeros(6)
        else:
            # optionally run per-region lag scan; else use global_lag
            if cfg.get("pwl_region_lag_scan", False):
                # subset arrays before building features
                Wr = W[idx]
                Xr = X[idx]
                # affine features with bias
                score, lag_k, A, r2_k, _ = fit_with_lag_scan(
                    lambda W_: build_features_linear(W_, with_bias=True), Wr, Xr, cfg
                )
                Ck = A[:6, :].T
                bk = A[6, :].reshape(6)
            else:
                # fixed lag
                Wr, Xr = apply_lag(W[idx], X[idx], global_lag)
                Phi, ridge_mask = build_features_linear(Wr, with_bias=True)
                if cfg['huber_delta'] > 0:
                    A = weighted_ridge(Phi, Xr, None, cfg['ridge'], ridge_mask)
                    for _ in range(cfg['max_irwls_iters']):
                        res = Xr - Phi @ A
                        w = huber_weights(res, cfg['huber_delta'])
                        A_new = weighted_ridge(Phi, Xr, w, cfg['ridge'], ridge_mask)
                        if np.linalg.norm(A_new - A) / (np.linalg.norm(A) + 1e-12) < 1e-5:
                            A = A_new
                            break
                        A = A_new
                else:
                    A = weighted_ridge(Phi, Xr, None, cfg['ridge'], ridge_mask)
                r2_k, _ = r2_multitarget(Phi, Xr, A)
                Ck = A[:6, :].T
                bk = A[6, :].reshape(6)
                lag_k = global_lag

        # optional post-processing for Ck
        if cfg['symmetrize']:
            Ck = 0.5 * (Ck + Ck.T)
        if cfg['spd_project']:
            Ck = project_spd(Ck)

        regions.append({
            "idx": idx,
            "lag": lag_k,
            "C": Ck,
            "b": bk,
            "r2": r2_k
        })

    # Compute overall R2 on aligned data using chosen gating (hard or soft)
    if cfg.get("pwl_soft_gating", False):
        # soft RBF weights from centroids learned on Wc
        # weights_k(n) ∝ exp(-gamma * || (Wc[n]-mu_k)/scale ||^2)
        gamma = float(cfg.get("pwl_rbf_gamma", 1.0))
        # normalized by per-dim scales (to make distances roughly isotropic)
        normed = [(Wc - Ctr[k]) / Scales[k] for k in range(K)]  # list N x 6
        d2 = np.stack([np.sum(n**2, axis=1) for n in normed], axis=1)  # (N,K)
        Wsoft = np.exp(-gamma * d2) + 1e-12
        Wsoft /= Wsoft.sum(axis=1, keepdims=True)

        # predict blended
        preds = np.zeros_like(Xc)
        for k, reg in enumerate(regions):
            # allow using region k for all samples with soft weight
            Ck, bk = reg["C"], reg["b"]
            preds += Wsoft[:, [k]] * (Wc @ Ck.T + bk)
        ss_res = ((Xc - preds) ** 2).sum(axis=0)
        ss_tot = ((Xc - Xc.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
        overall_r2 = 1.0 - ss_res / ss_tot
    else:
        # hard: use each sample’s region model
        preds = np.zeros_like(Xc)
        # build a quick map from region to model; if a region missing, use first
        if not regions:
            overall_r2 = np.zeros(6)
        else:
            # region id → model
            region_models = {}
            for k, reg in enumerate(regions):
                region_models[k] = (reg["C"], reg["b"])
            for n in range(Wc.shape[0]):
                k = int(hard_assign[n])
                if k not in region_models:
                    k = list(region_models.keys())[0]
                Ck, bk = region_models[k]
                preds[n] = Wc[n] @ Ck.T + bk
            ss_res = ((Xc - preds) ** 2).sum(axis=0)
            ss_tot = ((Xc - Xc.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
            overall_r2 = 1.0 - ss_res / ss_tot

    return {
        "K": K,
        "centroids": Ctr,
        "scales": Scales,
        "regions": regions,
        "overall_r2": overall_r2,
        "assign": hard_assign
    }

# ---------- Main pipeline ----------

def main():
    cfg = CONFIG
    os.makedirs(os.path.dirname(cfg['out']), exist_ok=True)

    df = pd.read_csv(cfg['csv'])
    W = df[cfg['wrench_cols']].to_numpy(float)  # (N,6)
    X = df[cfg['pose_cols']].to_numpy(float)    # (N,6)

    if cfg['angles_deg']:
        X = to_radians(X, [3, 4, 5])

    W, X = standardize_missing(W, X)
    if cfg['demean']:
        W -= W.mean(0, keepdims=True)
        X -= X.mean(0, keepdims=True)

    # ---------- (1) Basic linear ----------
    score_b, lag_b, A_b, r2_b, _ = fit_with_lag_scan(
        lambda W_: build_features_linear(W_, with_bias=False), W, X, cfg
    )
    C_basic = A_b.T
    if cfg['symmetrize']:
        C_basic = 0.5 * (C_basic + C_basic.T)
    if cfg['spd_project']:
        C_basic = project_spd(C_basic)
    out_basic = cfg['out']
    pd.DataFrame(C_basic,
        index=['x','y','z','rx','ry','rz'],
        columns=['Fx','Fy','Fz','Tx','Ty','Tz']).to_csv(out_basic, float_format='%.10e')
    print(f"[BASIC]  Best lag: {lag_b}  mean R2: {np.mean(r2_b):.4f}")

    # ---------- (2) Affine ----------
    score_a, lag_a, A_a, r2_a, _ = fit_with_lag_scan(
        lambda W_: build_features_linear(W_, with_bias=True), W, X, cfg
    )
    C_aff = A_a[:6, :].T
    b_aff = A_a[6, :].reshape(6)
    if cfg['symmetrize']:
        C_aff = 0.5 * (C_aff + C_aff.T)
    if cfg['spd_project']:
        C_aff = project_spd(C_aff)
    stem = out_basic[:-4] if out_basic.lower().endswith('.csv') else out_basic
    out_aff_C = f"{stem}_affine_C.csv"
    out_aff_b = f"{stem}_affine_bias.csv"
    pd.DataFrame(C_aff,
        index=['x','y','z','rx','ry','rz'],
        columns=['Fx','Fy','Fz','Tx','Ty','Tz']).to_csv(out_aff_C, float_format='%.10e')
    pd.DataFrame(b_aff[None, :],
        index=['bias'], columns=['x','y','z','rx','ry','rz']).T.to_csv(out_aff_b, float_format='%.10e')
    print(f"[AFFINE] Best lag: {lag_a}  mean R2: {np.mean(r2_a):.4f}")

    # ---------- (3) Quadratic (poly2) ----------
    score_p, lag_p, A_p, r2_p, mapping = fit_with_lag_scan(
        lambda W_: build_features_poly2(W_, with_bias=True), W, X, cfg
    )
    lin_idx = mapping["lin_idx"]; quad_pairs = mapping["quad_pairs"]; bias_idx = mapping["bias_idx"]
    A_lin = A_p[lin_idx, :]
    A_quad = A_p[len(range(lin_idx.start, lin_idx.stop)) : len(range(lin_idx.start, lin_idx.stop)) + len(quad_pairs), :]
    b_poly = A_p[bias_idx, :].reshape(6)
    C_poly = A_lin.T
    Q = np.zeros((6, 6, 6))
    for q_idx, (i, j) in enumerate(quad_pairs):
        for k in range(6):
            coeff = A_quad[q_idx, k]
            if i == j:
                Q[k, i, i] += 2.0 * coeff
            else:
                Q[k, i, j] += coeff
                Q[k, j, i] += coeff
    if cfg['symmetrize']:
        C_poly = 0.5 * (C_poly + C_poly.T)
    if cfg['spd_project']:
        C_poly = project_spd(C_poly)
    out_poly_C = f"{stem}_poly2_C.csv"
    out_poly_b = f"{stem}_poly2_bias.csv"
    out_poly_Q = f"{stem}_poly2_Q.npz"
    pd.DataFrame(C_poly,
        index=['x','y','z','rx','ry','rz'],
        columns=['Fx','Fy','Fz','Tx','Ty','Tz']).to_csv(out_poly_C, float_format='%.10e')
    pd.DataFrame(b_poly[None, :],
        index=['bias'], columns=['x','y','z','rx','ry','rz']).T.to_csv(out_poly_b, float_format='%.10e')
    np.savez(out_poly_Q, Q=Q)
    print(f"[POLY2] Best lag: {lag_p}  mean R2: {np.mean(r2_p):.4f}")

    # ---------- (4) Piecewise-linear (K regions) ----------
    if cfg.get("pwl_enable", True):
        global_affine = {"C": C_aff, "b": b_aff, "lag": lag_a}
        pwl = fit_piecewise_affine(W, X, cfg, global_affine=global_affine)

        # Save gating params and per-region models
        out_pwl_info = f"{stem}_pwl_info.npz"     # centroids, scales, assignments, meta
        out_pwl_csvs = []
        np.savez(out_pwl_info,
                 centroids=pwl["centroids"],
                 scales=pwl["scales"],
                 assign=pwl["assign"],
                 K=pwl["K"],
                 overall_r2=pwl["overall_r2"])
        for r_idx, reg in enumerate(pwl["regions"]):
            out_Ck = f"{stem}_pwl_region{r_idx}_C.csv"
            out_bk = f"{stem}_pwl_region{r_idx}_bias.csv"
            pd.DataFrame(reg["C"],
                index=['x','y','z','rx','ry','rz'],
                columns=['Fx','Fy','Fz','Tx','Ty','Tz']).to_csv(out_Ck, float_format='%.10e')
            pd.DataFrame(reg["b"][None, :],
                index=['bias'], columns=['x','y','z','rx','ry','rz']).T.to_csv(out_bk, float_format='%.10e')
            out_pwl_csvs += [out_Ck, out_bk]

        print(f"[PWL] K={pwl['K']}  overall mean R2: {np.mean(pwl['overall_r2']):.4f}")
        print(f"[PWL] Saved gating + assignments -> {out_pwl_info}")
        if out_pwl_csvs:
            print("[PWL] Saved region models:")
            for p in out_pwl_csvs:
                print("   ", p)
        else:
            print("[PWL] No region models saved (likely insufficient points per region).")

if __name__ == '__main__':
    main()
