#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate compliance from (wrench, relative pose) time series:

(1) Linear (no bias):     x ≈ C w
(2) Affine (with bias):   x ≈ C w + b
(3) Quadratic (poly2):    x ≈ C w + 1/2 * w^T Q w + b
    where Q is symmetric, one 6x6 matrix per output dimension.

Keeps CONFIG dict (no argparse). Uses lag-scan + Huber IRWLS + ridge.
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
    "symmetrize": False,   # if True, symmetrize C (not used for Q; Q is constructed symmetric by design)
    "spd_project": True,   # if True, project C to SPD
    "out": "./compliance_modeling/compliance_matrix.csv",
}

# ---------- Utils ----------

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
    """
    Solve:  A = argmin_A ||diag(sqrt(w)) (X - W A)||_F^2 + ridge * ||A||_F^2
    W: (N, D), X: (N, 6), A: (D, 6)
    ridge_mask: per-feature penalty scaling (0 to avoid penalizing that feature).
    """
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
    # residuals: (N,6)
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

# ---------- Feature builders ----------

def build_features_linear(W, with_bias=False):
    if with_bias:
        return np.hstack([W, np.ones((W.shape[0], 1))]), np.r_[np.ones(W.shape[1]), 0.0]
    else:
        return W, np.ones(W.shape[1])

def build_features_poly2(W, with_bias=True):
    """
    Build [w, (wi*wj for i<=j), 1?]
    Returns:
      Phi: (N, D_lin + D_quad [+1])
      ridge_mask: 1 per feature (0 for bias to avoid penalizing it)
      mapping: dict with keys:
         - 'lin_idx' -> slice for linear features
         - 'quad_pairs' -> list of (i,j) for i<=j in the order used
         - 'bias_idx' -> int or None
    """
    N, D = W.shape
    # linear part
    parts = [W]
    lin_idx = slice(0, D)
    # quadratic upper-triangular (including diag)
    quad_pairs = []
    quad_cols = []
    for i in range(D):
        for j in range(i, D):
            quad_pairs.append((i, j))
            quad_cols.append((W[:, i] * W[:, j])[:, None])
    if quad_cols:
        parts.append(np.hstack(quad_cols))
    # bias
    bias_idx = None
    if with_bias:
        bias_idx = D + len(quad_pairs)
        parts.append(np.ones((N, 1)))

    Phi = np.hstack(parts)
    ridge_mask = np.ones(Phi.shape[1])
    if with_bias:
        ridge_mask[bias_idx] = 0.0

    mapping = {
        "lin_idx": lin_idx,
        "quad_pairs": quad_pairs,
        "bias_idx": bias_idx,
    }
    return Phi, ridge_mask, mapping

# ---------- Generic lag-scan fitter ----------

def fit_with_lag_scan(Phi_builder, W, X, cfg):
    """
    Phi_builder(W) -> (Phi, ridge_mask, mapping) OR (Phi, ridge_mask) for linear.
    Runs Huber IRWLS + ridge for each lag; returns the best by mean R^2.
    """
    best = (-np.inf, 0, None, None, None)  # score, lag, A, r2, mapping

    for lag in range(-cfg['lag_scan'], cfg['lag_scan'] + 1):
        Wl, Xl = apply_lag(W, X, lag)
        if len(Wl) < 10:
            continue

        Phi_out = Phi_builder(Wl)
        if len(Phi_out) == 2:
            Phi, ridge_mask = Phi_out
            mapping = None
        else:
            Phi, ridge_mask, mapping = Phi_out

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
            best = (score, lag, A, r2, mapping)

    return best  # (score, lag, A, r2, mapping)

# ---------- Main ----------

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

    # ---------- (1) Linear ----------
    score_b, lag_b, A_b, r2_b, _ = fit_with_lag_scan(
        lambda W_: build_features_linear(W_, with_bias=False), W, X, cfg
    )
    if A_b is None:
        raise RuntimeError('Failed to estimate basic compliance matrix.')

    C_basic = A_b.T  # (6,6)
    if cfg['symmetrize']:
        C_basic = 0.5 * (C_basic + C_basic.T)
    if cfg['spd_project']:
        C_basic = project_spd(C_basic)

    # Save basic
    out_basic = cfg['out']
    pd.DataFrame(
        C_basic,
        index=['x','y','z','rx','ry','rz'],
        columns=['Fx','Fy','Fz','Tx','Ty','Tz']
    ).to_csv(out_basic, float_format='%.10e')

    print(f"[BASIC]  Best lag: {lag_b}  mean R2: {np.mean(r2_b):.4f}")
    print(f"[BASIC]  R2 per dimension: {np.round(r2_b, 4).tolist()}")

    # ---------- (2) Affine ----------
    score_a, lag_a, A_a, r2_a, _ = fit_with_lag_scan(
        lambda W_: build_features_linear(W_, with_bias=True), W, X, cfg
    )
    if A_a is None:
        raise RuntimeError('Failed to estimate affine model.')

    C_aff = A_a[:6, :].T  # (6,6)
    b_aff = A_a[6, :].reshape(6)
    if cfg['symmetrize']:
        C_aff = 0.5 * (C_aff + C_aff.T)
    if cfg['spd_project']:
        C_aff = project_spd(C_aff)

    # Save affine
    stem = out_basic[:-4] if out_basic.lower().endswith('.csv') else out_basic
    out_aff_C = f"{stem}_affine_C.csv"
    out_aff_b = f"{stem}_affine_bias.csv"
    pd.DataFrame(C_aff,
        index=['x','y','z','rx','ry','rz'],
        columns=['Fx','Fy','Fz','Tx','Ty','Tz']).to_csv(out_aff_C, float_format='%.10e')
    pd.DataFrame(b_aff[None, :],
        index=['bias'], columns=['x','y','z','rx','ry','rz']).T.to_csv(out_aff_b, float_format='%.10e')

    print(f"[AFFINE] Best lag: {lag_a}  mean R2: {np.mean(r2_a):.4f}")
    print(f"[AFFINE] R2 per dimension: {np.round(r2_a, 4).tolist()}")

    # ---------- (3) Quadratic (poly2) ----------
    score_p, lag_p, A_p, r2_p, mapping = fit_with_lag_scan(
        lambda W_: build_features_poly2(W_, with_bias=True), W, X, cfg
    )
    if A_p is None or mapping is None:
        raise RuntimeError('Failed to estimate quadratic (poly2) model.')

    lin_idx = mapping["lin_idx"]
    quad_pairs = mapping["quad_pairs"]
    bias_idx = mapping["bias_idx"]

    # Decompose coefficients:
    # A_p: (D_lin + D_quad + 1, 6). Rows: [linear(6), quad(i<=j), bias]
    A_lin = A_p[lin_idx, :]                   # (6,6) -> C^T
    A_quad = A_p[len(range(lin_idx.start, lin_idx.stop)) : len(range(lin_idx.start, lin_idx.stop)) + len(quad_pairs), :]  # (Dq,6)
    b_poly = A_p[bias_idx, :].reshape(6)      # (6,)

    C_poly = A_lin.T  # (6,6)

    # Build symmetric Q for each output k
    Q = np.zeros((6, 6, 6))  # (k, i, j)
    # A_quad rows correspond to features wi*wj for (i<=j) in quad_pairs order
    for q_idx, (i, j) in enumerate(quad_pairs):
        # For each output dimension k, coefficient theta_{ij,k} belongs to X[:,k]
        for k in range(6):
            coeff = A_quad[q_idx, k]
            if i == j:
                # 1/2 * Q_ii * wi^2 = coeff * wi^2  => Q_ii = 2 * coeff
                Q[k, i, i] += 2.0 * coeff
            else:
                # 1/2 * (Q_ij+Q_ji) * wi * wj = coeff * wi * wj
                # enforce symmetry: Q_ij = Q_ji = coeff
                Q[k, i, j] += coeff
                Q[k, j, i] += coeff

    # Optional symmetrize C (not Q; already symmetric)
    if cfg['symmetrize']:
        C_poly = 0.5 * (C_poly + C_poly.T)
    if cfg['spd_project']:
        C_poly = project_spd(C_poly)

    # Save poly2
    out_poly_C = f"{stem}_poly2_C.csv"
    out_poly_b = f"{stem}_poly2_bias.csv"
    out_poly_Q = f"{stem}_poly2_Q.npz"  # save full tensor in NPZ

    pd.DataFrame(C_poly,
        index=['x','y','z','rx','ry','rz'],
        columns=['Fx','Fy','Fz','Tx','Ty','Tz']).to_csv(out_poly_C, float_format='%.10e')
    pd.DataFrame(b_poly[None, :],
        index=['bias'], columns=['x','y','z','rx','ry','rz']).T.to_csv(out_poly_b, float_format='%.10e')
    # Q saved as: Q[k] is 6x6 for output k in order [x,y,z,rx,ry,rz]
    np.savez(out_poly_Q, Q=Q)

    print(f"[POLY2] Best lag: {lag_p}  mean R2: {np.mean(r2_p):.4f}")
    print(f"[POLY2] R2 per dimension: {np.round(r2_p, 4).tolist()}")
    print("[POLY2] Saved:")
    print(f"   C  -> {out_poly_C}")
    print(f"   b  -> {out_poly_b}")
    print(f"   Q  -> {out_poly_Q} (npz; Q.shape = (6,6,6))")

if __name__ == '__main__':
    main()
