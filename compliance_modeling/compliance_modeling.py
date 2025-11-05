#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate a 6x6 compliance matrix C from (wrench, relative pose) time series.
Now uses a CONFIG dictionary instead of argparse.

Model: x_t ≈ C * w_t  ⇒  X ≈ W @ C^T
"""

import numpy as np
import pandas as pd

CONFIG = {
    # "csv": "/home/rp/abhay_ws/RCC_modeling/FTS/data/combined_fts_wrench_kuka_pose_aligned_by_FZ_peaks.csv",
    "csv": "./data/RCC_kuka_ati_15_trials.csv",
    "wrench_cols": ["Fx_ati", "Fy_ati", "Fz_ati", "Tx_ati", "Ty_ati", "Tz_ati"],
    "pose_cols": ["X", "Y", "Z", "C", "B", "A"],
    "angles_deg": True,
    "demean": True,
    "lag_scan": 10,
    "ridge": 1e-3,
    "huber_delta": 1.0,
    "max_irwls_iters": 100,
    "symmetrize": False,
    "spd_project": True,
    "out": "./compliance_modeling/compliance_matrix.csv",
}

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

def weighted_ridge(W, X, w=None, ridge=0.0):
    if w is not None:
        sw = np.sqrt(w).reshape(-1, 1)
        Ww = W * sw
        Xw = X * sw
    else:
        Ww, Xw = W, X
    WTW = Ww.T @ Ww
    if ridge > 0:
        WTW += ridge * np.eye(6)
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

def select_best_lag(W, X, K, ridge, huber_delta, max_iters):
    best = (-np.inf, 0, None, None)
    for lag in range(-K, K + 1):
        Wl, Xl = apply_lag(W, X, lag)
        if len(Wl) < 10:
            continue
        if huber_delta > 0:
            A = weighted_ridge(Wl, Xl, None, ridge)
            for _ in range(max_iters):
                res = Xl - Wl @ A
                w = huber_weights(res, huber_delta)
                A_new = weighted_ridge(Wl, Xl, w, ridge)
                if np.linalg.norm(A_new - A) / (np.linalg.norm(A) + 1e-12) < 1e-5:
                    A = A_new
                    break
                A = A_new
        else:
            A = weighted_ridge(Wl, Xl, None, ridge)
        r2, _ = r2_multitarget(Wl, Xl, A)
        score = float(np.mean(r2))
        if score > best[0]:
            best = (score, lag, A, r2)
    return best[1], best[2], best[3]

def main():
    cfg = CONFIG
    df = pd.read_csv(cfg['csv'])
    W = df[cfg['wrench_cols']].to_numpy(float)
    X = df[cfg['pose_cols']].to_numpy(float)

    if cfg['angles_deg']:
        X = to_radians(X, [3, 4, 5])

    W, X = standardize_missing(W, X)
    if cfg['demean']:
        W -= W.mean(0, keepdims=True)
        X -= X.mean(0, keepdims=True)

    lag, A, r2 = select_best_lag(W, X, cfg['lag_scan'], cfg['ridge'], cfg['huber_delta'], cfg['max_irwls_iters'])
    if A is None:
        raise RuntimeError('Failed to estimate compliance matrix.')

    C = A.T
    if cfg['symmetrize']:
        C = 0.5 * (C + C.T)
    if cfg['spd_project']:
        C = project_spd(C)

    print(f"Best lag: {lag}")
    print(f"R2 per dimension: {np.round(r2, 4).tolist()}")
    print("Compliance Matrix (C):\n", np.array2string(C, formatter={'float_kind':lambda x: f'{x: .6e}'}))

    pd.DataFrame(C, index=['x','y','z','rx','ry','rz'], columns=['Fx','Fy','Fz','Tx','Ty','Tz']).to_csv(cfg['out'], float_format='%.10e')
    print(f"Saved to {cfg['out']}")

if __name__ == '__main__':
    main()
