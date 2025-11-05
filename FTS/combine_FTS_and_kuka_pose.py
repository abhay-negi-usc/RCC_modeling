import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict


data_dir = "/home/rp/abhay_ws/RCC_modeling/FTS/data/"
# FTS_data = "ati_192-168-10-133_2025-11-04_10-40-09.csv"
# kuka_data = "RCC_new_compliance_data_collection_2013-01-01_00-02-26_processed.csv"

trial_number = 0 
out_path = Path(data_dir) / f"RCC_perturbation_{trial_number}.csv"

def smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def first_peak_index(y: np.ndarray, min_height: float | None = None) -> int:
    """Return index of the first local maximum above min_height.
    Falls back to the global maximum if none found.
    """
    y = np.asarray(y)
    if y.size < 3:
        return int(np.argmax(y))
    if min_height is None:
        # Use median + 1*std as a conservative threshold to avoid noise.
        min_height = np.median(y) + 1.0 * np.std(y)
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_height:
            return i
    return int(np.argmax(y))


def find_all_peaks(y: np.ndarray, min_height: float, min_distance: int) -> np.ndarray:
    """Detect local maxima indices above min_height with a minimum sample spacing.
    Simple non-maximum suppression without SciPy.
    """
    y = np.asarray(y)
    if y.size < 3:
        return np.array([int(np.argmax(y))]) if y.size else np.array([], dtype=int)
    # Candidate local maxima
    cand = [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] >= min_height]
    if not cand:
        return np.array([], dtype=int)
    cand = np.array(cand, dtype=int)
    amps = y[cand]
    # Select by amplitude with non-maximum suppression
    order = np.argsort(-amps)  # descending by amplitude
    selected = []
    taken = np.zeros(len(y), dtype=bool)
    for idx in order:
        i = cand[idx]
        if not taken[max(0, i - min_distance): min(len(y), i + min_distance + 1)].any():
            selected.append(i)
            taken[max(0, i - min_distance): min(len(y), i + min_distance + 1)] = True
    selected = np.array(sorted(selected), dtype=int)
    return selected


def robust_lag_from_peak_times(t1: np.ndarray, t2: np.ndarray) -> float:
    """Estimate constant time lag dt such that t2 + dt aligns to t1, by matching nearest peaks.
    Returns median of nearest-neighbor deltas; robust to outliers.
    """
    if len(t1) == 0 or len(t2) == 0:
        return 0.0
    t2_sorted = np.sort(t2)
    deltas = []
    for x in t1:
        j = np.searchsorted(t2_sorted, x)
        cand = []
        if j < len(t2_sorted):
            cand.append(t2_sorted[j])
        if j > 0:
            cand.append(t2_sorted[j - 1])
        if cand:
            k = min(cand, key=lambda v: abs(v - x))
            deltas.append(x - k)
    if not deltas:
        return 0.0
    deltas = np.array(deltas)
    med = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - med))) if len(deltas) > 1 else 0.0
    # Filter gross outliers (3*MAD or 0.1s if MAD is tiny)
    tol = 3.0 * mad if mad > 0 else 0.1
    keep = np.abs(deltas - med) <= tol
    if keep.any():
        return float(np.median(deltas[keep]))
    return med


# New: reusable function for combining a single FTS/KUKA pair

def combine_fts_and_kuka_pose(
    fts_csv: str | Path,
    kuka_csv: str | Path,
    out_csv: Optional[str | Path] = None,
    plot: bool = False,
    offset_sec: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Combine a single pair of FTS (ATI) wrench data and KUKA pose data.

    - Aligns timelines by applying a manual time offset to KUKA time: t_kuka_shifted = t_kuka + offset_sec
    - Interpolates/upsamples KUKA pose (and selected signals) onto FTS time base
    - Outputs overlap-only combined dataframe (pose populated region)

    Returns (combined_overlap_df, metadata)
    """
    # Helpers for angle-safe interpolation (avoid 2π wrap jumps)
    def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _angle_interp(t_out: np.ndarray, t_in: np.ndarray, ang_in: np.ndarray) -> np.ndarray:
        ang_in = np.asarray(ang_in, dtype=float)
        ang_unwrapped = np.unwrap(ang_in)
        ang_out_unwrapped = np.interp(t_out, t_in, ang_unwrapped, left=np.nan, right=np.nan)
        return _wrap_to_pi(ang_out_unwrapped)

    fts_csv = str(fts_csv)
    kuka_csv = str(kuka_csv)

    df_fts = pd.read_csv(fts_csv)
    df_kuka = pd.read_csv(kuka_csv)

    # --- Prepare timelines ---
    # FTS time
    if "t_epoch" in df_fts.columns:
        t_fts = df_fts["t_epoch"].to_numpy()
    else:
        # fall back to ISO time if present
        if "iso_time" in df_fts.columns:
            t_fts = pd.to_datetime(df_fts["iso_time"]).astype("int64") / 1e9
        else:
            raise ValueError("FTS file missing 't_epoch' or 'iso_time' column")
    # Normalize to start at 0
    t_fts_rel = t_fts - t_fts[0]

    # KUKA time (already ~seconds)
    if "time_sec" not in df_kuka.columns:
        raise ValueError("KUKA file is missing 'time_sec' column.")
    t_kuka = df_kuka["time_sec"].to_numpy()

    # --- Apply manual time offset (positive shifts KUKA forward) ---
    t_kuka_shifted = t_kuka + float(offset_sec)

    # --- Build combined dataframe on FTS timeline ---
    Fmag_fts = np.linalg.norm(df_fts[["Fx", "Fy", "Fz"]].to_numpy(), axis=1)
    combined = pd.DataFrame(
        {
            "time": t_fts_rel,
            "Fx": df_fts["Fx"],
            "Fy": df_fts["Fy"],
            "Fz": df_fts["Fz"],
            "Tx": df_fts["Tx"],
            "Ty": df_fts["Ty"],
            "Tz": df_fts["Tz"],
            "Fmag": Fmag_fts,
        }
    )

    # Interpolate (upsample) KUKA pose and selected signals onto FTS time base
    order_kuka = np.argsort(t_kuka_shifted)
    t_kuka_sorted = t_kuka_shifted[order_kuka]

    # Positions (linear)
    for col in ["X", "Y", "Z"]:
        if col in df_kuka.columns:
            series = df_kuka[col].to_numpy()[order_kuka]
            combined[col] = np.interp(
                t_fts_rel, t_kuka_sorted, series, left=np.nan, right=np.nan
            )

    # Orientations (angle-safe)
    for col in ["A", "B", "C"]:
        if col in df_kuka.columns:
            series = df_kuka[col].to_numpy()[order_kuka]
            combined[col] = _angle_interp(t_fts_rel, t_kuka_sorted, series)

    # Also upsample lowercase orientation a, b, c if present (linear interpolation)
    for col in ["a", "b", "c"]:
        if col in df_kuka.columns:
            series = df_kuka[col].to_numpy()[order_kuka]
            combined[col] = np.interp(t_fts_rel, t_kuka_sorted, series, left=np.nan, right=np.nan)

    # For diagnostics and optional modeling: upsample KUKA forces too if present
    if set(["FX", "FY"]).issubset(df_kuka.columns):
        combined["FX_kuka_at_fts"] = np.interp(
            t_fts_rel, t_kuka_sorted, df_kuka["FX"].to_numpy()[order_kuka], left=np.nan, right=np.nan
        )
        combined["FY_kuka_at_fts"] = np.interp(
            t_fts_rel, t_kuka_sorted, df_kuka["FY"].to_numpy()[order_kuka], left=np.nan, right=np.nan
        )
    if "FZ" in df_kuka.columns:
        combined["FZ_kuka_at_fts"] = np.interp(
            t_fts_rel, t_kuka_sorted, df_kuka["FZ"].to_numpy()[order_kuka], left=np.nan, right=np.nan
        )

    # Determine overlap where KUKA pose exists after shifting
    kuka_t_min, kuka_t_max = float(t_kuka_sorted[0]), float(t_kuka_sorted[-1])
    overlap_mask = (combined["time"] >= kuka_t_min) & (combined["time"] <= kuka_t_max)
    combined_overlap = combined.loc[overlap_mask].reset_index(drop=True)

    # Metadata
    meta: Dict[str, float] = {
        "manual_offset_sec": float(offset_sec),
        "kuka_time_min": kuka_t_min,
        "kuka_time_max": kuka_t_max,
        "rows_total": int(len(combined)),
        "rows_overlap": int(len(combined_overlap)),
        "kuka_upsampled_to_fts": True,
    }

    if out_csv is not None:
        out_csv = str(out_csv)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        combined_overlap.to_csv(out_csv, index=False)

    if plot:
        fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex=True)
        axes = axes.reshape(4, 3)

        def plot_row(row, cols, labels, units=""):
            for j, (col, lab) in enumerate(zip(cols, labels)):
                ax = axes[row, j]
                if col in combined.columns:
                    ax.plot(combined["time"], combined[col], lw=1.0)
                ax.set_title(f"{lab}")
                if units:
                    ax.set_ylabel(units)
                ax.grid(True, alpha=0.3)

        # Row 0: Forces
        plot_row(0, ["Fx", "Fy", "Fz"], ["Fx (ATI)", "Fy (ATI)", "Fz (ATI)"], units="N")
        # Row 1: Torques
        plot_row(1, ["Tx", "Ty", "Tz"], ["Tx (ATI)", "Ty (ATI)", "Tz (ATI)"], units="Nm")
        # Row 2: Position
        plot_row(2, ["X", "Y", "Z"], ["X (KUKA)", "Y (KUKA)", "Z (KUKA)"], units="m")
        # Row 3: Orientation (uppercase)
        plot_row(3, ["A", "B", "C"], ["A (KUKA)", "B (KUKA)", "C (KUKA)"], units="rad")

        # Overlay lowercase a, b, c if present
        if all(col in combined.columns for col in ["a", "b", "c"]):
            axes[3, 0].plot(combined["time"], combined["a"], lw=1.0, ls="--")
            axes[3, 1].plot(combined["time"], combined["b"], lw=1.0, ls="--")
            axes[3, 2].plot(combined["time"], combined["c"], lw=1.0, ls="--")

        axes[-1, 0].set_xlabel("Time (s, FTS base)")
        axes[-1, 1].set_xlabel("Time (s, FTS base)")
        axes[-1, 2].set_xlabel("Time (s, FTS base)")
        fig.suptitle("FTS and KUKA (manual time offset applied)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

    return combined_overlap, meta


# New: multi-trial combiner

def combine_trials_kuka_ati(
    kuka_csv: str | Path,
    ati_csv: str | Path,
    out_csv: Optional[str | Path] = None,
    plot: bool = False,
    offsets: Optional[dict | list] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Combine multi-trial KUKA pose CSV and ATI/FTS wrench CSV.

    For each trial value present in both files:
      - Apply provided time offset per trial to align timelines (no auto-peak detection)
      - Choose the higher-frequency stream as the base timeline
      - Upsample the lower-frequency signals onto the base
      - Restrict to overlapping time window only
      - Bias wrench columns by the first sample in the overlap (forces/torques start at 0)
    Concatenate per-trial results and save a single CSV with original 'trial' values.

    offsets can be either:
      - dict: {trial_id: offset_sec}
      - list: [offset_sec for trial in sorted(common_trials)]
    Positive offset shifts KUKA forward in time (t_kuka_shifted = t_kuka + offset).
    """

    def _wrap_to_pi(a: np.ndarray) -> np.ndarray:
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _angle_interp(t_out: np.ndarray, t_in: np.ndarray, ang_in: np.ndarray) -> np.ndarray:
        ang_in = np.asarray(ang_in, dtype=float)
        ang_unwrapped = np.unwrap(ang_in)
        ang_out_unwrapped = np.interp(t_out, t_in, ang_unwrapped, left=np.nan, right=np.nan)
        return _wrap_to_pi(ang_out_unwrapped)

    kuka_csv = str(kuka_csv)
    ati_csv = str(ati_csv)

    df_kuka_all = pd.read_csv(kuka_csv)
    df_fts_all = pd.read_csv(ati_csv)

    if "trial" not in df_kuka_all.columns or "trial" not in df_fts_all.columns:
        raise ValueError("Both CSVs must contain a 'trial' column.")

    if "time_sec" not in df_kuka_all.columns:
        raise ValueError("KUKA CSV must contain a 'time_sec' column (seconds, relative or absolute).")

    # Determine FTS time column
    fts_time_col = None
    if "t_epoch" in df_fts_all.columns:
        fts_time_col = "t_epoch"
    elif "iso_time" in df_fts_all.columns:
        fts_time_col = "iso_time"
    elif "time_sec" in df_fts_all.columns:
        fts_time_col = "time_sec"
    else:
        raise ValueError("FTS/ATI CSV must have 't_epoch', 'iso_time' or 'time_sec' time column.")

    # Common trials only
    common_trials = sorted(set(df_kuka_all["trial"]).intersection(set(df_fts_all["trial"])))
    if not common_trials:
        raise ValueError("No common trial IDs found between the two CSVs.")

    # Map offsets to trials
    if offsets is None:
        raise ValueError("You must provide 'offsets' (dict or list) for manual alignment.")
    if isinstance(offsets, dict):
        tr_to_offset = {int(k): float(v) for k, v in offsets.items()}
    elif isinstance(offsets, list):
        if len(offsets) != len(common_trials):
            raise ValueError("If 'offsets' is a list, it must match the number of common trials.")
        tr_to_offset = {int(tr): float(off) for tr, off in zip(common_trials, offsets)}
    else:
        raise ValueError("'offsets' must be a dict or a list.")

    missing = [tr for tr in common_trials if tr not in tr_to_offset]
    if missing:
        raise ValueError(f"Missing offsets for trials: {missing}")

    combined_parts: list[pd.DataFrame] = []
    trials_done: list[int] = []

    for tr in common_trials:
        df_kuka = df_kuka_all[df_kuka_all["trial"] == tr].copy()
        df_fts = df_fts_all[df_fts_all["trial"] == tr].copy()
        if df_kuka.empty or df_fts.empty:
            continue

        # Timelines (make each relative to first sample of that trial)
        t_kuka = df_kuka["time_sec"].to_numpy(dtype=float)
        t_kuka_rel = t_kuka - (t_kuka[0] if len(t_kuka) else 0.0)

        if fts_time_col == "iso_time":
            t_fts = pd.to_datetime(df_fts[fts_time_col]).astype("int64").to_numpy() / 1e9
        else:
            t_fts = df_fts[fts_time_col].to_numpy(dtype=float)
        t_fts_rel = t_fts - (t_fts[0] if len(t_fts) else 0.0)

        # Manual offset for this trial
        offset = float(tr_to_offset[tr])

        # Choose base timeline = higher frequency (smaller dt)
        dt_fts = float(np.median(np.diff(t_fts_rel))) if len(t_fts_rel) > 1 else 1e-3
        dt_kuka = float(np.median(np.diff(t_kuka_rel))) if len(t_kuka_rel) > 1 else 5e-3
        use_fts_base = dt_fts <= dt_kuka

        if use_fts_base:
            base_t = t_fts_rel
            # Shift KUKA forward by offset to align with FTS timeline
            t_kuka_shifted = t_kuka_rel + offset
            # Overlap window on base timeline
            kuka_t_min, kuka_t_max = float(t_kuka_shifted[0]), float(t_kuka_shifted[-1])
            mask = (base_t >= kuka_t_min) & (base_t <= kuka_t_max)
            if not mask.any():
                continue
            t_out = base_t[mask]

            # Build combined rows on base timeline
            part = pd.DataFrame({
                "trial": tr,
                "time": t_out,
            })
            # Wrench from FTS directly (already on base) -> save as *_ati
            for col in ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]:
                if col in df_fts.columns:
                    part[f"{col}_ati"] = df_fts[col].to_numpy(dtype=float)[mask]

            # Pose from KUKA interpolated onto base
            order_kuka = np.argsort(t_kuka_shifted)
            t_kuka_sorted = t_kuka_shifted[order_kuka]
            for col in ["X", "Y", "Z"]:
                if col in df_kuka.columns:
                    series = df_kuka[col].to_numpy(dtype=float)[order_kuka]
                    part[col] = np.interp(t_out, t_kuka_sorted, series, left=np.nan, right=np.nan)
            for col in ["A", "B", "C"]:
                if col in df_kuka.columns:
                    series = df_kuka[col].to_numpy(dtype=float)[order_kuka]
                    part[col] = _angle_interp(t_out, t_kuka_sorted, series)
            # Also interpolate lowercase a, b, c if present (linear)
            for col in ["a", "b", "c"]:
                if col in df_kuka.columns:
                    series = df_kuka[col].to_numpy(dtype=float)[order_kuka]
                    part[col] = np.interp(t_out, t_kuka_sorted, series, left=np.nan, right=np.nan)

            # KUKA wrench interpolated onto base if present -> save as *_kuka
            if set(["FX", "FY", "FZ", "TX", "TY", "TZ"]).issubset(df_kuka.columns):
                for src, dst in zip(["FX", "FY", "FZ", "TX", "TY", "TZ"], ["Fx_kuka", "Fy_kuka", "Fz_kuka", "Tx_kuka", "Ty_kuka", "Tz_kuka"]):
                    series = df_kuka[src].to_numpy(dtype=float)[order_kuka]
                    part[dst] = np.interp(t_out, t_kuka_sorted, series, left=np.nan, right=np.nan)
        else:
            base_t = t_kuka_rel
            # Shift FTS backward by offset so that FTS aligns to KUKA timeline
            t_fts_shifted = t_fts_rel - offset
            fts_t_min, fts_t_max = float(t_fts_shifted[0]), float(t_fts_shifted[-1])
            mask = (base_t >= fts_t_min) & (base_t <= fts_t_max)
            if not mask.any():
                continue
            t_out = base_t[mask]

            # Build combined rows on base timeline
            part = pd.DataFrame({
                "trial": tr,
                "time": t_out,
            })
            # Pose from KUKA directly (already on base)
            for col in ["X", "Y", "Z", "A", "B", "C"]:
                if col in df_kuka.columns:
                    part[col] = df_kuka[col].to_numpy(dtype=float)[mask]
            # Also copy lowercase a, b, c if present
            for col in ["a", "b", "c"]:
                if col in df_kuka.columns:
                    part[col] = df_kuka[col].to_numpy(dtype=float)[mask]

            # KUKA wrench directly on base if present -> save as *_kuka
            if set(["FX", "FY", "FZ", "TX", "TY", "TZ"]).issubset(df_kuka.columns):
                for src, dst in zip(["FX", "FY", "FZ", "TX", "TY", "TZ"], ["Fx_kuka", "Fy_kuka", "Fz_kuka", "Tx_kuka", "Ty_kuka", "Tz_kuka"]):
                    part[dst] = df_kuka[src].to_numpy(dtype=float)[mask]

            # Wrench from FTS interpolated onto base -> save as *_ati
            order_fts = np.argsort(t_fts_shifted)
            t_fts_sorted = t_fts_shifted[order_fts]
            for col, dst in zip(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], ["Fx_ati", "Fy_ati", "Fz_ati", "Tx_ati", "Ty_ati", "Tz_ati"]):
                if col in df_fts.columns:
                    series = df_fts[col].to_numpy(dtype=float)[order_fts]
                    part[dst] = np.interp(t_out, t_fts_sorted, series, left=np.nan, right=np.nan)

        # Drop rows with any NaNs (outside interpolation range), then bias wrench
        part = part.dropna().reset_index(drop=True)
        if part.empty:
            continue

        # Bias each available wrench column set so first value is zero
        ati_cols = [c for c in ["Fx_ati", "Fy_ati", "Fz_ati", "Tx_ati", "Ty_ati", "Tz_ati"] if c in part.columns]
        kuka_cols = [c for c in ["Fx_kuka", "Fy_kuka", "Fz_kuka", "Tx_kuka", "Ty_kuka", "Tz_kuka"] if c in part.columns]
        for col in ati_cols + kuka_cols:
            part[col] = part[col].to_numpy(dtype=float) - float(part[col].iloc[0])

        # Force magnitudes
        if set(["Fx_ati", "Fy_ati", "Fz_ati"]).issubset(part.columns):
            part["Fmag_ati"] = np.linalg.norm(part[["Fx_ati", "Fy_ati", "Fz_ati"]].to_numpy(dtype=float), axis=1)
        if set(["Fx_kuka", "Fy_kuka", "Fz_kuka"]).issubset(part.columns):
            part["Fmag_kuka"] = np.linalg.norm(part[["Fx_kuka", "Fy_kuka", "Fz_kuka"]].to_numpy(dtype=float), axis=1)

        # Normalized pose columns (±2σ -> ±1)
        for pose_col in ["X", "Y", "Z", "A", "B", "C"]:
            if pose_col in part.columns:
                vals = part[pose_col].to_numpy(dtype=float)
                mean_val = float(np.nanmean(vals)) if len(vals) else 0.0
                std_val = float(np.nanstd(vals)) if len(vals) else 0.0
                denom = 2.0 * std_val if std_val and np.isfinite(std_val) else None
                if denom and denom > 0:
                    part[f"{pose_col}_norm"] = (vals - mean_val) / denom
                else:
                    part[f"{pose_col}_norm"] = 0.0

        combined_parts.append(part)
        trials_done.append(tr)

        if plot:
            # 4 rows now: Forces, Torques, Position, Orientation
            fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True)
            axes = axes.reshape(4, 3)
            # Forces (ATI and KUKA)
            if "Fx_ati" in part.columns:
                axes[0, 0].plot(part["time"], part["Fx_ati"], label="ATI")
                axes[0, 1].plot(part["time"], part["Fy_ati"], label="ATI")
                axes[0, 2].plot(part["time"], part["Fz_ati"], label="ATI")
            if "Fx_kuka" in part.columns:
                axes[0, 0].plot(part["time"], part["Fx_kuka"], label="KUKA")
                axes[0, 1].plot(part["time"], part["Fy_kuka"], label="KUKA")
                axes[0, 2].plot(part["time"], part["Fz_kuka"], label="KUKA")
            axes[0, 0].set_title(f"Trial {tr} Fx"); axes[0, 0].legend()
            axes[0, 1].set_title("Fy"); axes[0, 1].legend()
            axes[0, 2].set_title("Fz"); axes[0, 2].legend()
            # Torques
            if "Tx_ati" in part.columns:
                axes[1, 0].plot(part["time"], part["Tx_ati"], label="ATI")
                axes[1, 1].plot(part["time"], part["Ty_ati"], label="ATI")
                axes[1, 2].plot(part["time"], part["Tz_ati"], label="ATI")
            if "Tx_kuka" in part.columns:
                axes[1, 0].plot(part["time"], part["Tx_kuka"], label="KUKA")
                axes[1, 1].plot(part["time"], part["Ty_kuka"], label="KUKA")
                axes[1, 2].plot(part["time"], part["Tz_kuka"], label="KUKA")
            axes[1, 0].set_title("Tx"); axes[1, 0].legend()
            axes[1, 1].set_title("Ty"); axes[1, 1].legend()
            axes[1, 2].set_title("Tz"); axes[1, 2].legend()
            # Position
            axes[2, 0].plot(part["time"], part["X"]) if "X" in part.columns else None; axes[2, 0].set_title("X")
            axes[2, 1].plot(part["time"], part["Y"]) if "Y" in part.columns else None; axes[2, 1].set_title("Y")
            axes[2, 2].plot(part["time"], part["Z"]) if "Z" in part.columns else None; axes[2, 2].set_title("Z")
            # Orientation (uppercase)
            if all(c in part.columns for c in ["A", "B", "C"]):
                axes[3, 0].plot(part["time"], part["A"]) ; axes[3, 0].set_title("A")
                axes[3, 1].plot(part["time"], part["B"]) ; axes[3, 1].set_title("B")
                axes[3, 2].plot(part["time"], part["C"]) ; axes[3, 2].set_title("C")
            # Overlay lowercase if present
            if all(c in part.columns for c in ["a", "b", "c"]):
                axes[3, 0].plot(part["time"], part["a"], ls="--")
                axes[3, 1].plot(part["time"], part["b"], ls="--")
                axes[3, 2].plot(part["time"], part["c"], ls="--")

            for ax in axes.ravel():
                ax.grid(True, alpha=0.3)
            axes[-1, 0].set_xlabel("Time (s)")
            axes[-1, 1].set_xlabel("Time (s)")
            axes[-1, 2].set_xlabel("Time (s)")
            fig.suptitle("Synchronized (biased) Wrench and Pose — manual offsets")
            fig.tight_layout()
            plt.show()

    if not combined_parts:
        raise ValueError("No trials could be synchronized and combined.")

    combined_all = pd.concat(combined_parts, ignore_index=True)

    if out_csv is not None:
        out_csv = str(out_csv)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        combined_all.to_csv(out_csv, index=False)

    meta = {
        "num_trials": int(len(trials_done)),
        "trials": list(trials_done),
        "rows_out": int(len(combined_all)),
    }
    return combined_all, meta


if __name__ == "__main__":
    # Configuration dict (edit these paths/options)
    CONFIG = {
        "kuka_csv": "./data/RCC_kuka_15_trials_processed.csv",  # KUKA pose CSV (must contain 'trial' and 'time_sec')
        "ati_csv": "./data/RCC_ati_15_trials.csv",    # ATI/FTS wrench CSV (must contain 'trial' and time column)
        "out_csv": "./data/RCC_kuka_ati_15_trials.csv",                   # Output CSV path; if None, defaults next to ati_csv
        "plot": False,                     # Set True for per-trial diagnostic plots
        # Positive offsets shift KUKA forward in time. Provide one per trial in sorted(common_trials).
        # Example: for 15 trials -> 15 values. Adjust as needed.
        "offsets": [23.5582-10.351, 120.2-115.0, 28.1-24.91, 59.5-57.2, 307.7-303.9, 13.0-9.3, 11.22-8.52, 57.53-55.84, 334.54-332.87, 54.52-53.09, 12.02-10.92, 198.91-196.90, 222.74-220.64, 54.11-52.47, 10.82-8.94],
    }

    out = CONFIG["out_csv"]
    if out is None:
        out = str(Path(CONFIG["ati_csv"]).with_name("combined_trials_from_kuka_ati.csv"))

    df_comb, meta = combine_trials_kuka_ati(
        CONFIG["kuka_csv"], CONFIG["ati_csv"], out_csv=out, plot=CONFIG["plot"], offsets=CONFIG["offsets"]
    )
    print(f"Saved: {out}")
    print(f"Trials combined: {meta['num_trials']} -> {meta['trials']}")
    print(f"Rows saved: {meta['rows_out']}")


