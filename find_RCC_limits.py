import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class SliceStats:
    bin_centers: np.ndarray          # (S,)
    counts: np.ndarray               # (S,)
    min_: Dict[int, np.ndarray]      # j -> (S,)
    max_: Dict[int, np.ndarray]      # j -> (S,)
    median: Dict[int, np.ndarray]    # j -> (S,)

def _make_bins(x: np.ndarray, max_slices: int, binning: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bin edges, centers, and bin index per sample for a 1D array."""
    x = np.asarray(x).ravel()
    S = int(max(2, max_slices))
    if binning == "quantile":
        # quantile bins (robust to skew)
        q = np.linspace(0, 100, S + 1)
        edges = np.percentile(x, q)
        # ensure strictly increasing edges
        edges = np.unique(edges)
        if len(edges) < 3:
            # fallback to equal-width if quantiles collapse
            edges = np.linspace(x.min(), x.max(), S + 1)
    elif binning == "equal":
        edges = np.linspace(x.min(), x.max(), S + 1)
    else:
        raise ValueError("binning must be 'quantile' or 'equal'")

    # digitize: bins are [edges[k], edges[k+1]), last bin includes right edge
    idx = np.digitize(x, edges[1:-1], right=False)
    # bin centers
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers, idx

def _agg_per_bin(y: np.ndarray, bin_idx: np.ndarray, S: int, robust: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate y by bins: min/max (or robust percentiles), and median + counts.
    robust in [0,50): if >0, uses (robust, 100-robust) percentiles instead of min/max.
    """
    y = np.asarray(y).ravel()
    minv = np.full(S, np.nan)
    maxv = np.full(S, np.nan)
    medv = np.full(S, np.nan)
    counts = np.zeros(S, dtype=int)

    for b in range(S):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        yb = y[mask]
        counts[b] = yb.size
        if robust > 0:
            lo, hi = np.percentile(yb, [robust, 100 - robust])
            minv[b], maxv[b] = lo, hi
        else:
            minv[b], maxv[b] = np.min(yb), np.max(yb)
        medv[b] = np.median(yb)
    return minv, maxv, medv, counts

def slice_limits(
    X: np.ndarray,
    max_slices: int = 12,
    binning: str = "quantile",
    robust_pct: float = 2.5,
    min_bin_count: int = 10,
) -> Dict[int, SliceStats]:
    """
    For each dimension i, slice X along column i into at most `max_slices` bins,
    then compute per-slice limits (min/max or robust percentiles) for every other
    dimension j != i.

    Parameters
    ----------
    X : (M, N) array
        Dataset with M samples, N features.
    max_slices : int
        Maximum number of slices (bins) per slicing dimension.
    binning : {'quantile', 'equal'}
        Quantile bins (equal-mass) or equal-width bins.
    robust_pct : float in [0, 50)
        If >0, use (robust_pct, 100-robust_pct) percentiles instead of strict min/max.
        Helps avoid outlier-driven “spikes” in limit bands.
    min_bin_count : int
        Bins with fewer than this many samples will be left as NaN (insufficient data).

    Returns
    -------
    per_dim : dict
        Keys are slicing dimension i. Values are SliceStats with:
        - bin_centers: (S,)
        - counts: (S,)
        - min_/max_/median: dict j -> (S,) arrays
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (M, N).")
    M, N = X.shape
    S = int(max(2, max_slices))
    robust = float(robust_pct)
    if not (0 <= robust < 50):
        raise ValueError("robust_pct must be in [0, 50).")

    per_dim: Dict[int, SliceStats] = {}
    for i in range(N):
        edges, centers, idx = _make_bins(X[:, i], S, binning=binning)

        min_by_j: Dict[int, np.ndarray] = {}
        max_by_j: Dict[int, np.ndarray] = {}
        med_by_j: Dict[int, np.ndarray] = {}
        counts_global = np.zeros(S, dtype=int)

        # pre-compute per-bin masks once for speed
        bin_masks = [(idx == b) for b in range(S)]
        counts_global = np.array([np.count_nonzero(m) for m in bin_masks], dtype=int)

        for j in range(N):
            if j == i:
                continue
            y = X[:, j]
            minv, maxv, medv, _ = _agg_per_bin(y, idx, S, robust=robust)
            # wipe bins with too few samples
            bad = counts_global < min_bin_count
            minv[bad] = np.nan
            maxv[bad] = np.nan
            medv[bad] = np.nan

            min_by_j[j] = minv
            max_by_j[j] = maxv
            med_by_j[j] = medv

        per_dim[i] = SliceStats(
            bin_centers=centers, counts=counts_global,
            min_=min_by_j, max_=max_by_j, median=med_by_j
        )
    return per_dim

def plot_slice_limits(
    X: np.ndarray,
    stats: Dict[int, SliceStats],
    feature_names: Optional[List[str]] = None,
    show_median: bool = True,
    figsize_scale: float = 2.2,
    suptitle: Optional[str] = "Limit bands via slicing",
    tight_layout: bool = True,
):
    """
    Make an N x N grid:
      - Row i, Col j (i != j): shaded band shows min/max of feature j across slices of feature i.
        Optional line is the per-slice median of feature j.
      - Diagonal (i == j): histogram of feature i for reference.

    Parameters
    ----------
    X : (M, N) array
    stats : dict from `slice_limits`
    feature_names : list of str, optional
    show_median : bool
    figsize_scale : float
    suptitle : str or None
    tight_layout : bool
    """
    X = np.asarray(X)
    M, N = X.shape
    if feature_names is None:
        feature_names = [f"x{c}" for c in range(N)]
    
    # Compute global min/max for each dimension
    global_min = np.min(X, axis=0)
    global_max = np.max(X, axis=0)

    fig, axes = plt.subplots(N, N, figsize=(figsize_scale*N, figsize_scale*N), squeeze=False)

    for i in range(N):
        si = stats[i]
        x_centers = si.bin_centers
        # For diagonal: histogram of slicing var
        ax = axes[i, i]
        ax.hist(X[:, i], bins=40, alpha=0.7)
        ax.set_title(f"{feature_names[i]} (hist)")
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel("count")

        # Off-diagonals: bands of j vs i
        for j in range(N):
            if j == i:
                continue
            ax = axes[i, j]
            lo = si.min_[j]
            hi = si.max_[j]
            ax.fill_between(x_centers, lo, hi, alpha=0.25, linewidth=0)
            if show_median:
                med = si.median[j]
                ax.plot(x_centers, med, lw=1.5)
            
            # Add dashed lines for global min/max of dimension j
            ax.axhline(y=global_min[j], color='red', linestyle='--', alpha=0.7, linewidth=1, label=f'Global min')
            ax.axhline(y=global_max[j], color='blue', linestyle='--', alpha=0.7, linewidth=1, label=f'Global max')

            ax.set_xlabel(f"{feature_names[i]} slices")
            ax.set_ylabel(feature_names[j])
            ax.set_title(f"{feature_names[j]} vs {feature_names[i]}")
            # hide NaN-only panels nicely
            if np.all(np.isnan(lo)) and np.all(np.isnan(hi)):
                ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(x_centers.min(), x_centers.max())
            else:
                # Add legend only to panels with data
                ax.legend(loc='upper right', fontsize='small')

    if suptitle:
        fig.suptitle(suptitle, y=0.99)
    if tight_layout:
        plt.tight_layout()
    return fig, axes

# ----- Example usage -----
if __name__ == "__main__":

    data_path = "./data/RCC_mounted_data_2013-01-08_11-04-47_processed.csv"
    df = pd.read_csv(data_path)
    dimension_labels = ['X', 'Y', 'Z', 'A', 'B', 'C']
    X = df[dimension_labels].to_numpy()
    stats = slice_limits(X, max_slices=16, binning="quantile", robust_pct=2.5, min_bin_count=25)
    plot_slice_limits(X, stats, feature_names=dimension_labels, suptitle="Limit envelopes (robust 2.5–97.5%)")
    plt.show()
