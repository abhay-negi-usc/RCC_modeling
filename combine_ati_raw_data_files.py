import os
import glob
import pandas as pd

# Directory containing ATI (FTS) CSV logs
# Default to the project FTS data folder used elsewhere in this repo
DATA_DIR = "/home/rp/abhay_ws/RCC_modeling/FTS/data/fts"
FALLBACK_DIR = "/home/rp/abhay_ws/RCC_modeling/FTS/data"

# Pattern for ATI files (e.g., ati_192-168-10-133_YYYY-MM-DD_HH-MM-SS.csv)
FILE_PATTERN = "ati_*.csv"

# Output base directory (match the convention used by the KUKA combiner)
OUTPUT_DIR = "./data"

# Configuration dictionary. Provide the file paths in desired order here.
# Paths can be absolute or relative to DATA_DIR/FALLBACK_DIR.
CONFIG = {
    "files": [
        "./fts/ati_192-168-10-133_2025-11-04_19-55-31.csv",
        "./fts/ati_192-168-10-133_2025-11-04_20-09-20.csv",
        "./fts/ati_192-168-10-133_2025-11-04_20-30-01.csv",
        "./fts/ati_192-168-10-133_2025-11-04_20-42-44.csv",
        "./fts/ati_192-168-10-133_2025-11-04_20-55-08.csv",
        "./fts/ati_192-168-10-133_2025-11-04_21-21-44.csv",
        "./fts/ati_192-168-10-133_2025-11-04_21-32-25.csv",
        "./fts/ati_192-168-10-133_2025-11-04_21-45-05.csv",
        "./fts/ati_192-168-10-133_2025-11-04_22-08-06.csv",
        "./fts/ati_192-168-10-133_2025-11-04_22-18-05.csv",
        "./fts/ati_192-168-10-133_2025-11-04_22-57-57.csv",
        "./fts/ati_192-168-10-133_2025-11-04_23-15-10.csv",
        "./fts/ati_192-168-10-133_2025-11-04_23-24-09.csv",
        "./fts/ati_192-168-10-133_2025-11-04_23-40-08.csv",
        "./fts/ati_192-168-10-133_2025-11-04_23-46-53.csv",
    ],
    # Optional explicit output path. If None, defaults to ./data/RCC_ati_<N>_trials.csv
    "output_path": None,
}


def _resolve_path(p: str) -> str:
    """Resolve a file path that may be absolute or relative to known data dirs."""
    if os.path.isabs(p) and os.path.exists(p):
        return p
    cand = os.path.join(DATA_DIR, p)
    if os.path.exists(cand):
        return cand
    cand = os.path.join(FALLBACK_DIR, p)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"File not found: {p}")


def main():
    # Determine input files in desired order from CONFIG; fallback to auto-discovery
    files_in = CONFIG.get("files") or []

    if not files_in:
        # Auto-discover files (sorted)
        search_glob = os.path.join(DATA_DIR, FILE_PATTERN)
        files_in = sorted(glob.glob(search_glob))
        if not files_in:
            files_in = sorted(glob.glob(os.path.join(FALLBACK_DIR, FILE_PATTERN)))

    if not files_in:
        raise FileNotFoundError(
            f"No ATI CSV files provided or found matching {FILE_PATTERN} in {DATA_DIR} or {FALLBACK_DIR}."
        )

    # Resolve to absolute existing paths (preserve provided order)
    files = [_resolve_path(p) if not (os.path.isabs(p) and os.path.exists(p)) else p for p in files_in]

    # Read and append trial index; keep all columns as-is
    dfs = []
    for i, fp in enumerate(files, start=1):
        df = pd.read_csv(fp)
        # Add/overwrite 1-based trial index corresponding to file order (column appended at end)
        df["trial"] = i
        dfs.append(df)

    # Concatenate while preserving all columns (union); pandas will align by column names
    combined = pd.concat(dfs, ignore_index=True, sort=False)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_path = CONFIG.get("output_path") or os.path.join(OUTPUT_DIR, f"RCC_ati_{len(files)}_trials.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved combined ATI CSV: {out_path}")
    print(f"Files combined in order ({len(files)}):")
    for i, fp in enumerate(files, start=1):
        print(f"  {i:02d}: {fp}")


if __name__ == "__main__":
    main()
