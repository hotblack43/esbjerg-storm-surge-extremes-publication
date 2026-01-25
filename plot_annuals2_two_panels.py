#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPO = Path(".").resolve()
IN_DIR = REPO / "OUTPUT" / "ANNUALS2"
OUT_DIR = REPO / "FIGURES" / "ALLANNAUL"


def read_table_any(path: Path) -> pd.DataFrame:
    """
    Read a table-like file into a DataFrame.
    Supports:
      - .csv (comma)
      - .tsv (tab)
      - .txt / unknown: tries comma, then tab, then whitespace
      - .rds (if pyreadr installed)
    """
    suf = path.suffix.lower()

    if suf == ".rds":
        try:
            import pyreadr  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "File is .rds but pyreadr is not installed. "
                "Install with: python3 -m pip install pyreadr"
            ) from e
        res = pyreadr.read_r(str(path))
        if not res:
            raise RuntimeError("pyreadr returned no objects")
        df = next(iter(res.values()))
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError("RDS did not contain a data.frame-like object")
        return df

    # CSV/TSV/other text: try a cascade of separators
    try_seps = [",", "\t", r"\s+"]

    last_err = None
    for sep in try_seps:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            df = df.dropna(axis=1, how="all")
            if df.shape[1] >= 3:
                return df
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Could not parse {path.name} as a table. Last error: {last_err}")


def to_numeric(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def plot_file(path: Path) -> None:
    df = read_table_any(path)

    if df.shape[1] < 3:
        print(f"SKIP {path.name}: need >= 3 columns, got {df.shape[1]}")
        return

    x = to_numeric(df.iloc[:, 0])
    y1 = to_numeric(df.iloc[:, -2])
    y2 = to_numeric(df.iloc[:, -1])

    # Keep rows where x is finite; y can be NaN (we'll mask per-panel)
    mx = np.isfinite(x)
    if mx.sum() < 2:
        print(f"SKIP {path.name}: not enough numeric x values")
        return

    x = x[mx]
    y1 = y1[mx]
    y2 = y2[mx]

    # Sort by x for readability (still points-only)
    order = np.argsort(x)
    x = x[order]
    y1 = y1[order]
    y2 = y2[order]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=150, sharex=True)

    # Points only (no lines)
    m1 = np.isfinite(y1)
    axes[0].scatter(x[m1], y1[m1], s=10)
    axes[0].set_ylabel(df.columns[-2] if df.columns[-2] else "col[-2]")

    m2 = np.isfinite(y2)
    axes[1].scatter(x[m2], y2[m2], s=10)
    axes[1].set_ylabel(df.columns[-1] if df.columns[-1] else "col[-1]")
    axes[1].set_xlabel(df.columns[0] if df.columns[0] else "col[0]")

    fig.suptitle(path.stem)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / f"{path.stem}_two_panels_points.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"WROTE {out_png}")


def main() -> int:
    if not IN_DIR.exists():
        print(f"ERROR: input directory does not exist: {IN_DIR}")
        return 2

    files = sorted([p for p in IN_DIR.iterdir() if p.is_file()])
    if not files:
        print(f"ERROR: no files found in: {IN_DIR}")
        return 2

    n_ok = 0
    n_fail = 0

    for p in files:
        try:
            plot_file(p)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"FAIL {p.name}: {e}", file=sys.stderr)

    print(f"Done. OK={n_ok}, FAIL={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

