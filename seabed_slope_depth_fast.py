#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# uv run seabed_slope_depth_fast.py --column slope_median
# or
# uv run seabed_slope_depth_fast.py --column depth_band_median_m
#

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------------------
# Suppress ALL warnings (including MatplotlibDeprecationWarning spam)
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore")

import numpy as np

try:
    import polars as pl
except Exception as e:
    raise RuntimeError("Missing dependency 'polars'. Install with: uv pip install polars") from e

try:
    import pyreadr  # reads .rds
except Exception as e:
    raise RuntimeError("Missing dependency 'pyreadr'. Install with: uv pip install pyreadr") from e

try:
    from scipy import stats
except Exception as e:
    raise RuntimeError("Missing dependency 'scipy'. Install with: uv pip install scipy") from e

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError("Missing dependency 'matplotlib'. Install with: uv pip install matplotlib") from e


# ----------------------------
# I/O helpers / formatting
# ----------------------------

def format_p_value(p: float) -> str:
    """
    Show p-values in scientific notation when they'd otherwise look like zeros
    under fixed decimal formatting.

    Rule:
      - p < 1e-4 => scientific (e.g. 3.2e-07)
      - else     => fixed 4 decimals
    """
    if p is None or not np.isfinite(p):
        return "p = NA"
    if p <= 0:
        return "p < 1e-300"
    if p < 1e-4:
        return f"p = {p:.2e}"
    return f"p = {p:.4f}"


def ensure_filename_column_pandas(pdf, df_name: str = "df"):
    """
    If pandas df has 'filename' do nothing.
    Else if pandas df has 'filename_x', rename it to 'filename'.
    Else: error.
    """
    cols = list(pdf.columns)
    if "filename" in cols:
        return pdf
    if "filename_x" in cols:
        return pdf.rename(columns={"filename_x": "filename"})
    raise ValueError(f"{df_name} has neither 'filename' nor 'filename_x' column")


def normalise_filename_expr(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.strip_chars().str.to_lowercase()


def read_rds_as_polars(path: str, df_name: str) -> pl.DataFrame:
    """
    Read an .rds file into a Polars DataFrame WITHOUT requiring pyarrow.

    We avoid pl.from_pandas() because pandas nullable dtypes (Int64, boolean, string)
    trigger the pyarrow requirement. Instead, we construct a Polars DataFrame column-by-column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    res = pyreadr.read_r(path)
    if len(res.keys()) < 1:
        raise ValueError(f"{df_name}: RDS contained no objects: {path}")

    pdf = list(res.values())[0]  # pandas DataFrame
    pdf = ensure_filename_column_pandas(pdf, df_name=df_name)

    data = {}
    for c in pdf.columns:
        s = pdf[c]
        dtype_name = getattr(s.dtype, "name", str(s.dtype))

        # pandas nullable / extension dtypes -> convert to object arrays
        if dtype_name in ("Int64", "UInt64", "boolean", "string"):
            data[c] = s.astype(object).to_numpy()
        else:
            data[c] = s.to_numpy()

    return pl.DataFrame(data)


# ----------------------------
# Core stats
# ----------------------------

@dataclass
class ResultRow:
    column: str
    inner_radius_km: float
    outer_radius_km: float
    beta_group1_mean: float
    beta_group2_mean: float
    delta_beta: float
    t_p_value: float
    group1: str
    group2: str
    type: str


def choose_groups(column_to_use: str, x: np.ndarray) -> Tuple[np.ndarray, str, str]:
    """
    Median-split grouping logic matching the Rmd:
      - slope* => gentle/steep
      - depth* => deep/shallow  (x < med => deep, else shallow)
      - else   => low/high
    Returns: group labels array, group1_name, group2_name
    """
    med = np.nanmedian(x)
    col_lower = column_to_use.lower()

    if "slope" in col_lower:
        grp = np.where(x < med, "gentle", "steep")
        return grp, "gentle", "steep"

    if "depth" in col_lower:
        grp = np.where(x < med, "deep", "shallow")
        return grp, "deep", "shallow"

    grp = np.where(x < med, "low", "high")
    return grp, "low", "high"


def welch_t_pvalue(y1: np.ndarray, y2: np.ndarray) -> float:
    y1 = y1[~np.isnan(y1)]
    y2 = y2[~np.isnan(y2)]
    if y1.size < 2 or y2.size < 2:
        return np.nan
    t = stats.ttest_ind(y1, y2, equal_var=False, nan_policy="omit")
    return float(t.pvalue)


def compute_results_for_type(
    df_joined: pl.DataFrame,
    column_to_use: str,
    type_label: str,
) -> List[ResultRow]:
    needed = {"inner_radius_km", "outer_radius_km", "beta1", column_to_use}
    missing = [c for c in needed if c not in df_joined.columns]
    if missing:
        raise ValueError(f"{type_label}: missing columns: {missing}")

    rings = (
        df_joined.select(["inner_radius_km", "outer_radius_km"])
        .unique()
        .filter(pl.col("outer_radius_km") > pl.col("inner_radius_km"))
        .sort(["inner_radius_km", "outer_radius_km"])
        .to_numpy()
    )

    out: List[ResultRow] = []

    for inner, outer in rings:
        sub = df_joined.filter(
            (pl.col("inner_radius_km") == inner) & (pl.col("outer_radius_km") == outer)
        ).select([column_to_use, "beta1"])

        if sub.height < 4:
            continue

        x = sub[column_to_use].to_numpy()
        y = sub["beta1"].to_numpy()

        if np.all(np.isnan(x)):
            continue

        grp, g1, g2 = choose_groups(column_to_use, x)
        n1 = int(np.sum(grp == g1))
        n2 = int(np.sum(grp == g2))
        if n1 < 2 or n2 < 2:
            continue

        y1 = y[grp == g1]
        y2 = y[grp == g2]

        p = welch_t_pvalue(y1, y2)
        if np.isnan(p):
            continue

        m1 = float(np.nanmean(y1))
        m2 = float(np.nanmean(y2))

        out.append(
            ResultRow(
                column=column_to_use,
                inner_radius_km=float(inner),
                outer_radius_km=float(outer),
                beta_group1_mean=m1,
                beta_group2_mean=m2,
                delta_beta=m2 - m1,
                t_p_value=float(p),
                group1=g1,
                group2=g2,
                type=type_label,
            )
        )

    return out


def rows_to_polars(rows: List[ResultRow]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            {
                "column": [],
                "inner_radius_km": [],
                "outer_radius_km": [],
                "beta_group1_mean": [],
                "beta_group2_mean": [],
                "delta_beta": [],
                "t_p_value": [],
                "group1": [],
                "group2": [],
                "type": [],
            }
        )

    return pl.DataFrame(
        {
            "column": [r.column for r in rows],
            "inner_radius_km": [r.inner_radius_km for r in rows],
            "outer_radius_km": [r.outer_radius_km for r in rows],
            "beta_group1_mean": [r.beta_group1_mean for r in rows],
            "beta_group2_mean": [r.beta_group2_mean for r in rows],
            "delta_beta": [r.delta_beta for r in rows],
            "t_p_value": [r.t_p_value for r in rows],
            "group1": [r.group1 for r in rows],
            "group2": [r.group2 for r in rows],
            "type": [r.type for r in rows],
        }
    )


# ----------------------------
# Plotting
# ----------------------------

def _boxplot_colours_for_groups(group_names: List[str]) -> List[str]:
    cols = []
    for g in group_names:
        gl = g.lower()
        if gl in ("gentle", "shallow"):
            cols.append("#90ee90")
        elif gl in ("steep", "deep"):
            cols.append("#f08080")
        else:
            cols.append("#cccccc")
    return cols


def _boxplot(ax, data, group_labels):
    """
    Compatibility wrapper:
    - Matplotlib 3.9+ uses tick_labels=
    - Older Matplotlib uses labels=
    """
    try:
        return ax.boxplot(
            data,
            tick_labels=group_labels,
            patch_artist=True,
            widths=0.6,
            showfliers=True,
        )
    except TypeError:
        return ax.boxplot(
            data,
            labels=group_labels,
            patch_artist=True,
            widths=0.6,
            showfliers=True,
        )


def save_split_boxplots(
    df_stat: pl.DataFrame,
    df_non: pl.DataFrame,
    results_stat: pl.DataFrame,
    results_non: pl.DataFrame,
    column_to_use: str,
    outdir: str,
):
    os.makedirs(outdir, exist_ok=True)

    rings = (
        pl.concat(
            [
                df_stat.select(["inner_radius_km", "outer_radius_km"]).unique(),
                df_non.select(["inner_radius_km", "outer_radius_km"]).unique(),
            ],
            how="vertical",
        )
        .unique()
        .filter(pl.col("outer_radius_km") > pl.col("inner_radius_km"))
        .sort(["inner_radius_km", "outer_radius_km"])
        .to_numpy()
    )

    for inner, outer in rings:
        sub_stat = df_stat.filter(
            (pl.col("inner_radius_km") == inner) & (pl.col("outer_radius_km") == outer)
        ).select([column_to_use, "beta1"])
        sub_non = df_non.filter(
            (pl.col("inner_radius_km") == inner) & (pl.col("outer_radius_km") == outer)
        ).select([column_to_use, "beta1"])

        if sub_stat.height < 4 and sub_non.height < 4:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(8, 5), constrained_layout=True)

        panels = [
            ("Stationary", sub_stat, results_stat),
            ("Non-stationary", sub_non, results_non),
        ]

        for ax, (label, sub, res) in zip(axes, panels):
            if sub.height < 4:
                ax.set_axis_off()
                continue

            x = sub[column_to_use].to_numpy()
            y = sub["beta1"].to_numpy()
            if np.all(np.isnan(x)) or np.all(np.isnan(y)):
                ax.set_axis_off()
                continue

            grp, g1, g2 = choose_groups(column_to_use, x)
            if (np.sum(grp == g1) < 2) or (np.sum(grp == g2) < 2):
                ax.set_axis_off()
                continue

            y1 = y[grp == g1]
            y2 = y[grp == g2]

            p_rows = res.filter(
                (pl.col("inner_radius_km") == float(inner))
                & (pl.col("outer_radius_km") == float(outer))
                & (pl.col("column") == column_to_use)
            )
            p_val = p_rows["t_p_value"][0] if p_rows.height > 0 else np.nan

            group_labels = [g1, g2]
            data = [y1[~np.isnan(y1)], y2[~np.isnan(y2)]]

            bp = _boxplot(ax, data, group_labels)

            colours = _boxplot_colours_for_groups(group_labels)
            for patch, c in zip(bp["boxes"], colours):
                patch.set_facecolor(c)
                patch.set_edgecolor("black")

            y_all = np.concatenate([data[0], data[1]]) if data[0].size + data[1].size else np.array([0.0])
            y_max = np.nanquantile(y_all, 0.99) if y_all.size else 0.0

            ax.text(
                1.5,
                y_max,
                format_p_value(float(p_val)),
                ha="center",
                va="bottom",
                fontsize=11,
            )

            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("beta")

        fig.suptitle(
            f"beta by median split on {column_to_use} ({inner} to {outer} km)",
            fontsize=13,
        )

        fname = os.path.join(outdir, f"boxplot_SPLIT_{column_to_use}_r{inner}to{outer}.png")
        fig.savefig(fname, dpi=300)
        plt.close(fig)


def save_all_boxplots(
    df_all: pl.DataFrame,
    results_all: pl.DataFrame,
    column_to_use: str,
    outdir: str,
):
    os.makedirs(outdir, exist_ok=True)

    rings = (
        df_all.select(["inner_radius_km", "outer_radius_km"])
        .unique()
        .filter(pl.col("outer_radius_km") > pl.col("inner_radius_km"))
        .sort(["inner_radius_km", "outer_radius_km"])
        .to_numpy()
    )

    for inner, outer in rings:
        sub = df_all.filter(
            (pl.col("inner_radius_km") == inner) & (pl.col("outer_radius_km") == outer)
        ).select([column_to_use, "beta1"])

        if sub.height < 4:
            continue

        x = sub[column_to_use].to_numpy()
        y = sub["beta1"].to_numpy()
        if np.all(np.isnan(x)) or np.all(np.isnan(y)):
            continue

        grp, g1, g2 = choose_groups(column_to_use, x)
        if (np.sum(grp == g1) < 2) or (np.sum(grp == g2) < 2):
            continue

        y1 = y[grp == g1]
        y2 = y[grp == g2]

        p_rows = results_all.filter(
            (pl.col("inner_radius_km") == float(inner))
            & (pl.col("outer_radius_km") == float(outer))
            & (pl.col("column") == column_to_use)
        )
        p_val = p_rows["t_p_value"][0] if p_rows.height > 0 else np.nan

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), constrained_layout=True)

        group_labels = [g1, g2]
        data = [y1[~np.isnan(y1)], y2[~np.isnan(y2)]]

        bp = _boxplot(ax, data, group_labels)

        colours = _boxplot_colours_for_groups(group_labels)
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_edgecolor("black")

        y_all = np.concatenate([data[0], data[1]]) if data[0].size + data[1].size else np.array([0.0])
        y_max = np.nanquantile(y_all, 0.99) if y_all.size else 0.0

        ax.text(
            1.5,
            y_max,
            format_p_value(float(p_val)),
            ha="center",
            va="bottom",
            fontsize=11,
        )

        ax.set_title(f"beta by median split on {column_to_use} ({inner} to {outer} km)")
        ax.set_xlabel("")
        ax.set_ylabel("beta")

        fname = os.path.join(outdir, f"boxplot_ALL_{column_to_use}_r{inner}to{outer}.png")
        fig.savefig(fname, dpi=300)
        plt.close(fig)


def save_split_heatmap_logp(
    results_stat: pl.DataFrame,
    results_non: pl.DataFrame,
    column_to_use: str,
    outdir: str,
):
    """
    Split (two-panel) heatmap with graded intensity, AND with reduced intensity
    for non-significant cells (p > 0.05).

    - We plot -log10(p): larger => more significant
    - Alpha increases smoothly with -log10(p)
    - If p > 0.05, alpha is capped (fainter)
    """
    os.makedirs(outdir, exist_ok=True)

    res = pl.concat([results_stat, results_non], how="vertical")
    if res.height == 0:
        return

    res = res.filter(pl.col("column") == column_to_use)
    if res.height == 0:
        return

    res = res.with_columns([
        pl.max_horizontal([pl.col("t_p_value"), pl.lit(1e-12)]).alias("p_safe"),
    ]).with_columns([
        (-pl.col("p_safe").log10()).alias("neglog10p"),
    ])

    inner_levels = sorted(res["inner_radius_km"].unique().to_list())
    outer_levels = sorted(res["outer_radius_km"].unique().to_list())

    x_tick_idx = list(range(0, len(outer_levels), 2))
    y_tick_idx = list(range(0, len(inner_levels), 2))

    def make_grid(sub: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        grid = np.full((len(inner_levels), len(outer_levels)), np.nan, dtype=float)

        # Alpha grading parameters (tune if you like)
        alpha_min = 0.08

        # Saturate alpha at this significance level (p = 1e-6 gives -log10(p)=6)
        alpha_max_at = 6.0

        # For non-significant (p>0.05), cap intensity so those cells are visibly fainter
        alpha_cap_nonsig = 0.22

        alpha = np.full((len(inner_levels), len(outer_levels)), alpha_min, dtype=float)

        inner_to_i = {v: i for i, v in enumerate(inner_levels)}
        outer_to_j = {v: j for j, v in enumerate(outer_levels)}

        # Iterate values including p_safe so we can apply the 5% rule
        for inner, outer, p_safe, val in sub.select(
            ["inner_radius_km", "outer_radius_km", "p_safe", "neglog10p"]
        ).iter_rows():

            i = inner_to_i[inner]
            j = outer_to_j[outer]

            grid[i, j] = val

            # Base alpha from significance (graded)
            a = alpha_min + (1.0 - alpha_min) * min(max(val / alpha_max_at, 0.0), 1.0)

            # If not significant at 5%, cap alpha (less intense)
            if p_safe > 0.05:
                a = min(a, alpha_cap_nonsig)

            alpha[i, j] = a

        return grid, alpha

    panels = [
        ("Stationary", res.filter(pl.col("type") == "Stationary")),
        ("Non-stationary", res.filter(pl.col("type") == "Non-stationary")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Colour scale for -log10(p)
    vmin, vmax = 0.0, 6.0

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(alpha=0.0)  # masked values transparent

    im_for_cbar = None

    for ax, (title, sub) in zip(axes, panels):
        grid, alpha = make_grid(sub)
        grid_masked = np.ma.masked_invalid(grid)

        im = ax.imshow(
            grid_masked,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            cmap=cmap,
            alpha=alpha,  # graded + capped for nonsig
        )

        if im_for_cbar is None:
            im_for_cbar = im

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Outer radius (km)")
        ax.set_ylabel("Inner radius (km)")

        ax.set_xticks(x_tick_idx)
        ax.set_xticklabels([str(outer_levels[i]) for i in x_tick_idx], rotation=45, ha="right")
        ax.set_yticks(y_tick_idx)
        ax.set_yticklabels([str(inner_levels[i]) for i in y_tick_idx])

    fig.suptitle(f"Median Split t-test p-values for {column_to_use} vs beta (-log10 p)", fontsize=13)

    cbar = fig.colorbar(im_for_cbar, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("-log10(p-value)")

    out = os.path.join(outdir, f"median_split_neglog10p_heatmap_SPLIT_{column_to_use}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Fast seabed slope/depth vs beta ring analysis (Python/Polars) + plots (median split)."
    )
    ap.add_argument(
        "--input_csv",
        default="/home/pth/pCloudDrive/WORKSHOP/ESBJERG2/PUBLICATION/DATA/seabed_slope_and_band_depths_ESBJERG3.csv",
        help="CSV with filename, inner_radius_km, outer_radius_km, slope/depth columns",
    )
    ap.add_argument(
        "--goodies_rds",
        default="/home/pth/WORKSHOP/ESBJERG2/OUTPUT/the_goodies.rds",
        help="RDS file containing goodies table (beta1 etc.)",
    )
    ap.add_argument(
        "--best_rds",
        default="/home/pth/WORKSHOP/ESBJERG2/OUTPUT/the_best.rds",
        help="RDS file containing best table (beta1 etc.)",
    )
    ap.add_argument(
        "--column",
        default="depth_band_median_m",
        help="Column to use for median split (e.g. depth_band_median_m or slope_median)",
    )
    ap.add_argument(
        "--outdir",
        default="OUTPUT",
        help="Output directory for CSVs",
    )
    ap.add_argument(
        "--fig_dir",
        default="FIGURES/BOXPLOTS",
        help="Figure output directory (like the Rmd fig_dir)",
    )
    ap.add_argument(
        "--if_all",
        action="store_true",
        help="Also compute + plot the ALL pooled case (like if_ALL in Rmd)",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(args.input_csv)

    df_input = pl.read_csv(args.input_csv, infer_schema_length=10000)

    required_input_cols = {"filename", "inner_radius_km", "outer_radius_km", args.column}
    missing = [c for c in required_input_cols if c not in df_input.columns]
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    df_input = df_input.with_columns(normalise_filename_expr("filename").alias("filename"))

    goodies = read_rds_as_polars(args.goodies_rds, "goodies")
    best = read_rds_as_polars(args.best_rds, "best")

    if "beta1" not in goodies.columns:
        raise ValueError("goodies RDS must contain column 'beta1'")
    if "beta1" not in best.columns:
        raise ValueError("best RDS must contain column 'beta1'")

    goodies = goodies.select(["filename", "beta1"]).with_columns(normalise_filename_expr("filename").alias("filename"))
    best = best.select(["filename", "beta1"]).with_columns(normalise_filename_expr("filename").alias("filename"))

    best_filenames = best.select(["filename"]).unique()
    goodies_exclusive = goodies.join(best_filenames, on="filename", how="anti")

    all_beta = pl.concat([goodies_exclusive, best], how="vertical").unique(subset=["filename"], keep="first")

    df_stat = df_input.join(goodies_exclusive, on="filename", how="inner")
    df_non = df_input.join(best, on="filename", how="inner")
    df_all = df_input.join(all_beta, on="filename", how="inner")

    rows_stat = compute_results_for_type(df_stat, args.column, "Stationary")
    rows_non = compute_results_for_type(df_non, args.column, "Non-stationary")

    out_stat = rows_to_polars(rows_stat)
    out_non = rows_to_polars(rows_non)

    p_stat = os.path.join(args.outdir, "results_stat.csv")
    p_non = os.path.join(args.outdir, "results_nonstat.csv")
    out_stat.write_csv(p_stat)
    out_non.write_csv(p_non)

    print(f"Wrote: {p_stat} ({out_stat.height} rows)")
    print(f"Wrote: {p_non} ({out_non.height} rows)")

    out_all = None
    if args.if_all:
        rows_all = compute_results_for_type(df_all, args.column, "All")
        out_all = rows_to_polars(rows_all)
        p_all = os.path.join(args.outdir, "results_all.csv")
        out_all.write_csv(p_all)
        print(f"Wrote: {p_all} ({out_all.height} rows)")

    save_split_boxplots(
        df_stat=df_stat,
        df_non=df_non,
        results_stat=out_stat,
        results_non=out_non,
        column_to_use=args.column,
        outdir=args.fig_dir,
    )
    print(f"Wrote split boxplots in: {args.fig_dir}")

    save_split_heatmap_logp(
        results_stat=out_stat,
        results_non=out_non,
        column_to_use=args.column,
        outdir=args.fig_dir,
    )
    print(f"Wrote split heatmap in: {args.fig_dir}")

    if args.if_all and out_all is not None:
        save_all_boxplots(
            df_all=df_all,
            results_all=out_all,
            column_to_use=args.column,
            outdir=args.fig_dir,
        )
        print(f"Wrote ALL boxplots in: {args.fig_dir}")

        print("Now run summarise_best_ring.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

