#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np

import polars as pl


def add_scores(df: pl.DataFrame) -> pl.DataFrame:
    # protect log10
    df = df.with_columns(
        pl.when(pl.col("t_p_value") <= 0)
        .then(1e-300)
        .otherwise(pl.col("t_p_value"))
        .alias("p_safe")
    )
    return df.with_columns([
        (-pl.col("p_safe").log10()).alias("neglog10p"),
        (pl.col("delta_beta").abs()).alias("abs_delta"),
        (pl.col("delta_beta").abs() * (-pl.col("p_safe").log10())).alias("score_balanced"),
    ])


def best_by(df: pl.DataFrame, sort_exprs, n=10) -> pl.DataFrame:
    return df.sort(sort_exprs, descending=True).head(n)


def main():
    ap = argparse.ArgumentParser(description="Summarise best (inner, outer) ring choices from results_*.csv")
    ap.add_argument("--outdir", default="OUTPUT", help="Directory containing results_stat.csv etc.")
    ap.add_argument("--column", default=None, help="Optional: filter to a specific column name")
    ap.add_argument("--topn", type=int, default=20, help="How many top rings to list per criterion/type")
    args = ap.parse_args()

    paths = {
        "Stationary": os.path.join(args.outdir, "results_stat.csv"),
        "Non-stationary": os.path.join(args.outdir, "results_nonstat.csv"),
        "All": os.path.join(args.outdir, "results_all.csv"),
    }

    tables = {}
    for k, p in paths.items():
        if os.path.exists(p):
            df = pl.read_csv(p)
            if args.column is not None:
                df = df.filter(pl.col("column") == args.column)
            if df.height > 0:
                tables[k] = add_scores(df)

    if not tables:
        raise SystemExit(f"No results_*.csv found in {args.outdir}")

    # For each type, compute top lists
    out_rows = []
    for typ, df in tables.items():
        # 1) smallest p => largest neglog10p
        top_p = best_by(df, ["neglog10p"], n=args.topn).with_columns([
            pl.lit(typ).alias("type"),
            pl.lit("min_p").alias("criterion"),
        ])

        # 2) largest abs(delta)
        top_d = best_by(df, ["abs_delta"], n=args.topn).with_columns([
            pl.lit(typ).alias("type"),
            pl.lit("max_abs_delta").alias("criterion"),
        ])

        # 3) balanced score
        top_b = best_by(df, ["score_balanced"], n=args.topn).with_columns([
            pl.lit(typ).alias("type"),
            pl.lit("balanced").alias("criterion"),
        ])

        out_rows.append(pl.concat([top_p, top_d, top_b], how="vertical"))

    out = pl.concat(out_rows, how="vertical").select([
        "type", "criterion", "column",
        "inner_radius_km", "outer_radius_km",
        "group1", "group2",
        "beta_group1_mean", "beta_group2_mean",
        "delta_beta", "t_p_value",
        "neglog10p", "abs_delta", "score_balanced",
    ])

    out_path = os.path.join(args.outdir, "best_rings_summary.csv")
    out.write_csv(out_path)

    # Also print the single "best" per type using balanced score
    print("\nSingle best ring per type (balanced score):")
    for typ, df in tables.items():
        best = df.sort("score_balanced", descending=True).head(1)
        if best.height == 1:
            r = best.row(0, named=True)
            print(
                f"- {typ}: inner={r['inner_radius_km']}, outer={r['outer_radius_km']}, "
                f"delta={r['delta_beta']:.4g}, p={r['t_p_value']:.4g}, score={r['score_balanced']:.4g}"
            )

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

