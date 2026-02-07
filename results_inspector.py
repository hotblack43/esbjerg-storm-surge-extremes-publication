#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}\nAvailable: {list(df.columns)}")


def load_inputs(base_dir: Path):
    """
    Uses filenames/columns exactly as in your outputs:
      - RESULTS/collected_S_vs_NS_test_results_RENAME_or_LOOSE.txt  (has LRT_p, AIC_s, AIC_ns, ...)
      - RESULTS/gev_gof_summary.csv                                 (has ks_p_S, ks_p_NS, ...)
      - RESULTS/merged.csv (optional)                               (metadata, not required)
    """
    results_file = base_dir / "RESULTS" / "collected_S_vs_NS_test_results_RENAME_or_LOOSE.txt"
    gof_file = base_dir / "RESULTS" / "gev_gof_summary.csv"
    merged_file = base_dir / "RESULTS" / "merged.csv"

    if not results_file.exists():
        raise FileNotFoundError(f"Missing: {results_file}")
    if not gof_file.exists():
        raise FileNotFoundError(f"Missing: {gof_file}")

    df_sel = pd.read_csv(results_file)
    df_gof = pd.read_csv(gof_file)

    # Validate expected columns (real names from your files)
    require_columns(
        df_sel,
        ["filename", "AIC_s", "AIC_ns", "BIC_s", "BIC_ns", "LRT_stat", "LRT_p", "beta1"],
        "collected_S_vs_NS_test_results_RENAME_or_LOOSE.txt",
    )
    require_columns(
        df_gof,
        ["filename", "n", "ks_stat_S", "ks_p_S", "ks_stat_NS", "ks_p_NS"],
        "gev_gof_summary.csv",
    )

    df_meta = None
    if merged_file.exists():
        df_meta = pd.read_csv(merged_file)
        # We won’t require any particular metadata columns; just merge if possible.
        if "filename" not in df_meta.columns:
            df_meta = None

    return df_sel, df_gof, df_meta


def build_master_table(df_sel: pd.DataFrame, df_gof: pd.DataFrame, df_meta: pd.DataFrame | None):
    # Merge selection + GOF
    df = df_sel.merge(df_gof, on="filename", how="inner", validate="one_to_one")

    # Optional metadata merge (left join; keep all matched stations)
    if df_meta is not None:
        # Avoid duplicate columns explosion by suffixing metadata columns
        df = df.merge(df_meta, on="filename", how="left", suffixes=("", "_meta"))

    # Derived fields (names that do NOT clash with your existing ones)
    df["delta_AIC"] = df["AIC_s"] - df["AIC_ns"]  # >0 favours NS
    df["delta_BIC"] = df["BIC_s"] - df["BIC_ns"]  # >0 favours NS
    df["delta_KS_p"] = df["ks_p_NS"] - df["ks_p_S"]  # >0 favours NS (better uniformity p)
    df["NS_selected_LRT"] = df["LRT_p"] < 0.05

    return df


def classify_2x2(df: pd.DataFrame, selection_rule: str = "LRT"):
    """
    2×2 classification:
      - Selection: NS vs S based on selection_rule
      - Adequacy: KS p-value >= 0.05 for the SELECTED model
    """
    if selection_rule.upper() == "LRT":
        ns_sel = df["NS_selected_LRT"].astype(bool)
        sel_name = "NS_selected_LRT"
    elif selection_rule.upper() == "AIC":
        ns_sel = (df["delta_AIC"] > 2.0)  # common threshold
        sel_name = "NS_selected_dAICgt2"
        df[sel_name] = ns_sel
    else:
        raise ValueError("selection_rule must be 'LRT' or 'AIC'")

    # Adequacy of the *selected* model
    adequate = np.where(ns_sel, df["ks_p_NS"], df["ks_p_S"]) >= 0.05

    # Attach classification columns
    df = df.copy()
    df["NS_selected"] = ns_sel
    df["selected_model"] = np.where(ns_sel, "NS", "S")
    df["selected_model_adequate"] = adequate

    df["class_2x2"] = np.select(
        [
            (df["NS_selected"] == True) & (df["selected_model_adequate"] == True),
            (df["NS_selected"] == True) & (df["selected_model_adequate"] == False),
            (df["NS_selected"] == False) & (df["selected_model_adequate"] == True),
            (df["NS_selected"] == False) & (df["selected_model_adequate"] == False),
        ],
        [
            "NS_selected__adequate",
            "NS_selected__poor",
            "S_selected__adequate",
            "S_selected__poor",
        ],
        default="UNKNOWN",
    )

    # Crosstab
    ct = pd.crosstab(
        df["NS_selected"].map({True: "NS selected", False: "S selected"}),
        df["selected_model_adequate"].map({True: "adequate", False: "poor"}),
        rownames=["Selection"],
        colnames=["Adequacy (KS p>=0.05 on selected model)"],
    )

    return df, ct, sel_name


def robustness_summary(df: pd.DataFrame):
    """
    Compare NS prevalence in:
      - all stations
      - only those where the selected model is adequate
    """
    all_frac = float(df["NS_selected"].mean())
    good = df[df["selected_model_adequate"] == True].copy()
    good_frac = float(good["NS_selected"].mean()) if len(good) else np.nan

    out = {
        "n_all": int(len(df)),
        "n_adequate": int(len(good)),
        "NS_frac_all": all_frac,
        "NS_frac_adequate_only": good_frac,
    }
    return out


def plot_deltaAIC_vs_deltaKSp(df: pd.DataFrame, outpng: Path, use_only_LRT_ns: bool = False):
    """
    Scatter:
      x = delta_AIC (AIC_s - AIC_ns), >0 favours NS
      y = delta_KS_p (ks_p_NS - ks_p_S), >0 favours NS in PIT uniformity p-values
    Optionally restrict to stations where NS_selected_LRT is True.
    """
    d = df.copy()
    if use_only_LRT_ns:
        d = d[d["NS_selected_LRT"] == True].copy()

    x = d["delta_AIC"].to_numpy(dtype=float)
    y = d["delta_KS_p"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(7.5, 6), dpi=150)
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(x, y, s=10, alpha=0.6)

    ax.axvline(0.0, linewidth=1.6)
    ax.axhline(0.0, linewidth=1.6)

    ax.set_xlabel("ΔAIC = AIC_s − AIC_ns ( >0 favours NS )")
    ax.set_ylabel("ΔKS p = ks_p_NS − ks_p_S ( >0 favours NS )")
    ax.set_title("Likelihood improvement vs PIT improvement")

    # Quadrant counts
    q1 = int(np.sum((x > 0) & (y > 0)))
    q2 = int(np.sum((x <= 0) & (y > 0)))
    q3 = int(np.sum((x <= 0) & (y <= 0)))
    q4 = int(np.sum((x > 0) & (y <= 0)))

    ax.text(
        0.02, 0.98,
        f"Quadrants:\n"
        f"ΔAIC>0 & ΔKS>0: {q1}\n"
        f"ΔAIC<=0 & ΔKS>0: {q2}\n"
        f"ΔAIC<=0 & ΔKS<=0: {q3}\n"
        f"ΔAIC>0 & ΔKS<=0: {q4}",
        transform=ax.transAxes,
        va="top",
        ha="left",
    )

    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_dir",
        default=str(Path.home() / "WORKSHOP/esbjerg-storm-surge-extremes-publication"),
        help="Base directory of your repo (default: ~/WORKSHOP/esbjerg-storm-surge-extremes-publication)",
    )
    ap.add_argument(
        "--selection_rule",
        default="LRT",
        choices=["LRT", "AIC"],
        help="How to define NS vs S for the 2x2 table (default: LRT)",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Where to write outputs (default: <base_dir>/RESULTS/INSPECTOR)",
    )
    ap.add_argument(
        "--only_LRT_NS_for_scatter",
        action="store_true",
        help="If set, scatter uses only stations with LRT_p < 0.05",
    )
    args = ap.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (base_dir / "RESULTS" / "INSPECTOR")
    safe_mkdir(out_dir)

    df_sel, df_gof, df_meta = load_inputs(base_dir)
    df = build_master_table(df_sel, df_gof, df_meta)

    # --- Suggestion 1: 2x2 classification table ---
    df2, ct, sel_name = classify_2x2(df, selection_rule=args.selection_rule)

    print("\n=== 2×2 classification (Selection × Adequacy) ===")
    print(f"Selection rule used: {args.selection_rule}  (column: {sel_name})")
    print(ct)

    # Save the table and the classified station list
    (out_dir / "2x2_crosstab.csv").write_text(ct.to_csv(), encoding="utf-8")
    df2.to_csv(out_dir / "stations_classified_2x2.csv", index=False)

    # --- Suggestion 2: robustness check ---
    rob = robustness_summary(df2)
    print("\n=== Robustness: NS prevalence ===")
    print(f"n_all = {rob['n_all']}")
    print(f"n_adequate(selected model) = {rob['n_adequate']}")
    print(f"NS fraction (all) = {rob['NS_frac_all']:.3f}")
    print(f"NS fraction (adequate only) = {rob['NS_frac_adequate_only']:.3f}")

    pd.DataFrame([rob]).to_csv(out_dir / "robustness_summary.csv", index=False)

    # --- Suggestion 3: ΔAIC vs PIT improvement scatter ---
    scatter_png = out_dir / "deltaAIC_vs_deltaKSp.png"
    plot_deltaAIC_vs_deltaKSp(df2, scatter_png, use_only_LRT_ns=args.only_LRT_NS_for_scatter)
    print(f"\nWrote: {scatter_png}")

    print(f"\nWrote: {out_dir / 'stations_classified_2x2.csv'}")
    print(f"Wrote: {out_dir / '2x2_crosstab.csv'}")
    print(f"Wrote: {out_dir / 'robustness_summary.csv'}")


if __name__ == "__main__":
    main()

