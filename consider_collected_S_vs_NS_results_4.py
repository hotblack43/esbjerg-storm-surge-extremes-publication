#!/usr/bin/env python3
# consider_collected_S_vs_NS_results_4.py
#
# Python translation of: consider_collected_S_vs_NS_results_4.Rmd
# Reads RESULTS/merged.csv, applies the same (active) cleanup filters,
# computes AIC/BIC winners + deltas, selects "best" (clear NS cases),
# makes the same plots, and writes outputs in OUTPUT/.
#
# Notes:
# - The original R code saves .rds files. In Python we *try* to write RDS
#   using pyreadr if available; otherwise we fall back to .pkl.
# - The Rmd contains a small plotting typo: goodies$deltaBIC (missing underscore).
#   Here we use delta_BIC so the plot runs.

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def drop_rows_matching_firstcol(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """
    R equivalent:
      df <- df[!grepl("pattern", df[[1]]), ]
    We match against the *first column* of df, converted to string.
    """
    first_col = df.columns[0]
    s = df[first_col].astype(str)
    mask = ~s.str.contains(pattern, case=False, regex=True, na=False)
    return df.loc[mask].copy()


def try_write_rds(df: pd.DataFrame, out_path: Path) -> bool:
    """
    Try to write an RDS file using pyreadr (if installed).
    Returns True if successful, False otherwise.
    """
    try:
        import pyreadr  # type: ignore
    except Exception:
        return False

    try:
        pyreadr.write_rds(str(out_path), df)
        return True
    except Exception:
        return False


def main() -> int:
    # R: setwd("./")
    # Here: assume script is run from project root; if not, we cd to script dir.
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    in_csv = Path("RESULTS/merged.csv")
    if not in_csv.exists():
        print(f"ERROR: Cannot find input CSV: {in_csv.resolve()}", file=sys.stderr)
        return 2

    df = pd.read_csv(in_csv)

    # -----------------------------
    # Clean up the df (ACTIVE filters only, as in the Rmd)
    # -----------------------------
    if_cleanup = True
    if if_cleanup:
        active_patterns = [
            #           "aomorika",
            #           "cristobal",                 # offset after 2000
            "churchill",  # due to the ice
            "fernandina_beach-240a-usa-uhslc_rq",  # least data of two
            #           "furuogrund-2055",
            #           "harlingen",
            #           "hanasaki",
            #           "hornbaek",
            #           "hiroshima",                 # jump at 1958ish
            #           "izuhara",
            #           "kamaishi",
            #           "kungsholmsfort-kun",        # form of duplicate
            #           "kobe",
            #           "kozushima",
            #           "la_libertad",               # odd gradual msl slope in 1980s
            "landsort1-lan",  # omitting the detrended version
            #           "maisaka",
            #           "miyakejima",
            #           "miyako",
            #           "naze",                      # few first years MSL is off, could be fixed
            #           "ontario",
            #           "ofunato",
            #           "ominato",
            #           "onahama",
            #           "oita",
            #           "sitka_ak",
            #           "tokyo",
            #           "tomakomainishiko",          # sudden drop near 212, can be fixed
            #           "yakutat-9",
            #           "ayukawa-ma11-jpn",
        ]

        for pat in active_patterns:
            df = drop_rows_matching_firstcol(df, pat)

    # -----------------------------
    # Find clear NS cases (goodies)
    # -----------------------------
    goodies = df.copy()

    # R: goodies$AICbest <- '' ; goodies$BICbest <- ''
    goodies["AICbest"] = ""
    goodies["BICbest"] = ""

    # R: idx <- which((df$AIC_s > df$AIC_ns)) ; goodies$AICbest[idx] <- '*'
    #    jdx <- which((df$BIC_s > df$BIC_ns)) ; goodies$BICbest[jdx] <- '*'
    required_cols = ["AIC_s", "AIC_ns", "BIC_s", "BIC_ns", "LRT_p", "LRT_stat", "beta1"]
    missing = [c for c in required_cols if c not in goodies.columns]
    if missing:
        print(
            "ERROR: Missing required columns in RESULTS/merged.csv:\n"
            f"  Missing: {missing}\n"
            f"  Present: {list(goodies.columns)}",
            file=sys.stderr,
        )
        return 3

    aic_better = goodies["AIC_s"] > goodies["AIC_ns"]
    bic_better = goodies["BIC_s"] > goodies["BIC_ns"]
    goodies.loc[aic_better, "AICbest"] = "*"
    goodies.loc[bic_better, "BICbest"] = "*"

    # -----------------------------
    # Post calc
    # -----------------------------
    goodies["delta_AIC"] = goodies["AIC_s"] - goodies["AIC_ns"]
    goodies["delta_BIC"] = goodies["BIC_s"] - goodies["BIC_ns"]

    # R:
    # idx <- which(delta_AIC > 0 & delta_BIC > 0 & LRT_p < 0.05)
    # best <- goodies[idx,]
    best = goodies.loc[
        (goodies["delta_AIC"] > 0)
        & (goodies["delta_BIC"] > 0)
        & (goodies["LRT_p"] < 0.05)
    ].copy()

    # -----------------------------
    # Plots (same spirit as R code)
    # -----------------------------
    ensure_dir(Path("FIGURES"))
    ensure_dir(Path("OUTPUT"))

    # R: plot(df$LRT_p, df$LRT_stat, pch=19); abline(v=0.05)
    plt.figure()
    plt.scatter(df["LRT_p"], df["LRT_stat"], s=20)
    plt.axvline(0.05, linestyle="--")
    plt.xlabel("LRT_p")
    plt.ylabel("LRT_stat")
    plt.title("LRT p vs LRT statistic")
    plt.tight_layout()
    plt.savefig("FIGURES/scatter_LRTp_vs_LRTstat.png", dpi=150)
    plt.close()

    # R: hist(df$beta1, breaks=17)
    plt.figure()
    plt.hist(df["beta1"].dropna().values, bins=17)
    plt.xlabel("beta1")
    plt.ylabel("N")
    plt.title("Histogram of beta1 (df)")
    plt.tight_layout()
    plt.savefig("FIGURES/hist_beta_df.png", dpi=150)
    plt.close()

    # R: hist(goodies$beta1, breaks=19)
    plt.figure()
    plt.hist(goodies["beta1"].dropna().values, bins=19)
    plt.xlabel("beta1")
    plt.ylabel("N")
    plt.title("Histogram of beta1 (goodies)")
    plt.tight_layout()
    plt.savefig("FIGURES/hist_beta_goodies.png", dpi=150)
    plt.close()

    # R:
    # png("FIGURES/hist_beta_best.png")
    # par(mar=c(5,5,4,2))
    # hist(best$beta1, breaks=11, xlab=expression(beta), ylab="N", main="Significantly non-stationary", ...)
    plt.figure()
    plt.hist(best["beta1"].dropna().values, bins=11)
    plt.xlabel(r"$\beta$")
    plt.ylabel("N")
    plt.title("Significantly non-stationary")
    plt.tight_layout()
    plt.savefig("FIGURES/hist_beta_best.png", dpi=150)
    plt.close()

    # R (typo in Rmd fixed here):
    # plot(goodies$delta_AIC, goodies$deltaBIC, pch=19, ylab="AIC_S - AIC_NS", xlab="Station")
    # abline(h=0, col='red', lty=2)
    #
    # The ylab/xlab in the Rmd are also swapped vs the plotted variables.
    # We keep the variables consistent (delta_AIC vs delta_BIC) and label them clearly.
    plt.figure()
    plt.scatter(goodies["delta_AIC"], goodies["delta_BIC"], s=20)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("delta_AIC = AIC_s - AIC_ns")
    plt.ylabel("delta_BIC = BIC_s - BIC_ns")
    plt.title("Delta AIC vs Delta BIC (goodies)")
    plt.tight_layout()
    plt.savefig("FIGURES/scatter_deltaAIC_vs_deltaBIC.png", dpi=150)
    plt.close()

    # R:
    # plot(goodies$LRT_p, goodies$delta_AIC, pch=19, log='y', xlim=c(0,0.2), ylab="delta AIC", xlab="LRT p")
    # abline(v=0.05, lty=2, col=2, lwd=3)
    # abline(h=1.75, lty=2, col=2, lwd=3)
    plt.figure()
    x = goodies["LRT_p"].values
    y = goodies["delta_AIC"].values

    # Matplotlib can't log-scale with non-positive values; we only plot positive delta_AIC on log scale.
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    plt.scatter(x[mask], y[mask], s=20)
    plt.yscale("log")
    plt.xlim(0, 0.2)
    plt.xlabel("LRT p")
    plt.ylabel("delta AIC")
    plt.title("LRT p vs delta AIC (log y)")
    plt.axvline(0.05, linestyle="--", linewidth=2)
    plt.axhline(1.75, linestyle="--", linewidth=2)
    plt.tight_layout()
    plt.savefig("FIGURES/scatter_LRTp_vs_deltaAIC_logy.png", dpi=150)
    plt.close()

    # -----------------------------
    # Recompute "best" exactly like the last R chunk (same condition, repeated)
    # -----------------------------
    best2 = goodies.loc[
        (goodies["LRT_p"] < 0.05)
        & (goodies["delta_BIC"] > 0)
        & (goodies["delta_AIC"] > 0)
    ].copy()

    # -----------------------------
    # Save outputs
    # -----------------------------
    # R:
    # saveRDS(best, 'OUTPUT/the_best.rds')
    # saveRDS(goodies, 'OUTPUT/the_goodies.rds')
    # write.csv(goodies, "OUTPUT/the_goodies.csv", row.names = FALSE)
    wrote_best_rds = try_write_rds(best2, Path("OUTPUT/the_best.rds"))
    wrote_goodies_rds = try_write_rds(goodies, Path("OUTPUT/the_goodies.rds"))

    if not wrote_best_rds:
        best2.to_pickle("OUTPUT/the_best.pkl")
    if not wrote_goodies_rds:
        goodies.to_pickle("OUTPUT/the_goodies.pkl")

    goodies.to_csv("OUTPUT/the_goodies.csv", index=False)

    # R: print(round(nrow(best)/nrow(goodies)*100,0))
    pct = 0.0
    if len(goodies) > 0:
        pct = round((len(best2) / len(goodies)) * 100.0, 0)
    print(pct)

    # Also print some basic counts (harmless extra)
    print(f"goodies: {len(goodies)} rows")
    print(f"best:    {len(best2)} rows")
    if wrote_best_rds and wrote_goodies_rds:
        print("Wrote RDS outputs via pyreadr.")
    else:
        print(
            "pyreadr not available (or failed). Wrote .pkl fallback(s) instead of .rds."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
