#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import chi2

import statsmodels.api as sm


# -----------------------------
# Config (mirrors your R script)
# -----------------------------
#BASE_DIR = Path.home() / "./NEW"
BASE_DIR = Path.home() / "~/WORKSHOP/esbjerg-storm-surge-extremes-publication"
INPUT_DIR = BASE_DIR / "OUTPUT" / "ANNUALS2"
FIG_DIR = BASE_DIR / "FIGURES"
RES_DIR = BASE_DIR / "RESULTS"
OUT_DIR = BASE_DIR / "OUTPUT"
BETAS_DIR = OUT_DIR / "BETAS"
MIDDLE_DIR = FIG_DIR / "JUSTTHEMIDDLE3panelONE"

SUMMARY_FILE = RES_DIR / "collected_S_vs_NS_test_results_RENAME_or_LOOSE.txt"

N_BOOT = 1333
RANDOM_SEED = 134


# -----------------------------
# Utilities
# -----------------------------
def ensure_dirs():
    for d in [FIG_DIR, RES_DIR, OUT_DIR, BETAS_DIR, MIDDLE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def safe_val(x):
    try:
        if x is None:
            return np.nan
        x = float(x)
        if np.isfinite(x):
            return x
        return np.nan
    except Exception:
        return np.nan


# -----------------------------
# GEV log-likelihood (xi convention like extRemes)
# Parameters: mu, sigma>0, xi
# -----------------------------
def gev_logpdf(y, mu, sigma, xi):
    y = np.asarray(y, dtype=float)
    sigma = float(sigma)
    xi = float(xi)

    if sigma <= 0 or not np.isfinite(sigma) or not np.isfinite(xi):
        return -np.inf * np.ones_like(y)

    # Gumbel limit for |xi| small
    eps = 1e-8
    z = (y - mu) / sigma

    if abs(xi) < eps:
        # log f = -log(sigma) - z - exp(-z)
        return -np.log(sigma) - z - np.exp(-z)

    t = 1.0 + xi * z
    # support constraint
    if np.any(t <= 0):
        return -np.inf * np.ones_like(y)

    # log f = -log(sigma) - (1 + 1/xi)*log(t) - t^(-1/xi)
    return -np.log(sigma) - (1.0 + 1.0 / xi) * np.log(t) - np.power(t, -1.0 / xi)


def gev_negloglik_stationary(theta, y):
    mu, log_sigma, xi = theta
    sigma = np.exp(log_sigma)
    lp = gev_logpdf(y, mu=mu, sigma=sigma, xi=xi)
    if not np.all(np.isfinite(lp)):
        return 1e50
    return -np.sum(lp)


def gev_negloglik_nonstationary(theta, y, x):
    beta0, beta1, log_sigma, xi = theta
    sigma = np.exp(log_sigma)
    mu = beta0 + beta1 * x
    lp = gev_logpdf(y, mu=mu, sigma=sigma, xi=xi)
    if not np.all(np.isfinite(lp)):
        return 1e50
    return -np.sum(lp)


def fit_stationary_gev(y):
    y = np.asarray(y, dtype=float)
    n = len(y)

    # crude initial guesses
    mu0 = np.mean(y)
    sigma0 = np.std(y, ddof=1)
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = 1.0
    xi0 = 0.0

    x0 = np.array([mu0, np.log(sigma0), xi0], dtype=float)

    # bounds: log_sigma free; xi bounded moderately to avoid runaway
    bounds = [
        (None, None),        # mu
        (None, None),        # log_sigma
        (-1.0, 1.0),         # xi
    ]

    res = minimize(
        gev_negloglik_stationary,
        x0=x0,
        args=(y,),
        method="L-BFGS-B",
        bounds=bounds,
    )

    if not res.success:
        raise RuntimeError(f"Stationary GEV fit failed: {res.message}")

    mu, log_sigma, xi = res.x
    sigma = float(np.exp(log_sigma))
    nll = float(res.fun)
    loglik = -nll

    k = 3
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik

    return {
        "loc": float(mu),
        "scale": sigma,
        "shape": float(xi),
        "loglik": float(loglik),
        "aic": float(aic),
        "bic": float(bic),
    }


def fit_nonstationary_gev(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)

    # initial guesses from OLS
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    beta0_init = float(ols.params[0])
    beta1_init = float(ols.params[1])
    sigma0 = float(np.std(y - ols.fittedvalues, ddof=1))
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = max(1.0, float(np.std(y, ddof=1)))
    xi0 = 0.0

    x0 = np.array([beta0_init, beta1_init, np.log(sigma0), xi0], dtype=float)

    bounds = [
        (None, None),        # beta0
        (None, None),        # beta1
        (None, None),        # log_sigma
        (-1.0, 1.0),         # xi
    ]

    res = minimize(
        gev_negloglik_nonstationary,
        x0=x0,
        args=(y, x),
        method="L-BFGS-B",
        bounds=bounds,
    )

    if not res.success:
        raise RuntimeError(f"Non-stationary GEV fit failed: {res.message}")

    beta0, beta1, log_sigma, xi = res.x
    sigma = float(np.exp(log_sigma))
    nll = float(res.fun)
    loglik = -nll

    k = 4
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik

    return {
        "beta0": float(beta0),
        "beta1": float(beta1),
        "scale": sigma,
        "shape": float(xi),
        "loglik": float(loglik),
        "aic": float(aic),
        "bic": float(bic),
    }


def compare_gev_models(residuals, covariate):
    fit_s = fit_stationary_gev(residuals)
    fit_ns = fit_nonstationary_gev(residuals, covariate)

    lrt = 2.0 * (fit_ns["loglik"] - fit_s["loglik"])
    pval = float(chi2.sf(lrt, df=1))

    return {
        "Stationary": fit_s,
        "NonStationary": fit_ns,
        "LRT_statistic": float(lrt),
        "LRT_p_value": pval,
    }


# -----------------------------
# Robust slopes (like your gogetslopes)
# -----------------------------
def robust_slopes(df):
    # rlm(mean_sea_level ~ storm_year)
    X1 = sm.add_constant(df["storm_year"].to_numpy(dtype=float))
    y1 = df["mean_sea_level"].to_numpy(dtype=float)
    rlm1 = sm.RLM(y1, X1, M=sm.robust.norms.HuberT()).fit()
    slope1 = float(rlm1.params[1])
    slope_se1 = float(rlm1.bse[1])

    # rlm(max_residual ~ mean_sea_level)
    X2 = sm.add_constant(df["mean_sea_level"].to_numpy(dtype=float))
    y2 = df["max_residual"].to_numpy(dtype=float)
    rlm2 = sm.RLM(y2, X2, M=sm.robust.norms.HuberT()).fit()
    slope2 = float(rlm2.params[1])
    slope_se2 = float(rlm2.bse[1])

    return slope1, slope_se1, slope2, slope_se2


# -----------------------------
# Plotting (matplotlib)
# -----------------------------
def contiguous_year_lines(ax, years, vals):
    years = np.asarray(years, dtype=float)
    vals = np.asarray(vals, dtype=float)
    m = np.isfinite(years) & np.isfinite(vals)
    years = years[m]
    vals = vals[m]
    if len(years) < 2:
        return

    order = np.argsort(years)
    years = years[order]
    vals = vals[order]

    i = 0
    n = len(years)
    while i < n:
        j = i
        while j + 1 < n and (years[j + 1] - years[j]) == 1:
            j += 1
        if j - i + 1 >= 2:
            ax.plot(years[i:j + 1], vals[i:j + 1])
        i = j + 1


def plot_3panel_summary(df, covariate, residuals, beta1_samples, outname, name_clean):
    figpath = FIG_DIR / f"{outname}_3panel_bootstrap_beta1_baseR.png"

    fig = plt.figure(figsize=(12, 4), dpi=150)
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    # Panel 1: MSL vs year (points + contiguous lines). CI from OLS.
    ax1 = fig.add_subplot(gs[0, 0])
    years = df["storm_year"].to_numpy(dtype=float)
    msl = df["mean_sea_level"].to_numpy(dtype=float)

    ax1.scatter(years, msl, s=12)
    contiguous_year_lines(ax1, years, msl)

    X = sm.add_constant(years)
    ols = sm.OLS(msl, X).fit()

    xgrid = np.linspace(np.nanmin(years), np.nanmax(years), 100)
    Xg = sm.add_constant(xgrid)
    pred = ols.get_prediction(Xg).summary_frame(alpha=0.05)
    ax1.fill_between(xgrid, pred["mean_ci_lower"].to_numpy(), pred["mean_ci_upper"].to_numpy(), alpha=0.2)
    ax1.plot(xgrid, pred["mean"].to_numpy(), linewidth=2)

    ax1.set_xlabel("Year")
    ax1.set_ylabel("MSL [cm]")

    # Panel 2: Residuals vs MSL (CI from OLS)
    ax2 = fig.add_subplot(gs[0, 1])

    x = np.asarray(covariate, dtype=float)
    y = np.asarray(residuals, dtype=float)

    ax2.scatter(x, y, s=12, alpha=0.5)
    X2 = sm.add_constant(x)
    ols2 = sm.OLS(y, X2).fit()

    x2grid = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    X2g = sm.add_constant(x2grid)
    pred2 = ols2.get_prediction(X2g).summary_frame(alpha=0.05)

    ax2.fill_between(x2grid, pred2["mean_ci_lower"].to_numpy(), pred2["mean_ci_upper"].to_numpy(), alpha=0.2)
    ax2.plot(x2grid, pred2["mean"].to_numpy(), linewidth=2)

    ax2.set_title(name_clean)
    ax2.set_xlabel("MSL [cm]")
    ax2.set_ylabel("Residual [cm]")

    # Panel 3: Histogram of beta1_samples (clipped [-3, 3], 81 bins)
    ax3 = fig.add_subplot(gs[0, 2])

    b = np.asarray(beta1_samples, dtype=float)
    x1, x2 = -3.0, 3.0
    n_bins = 81
    b_clip = b[(b >= x1) & (b <= x2)]
    if len(b_clip) == 0:
        b_clip = b  # fallback

    left = b_clip[b_clip < 0]
    right = b_clip[b_clip >= 0]
    pct_left = 100.0 * len(left) / max(1, len(b_clip))
    pct_right = 100.0 * len(right) / max(1, len(b_clip))

    if pct_right >= pct_left:
        legend_text = f"Data < 0: {pct_left:.1f}%"
    else:
        legend_text = f"Data >= 0: {pct_right:.1f}%"

    ax3.hist(b_clip, bins=np.linspace(x1, x2, n_bins + 1), edgecolor="black")
    ax3.axvline(0.0, linewidth=3)
    ax3.set_xlim(x1, x2)
    ax3.set_xlabel("β [cm/cm]")
    ax3.legend([legend_text], frameon=False, loc="upper right")

    fig.savefig(figpath, bbox_inches="tight")
    plt.close(fig)


def save_middle_panel(covariate, residuals, outname, name_clean):
    figpath = MIDDLE_DIR / f"{outname}_middle_panel.png"

    x = np.asarray(covariate, dtype=float)
    y = np.asarray(residuals, dtype=float)

    fig = plt.figure(figsize=(4, 4), dpi=175)
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(x, y, s=12, alpha=0.5)

    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    xgrid = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    Xg = sm.add_constant(xgrid)
    pred = ols.get_prediction(Xg).summary_frame(alpha=0.05)

    ax.fill_between(xgrid, pred["mean_ci_lower"].to_numpy(), pred["mean_ci_upper"].to_numpy(), alpha=0.2)
    ax.plot(xgrid, pred["mean"].to_numpy(), linewidth=2)

    ax.set_title(name_clean)
    ax.set_xlabel("MSL [cm]")
    ax.set_ylabel("Residual [cm]")

    fig.savefig(figpath, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Merge metadata (like your merge_metadata_files)
# -----------------------------
def merge_metadata_files(file1, file2, output_file):
    df1 = pd.read_csv(file1, dtype=str)
    df2 = pd.read_csv(file2, dtype=str)

    if "filename" not in df1.columns:
        raise ValueError("file1 must contain a 'filename' column")
    if "filename" not in df2.columns:
        raise ValueError("file2 must contain a 'filename' column")

    df1["match_name"] = df1["filename"].str.replace(r"_annual$", "", regex=True)

    unmatched = sorted(set(df1["match_name"]) - set(df2["filename"]))
    if len(unmatched) > 0:
        raise RuntimeError(
            "These stripped filenames in file1 are missing from file2:\n"
            + ", ".join(unmatched)
        )

    merged = df1.merge(df2, how="left", left_on="match_name", right_on="filename", sort=False)

    # mirror your R: drop match_name and station_name
    if "match_name" in merged.columns:
        merged = merged.drop(columns=["match_name"])
    if "station_name" in merged.columns:
        merged = merged.drop(columns=["station_name"])

    merged.to_csv(output_file, index=False)
    print(f"✅ Merge completed and written to: {output_file}")


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dirs()

    # Remove existing summary file if present
    if SUMMARY_FILE.exists():
        SUMMARY_FILE.unlink()

    header = (
        "filename,AIC_s,BIC_s,AIC_ns,BIC_ns,LRT_stat,LRT_p,"
        "beta1,slope_MSL,errMSL,slope_res,errresid"
    )
    SUMMARY_FILE.write_text(header + "\n", encoding="utf-8")

    input_files = sorted(INPUT_DIR.glob("*_annual.csv"))

    if len(input_files) == 0:
        raise RuntimeError(f"No input files found in: {INPUT_DIR}")

    rng = np.random.default_rng(RANDOM_SEED)

    for filepath in input_files:
        try:
            df = pd.read_csv(filepath)

            required = {"storm_year", "max_residual", "mean_sea_level"}
            if not required.issubset(set(df.columns)):
                print(f"Skipping {filepath} - incorrect format")
                continue

            df = df.dropna(subset=["storm_year", "max_residual", "mean_sea_level"]).copy()

            residuals = df["max_residual"].to_numpy(dtype=float)
            covariate_raw = df["mean_sea_level"].to_numpy(dtype=float)
            covariate = covariate_raw - np.nanmean(covariate_raw)

            results = compare_gev_models(residuals, covariate)

            outname = filepath.stem  # includes "_annual" like in your R outname
            name_clean = outname.replace("_", " ")
            # rough mimic of toTitleCase + removing "-.*" (you didn’t have '-' here anyway)
            name_clean = " ".join([w.capitalize() for w in name_clean.split()])

            slope1, slope_se1, slope2, slope_se2 = robust_slopes(df)

            line = (
                f"{outname},"
                f"{safe_val(results['Stationary']['aic']):.2f},"
                f"{safe_val(results['Stationary']['bic']):.2f},"
                f"{safe_val(results['NonStationary']['aic']):.2f},"
                f"{safe_val(results['NonStationary']['bic']):.2f},"
                f"{safe_val(results['LRT_statistic']):.2f},"
                f"{safe_val(results['LRT_p_value']):.9f},"
                f"{safe_val(results['NonStationary']['beta1']):.6f},"
                f"{safe_val(slope1):.9f},"
                f"{safe_val(slope_se1):.9f},"
                f"{safe_val(slope2):.9f},"
                f"{safe_val(slope_se2):.9f}"
            )

            with SUMMARY_FILE.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

            # Bootstrap beta1
            beta1_samples = []
            n = len(residuals)

            for _ in range(N_BOOT):
                idx = rng.integers(0, n, size=n)
                res_r = residuals[idx]
                cov_r = covariate[idx]
                try:
                    fit_r = fit_nonstationary_gev(res_r, cov_r)
                    b1 = fit_r["beta1"]
                    if np.isfinite(b1):
                        beta1_samples.append(float(b1))
                except Exception:
                    pass

            beta1_samples = np.asarray(beta1_samples, dtype=float)

            betas_out = BETAS_DIR / f"{outname}_beta1_samples.csv"
            pd.Series(beta1_samples).to_csv(betas_out, index=False, header=False)

            n_zeros = int(np.sum(np.abs(beta1_samples) < 1e-8))
            print(f"Number of exact zeros: {n_zeros}")

            # plots
            try:
                plot_3panel_summary(df, covariate, residuals, beta1_samples, outname, name_clean)
            except Exception as e:
                print(f"(plot_3panel_summary failed for {outname}): {e}")

            try:
                save_middle_panel(covariate, residuals, outname, name_clean)
            except Exception as e:
                print(f"(save_middle_panel failed for {outname}): {e}")

            print(f"✔ Processed {filepath}")

        except Exception as e:
            print(f"✖ Failed on {filepath}: {e}")

    # merge metadata
    try:
        file2 = BASE_DIR / "DATA" / "gesla_station_index.csv"
        out_merge = RES_DIR / "merged.csv"
        merge_metadata_files(SUMMARY_FILE, file2, out_merge)
    except Exception as e:
        print(f"✖ Merge step failed: {e}")


if __name__ == "__main__":
    # keep output quieter like your tryCatch approach
    warnings.filterwarnings("ignore")
    main()

