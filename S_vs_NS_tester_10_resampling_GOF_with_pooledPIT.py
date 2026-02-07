#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
S_vs_NS_tester_10_resampling_GOF_with_pooledPIT.py

- Keeps your fitting / bootstrap / existing plots as-is.
- Replaces the “PIT GOF” idea with the “GEV-native” wording:
    * u = F(y) are the GEV PP-values (these are also PIT values).
    * GEV PP plot uses sorted u vs plotting positions.
    * GEV QQ plot uses standardised maxima vs standard GEV(0,1,xi) quantiles.
- Adds:
    * Per-station 3-panel GEV GOF plot: histogram (SIDE-BY-SIDE S vs NS), PP, QQ.
    * Station-wise KS(p) summary on u~Uniform(0,1).
    * Global (station-wise) p-value figure.
    * NEW: Global pooled PIT/PP figure (pooled u-values across all stations):
        - pooled histogram (SIDE-BY-SIDE S vs NS)
        - pooled PP plot (S vs NS)

Outputs:
  FIGURES/GOF_GEV/<station>_GEV_GOF.png
  FIGURES/GOF_GEV/global_GEV_PP_KS_pvalues.png
  FIGURES/GOF_GEV/global_PIT_pooled_hist_and_PP.png
  RESULTS/gev_gof_summary.csv
  RESULTS/collected_S_vs_NS_test_results_RENAME_or_LOOSE.txt
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import chi2, kstest

import statsmodels.api as sm


# -----------------------------
# Config (mirrors your R script)
# -----------------------------
BASE_DIR = Path.home() / "./WORKSHOP/esbjerg-storm-surge-extremes-publication"
INPUT_DIR = BASE_DIR / "OUTPUT" / "ANNUALS2"
FIG_DIR = BASE_DIR / "FIGURES"
RES_DIR = BASE_DIR / "RESULTS"
OUT_DIR = BASE_DIR / "OUTPUT"
BETAS_DIR = OUT_DIR / "BETAS"
MIDDLE_DIR = FIG_DIR / "JUSTTHEMIDDLE3panelONE"

SUMMARY_FILE = RES_DIR / "collected_S_vs_NS_test_results_RENAME_or_LOOSE.txt"

N_BOOT = 1333
RANDOM_SEED = 134

# GOF outputs
GOF_DIR = FIG_DIR / "GOF_GEV"
GOF_SUMMARY_FILE = RES_DIR / "gev_gof_summary.csv"
GLOBAL_PVALUES_FIG = GOF_DIR / "global_GEV_PP_KS_pvalues.png"
GLOBAL_POOLED_FIG = GOF_DIR / "global_PIT_pooled_hist_and_PP.png"


# -----------------------------
# Utilities
# -----------------------------
def ensure_dirs():
    for d in [FIG_DIR, RES_DIR, OUT_DIR, BETAS_DIR, MIDDLE_DIR, GOF_DIR]:
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


def clamp01(u, eps=1e-12):
    u = np.asarray(u, dtype=float)
    u = np.where(np.isfinite(u), u, np.nan)
    return np.clip(u, eps, 1.0 - eps)


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

    eps = 1e-8
    z = (y - mu) / sigma

    if abs(xi) < eps:
        # Gumbel limit: log f = -log(sigma) - z - exp(-z)
        return -np.log(sigma) - z - np.exp(-z)

    t = 1.0 + xi * z
    if np.any(t <= 0):
        return -np.inf * np.ones_like(y)

    # log f = -log(sigma) - (1 + 1/xi)*log(t) - t^(-1/xi)
    return -np.log(sigma) - (1.0 + 1.0 / xi) * np.log(t) - np.power(t, -1.0 / xi)


def gev_cdf(y, mu, sigma, xi):
    """
    GEV CDF consistent with gev_logpdf above (extRemes-style parameterisation).
    Returns P(Y <= y).
    """
    y = np.asarray(y, dtype=float)
    sigma = float(sigma)
    xi = float(xi)

    if sigma <= 0 or not np.isfinite(sigma) or not np.isfinite(xi):
        return np.full_like(y, np.nan, dtype=float)

    eps = 1e-8
    z = (y - mu) / sigma

    if abs(xi) < eps:
        # Gumbel: G(y)=exp(-exp(-z))
        return np.exp(-np.exp(-z))

    t = 1.0 + xi * z

    if xi > 0:
        # support: y > mu - sigma/xi ; below support -> CDF 0
        return np.where(t <= 0, 0.0, np.exp(-np.power(t, -1.0 / xi)))
    else:
        # xi < 0 support: y < mu - sigma/xi ; above support -> CDF 1
        return np.where(t <= 0, 1.0, np.exp(-np.power(t, -1.0 / xi)))


def gev_ppf(u, mu, sigma, xi):
    """
    GEV quantile function consistent with the CDF above.
    """
    u = np.asarray(u, dtype=float)
    sigma = float(sigma)
    xi = float(xi)

    if sigma <= 0 or not np.isfinite(sigma) or not np.isfinite(xi):
        return np.full_like(u, np.nan, dtype=float)

    u = clamp01(u)
    eps = 1e-8

    if abs(xi) < eps:
        # Gumbel inverse: y = mu - sigma * log(-log(u))
        return mu - sigma * np.log(-np.log(u))

    # General: y = mu + (sigma/xi) * ( (-log u)^(-xi) - 1 )
    return mu + (sigma / xi) * (np.power(-np.log(u), -xi) - 1.0)


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

    mu0 = np.mean(y)
    sigma0 = np.std(y, ddof=1)
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = 1.0
    xi0 = 0.0

    x0 = np.array([mu0, np.log(sigma0), xi0], dtype=float)

    bounds = [
        (None, None),   # mu
        (None, None),   # log_sigma
        (-1.0, 1.0),    # xi
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
    loglik = -float(res.fun)

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

    # Initial guesses from OLS
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
        (None, None),   # beta0
        (None, None),   # beta1
        (None, None),   # log_sigma
        (-1.0, 1.0),    # xi
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
    loglik = -float(res.fun)

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
# Robust slopes (as in your script)
# -----------------------------
def robust_slopes(df):
    X1 = sm.add_constant(df["storm_year"].to_numpy(dtype=float))
    y1 = df["mean_sea_level"].to_numpy(dtype=float)
    rlm1 = sm.RLM(y1, X1, M=sm.robust.norms.HuberT()).fit()
    slope1 = float(rlm1.params[1])
    slope_se1 = float(rlm1.bse[1])

    X2 = sm.add_constant(df["mean_sea_level"].to_numpy(dtype=float))
    y2 = df["max_residual"].to_numpy(dtype=float)
    rlm2 = sm.RLM(y2, X2, M=sm.robust.norms.HuberT()).fit()
    slope2 = float(rlm2.params[1])
    slope_se2 = float(rlm2.bse[1])

    return slope1, slope_se1, slope2, slope_se2


# -----------------------------
# Plotting (your existing)
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
            ax.plot(years[i : j + 1], vals[i : j + 1])
        i = j + 1


def plot_3panel_summary(df, covariate, residuals, beta1_samples, outname, name_clean):
    figpath = FIG_DIR / f"{outname}_3panel_bootstrap_beta1_baseR.png"

    fig = plt.figure(figsize=(12, 4), dpi=150)
    gs = fig.add_gridspec(1, 3, wspace=0.35)

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
    ax1.fill_between(
        xgrid,
        pred["mean_ci_lower"].to_numpy(),
        pred["mean_ci_upper"].to_numpy(),
        alpha=0.2,
    )
    ax1.plot(xgrid, pred["mean"].to_numpy(), linewidth=2)

    ax1.set_xlabel("Year")
    ax1.set_ylabel("MSL [cm]")

    ax2 = fig.add_subplot(gs[0, 1])

    x = np.asarray(covariate, dtype=float)
    y = np.asarray(residuals, dtype=float)

    ax2.scatter(x, y, s=12, alpha=0.5)
    X2 = sm.add_constant(x)
    ols2 = sm.OLS(y, X2).fit()

    x2grid = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    X2g = sm.add_constant(x2grid)
    pred2 = ols2.get_prediction(X2g).summary_frame(alpha=0.05)

    ax2.fill_between(
        x2grid,
        pred2["mean_ci_lower"].to_numpy(),
        pred2["mean_ci_upper"].to_numpy(),
        alpha=0.2,
    )
    ax2.plot(x2grid, pred2["mean"].to_numpy(), linewidth=2)

    ax2.set_title(name_clean)
    ax2.set_xlabel("MSL [cm]")
    ax2.set_ylabel("Residual [cm]")

    ax3 = fig.add_subplot(gs[0, 2])

    b = np.asarray(beta1_samples, dtype=float)
    x1, x2 = -3.0, 3.0
    n_bins = 81
    b_clip = b[(b >= x1) & (b <= x2)]
    if len(b_clip) == 0:
        b_clip = b

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

    ax.fill_between(
        xgrid,
        pred["mean_ci_lower"].to_numpy(),
        pred["mean_ci_upper"].to_numpy(),
        alpha=0.2,
    )
    ax.plot(xgrid, pred["mean"].to_numpy(), linewidth=2)

    ax.set_title(name_clean)
    ax.set_xlabel("MSL [cm]")
    ax.set_ylabel("Residual [cm]")

    fig.savefig(figpath, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# GEV GOF diagnostics (PP + QQ + histogram)
# -----------------------------
def gev_pp_values_stationary(y, fit_s):
    # “GEV PP values” == CDF values (also PIT values) under the stationary fit
    mu = float(fit_s["loc"])
    sigma = float(fit_s["scale"])
    xi = float(fit_s["shape"])
    u = gev_cdf(y, mu=mu, sigma=sigma, xi=xi)
    return clamp01(u)


def gev_pp_values_nonstationary(y, x, fit_ns):
    # CDF values under the covariate-dependent fit
    beta0 = float(fit_ns["beta0"])
    beta1 = float(fit_ns["beta1"])
    sigma = float(fit_ns["scale"])
    xi = float(fit_ns["shape"])
    mu = beta0 + beta1 * x
    u = gev_cdf(y, mu=mu, sigma=sigma, xi=xi)
    return clamp01(u)


def standardised_maxima(y, mu, sigma):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = float(sigma)
    return (y - mu) / sigma


def ks_uniform(u):
    u = np.asarray(u, dtype=float)
    u = u[np.isfinite(u)]
    if len(u) < 5:
        return np.nan, np.nan
    stat, p = kstest(u, "uniform")
    return float(stat), float(p)


def plot_gev_gof_3panel(u_s, u_ns, z_s, z_ns, xi_s, xi_ns, outname, name_clean):
    """
    3 panels, “GEV-native” wording:
      (1) Histogram of PP-values (CDF values) – SIDE-BY-SIDE S vs NS
      (2) GEV PP plot (CDF values vs Uniform)
      (3) GEV QQ plot using standardised maxima vs standard GEV quantiles
    """
    figpath = GOF_DIR / f"{outname}_GEV_GOF.png"

    fig = plt.figure(figsize=(12, 4), dpi=150)
    gs = fig.add_gridspec(1, 3, wspace=0.35)

    # -------- Panel 1: Histogram (side-by-side) --------
    ax1 = fig.add_subplot(gs[0, 0])

    u_s = np.asarray(u_s, dtype=float)
    u_ns = np.asarray(u_ns, dtype=float)

    bins = np.linspace(0.0, 1.0, 11)  # 10 bins
    counts_s, _ = np.histogram(u_s[np.isfinite(u_s)], bins=bins)
    counts_ns, _ = np.histogram(u_ns[np.isfinite(u_ns)], bins=bins)

    bin_left = bins[:-1]
    bin_right = bins[1:]
    bin_width = bin_right - bin_left
    bin_center = bin_left + 0.5 * bin_width

    gap = 0.08  # fraction of bin width reserved as internal spacing
    bar_w = (1.0 - gap) * 0.5 * bin_width
    offset = 0.5 * bar_w

    ax1.bar(
        bin_center - offset,
        counts_s,
        width=bar_w,
        align="center",
        edgecolor="black",
        alpha=0.75,
        label="S",
    )
    ax1.bar(
        bin_center + offset,
        counts_ns,
        width=bar_w,
        align="center",
        edgecolor="black",
        alpha=0.75,
        label="NS",
    )

    ax1.set_xlim(0, 1)
    ax1.set_title(f"{name_clean}\nGEV PP-values histogram")
    ax1.set_xlabel("u = F(y)")
    ax1.set_ylabel("Count")
    ax1.legend(frameon=False)

    # -------- Panel 2: GEV PP plot --------
    ax2 = fig.add_subplot(gs[0, 1])

    def ppplot(ax, u, lab):
        u = np.sort(u[np.isfinite(u)])
        n = len(u)
        if n < 2:
            return
        theo = (np.arange(1, n + 1) - 0.5) / n
        ax.plot(theo, u, marker="o", markersize=2.8, linewidth=0.8, label=lab)

    ppplot(ax2, u_s, "S")
    ppplot(ax2, u_ns, "NS")
    ax2.plot([0, 1], [0, 1], linewidth=1.6)
    ax2.set_title("GEV PP plot (via CDF values)")
    ax2.set_xlabel("Theoretical p (Uniform)")
    ax2.set_ylabel("Empirical p")
    ax2.legend(frameon=False)

    # -------- Panel 3: GEV QQ plot (standardised maxima) --------
    ax3 = fig.add_subplot(gs[0, 2])

    def qqplot_standard(ax, z, xi, lab):
        z = np.sort(np.asarray(z, dtype=float))
        z = z[np.isfinite(z)]
        n = len(z)
        if n < 2:
            return
        p = (np.arange(1, n + 1) - 0.5) / n
        q = gev_ppf(p, mu=0.0, sigma=1.0, xi=xi)
        ax.plot(q, z, marker="o", markersize=2.8, linewidth=0.8, label=lab)

    qqplot_standard(ax3, z_s, xi_s, "S (standardised)")
    qqplot_standard(ax3, z_ns, xi_ns, "NS (standardised)")

    # 1:1 guide line based on combined range
    allz = np.concatenate([np.asarray(z_s, dtype=float), np.asarray(z_ns, dtype=float)])
    allz = allz[np.isfinite(allz)]
    if len(allz) > 0:
        lo = np.nanpercentile(allz, 1)
        hi = np.nanpercentile(allz, 99)
        ax3.plot([lo, hi], [lo, hi], linewidth=1.6)
        ax3.set_xlim(lo, hi)
        ax3.set_ylim(lo, hi)

    ax3.set_title("GEV QQ plot (standardised maxima)")
    ax3.set_xlabel("Theoretical quantile (GEV(0,1,ξ))")
    ax3.set_ylabel("Empirical quantile")
    ax3.legend(frameon=False)

    fig.savefig(figpath, bbox_inches="tight")
    plt.close(fig)


def plot_global_pvalues(pvals_s, pvals_ns):
    """
    Global summary: distribution of KS p-values for Uniformity of GEV PP-values.
    """
    pvals_s = np.asarray(pvals_s, dtype=float)
    pvals_ns = np.asarray(pvals_ns, dtype=float)

    pvals_s = pvals_s[np.isfinite(pvals_s)]
    pvals_ns = pvals_ns[np.isfinite(pvals_ns)]

    fig = plt.figure(figsize=(9, 4), dpi=150)
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 11)
    ax1.hist(pvals_s, bins=bins, alpha=0.7, edgecolor="black", label="S")
    ax1.hist(pvals_ns, bins=bins, alpha=0.7, edgecolor="black", label="NS")
    ax1.axvline(0.05, linewidth=1.6)
    ax1.set_xlabel("KS p-value for Uniformity of GEV PP-values")
    ax1.set_ylabel("Number of stations")
    ax1.set_title("GEV PP uniformity across stations (KS)")
    ax1.legend(frameon=False)

    nS = len(pvals_s)
    nN = len(pvals_ns)
    fS = int(np.sum(pvals_s < 0.05))
    fN = int(np.sum(pvals_ns < 0.05))
    ax1.text(
        0.02,
        0.98,
        f"S: p<0.05: {fS}/{nS}\nNS: p<0.05: {fN}/{nN}",
        transform=ax1.transAxes,
        va="top",
        ha="left",
    )

    ax2 = fig.add_subplot(gs[0, 1])

    def ecdf(ax, p, lab):
        p = np.sort(p)
        n = len(p)
        if n < 2:
            return
        y = np.arange(1, n + 1) / n
        ax.plot(p, y, linewidth=1.6, label=lab)

    ecdf(ax2, pvals_s, "S")
    ecdf(ax2, pvals_ns, "NS")
    ax2.axvline(0.05, linewidth=1.6)
    ax2.set_xlabel("KS p-value")
    ax2.set_ylabel("ECDF")
    ax2.set_title("ECDF of KS p-values")
    ax2.legend(frameon=False)

    fig.savefig(GLOBAL_PVALUES_FIG, bbox_inches="tight")
    plt.close(fig)


def plot_global_pooled_pit(all_u_s, all_u_ns):
    """
    NEW: Global pooled PIT/PP figure (pooled u-values across all stations).

    Panel A: pooled histogram (SIDE-BY-SIDE bars)
    Panel B: pooled PP plot
    """
    # Flatten while keeping only finite
    if len(all_u_s) == 0 or len(all_u_ns) == 0:
        raise RuntimeError("No pooled u-values collected.")

    uS = np.concatenate([u[np.isfinite(u)] for u in all_u_s]) if len(all_u_s) else np.array([])
    uN = np.concatenate([u[np.isfinite(u)] for u in all_u_ns]) if len(all_u_ns) else np.array([])

    fig = plt.figure(figsize=(10, 4), dpi=150)
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    # Panel 1: pooled histogram (side-by-side)
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0.0, 1.0, 11)

    cS, _ = np.histogram(uS, bins=bins)
    cN, _ = np.histogram(uN, bins=bins)

    bin_left = bins[:-1]
    bin_right = bins[1:]
    bin_width = bin_right - bin_left
    bin_center = bin_left + 0.5 * bin_width

    gap = 0.08
    bar_w = (1.0 - gap) * 0.5 * bin_width
    offset = 0.5 * bar_w

    ax1.bar(bin_center - offset, cS, width=bar_w, align="center",
            edgecolor="black", alpha=0.75, label="S")
    ax1.bar(bin_center + offset, cN, width=bar_w, align="center",
            edgecolor="black", alpha=0.75, label="NS")

    ax1.set_xlim(0, 1)
    ax1.set_xlabel("u = F(y)")
    ax1.set_ylabel("Count (pooled over stations)")
    ax1.set_title("Global pooled PIT histogram")
    ax1.legend(frameon=False)

    # Panel 2: pooled PP plot
    ax2 = fig.add_subplot(gs[0, 1])

    def ppplot(ax, u, label):
        u = np.sort(u[np.isfinite(u)])
        n = len(u)
        if n < 2:
            return
        theo = (np.arange(1, n + 1) - 0.5) / n
        ax.plot(theo, u, marker="o", markersize=1.8, linewidth=0.6, label=label)

    ppplot(ax2, uS, "S")
    ppplot(ax2, uN, "NS")
    ax2.plot([0, 1], [0, 1], linewidth=1.6)
    ax2.set_xlabel("Theoretical p (Uniform)")
    ax2.set_ylabel("Empirical p")
    ax2.set_title("Global pooled PIT PP plot")
    ax2.legend(frameon=False)

    fig.savefig(GLOBAL_POOLED_FIG, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Merge metadata (your existing)
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

    merged = df1.merge(
        df2, how="left", left_on="match_name", right_on="filename", sort=False
    )

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

    # GOF summary
    GOF_SUMMARY_FILE.write_text(
        "filename,n,ks_stat_S,ks_p_S,ks_stat_NS,ks_p_NS\n",
        encoding="utf-8",
    )

    input_files = sorted(INPUT_DIR.glob("*_annual.csv"))
    if len(input_files) == 0:
        raise RuntimeError(f"No input files found in: {INPUT_DIR}")

    rng = np.random.default_rng(RANDOM_SEED)

    # Collect station-wise p-values for global figure
    all_pvals_s = []
    all_pvals_ns = []

    # NEW: collect pooled u-values for global pooled PIT histogram + PP plot
    all_u_s = []
    all_u_ns = []

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

            outname = filepath.stem
            name_clean = outname.replace("_", " ")
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

            # Bootstrap beta1 (unchanged)
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

            # Your existing plots (unchanged)
            try:
                plot_3panel_summary(df, covariate, residuals, beta1_samples, outname, name_clean)
            except Exception as e:
                print(f"(plot_3panel_summary failed for {outname}): {e}")

            try:
                save_middle_panel(covariate, residuals, outname, name_clean)
            except Exception as e:
                print(f"(save_middle_panel failed for {outname}): {e}")

            # -----------------------------
            # NEW: GEV GOF plots + KS summary
            # -----------------------------
            try:
                fit_s = results["Stationary"]
                fit_ns = results["NonStationary"]

                # PP-values (CDF values) for the GEV fits (also PIT values)
                u_s = gev_pp_values_stationary(residuals, fit_s)
                u_ns = gev_pp_values_nonstationary(residuals, covariate, fit_ns)

                # Collect for pooled global PIT histogram + PP plot
                all_u_s.append(u_s)
                all_u_ns.append(u_ns)

                # Station-wise KS summary
                ksS, pS = ks_uniform(u_s)
                ksN, pN = ks_uniform(u_ns)

                all_pvals_s.append(pS)
                all_pvals_ns.append(pN)

                with GOF_SUMMARY_FILE.open("a", encoding="utf-8") as f:
                    f.write(
                        f"{outname},{len(residuals)},"
                        f"{safe_val(ksS):.6f},{safe_val(pS):.6g},"
                        f"{safe_val(ksN):.6f},{safe_val(pN):.6g}\n"
                    )

                # Standardised maxima for GEV QQ plot
                mu_s = float(fit_s["loc"])
                sigma_s = float(fit_s["scale"])
                xi_s = float(fit_s["shape"])
                z_s = standardised_maxima(residuals, mu_s, sigma_s)

                beta0 = float(fit_ns["beta0"])
                beta1 = float(fit_ns["beta1"])
                sigma_ns = float(fit_ns["scale"])
                xi_ns = float(fit_ns["shape"])
                mu_ns = beta0 + beta1 * covariate
                z_ns = standardised_maxima(residuals, mu_ns, sigma_ns)

                plot_gev_gof_3panel(u_s, u_ns, z_s, z_ns, xi_s, xi_ns, outname, name_clean)

            except Exception as e:
                print(f"(GEV GOF failed for {outname}): {e}")

            print(f"✔ Processed {filepath}")

        except Exception as e:
            print(f"✖ Failed on {filepath}: {e}")

    # Global p-values figure
    try:
        plot_global_pvalues(all_pvals_s, all_pvals_ns)
        print(f"✅ Wrote global GOF figure: {GLOBAL_PVALUES_FIG}")
    except Exception as e:
        print(f"✖ Global GOF figure failed: {e}")

    # NEW: Global pooled PIT histogram + PP plot
    try:
        plot_global_pooled_pit(all_u_s, all_u_ns)
        print(f"✅ Wrote global pooled PIT figure: {GLOBAL_POOLED_FIG}")
    except Exception as e:
        print(f"✖ Global pooled PIT figure failed: {e}")

    # merge metadata (unchanged)
    try:
        file2 = BASE_DIR / "DATA" / "gesla_station_index.csv"
        out_merge = RES_DIR / "merged.csv"
        merge_metadata_files(SUMMARY_FILE, file2, out_merge)
    except Exception as e:
        print(f"✖ Merge step failed: {e}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

