#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize

import statsmodels.api as sm


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path.home() / "./WORKSHOP/esbjerg-storm-surge-extremes-publication"
INPUT_DIR = BASE_DIR / "OUTPUT" / "ANNUALS2"
FIG_DIR = BASE_DIR / "FIGURES" / "GOF_GEV"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = FIG_DIR / "PIT_u_vs_series_length.png"


# -----------------------------
# GEV pieces (same conventions)
# -----------------------------
def gev_logpdf(y, mu, sigma, xi):
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    xi = float(xi)

    if np.any(sigma <= 0) or not np.isfinite(xi):
        return -np.inf * np.ones_like(y)

    eps = 1e-8
    z = (y - mu) / sigma

    if abs(xi) < eps:
        return -np.log(sigma) - z - np.exp(-z)

    t = 1.0 + xi * z
    if np.any(t <= 0):
        return -np.inf * np.ones_like(y)

    return -np.log(sigma) - (1.0 + 1.0 / xi) * np.log(t) - np.power(t, -1.0 / xi)


def gev_cdf(y, mu, sigma, xi):
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    xi = float(xi)

    if np.any(sigma <= 0) or not np.isfinite(xi):
        return np.full_like(y, np.nan, dtype=float)

    eps = 1e-8
    z = (y - mu) / sigma

    if abs(xi) < eps:
        return np.exp(-np.exp(-z))

    t = 1.0 + xi * z
    out = np.full_like(y, np.nan, dtype=float)
    ok = t > 0
    out[ok] = np.exp(-np.power(t[ok], -1.0 / xi))
    return out


def nll_stationary(theta, y):
    mu, log_sigma, xi = theta
    sigma = np.exp(log_sigma)
    lp = gev_logpdf(y, mu=mu, sigma=sigma, xi=xi)
    if not np.all(np.isfinite(lp)):
        return 1e50
    return -np.sum(lp)


def nll_ns_mu_sigma(theta, y, x):
    beta0, beta1, gamma0, gamma1, xi = theta
    mu = beta0 + beta1 * x
    sigma = np.exp(gamma0 + gamma1 * x)
    lp = gev_logpdf(y, mu=mu, sigma=sigma, xi=xi)
    if not np.all(np.isfinite(lp)):
        return 1e50
    return -np.sum(lp)


def fit_stationary(y):
    y = np.asarray(y, float)
    mu0 = float(np.mean(y))
    s0 = float(np.std(y, ddof=1))
    if not np.isfinite(s0) or s0 <= 0:
        s0 = 1.0

    x0 = np.array([mu0, np.log(s0), 0.0], float)
    bounds = [(None, None), (None, None), (-1.0, 1.0)]
    res = minimize(nll_stationary, x0=x0, args=(y,), method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(res.message)
    mu, log_sigma, xi = res.x
    return dict(mu=float(mu), sigma=float(np.exp(log_sigma)), xi=float(xi))


def fit_ns_mu_sigma(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)

    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    beta0_init = float(ols.params[0])
    beta1_init = float(ols.params[1])

    resid = y - ols.fittedvalues
    s0 = float(np.std(resid, ddof=1))
    if not np.isfinite(s0) or s0 <= 0:
        s0 = max(1.0, float(np.std(y, ddof=1)))

    gamma0_init = float(np.log(s0))
    gamma1_init = 0.0
    xi0 = 0.0

    x0 = np.array([beta0_init, beta1_init, gamma0_init, gamma1_init, xi0], float)
    bounds = [(None, None), (None, None), (None, None), (None, None), (-1.0, 1.0)]
    res = minimize(nll_ns_mu_sigma, x0=x0, args=(y, x), method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(res.message)

    beta0, beta1, gamma0, gamma1, xi = res.x
    return dict(
        beta0=float(beta0), beta1=float(beta1),
        gamma0=float(gamma0), gamma1=float(gamma1),
        xi=float(xi)
    )


# -----------------------------
# Main diagnostic
# -----------------------------
def station_summaries_from_refit(fp):
    df = pd.read_csv(fp)
    df = df.dropna(subset=["max_residual", "mean_sea_level"]).copy()

    y = df["max_residual"].to_numpy(float)
    x_raw = df["mean_sea_level"].to_numpy(float)
    x = x_raw - np.nanmean(x_raw)  # centre only, as in your workflow

    n = int(len(y))
    fit_s = fit_stationary(y)
    fit_ns = fit_ns_mu_sigma(y, x)

    u_s = gev_cdf(y, mu=fit_s["mu"], sigma=fit_s["sigma"], xi=fit_s["xi"])

    mu_ns = fit_ns["beta0"] + fit_ns["beta1"] * x
    sig_ns = np.exp(fit_ns["gamma0"] + fit_ns["gamma1"] * x)
    u_ns = gev_cdf(y, mu=mu_ns, sigma=sig_ns, xi=fit_ns["xi"])

    # Use only finite u in (0,1)
    u_s = u_s[np.isfinite(u_s) & (u_s > 0) & (u_s < 1)]
    u_ns = u_ns[np.isfinite(u_ns) & (u_ns > 0) & (u_ns < 1)]

    def summarise(u):
        if len(u) == 0:
            return np.nan, np.nan
        mean_u = float(np.mean(u))
        frac_low = float(np.mean(u < 0.1))
        return mean_u, frac_low

    mean_s, frac_s = summarise(u_s)
    mean_ns, frac_ns = summarise(u_ns)

    return dict(
        filename=fp.stem,
        n=n,
        mean_u_S=mean_s,
        frac_u_lt_0p1_S=frac_s,
        mean_u_NS=mean_ns,
        frac_u_lt_0p1_NS=frac_ns,
    )


def main():
    files = sorted(INPUT_DIR.glob("*_annual.csv"))
    if not files:
        raise RuntimeError(f"No annual files found in {INPUT_DIR}")

    rows = []
    for fp in files:
        try:
            rows.append(station_summaries_from_refit(fp))
            print(f"✔ {fp.stem}")
        except Exception as e:
            print(f"✖ {fp.stem}: {e}")

    out = pd.DataFrame(rows).dropna(subset=["n"])
    if out.empty:
        raise RuntimeError("No station summaries produced.")

    # Plot
    fig = plt.figure(figsize=(12, 5), dpi=160)
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(out["n"], out["frac_u_lt_0p1_S"], s=14, alpha=0.7, label="S")
    ax1.scatter(out["n"], out["frac_u_lt_0p1_NS"], s=14, alpha=0.7, label="NS (μ+σ)")
    ax1.axhline(0.1, linewidth=2)
    ax1.set_xlabel("Series length n (annual maxima)")
    ax1.set_ylabel("Fraction of PIT values u < 0.1")
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(out["n"], out["mean_u_S"], s=14, alpha=0.7, label="S")
    ax2.scatter(out["n"], out["mean_u_NS"], s=14, alpha=0.7, label="NS (μ+σ)")
    ax2.axhline(0.5, linewidth=2)
    ax2.set_xlabel("Series length n (annual maxima)")
    ax2.set_ylabel("Mean PIT value")
    ax2.legend(frameon=False)

    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {OUT_PNG}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

