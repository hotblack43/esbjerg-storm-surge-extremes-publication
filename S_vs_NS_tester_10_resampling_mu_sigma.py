#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import chi2, kstest

import statsmodels.api as sm


# -----------------------------
# Paths (match your repo layout)
# -----------------------------
BASE_DIR = Path.home() / "./WORKSHOP/esbjerg-storm-surge-extremes-publication"
INPUT_DIR = BASE_DIR / "OUTPUT" / "ANNUALS2"
FIG_DIR = BASE_DIR / "FIGURES" / "GOF_GEV"
RES_DIR = BASE_DIR / "RESULTS"

OUT_SUMMARY = RES_DIR / "gev_gof_summary.csv"
OUT_POOLED = FIG_DIR / "global_PIT_pooled_hist_and_PP.png"
OUT_POOLED_KS = FIG_DIR / "global_GEV_PP_KS_pvalues.png"

RANDOM_SEED = 134


# -----------------------------
# Helpers
# -----------------------------
def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)


def gev_cdf(y, mu, sigma, xi):
    """
    Vectorised GEV CDF with Gumbel limit at xi ~ 0.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    xi = float(xi)

    if np.any(sigma <= 0) or not np.isfinite(xi):
        return np.full_like(y, np.nan, dtype=float)

    eps = 1e-8
    z = (y - mu) / sigma

    if abs(xi) < eps:
        # Gumbel: F = exp(-exp(-z))
        return np.exp(-np.exp(-z))

    t = 1.0 + xi * z
    out = np.full_like(y, np.nan, dtype=float)
    ok = t > 0
    out[ok] = np.exp(-np.power(t[ok], -1.0 / xi))
    return out


def gev_ppf(p, mu, sigma, xi):
    """
    Vectorised GEV quantile function (PPF). Used for QQ diagnostics.
    """
    p = np.asarray(p, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    xi = float(xi)

    eps = 1e-8
    if abs(xi) < eps:
        # Gumbel: Q(p) = mu - sigma * log(-log(p))
        return mu - sigma * np.log(-np.log(p))

    # General xi != 0:
    # Q(p) = mu + sigma/xi * [ (-log p)^(-xi) - 1 ]
    return mu + (sigma / xi) * (np.power(-np.log(p), -xi) - 1.0)


def gev_logpdf(y, mu, sigma, xi):
    """
    Vectorised log-pdf used in likelihood.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
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
    y = np.asarray(y, dtype=float)
    n = len(y)

    mu0 = float(np.mean(y))
    s0 = float(np.std(y, ddof=1))
    if not np.isfinite(s0) or s0 <= 0:
        s0 = 1.0

    x0 = np.array([mu0, np.log(s0), 0.0], dtype=float)
    bounds = [(None, None), (None, None), (-1.0, 1.0)]

    res = minimize(nll_stationary, x0=x0, args=(y,), method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(res.message)

    mu, log_sigma, xi = res.x
    sigma = float(np.exp(log_sigma))
    loglik = float(-res.fun)

    k = 3
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik

    return dict(mu=float(mu), sigma=sigma, xi=float(xi), loglik=loglik, aic=aic, bic=bic)


def fit_ns_mu_sigma(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)

    # Start mu from OLS, sigma from residual spread, gamma1=0
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

    x0 = np.array([beta0_init, beta1_init, gamma0_init, gamma1_init, xi0], dtype=float)
    bounds = [(None, None), (None, None), (None, None), (None, None), (-1.0, 1.0)]

    res = minimize(nll_ns_mu_sigma, x0=x0, args=(y, x), method="L-BFGS-B", bounds=bounds)
    if not res.success:
        raise RuntimeError(res.message)

    beta0, beta1, gamma0, gamma1, xi = res.x
    loglik = float(-res.fun)

    k = 5
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik

    return dict(
        beta0=float(beta0), beta1=float(beta1),
        gamma0=float(gamma0), gamma1=float(gamma1),
        xi=float(xi), loglik=loglik, aic=aic, bic=bic
    )


def make_station_gof_figure(station, y, x, fit_s, fit_ns, out_png):
    """
    3 panels:
      (1) PIT histogram (S vs NS, side-by-side bars per bin)
      (2) PIT PP plot (S vs NS)
      (3) Standardised QQ plot: compare z=(y-mu_i)/sigma_i to GEV(0,1,xi)
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)

    # PIT
    u_s = gev_cdf(y, mu=fit_s["mu"], sigma=fit_s["sigma"], xi=fit_s["xi"])
    mu_ns = fit_ns["beta0"] + fit_ns["beta1"] * x
    sig_ns = np.exp(fit_ns["gamma0"] + fit_ns["gamma1"] * x)
    u_ns = gev_cdf(y, mu=mu_ns, sigma=sig_ns, xi=fit_ns["xi"])

    # Safety: drop non-finite / edge values for plotting
    m1 = np.isfinite(u_s) & (u_s > 0) & (u_s < 1)
    m2 = np.isfinite(u_ns) & (u_ns > 0) & (u_ns < 1)

    u_s_plot = u_s[m1]
    u_ns_plot = u_ns[m2]

    # PP plot: empirical vs theoretical for Uniform(0,1)
    def pp_points(u):
        u = np.sort(u)
        m = len(u)
        if m < 3:
            return None, None
        p_emp = (np.arange(1, m + 1) - 0.5) / m
        return p_emp, u  # for Uniform, theoretical quantiles are u itself; PP uses u vs p_emp
    # We'll plot u (x) vs p_emp (y) so diagonal is y=x.
    p_emp_s, u_sorted_s = pp_points(u_s_plot)
    p_emp_ns, u_sorted_ns = pp_points(u_ns_plot)

    # Standardised QQ: z = (y - mu_i)/sigma_i  should follow GEV(0,1,xi)
    # Use each model's own xi for its QQ line.
    def qq_data(y, mu, sigma):
        z = (y - mu) / sigma
        z = z[np.isfinite(z)]
        z = np.sort(z)
        m = len(z)
        if m < 3:
            return None, None
        p = (np.arange(1, m + 1) - 0.5) / m
        return p, z

    p_s, z_s = qq_data(y, fit_s["mu"], fit_s["sigma"])
    p_ns, z_ns = qq_data(y, mu_ns, sig_ns)

    # theoretical quantiles for GEV(0,1,xi)
    def q_theory(p, xi):
        return gev_ppf(p, mu=0.0, sigma=1.0, xi=xi)

    # Figure
    fig = plt.figure(figsize=(12, 4), dpi=160)
    gs = fig.add_gridspec(1, 3, wspace=0.33)

    # (1) PIT histogram side-by-side (not stacked)
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 11)
    h_s, _ = np.histogram(u_s_plot, bins=bins)
    h_ns, _ = np.histogram(u_ns_plot, bins=bins)

    centers = 0.5 * (bins[:-1] + bins[1:])
    width = (bins[1] - bins[0]) * 0.42

    ax1.bar(centers - width/2, h_s, width=width, label="S", alpha=0.7, edgecolor="black")
    ax1.bar(centers + width/2, h_ns, width=width, label="NS (μ+σ)", alpha=0.7, edgecolor="black")
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("PIT value u")
    ax1.set_ylabel("Count")
    ax1.legend(frameon=False)

    # (2) PP plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot([0, 1], [0, 1], linewidth=2)
    if p_emp_s is not None:
        ax2.plot(u_sorted_s, p_emp_s, label="S")
    if p_emp_ns is not None:
        ax2.plot(u_sorted_ns, p_emp_ns, label="NS (μ+σ)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Theoretical probability (u)")
    ax2.set_ylabel("Empirical probability")
    ax2.legend(frameon=False)
    ax2.set_title(station)

    # (3) QQ plot on standardised maxima
    ax3 = fig.add_subplot(gs[0, 2])
    # 1:1 line
    ax3.plot([-4, 6], [-4, 6], linewidth=2)

    if p_s is not None:
        ax3.plot(q_theory(p_s, fit_s["xi"]), z_s, label="S")
    if p_ns is not None:
        ax3.plot(q_theory(p_ns, fit_ns["xi"]), z_ns, label="NS (μ+σ)")

    ax3.set_xlabel("GEV(0,1,xi) theoretical quantile")
    ax3.set_ylabel("Standardised observed quantile")
    ax3.legend(frameon=False)

    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def pooled_plots(u_s_all, u_ns_all):
    """
    Create pooled PIT histogram + PP plot.
    """
    fig = plt.figure(figsize=(10, 4), dpi=160)
    gs = fig.add_gridspec(1, 2, wspace=0.28)

    # histogram (side-by-side)
    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 21)
    h_s, _ = np.histogram(u_s_all, bins=bins)
    h_ns, _ = np.histogram(u_ns_all, bins=bins)

    centers = 0.5 * (bins[:-1] + bins[1:])
    width = (bins[1] - bins[0]) * 0.42

    ax1.bar(centers - width/2, h_s, width=width, label="S", alpha=0.7, edgecolor="black")
    ax1.bar(centers + width/2, h_ns, width=width, label="NS (μ+σ)", alpha=0.7, edgecolor="black")
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("PIT value u (pooled)")
    ax1.set_ylabel("Count")
    ax1.legend(frameon=False)

    # PP plot pooled
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot([0, 1], [0, 1], linewidth=2)

    def pp(u):
        u = np.sort(u)
        m = len(u)
        p_emp = (np.arange(1, m + 1) - 0.5) / m
        return u, p_emp

    xs, ys = pp(u_s_all)
    xn, yn = pp(u_ns_all)

    ax2.plot(xs, ys, label="S")
    ax2.plot(xn, yn, label="NS (μ+σ)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Theoretical probability (u)")
    ax2.set_ylabel("Empirical probability")
    ax2.legend(frameon=False)

    fig.savefig(OUT_POOLED, bbox_inches="tight")
    plt.close(fig)


def ks_summary_plot(df):
    """
    A simple 2-panel summary: histogram of KS p-values and ECDF.
    """
    pS = df["ks_p_S"].to_numpy(float)
    pN = df["ks_p_NS"].to_numpy(float)

    fig = plt.figure(figsize=(10, 4), dpi=160)
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 21)
    ax1.hist(pS[np.isfinite(pS)], bins=bins, alpha=0.7, edgecolor="black", label="S")
    ax1.hist(pN[np.isfinite(pN)], bins=bins, alpha=0.7, edgecolor="black", label="NS (μ+σ)")
    ax1.set_xlabel("KS p-value (PIT ~ Uniform)")
    ax1.set_ylabel("Count")
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(gs[0, 1])

    def ecdf(a):
        a = np.sort(a[np.isfinite(a)])
        y = np.arange(1, len(a) + 1) / len(a)
        return a, y

    xs, ys = ecdf(pS)
    xn, yn = ecdf(pN)
    ax2.plot(xs, ys, label="S")
    ax2.plot(xn, yn, label="NS (μ+σ)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("KS p-value")
    ax2.set_ylabel("ECDF")
    ax2.legend(frameon=False)

    fig.savefig(OUT_POOLED_KS, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dirs()
    rng = np.random.default_rng(RANDOM_SEED)

    files = sorted(INPUT_DIR.glob("*_annual.csv"))
    if not files:
        raise RuntimeError(f"No files in {INPUT_DIR}")

    rows = []
    pooled_u_s = []
    pooled_u_ns = []

    for fp in files:
        try:
            df = pd.read_csv(fp)
            df = df.dropna(subset=["storm_year", "max_residual", "mean_sea_level"]).copy()

            y = df["max_residual"].to_numpy(float)
            x_raw = df["mean_sea_level"].to_numpy(float)
            x = x_raw - np.nanmean(x_raw)  # centre only (your convention)

            # Fit both
            fit_s = fit_stationary(y)
            fit_ns = fit_ns_mu_sigma(y, x)

            # PIT values for station
            u_s = gev_cdf(y, mu=fit_s["mu"], sigma=fit_s["sigma"], xi=fit_s["xi"])
            mu_ns = fit_ns["beta0"] + fit_ns["beta1"] * x
            sig_ns = np.exp(fit_ns["gamma0"] + fit_ns["gamma1"] * x)
            u_ns = gev_cdf(y, mu=mu_ns, sigma=sig_ns, xi=fit_ns["xi"])

            # Clean u for KS: only finite, inside (0,1)
            u_s_ks = u_s[np.isfinite(u_s) & (u_s > 0) & (u_s < 1)]
            u_ns_ks = u_ns[np.isfinite(u_ns) & (u_ns > 0) & (u_ns < 1)]

            ksS = kstest(u_s_ks, "uniform") if len(u_s_ks) >= 5 else (np.nan, np.nan)
            ksN = kstest(u_ns_ks, "uniform") if len(u_ns_ks) >= 5 else (np.nan, np.nan)

            # LRT: df=2 (beta1 and gamma1 added)
            lrt = 2.0 * (fit_ns["loglik"] - fit_s["loglik"])
            lrt_p = float(chi2.sf(lrt, df=2))

            station = fp.stem  # includes _annual
            out_png = FIG_DIR / f"{station}_GEV_GOF.png"

            make_station_gof_figure(station, y, x, fit_s, fit_ns, out_png)

            pooled_u_s.append(u_s_ks)
            pooled_u_ns.append(u_ns_ks)

            rows.append(
                dict(
                    filename=station,
                    n=int(len(y)),
                    AIC_s=float(fit_s["aic"]),
                    AIC_ns=float(fit_ns["aic"]),
                    BIC_s=float(fit_s["bic"]),
                    BIC_ns=float(fit_ns["bic"]),
                    LRT_stat=float(lrt),
                    LRT_p=float(lrt_p),
                    beta1=float(fit_ns["beta1"]),
                    gamma1=float(fit_ns["gamma1"]),
                    ks_stat_S=float(ksS.statistic) if np.isfinite(ksS.statistic) else np.nan,
                    ks_p_S=float(ksS.pvalue) if np.isfinite(ksS.pvalue) else np.nan,
                    ks_stat_NS=float(ksN.statistic) if np.isfinite(ksN.statistic) else np.nan,
                    ks_p_NS=float(ksN.pvalue) if np.isfinite(ksN.pvalue) else np.nan,
                )
            )

            print(f"✔ GOF: {station}")

        except Exception as e:
            print(f"✖ GOF failed for {fp.name}: {e}")

    # Write summary
    out = pd.DataFrame(rows)
    out.to_csv(OUT_SUMMARY, index=False)

    # Pooled plots
    if pooled_u_s and pooled_u_ns:
        uS = np.concatenate(pooled_u_s)
        uN = np.concatenate(pooled_u_ns)
        pooled_plots(uS, uN)
        ks_summary_plot(out)

    print(f"Wrote: {OUT_SUMMARY}")
    print(f"Wrote: {OUT_POOLED}")
    print(f"Wrote: {OUT_POOLED_KS}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()

