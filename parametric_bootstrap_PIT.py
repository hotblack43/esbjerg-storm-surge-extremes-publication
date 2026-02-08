#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parametric_bootstrap_PIT.py

Parametric bootstrap PIT/PP envelopes for GEV annual-maxima fits.

- Uses the SAME repo layout and input expectations as:
  S_vs_NS_tester_10_resampling_GOF_with_pooledPIT.py

Assumptions:
- Repo located at: ~/WORKSHOP/esbjerg-storm-surge-extremes-publication
- Annual CSVs in:  OUTPUT/ANNUALS2
- Required columns per CSV: storm_year, max_residual, mean_sea_level

Outputs:
- FIGURES/GOF_GEV_BOOTSTRAP/<station>_PIT_PP_bootstrap.png

Usage examples:
  uv run parametric_bootstrap_PIT.py --B 300
  uv run parametric_bootstrap_PIT.py --station esbjerg_dk_annual.csv --B 500
  uv run parametric_bootstrap_PIT.py --B 200 --max_stations 25

Notes:
- This makes the PP/PIT check "honest" in the sense that the envelope accounts for
  fitting + evaluating on the same sample (via simulate + refit).
- Computational cost can be high: B * (number of stations) * (2 fits).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import chi2

# -----------------------------
# Paths (match your S_vs_NS_tester_10 layout)
# -----------------------------
BASE_DIR = Path.home() / "./WORKSHOP/esbjerg-storm-surge-extremes-publication"
INPUT_DIR = BASE_DIR / "OUTPUT" / "ANNUALS2"
FIG_DIR = BASE_DIR / "FIGURES"
GOF_BOOT_DIR = FIG_DIR / "GOF_GEV_BOOTSTRAP"

# -----------------------------
# Numerics helpers
# -----------------------------
EPS = 1e-12

def clamp01(u: np.ndarray) -> np.ndarray:
    return np.clip(u, EPS, 1.0 - EPS)

# -----------------------------
# GEV logpdf / CDF / PPF (same conventions as your tester_10)
# Parameterisation:
#   mu, sigma>0, xi  (xi==0 => Gumbel limit)
# -----------------------------
def gev_logpdf(y: np.ndarray, mu: np.ndarray, sigma: float, xi: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if sigma <= 0:
        return np.full_like(y, -np.inf, dtype=float)

    z = (y - mu) / sigma

    if abs(xi) < 1e-9:
        # Gumbel
        t = np.exp(-z)
        return -np.log(sigma) - z - t

    t = 1.0 + xi * z
    # Support condition
    ok = t > 0
    out = np.full_like(y, -np.inf, dtype=float)
    tt = t[ok]
    out[ok] = -np.log(sigma) - (1.0 / xi + 1.0) * np.log(tt) - tt ** (-1.0 / xi)
    return out

def gev_cdf(y: np.ndarray, mu: np.ndarray, sigma: float, xi: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    z = (y - mu) / sigma

    if abs(xi) < 1e-9:
        # Gumbel
        return np.exp(-np.exp(-z))

    t = 1.0 + xi * z
    # Outside support -> CDF is 0 or 1 depending on xi sign
    out = np.empty_like(y, dtype=float)
    if xi > 0:
        out[t <= 0] = 0.0
        out[t > 0] = np.exp(-(t[t > 0]) ** (-1.0 / xi))
    else:
        out[t <= 0] = 1.0
        out[t > 0] = np.exp(-(t[t > 0]) ** (-1.0 / xi))
    return out

def gev_ppf(u: np.ndarray, mu: np.ndarray, sigma: float, xi: float) -> np.ndarray:
    u = clamp01(np.asarray(u, dtype=float))
    mu = np.asarray(mu, dtype=float)

    if abs(xi) < 1e-9:
        # Gumbel
        return mu - sigma * np.log(-np.log(u))

    return mu + (sigma / xi) * ((-np.log(u)) ** (-xi) - 1.0)

# -----------------------------
# Fitting: stationary and NS (mu = beta0 + beta1 * x)
# Keep bounds consistent with your tester scripts.
# -----------------------------
def fit_stationary_gev(y: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    mu0 = float(np.mean(y))
    sigma0 = float(np.std(y, ddof=1)) if len(y) > 1 else float(np.std(y))
    sigma0 = max(sigma0, 1e-6)
    xi0 = 0.0

    def nll(params):
        mu, log_sigma, xi = params
        sigma = np.exp(log_sigma)
        ll = np.sum(gev_logpdf(y, mu=np.full_like(y, mu), sigma=sigma, xi=xi))
        return -ll

    x0 = np.array([mu0, np.log(sigma0), xi0], dtype=float)

    bounds = [
        (None, None),              # mu
        (np.log(1e-8), np.log(1e8)),# log_sigma
        (-1.0, 1.0)                # xi
    ]

    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    mu_hat, log_sigma_hat, xi_hat = res.x
    sigma_hat = float(np.exp(log_sigma_hat))
    ll = -float(res.fun)

    k = 3
    n = len(y)
    aic = 2 * k - 2 * ll
    bic = np.log(max(n, 1)) * k - 2 * ll

    return {
        "ok": bool(res.success),
        "mu": float(mu_hat),
        "sigma": sigma_hat,
        "xi": float(xi_hat),
        "ll": ll,
        "aic": float(aic),
        "bic": float(bic),
        "opt": res,
    }

def fit_nonstationary_gev(y: np.ndarray, x: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    # OLS initialisation for mu(x)
    A = np.vstack([np.ones_like(x), x]).T
    beta_ols, *_ = np.linalg.lstsq(A, y, rcond=None)
    beta0_0, beta1_0 = beta_ols
    resid = y - (beta0_0 + beta1_0 * x)
    sigma0 = float(np.std(resid, ddof=1)) if len(y) > 1 else float(np.std(resid))
    sigma0 = max(sigma0, 1e-6)
    xi0 = 0.0

    def nll(params):
        beta0, beta1, log_sigma, xi = params
        sigma = np.exp(log_sigma)
        mu = beta0 + beta1 * x
        ll = np.sum(gev_logpdf(y, mu=mu, sigma=sigma, xi=xi))
        return -ll

    x0 = np.array([beta0_0, beta1_0, np.log(sigma0), xi0], dtype=float)

    bounds = [
        (None, None),               # beta0
        (None, None),               # beta1
        (np.log(1e-8), np.log(1e8)),# log_sigma
        (-1.0, 1.0)                 # xi
    ]

    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    beta0_hat, beta1_hat, log_sigma_hat, xi_hat = res.x
    sigma_hat = float(np.exp(log_sigma_hat))
    ll = -float(res.fun)

    k = 4
    n = len(y)
    aic = 2 * k - 2 * ll
    bic = np.log(max(n, 1)) * k - 2 * ll

    return {
        "ok": bool(res.success),
        "beta0": float(beta0_hat),
        "beta1": float(beta1_hat),
        "sigma": sigma_hat,
        "xi": float(xi_hat),
        "ll": ll,
        "aic": float(aic),
        "bic": float(bic),
        "opt": res,
    }

def compare_gev_models(fit_s: dict, fit_ns: dict) -> dict:
    # LRT df = 1
    lrt = 2.0 * (fit_ns["ll"] - fit_s["ll"])
    p = 1.0 - chi2.cdf(lrt, df=1)
    return {"lrt": float(lrt), "p": float(p)}

# -----------------------------
# Parametric bootstrap PIT / PP envelopes
# -----------------------------
def pp_curve(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (theoretical plotting positions p, empirical sorted u)."""
    u = np.sort(clamp01(u))
    n = len(u)
    p = (np.arange(1, n + 1) - 0.5) / n
    return p, u

def bootstrap_pp_envelope_stationary(
    y: np.ndarray,
    fit: dict,
    B: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (p, lo, hi) for the PP curve under parametric bootstrap with refit.
    lo/hi are pointwise quantiles of sorted u at each rank.
    """
    n = len(y)
    p = (np.arange(1, n + 1) - 0.5) / n

    mu0 = fit["mu"]
    sigma0 = fit["sigma"]
    xi0 = fit["xi"]

    U = np.empty((B, n), dtype=float)

    for b in range(B):
        u_sim = rng.uniform(EPS, 1.0 - EPS, size=n)
        y_sim = gev_ppf(u_sim, mu=mu0, sigma=sigma0, xi=xi0)

        fit_b = fit_stationary_gev(y_sim)
        if not fit_b["ok"]:
            U[b, :] = np.nan
            continue

        u_b = gev_cdf(y_sim, mu=np.full_like(y_sim, fit_b["mu"]), sigma=fit_b["sigma"], xi=fit_b["xi"])
        U[b, :] = np.sort(clamp01(u_b))

    # Drop failed reps
    U = U[~np.any(np.isnan(U), axis=1)]
    if U.shape[0] < max(20, 0.2 * B):
        raise RuntimeError(f"Too many failed bootstrap fits (kept {U.shape[0]}/{B}).")

    lo = np.quantile(U, 0.025, axis=0)
    hi = np.quantile(U, 0.975, axis=0)
    return p, lo, hi

def bootstrap_pp_envelope_nonstationary(
    y: np.ndarray,
    x: np.ndarray,
    fit: dict,
    B: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    p = (np.arange(1, n + 1) - 0.5) / n

    beta0 = fit["beta0"]
    beta1 = fit["beta1"]
    sigma0 = fit["sigma"]
    xi0 = fit["xi"]
    mu0 = beta0 + beta1 * x

    U = np.empty((B, n), dtype=float)

    for b in range(B):
        u_sim = rng.uniform(EPS, 1.0 - EPS, size=n)
        y_sim = gev_ppf(u_sim, mu=mu0, sigma=sigma0, xi=xi0)

        fit_b = fit_nonstationary_gev(y_sim, x)
        if not fit_b["ok"]:
            U[b, :] = np.nan
            continue

        mu_b = fit_b["beta0"] + fit_b["beta1"] * x
        u_b = gev_cdf(y_sim, mu=mu_b, sigma=fit_b["sigma"], xi=fit_b["xi"])
        U[b, :] = np.sort(clamp01(u_b))

    U = U[~np.any(np.isnan(U), axis=1)]
    if U.shape[0] < max(20, 0.2 * B):
        raise RuntimeError(f"Too many failed bootstrap fits (kept {U.shape[0]}/{B}).")

    lo = np.quantile(U, 0.025, axis=0)
    hi = np.quantile(U, 0.975, axis=0)
    return p, lo, hi

def bootstrap_hist_envelope(u_obs: np.ndarray, B_sorted_u: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build histogram envelope from bootstrap sorted-u arrays by binning each replicate.
    Returns (bin_edges, lo_counts, hi_counts).
    """
    # Convert sorted u per replicate to counts per bin
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts = []
    for row in B_sorted_u:
        c, _ = np.histogram(row, bins=edges)
        counts.append(c)
    counts = np.asarray(counts, dtype=float)
    lo = np.quantile(counts, 0.025, axis=0)
    hi = np.quantile(counts, 0.975, axis=0)
    return edges, lo, hi

# -----------------------------
# Per-station plotting
# -----------------------------
def plot_station_bootstrap(
    station_name: str,
    u_obs_s: np.ndarray,
    u_obs_ns: np.ndarray,
    pp_env_s: tuple[np.ndarray, np.ndarray, np.ndarray],
    pp_env_ns: tuple[np.ndarray, np.ndarray, np.ndarray],
    outpath: Path,
):
    p_s, lo_s, hi_s = pp_env_s
    p_ns, lo_ns, hi_ns = pp_env_ns

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    # ---- Left: histogram of PIT with uniform line (and optional envelopes via PP env not ideal)
    ax0 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 11)
    ax0.hist(u_obs_s, bins=bins, alpha=0.6, label="S")
    ax0.hist(u_obs_ns, bins=bins, alpha=0.6, label="NS")
    ax0.set_title(f"{station_name}: PIT histograms")
    ax0.set_xlabel("u = F(y)")
    ax0.set_ylabel("Count")
    ax0.legend(loc="best")

    # ---- Right: PP plot with bootstrap envelopes
    ax1 = fig.add_subplot(gs[0, 1])
    # Reference diagonal
    ax1.plot([0, 1], [0, 1], linewidth=2)

    # Stationary observed
    p_obs_s, u_sorted_s = pp_curve(u_obs_s)
    ax1.plot(p_obs_s, u_sorted_s, marker=".", linestyle="None", label="S (obs)")
    ax1.fill_between(p_s, lo_s, hi_s, alpha=0.25, label="S (95% boot env)")

    # Nonstationary observed
    p_obs_ns, u_sorted_ns = pp_curve(u_obs_ns)
    ax1.plot(p_obs_ns, u_sorted_ns, marker=".", linestyle="None", label="NS (obs)")
    ax1.fill_between(p_ns, lo_ns, hi_ns, alpha=0.25, label="NS (95% boot env)")

    ax1.set_title(f"{station_name}: PP plot with parametric bootstrap envelopes")
    ax1.set_xlabel("Theoretical p (Uniform plotting positions)")
    ax1.set_ylabel("Empirical u (sorted PIT)")
    ax1.legend(loc="best")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Main driver
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=200, help="Number of bootstrap replicates per model per station.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed.")
    parser.add_argument("--station", type=str, default="", help="Run only this one station filename (exact basename).")
    parser.add_argument("--max_stations", type=int, default=0, help="Limit number of stations processed (0 = all).")
    args = parser.parse_args()

    for d in [GOF_BOOT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    if not INPUT_DIR.exists():
        raise SystemExit(f"INPUT_DIR does not exist: {INPUT_DIR}")

    files = sorted(INPUT_DIR.glob("*.csv"))
    if args.station:
        files = [INPUT_DIR / args.station]
        if not files[0].exists():
            raise SystemExit(f"Station file not found: {files[0]}")

    if args.max_stations and args.max_stations > 0:
        files = files[: args.max_stations]

    if not files:
        raise SystemExit(f"No CSV files found in {INPUT_DIR}")

    print(f"Found {len(files)} station file(s). B={args.B}")

    required = {"storm_year", "max_residual", "mean_sea_level"}

    for filepath in files:
        station = filepath.stem
        df = pd.read_csv(filepath)
        if not required.issubset(df.columns):
            print(f"Skipping {filepath.name}: missing required columns")
            continue

        y = df["max_residual"].to_numpy(dtype=float)
        msl = df["mean_sea_level"].to_numpy(dtype=float)

        ok = np.isfinite(y) & np.isfinite(msl)
        y = y[ok]
        msl = msl[ok]

        if len(y) < 20:
            print(f"Skipping {station}: too few valid points (n={len(y)})")
            continue

        # Covariate centring (match your other scripts)
        x = msl - np.mean(msl)

        # Fit S and NS
        fit_s = fit_stationary_gev(y)
        fit_ns = fit_nonstationary_gev(y, x)

        if (not fit_s["ok"]) or (not fit_ns["ok"]):
            print(f"Skipping {station}: fit failed (S ok={fit_s['ok']}, NS ok={fit_ns['ok']})")
            continue

        # Observed PITs
        u_obs_s = clamp01(gev_cdf(y, mu=np.full_like(y, fit_s["mu"]), sigma=fit_s["sigma"], xi=fit_s["xi"]))
        mu_ns = fit_ns["beta0"] + fit_ns["beta1"] * x
        u_obs_ns = clamp01(gev_cdf(y, mu=mu_ns, sigma=fit_ns["sigma"], xi=fit_ns["xi"]))

        # Bootstrap envelopes (simulate from fitted, refit, compute PIT)
        try:
            pp_env_s = bootstrap_pp_envelope_stationary(y, fit_s, args.B, rng)
            pp_env_ns = bootstrap_pp_envelope_nonstationary(y, x, fit_ns, args.B, rng)
        except RuntimeError as e:
            print(f"{station}: bootstrap failed: {e}")
            continue

        outpath = GOF_BOOT_DIR / f"{station}_PIT_PP_bootstrap.png"
        plot_station_bootstrap(
            station_name=station,
            u_obs_s=u_obs_s,
            u_obs_ns=u_obs_ns,
            pp_env_s=pp_env_s,
            pp_env_ns=pp_env_ns,
            outpath=outpath,
        )

        print(f"Wrote: {outpath}")

if __name__ == "__main__":
    main()

