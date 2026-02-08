#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parametric_bootstrap_PIT_scores.py

Compute objective GOF scores from parametric-bootstrap PIT/PP envelopes for
stationary (S) and non-stationary (NS: MSL in location) GEV fits.

Inputs (repo layout matches your existing pipeline):
- BASE_DIR = ~/WORKSHOP/esbjerg-storm-surge-extremes-publication
- Annual CSVs in: OUTPUT/ANNUALS2/*.csv
- Required columns: storm_year, max_residual, mean_sea_level

Outputs:
- FIGURES/GOF_GEV_BOOTSTRAP/pp_boot_scores.csv

Optionally:
- FIGURES/GOF_GEV_BOOTSTRAP/PLOTS/<station>_PIT_PP_bootstrap.png

Usage:
  uv run parametric_bootstrap_PIT_scores.py --B 100
  uv run parametric_bootstrap_PIT_scores.py --B 50 --max_stations 10 --make_plots
  uv run parametric_bootstrap_PIT_scores.py --station esbjerg_dk_annual.csv --B 300 --make_plots

Notes:
- This addresses the "testing against the same fitted data" issue by using a
  simulate+refit parametric bootstrap to generate PP envelopes.
- The resulting score E = fraction of PP points outside envelope can be used to
  classify stations and then relate to depth/slope etc. WITHOUT subjective eyeballing.
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
# Paths (match your repo layout)
# -----------------------------
BASE_DIR = Path.home() / "./WORKSHOP/esbjerg-storm-surge-extremes-publication"
INPUT_DIR = BASE_DIR / "OUTPUT" / "ANNUALS2"
FIG_DIR = BASE_DIR / "FIGURES"
GOF_BOOT_DIR = FIG_DIR / "GOF_GEV_BOOTSTRAP"
PLOT_DIR = GOF_BOOT_DIR / "PLOTS"

# -----------------------------
# Numerics helpers
# -----------------------------
EPS = 1e-12

def clamp01(u: np.ndarray) -> np.ndarray:
    return np.clip(u, EPS, 1.0 - EPS)

def pp_positions(n: int) -> np.ndarray:
    # same plotting positions used in your earlier script
    return (np.arange(1, n + 1) - 0.5) / n

# -----------------------------
# GEV logpdf / CDF / PPF (consistent conventions)
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
        return np.exp(-np.exp(-z))

    t = 1.0 + xi * z
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
        return mu - sigma * np.log(-np.log(u))

    return mu + (sigma / xi) * ((-np.log(u)) ** (-xi) - 1.0)

# -----------------------------
# Fitting (same style as your tester scripts)
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
        (None, None),
        (np.log(1e-8), np.log(1e8)),
        (-1.0, 1.0),
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
        "sigma": float(sigma_hat),
        "xi": float(xi_hat),
        "ll": float(ll),
        "aic": float(aic),
        "bic": float(bic),
        "opt": res,
    }

def fit_nonstationary_gev(y: np.ndarray, x: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    # OLS initialisation
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
        (None, None),
        (None, None),
        (np.log(1e-8), np.log(1e8)),
        (-1.0, 1.0),
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
        "sigma": float(sigma_hat),
        "xi": float(xi_hat),
        "ll": float(ll),
        "aic": float(aic),
        "bic": float(bic),
        "opt": res,
    }

def compare_gev_models(fit_s: dict, fit_ns: dict) -> dict:
    lrt = 2.0 * (fit_ns["ll"] - fit_s["ll"])
    p = 1.0 - chi2.cdf(lrt, df=1)
    return {"lrt": float(lrt), "p": float(p)}

# -----------------------------
# Bootstrap PP envelope (simulate + refit + PIT)
# -----------------------------
def bootstrap_sorted_pits_stationary(
    n: int,
    mu0: float,
    sigma0: float,
    xi0: float,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns array U of shape (B_kept, n) containing sorted PIT values u_(i)
    from simulate+refit under the stationary model.
    """
    U = np.empty((B, n), dtype=float)

    for b in range(B):
        u_sim = rng.uniform(EPS, 1.0 - EPS, size=n)
        y_sim = gev_ppf(u_sim, mu=np.full(n, mu0), sigma=sigma0, xi=xi0)

        fit_b = fit_stationary_gev(y_sim)
        if not fit_b["ok"]:
            U[b, :] = np.nan
            continue

        u_b = gev_cdf(
            y_sim,
            mu=np.full(n, fit_b["mu"]),
            sigma=fit_b["sigma"],
            xi=fit_b["xi"],
        )
        U[b, :] = np.sort(clamp01(u_b))

    U = U[~np.any(np.isnan(U), axis=1)]
    return U

def bootstrap_sorted_pits_nonstationary(
    y: np.ndarray,
    x: np.ndarray,
    beta0: float,
    beta1: float,
    sigma0: float,
    xi0: float,
    B: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Returns array U of shape (B_kept, n) containing sorted PIT values u_(i)
    from simulate+refit under the nonstationary model with fixed x.
    """
    n = len(y)
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
    return U

def envelope_from_sorted_pits(U: np.ndarray, qlo: float = 0.025, qhi: float = 0.975) -> tuple[np.ndarray, np.ndarray]:
    lo = np.quantile(U, qlo, axis=0)
    hi = np.quantile(U, qhi, axis=0)
    return lo, hi

# -----------------------------
# Scores and classification
# -----------------------------
def pp_scores(u_sorted: np.ndarray, p: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """
    Compute:
      E   = fraction outside envelope
      MAD = mean absolute deviation from diagonal |u - p|
      MAX = maximum absolute deviation |u - p|
    """
    u_sorted = np.asarray(u_sorted, dtype=float)
    outside = (u_sorted < lo) | (u_sorted > hi)
    E = float(np.mean(outside))
    MAD = float(np.mean(np.abs(u_sorted - p)))
    MAX = float(np.max(np.abs(u_sorted - p)))
    return {"E": E, "MAD": MAD, "MAX": MAX}

def classify_station(E_s: float, E_ns: float, tau: float) -> int:
    """
    Simple classes:
      0: both OK (E_s<=tau and E_ns<=tau)
      1: NS improves (E_s>tau and E_ns<=tau)
      2: neither OK (E_s>tau and E_ns>tau)
      3: S OK, NS worse (E_s<=tau and E_ns>tau)
    """
    s_ok = E_s <= tau
    ns_ok = E_ns <= tau
    if s_ok and ns_ok:
        return 0
    if (not s_ok) and ns_ok:
        return 1
    if (not s_ok) and (not ns_ok):
        return 2
    return 3

# -----------------------------
# Optional plot
# -----------------------------
def plot_station(
    station: str,
    u_sorted_s: np.ndarray,
    u_sorted_ns: np.ndarray,
    p: np.ndarray,
    lo_s: np.ndarray,
    hi_s: np.ndarray,
    lo_ns: np.ndarray,
    hi_ns: np.ndarray,
    outpath: Path,
):
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(0, 1, 11)
    ax0.hist(u_sorted_s, bins=bins, alpha=0.6, label="S")
    ax0.hist(u_sorted_ns, bins=bins, alpha=0.6, label="NS")
    ax0.set_title(f"{station}: PIT histograms")
    ax0.set_xlabel("u = F(y)")
    ax0.set_ylabel("Count")
    ax0.legend(loc="best")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot([0, 1], [0, 1], linewidth=2)

    ax1.plot(p, u_sorted_s, marker=".", linestyle="None", label="S (obs)")
    ax1.fill_between(p, lo_s, hi_s, alpha=0.25, label="S (95% boot env)")

    ax1.plot(p, u_sorted_ns, marker=".", linestyle="None", label="NS (obs)")
    ax1.fill_between(p, lo_ns, hi_ns, alpha=0.25, label="NS (95% boot env)")

    ax1.set_title(f"{station}: PP plot with parametric bootstrap envelopes")
    ax1.set_xlabel("Theoretical p (Uniform plotting positions)")
    ax1.set_ylabel("Empirical u (sorted PIT)")
    ax1.legend(loc="best")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=100, help="Bootstrap replicates per station per model.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed.")
    parser.add_argument("--tau", type=float, default=0.10, help="Envelope exceedance threshold for class labels.")
    parser.add_argument("--station", type=str, default="", help="Process only this station basename (exact .csv filename).")
    parser.add_argument("--max_stations", type=int, default=0, help="Limit number of stations (0=all).")
    parser.add_argument("--make_plots", action="store_true", help="Also save per-station plot PNGs (slower).")
    parser.add_argument("--min_n", type=int, default=30, help="Minimum number of valid annual maxima required.")
    args = parser.parse_args()

    if not INPUT_DIR.exists():
        raise SystemExit(f"INPUT_DIR does not exist: {INPUT_DIR}")

    GOF_BOOT_DIR.mkdir(parents=True, exist_ok=True)
    if args.make_plots:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    files = sorted(INPUT_DIR.glob("*.csv"))
    if args.station:
        files = [INPUT_DIR / args.station]
        if not files[0].exists():
            raise SystemExit(f"Station file not found: {files[0]}")

    if args.max_stations and args.max_stations > 0:
        files = files[: args.max_stations]

    if not files:
        raise SystemExit(f"No station CSVs found in {INPUT_DIR}")

    required = {"storm_year", "max_residual", "mean_sea_level"}

    rows = []
    kept = 0
    skipped = 0

    for fp in files:
        station = fp.stem
        df = pd.read_csv(fp)

        if not required.issubset(df.columns):
            print(f"Skipping {fp.name}: missing required columns")
            skipped += 1
            continue

        y = df["max_residual"].to_numpy(dtype=float)
        msl = df["mean_sea_level"].to_numpy(dtype=float)

        ok = np.isfinite(y) & np.isfinite(msl)
        y = y[ok]
        msl = msl[ok]

        n = len(y)
        if n < args.min_n:
            print(f"Skipping {station}: too few valid points (n={n})")
            skipped += 1
            continue

        # centre covariate (match your main scripts)
        x = msl - np.mean(msl)

        fit_s = fit_stationary_gev(y)
        fit_ns = fit_nonstationary_gev(y, x)

        if (not fit_s["ok"]) or (not fit_ns["ok"]):
            print(f"Skipping {station}: fit failed (S ok={fit_s['ok']}, NS ok={fit_ns['ok']})")
            skipped += 1
            continue

        # observed PITs -> sort for PP
        u_obs_s = clamp01(gev_cdf(y, mu=np.full(n, fit_s["mu"]), sigma=fit_s["sigma"], xi=fit_s["xi"]))
        mu_ns = fit_ns["beta0"] + fit_ns["beta1"] * x
        u_obs_ns = clamp01(gev_cdf(y, mu=mu_ns, sigma=fit_ns["sigma"], xi=fit_ns["xi"]))

        u_sorted_s = np.sort(u_obs_s)
        u_sorted_ns = np.sort(u_obs_ns)
        p = pp_positions(n)

        # bootstrap envelopes
        U_s = bootstrap_sorted_pits_stationary(
            n=n,
            mu0=fit_s["mu"],
            sigma0=fit_s["sigma"],
            xi0=fit_s["xi"],
            B=args.B,
            rng=rng,
        )
        U_ns = bootstrap_sorted_pits_nonstationary(
            y=y,
            x=x,
            beta0=fit_ns["beta0"],
            beta1=fit_ns["beta1"],
            sigma0=fit_ns["sigma"],
            xi0=fit_ns["xi"],
            B=args.B,
            rng=rng,
        )

        # Basic sanity check
        if U_s.shape[0] < max(20, int(0.2 * args.B)) or U_ns.shape[0] < max(20, int(0.2 * args.B)):
            print(f"Skipping {station}: too many failed bootstrap fits (kept S={U_s.shape[0]}, NS={U_ns.shape[0]} of B={args.B})")
            skipped += 1
            continue

        lo_s, hi_s = envelope_from_sorted_pits(U_s)
        lo_ns, hi_ns = envelope_from_sorted_pits(U_ns)

        sc_s = pp_scores(u_sorted_s, p, lo_s, hi_s)
        sc_ns = pp_scores(u_sorted_ns, p, lo_ns, hi_ns)

        cls = classify_station(sc_s["E"], sc_ns["E"], tau=args.tau)

        # LRT info (useful later)
        lrt = compare_gev_models(fit_s, fit_ns)

        rows.append({
            "station": station,
            "filename": fp.name,
            "n": n,
            "B": args.B,
            "tau": args.tau,
            # fits
            "mu_S": fit_s["mu"],
            "sigma_S": fit_s["sigma"],
            "xi_S": fit_s["xi"],
            "beta0_NS": fit_ns["beta0"],
            "beta1_NS": fit_ns["beta1"],
            "sigma_NS": fit_ns["sigma"],
            "xi_NS": fit_ns["xi"],
            # likelihood and selection
            "ll_S": fit_s["ll"],
            "ll_NS": fit_ns["ll"],
            "aic_S": fit_s["aic"],
            "aic_NS": fit_ns["aic"],
            "bic_S": fit_s["bic"],
            "bic_NS": fit_ns["bic"],
            "lrt_stat": lrt["lrt"],
            "lrt_p": lrt["p"],
            "NS_selected_LRT_5pct": int(lrt["p"] < 0.05),
            # GOF scores
            "E_S": sc_s["E"],
            "MAD_S": sc_s["MAD"],
            "MAX_S": sc_s["MAX"],
            "E_NS": sc_ns["E"],
            "MAD_NS": sc_ns["MAD"],
            "MAX_NS": sc_ns["MAX"],
            "deltaE": sc_s["E"] - sc_ns["E"],
            "deltaMAD": sc_s["MAD"] - sc_ns["MAD"],
            "class": cls,
            # bootstrap kept counts
            "B_kept_S": int(U_s.shape[0]),
            "B_kept_NS": int(U_ns.shape[0]),
        })

        if args.make_plots:
            outpng = PLOT_DIR / f"{station}_PIT_PP_bootstrap.png"
            plot_station(
                station=station,
                u_sorted_s=u_sorted_s,
                u_sorted_ns=u_sorted_ns,
                p=p,
                lo_s=lo_s,
                hi_s=hi_s,
                lo_ns=lo_ns,
                hi_ns=hi_ns,
                outpath=outpng,
            )

        kept += 1
        if kept % 25 == 0:
            print(f"Processed {kept} stations...")

    outcsv = GOF_BOOT_DIR / "pp_boot_scores.csv"
    outdf = pd.DataFrame(rows)
    outdf.to_csv(outcsv, index=False)
    print(f"\nWrote: {outcsv}")
    print(f"Kept: {kept}   Skipped: {skipped}")

    if kept > 0:
        # Quick summary to stdout
        print("\nClass counts (0=both ok, 1=NS improves, 2=neither ok, 3=S ok NS worse):")
        print(outdf["class"].value_counts().sort_index())

        print("\nMedian deltaE (E_S - E_NS):", float(outdf["deltaE"].median()))
        print("Median deltaMAD (MAD_S - MAD_NS):", float(outdf["deltaMAD"].median()))

if __name__ == "__main__":
    main()

