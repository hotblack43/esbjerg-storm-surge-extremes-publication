#!/usr/bin/env python3
"""
QC for annual time series files in OUTPUT/ANNUALS2/
Flags: missing/integrity issues, robust outliers (MAD), spikes (diff), level shifts (robust breakpoint scan)

Outputs:
  OUTDIR/qc_summary.csv
  OUTDIR/qc_details_<basename>.csv
  OUTDIR/qc_plot_<basename>_MSL_ANNMAX.png  (optional, 2-panel)

Usage:
  python3 qc_annuals2_2.py --indir OUTPUT/ANNUALS2 --outdir OUTPUT/ANNUALS2_QC --plot 1
  uv run qc_annuals2_2.py --indir OUTPUT/ANNUALS2 --outdir OUTPUT/ANNUALS2_QC --plot 1 > qc_output.txt
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _finite(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)


def median_ignore_nan(x: np.ndarray) -> float:
    x = x[_finite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(x))


def mad_scale(x: np.ndarray) -> float:
    x = x[_finite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return float("nan")
    return float(1.4826 * mad)


def robust_z(x: np.ndarray) -> np.ndarray:
    z = np.full_like(x, np.nan, dtype=float)
    mask = _finite(x)
    if not np.any(mask):
        return z
    med = np.median(x[mask])
    s = mad_scale(x)
    if not np.isfinite(s) or s == 0:
        return z
    z[mask] = (x[mask] - med) / s
    return z


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = list(df.columns)
    cols_norm = [norm_name(c) for c in cols]
    cand_norm = [norm_name(c) for c in candidates]
    for cn in cand_norm:
        if cn in cols_norm:
            return cols[cols_norm.index(cn)]
    return None


def sniff_delimiter(sample: str) -> Optional[str]:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return None


def read_any_table(path: str) -> pd.DataFrame:
    with open(path, "r", errors="replace") as f:
        sample = f.read(4096)

    delim = sniff_delimiter(sample)

    if delim is not None:
        try:
            return pd.read_csv(path, sep=delim, engine="python")
        except Exception:
            pass

    try:
        return pd.read_csv(path)
    except Exception:
        pass

    try:
        return pd.read_csv(path, sep=";", engine="python")
    except Exception:
        pass

    try:
        return pd.read_csv(path, sep="\t", engine="python")
    except Exception:
        pass

    return pd.read_csv(path, delim_whitespace=True, engine="python")


@dataclass
class LevelShiftResult:
    change_indices: List[int]
    notes: List[str]
    ok: Optional[bool]


def detect_level_shifts_robust(
    years: np.ndarray,
    x: np.ndarray,
    min_seg: int = 6,
    shift_k: float = 3.0,
    max_cpts: int = 5,
    improvement_frac: float = 0.10,
) -> LevelShiftResult:
    notes: List[str] = []
    change_points: List[int] = []

    mask = _finite(years) & _finite(x)
    if np.sum(mask) < (2 * min_seg + 2):
        notes.append("too_short_for_shift_test")
        return LevelShiftResult(change_indices=[], notes=notes, ok=True)

    idx = np.where(mask)[0]
    x2 = x[idx]

    s = mad_scale(x2)
    if not np.isfinite(s) or s == 0:
        notes.append("mad_zero_shift_test")
        return LevelShiftResult(change_indices=[], notes=notes, ok=None)

    def segment_cost(a: np.ndarray) -> float:
        a = a[_finite(a)]
        if a.size == 0:
            return 0.0
        med = np.median(a)
        return float(np.sum(np.abs(a - med)))

    def best_split(xseg: np.ndarray) -> Tuple[Optional[int], float, float]:
        n = xseg.size
        base_cost = segment_cost(xseg)
        best_t = None
        best_cost = base_cost
        best_delta = 0.0

        for t in range(min_seg, n - min_seg + 1):
            left = xseg[:t]
            right = xseg[t:]
            if left.size < min_seg or right.size < min_seg:
                continue
            m1 = median_ignore_nan(left)
            m2 = median_ignore_nan(right)
            if not (np.isfinite(m1) and np.isfinite(m2)):
                continue
            delta = abs(m2 - m1)
            if delta < shift_k * s:
                continue
            cost = segment_cost(left) + segment_cost(right)
            if cost < best_cost:
                best_cost = cost
                best_t = t
                best_delta = delta

        return best_t, best_cost, best_delta

    segments: List[Tuple[int, int]] = [(0, x2.size)]
    base_total_cost = segment_cost(x2)

    for _ in range(max_cpts):
        best_overall = None  # (improvement, seg_index, split_at, delta, new_total_cost)
        current_total_cost = 0.0
        for a, b in segments:
            current_total_cost += segment_cost(x2[a:b])

        for si, (a, b) in enumerate(segments):
            xseg = x2[a:b]
            if xseg.size < 2 * min_seg:
                continue
            t, new_cost_seg, delta = best_split(xseg)
            if t is None:
                continue
            old_cost_seg = segment_cost(xseg)
            new_total_cost = current_total_cost - old_cost_seg + new_cost_seg
            improvement = current_total_cost - new_total_cost
            if improvement < improvement_frac * base_total_cost:
                continue
            candidate = (improvement, si, a + t, delta, new_total_cost)
            if best_overall is None or candidate[0] > best_overall[0]:
                best_overall = candidate

        if best_overall is None:
            break

        _, si, split_at, _, _ = best_overall
        a, b = segments[si]
        segments.pop(si)
        segments.insert(si, (a, split_at))
        segments.insert(si + 1, (split_at, b))
        change_points.append(split_at)

    change_points = sorted(set(change_points))
    if len(change_points) == 0:
        notes.append("no_level_shift_detected")
        return LevelShiftResult(change_indices=[], notes=notes, ok=True)

    change_indices = [int(idx[cp]) for cp in change_points]
    cp_years = [int(years[i]) for i in change_indices if np.isfinite(years[i])]
    notes.append("level_shift_years=" + "|".join(str(y) for y in cp_years))
    return LevelShiftResult(change_indices=change_indices, notes=notes, ok=False)


@dataclass
class SeriesQC:
    ok: bool
    flags: List[str]
    issues: List[str]
    outlier_level_years: List[int]
    outlier_spike_years: List[int]
    level_shift_years: List[int]


def qc_one_series(
    years: np.ndarray,
    x: np.ndarray,
    z_outlier: float = 6.0,
    z_spike: float = 6.0,
    min_seg: int = 6,
    shift_k: float = 3.0,
) -> SeriesQC:
    n = x.size
    flags = [""] * n
    issues: List[str] = []

    if not np.all(_finite(years)):
        issues.append("nonfinite_years")

    fy = years[_finite(years)]
    if fy.size > 0:
        if np.unique(fy).size != fy.size:
            issues.append("duplicate_years")
        if np.any(np.diff(fy) <= 0):
            issues.append("nonmonotone_years")
        if np.any(np.diff(fy) > 1):
            issues.append("gappy_years")

    bad_x = ~_finite(x)
    if np.any(bad_x):
        for i in np.where(bad_x)[0]:
            flags[i] += "NA_or_nonfinite;"
        issues.append("missing_values")

    z = robust_z(x)
    out_idx = np.where(_finite(z) & (np.abs(z) > z_outlier))[0]
    outlier_level_years: List[int] = []
    if out_idx.size > 0:
        for i in out_idx:
            flags[i] += "OUTLIER_LEVEL;"
        outlier_level_years = [int(years[i]) for i in out_idx if np.isfinite(years[i])]
        issues.append(f"outlier_level_n={out_idx.size}")

    d = np.full_like(x, np.nan, dtype=float)
    if n >= 2:
        d[1:] = np.diff(x)
    zd = robust_z(d)
    sp_idx = np.where(_finite(zd) & (np.abs(zd) > z_spike))[0]
    outlier_spike_years: List[int] = []
    if sp_idx.size > 0:
        for i in sp_idx:
            flags[i] += "OUTLIER_SPIKE;"
        outlier_spike_years = [int(years[i]) for i in sp_idx if np.isfinite(years[i])]
        issues.append(f"outlier_spike_n={sp_idx.size}")

    sh = detect_level_shifts_robust(years, x, min_seg=min_seg, shift_k=shift_k)
    level_shift_years: List[int] = []
    if sh.change_indices:
        for i in sh.change_indices:
            flags[i] += "LEVEL_SHIFT_CPT;"
        level_shift_years = [
            int(years[i]) for i in sh.change_indices if np.isfinite(years[i])
        ]
        issues.extend(sh.notes)
    else:
        issues.extend(sh.notes)

    strong_any = any(
        ("OUTLIER_LEVEL" in f or "OUTLIER_SPIKE" in f or "LEVEL_SHIFT_CPT" in f)
        for f in flags
    )
    integrity_bad = ("duplicate_years" in issues) or ("nonmonotone_years" in issues)
    ok = (not strong_any) and (not integrity_bad)

    issues = sorted(set(issues))

    return SeriesQC(
        ok=ok,
        flags=flags,
        issues=issues,
        outlier_level_years=sorted(set(outlier_level_years)),
        outlier_spike_years=sorted(set(outlier_spike_years)),
        level_shift_years=sorted(set(level_shift_years)),
    )


def plot_qc(
    years: np.ndarray, x: np.ndarray, flags: List[str], title: str, outfile: str
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 7))
    plt.plot(years, x, marker="o", linewidth=1)

    bad = np.array([f != "" for f in flags], dtype=bool)
    if np.any(bad):
        plt.scatter(
            years[bad],
            x[bad],
            s=90,
            facecolors="none",
            edgecolors="black",
            linewidths=2,
        )
        for y, xv, fl in zip(years[bad], x[bad], np.array(flags)[bad]):
            plt.text(y, xv, fl.rstrip(";"), fontsize=8, va="bottom", ha="center")

    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(title.split(":")[-1].strip())
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_qc_two_panel(
    years: np.ndarray,
    msl: np.ndarray,
    ann: np.ndarray,
    flags_msl: List[str],
    flags_ann: List[str],
    base: str,
    outfile: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # ---- Panel 1: MSL ----
    ax = axes[0]
    ax.plot(years, msl, marker="o", linewidth=1)

    bad = np.array([f != "" for f in flags_msl], dtype=bool)
    if np.any(bad):
        ax.scatter(
            years[bad],
            msl[bad],
            s=90,
            facecolors="none",
            edgecolors="black",
            linewidths=2,
        )
        for y, xv, fl in zip(years[bad], msl[bad], np.array(flags_msl)[bad]):
            ax.text(y, xv, fl.rstrip(";"), fontsize=8, va="bottom", ha="center")

    ax.set_title(f"{base}: MSL")
    ax.set_ylabel("MSL [cm]")

    # ---- Panel 2: ANNMAX ----
    ax = axes[1]
    ax.plot(years, ann, marker="o", linewidth=1)

    bad = np.array([f != "" for f in flags_ann], dtype=bool)
    if np.any(bad):
        ax.scatter(
            years[bad],
            ann[bad],
            s=90,
            facecolors="none",
            edgecolors="black",
            linewidths=2,
        )
        for y, xv, fl in zip(years[bad], ann[bad], np.array(flags_ann)[bad]):
            ax.text(y, xv, fl.rstrip(";"), fontsize=8, va="bottom", ha="center")

    ax.set_title(f"{base}: ANNMAX")
    ax.set_xlabel("Year")
    ax.set_ylabel("ANNMAX [cm]")

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="OUTPUT/ANNUALS2")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--pattern", default="*.aff")
    ap.add_argument("--also", default="*.csv,*.txt,*.dat,*.tsv")
    ap.add_argument("--z_outlier", type=float, default=6.0)
    ap.add_argument("--z_spike", type=float, default=6.0)
    ap.add_argument("--min_seg", type=int, default=6)
    ap.add_argument("--shift_k", type=float, default=3.0)
    ap.add_argument("--plot", type=int, default=0)
    args = ap.parse_args()

    indir = args.indir
    outdir = args.outdir if args.outdir is not None else os.path.join(indir, "QC")
    os.makedirs(outdir, exist_ok=True)

    patterns = [args.pattern] + [p.strip() for p in args.also.split(",") if p.strip()]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(indir, pat)))
    files = sorted({f for f in files if os.path.isfile(f)})

    if not files:
        raise SystemExit(f"No files found in {indir} with patterns: {patterns}")

    summary_rows: List[Dict[str, object]] = []

    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]

        try:
            df = read_any_table(path)
        except Exception as e:
            summary_rows.append(
                {
                    "file": os.path.basename(path),
                    "n": np.nan,
                    "ok_MSL": False,
                    "ok_ANNMAX": False,
                    "overall_ok": False,
                    "notes": f"READ_FAIL: {e}",
                }
            )
            continue

        # FIX: include storm_year (your files) as a year candidate
        year_col = pick_column(
            df,
            [
                "year",
                "yr",
                "yyyy",
                "annual_year",
                "summer_year",
                "storm_year",
                "stormyear",
                "season_year",
                "water_year",
            ],
        )
        msl_col = pick_column(
            df, ["msl", "mean_sea_level", "meansealevel", "gmsl", "msl_cm", "msl_mm"]
        )
        ann_col = pick_column(
            df,
            [
                "annmax",
                "annual_max",
                "annualmax",
                "max",
                "max_residual",
                "annual_residual_max",
                "rx1y",
            ],
        )

        if year_col is None or msl_col is None or ann_col is None:
            summary_rows.append(
                {
                    "file": os.path.basename(path),
                    "n": int(df.shape[0]),
                    "ok_MSL": False,
                    "ok_ANNMAX": False,
                    "overall_ok": False,
                    "notes": "MISSING_COLS: need year+MSL+ANNMAX. "
                    f"Have: {list(df.columns)}",
                }
            )
            continue

        years = pd.to_numeric(df[year_col], errors="coerce").to_numpy(dtype=float)
        msl = pd.to_numeric(df[msl_col], errors="coerce").to_numpy(dtype=float)
        ann = pd.to_numeric(df[ann_col], errors="coerce").to_numpy(dtype=float)

        order = np.argsort(years)
        years = years[order]
        msl = msl[order]
        ann = ann[order]

        qc_msl = qc_one_series(
            years, msl, args.z_outlier, args.z_spike, args.min_seg, args.shift_k
        )
        qc_ann = qc_one_series(
            years, ann, args.z_outlier, args.z_spike, args.min_seg, args.shift_k
        )

        overall_ok = bool(qc_msl.ok and qc_ann.ok)

        details = pd.DataFrame(
            {
                "storm_year": years,  # keep your naming in output
                "mean_sea_level": msl,
                "max_residual": ann,
                "flags_MSL": qc_msl.flags,
                "flags_ANNMAX": qc_ann.flags,
            }
        )
        details_out = os.path.join(outdir, f"qc_details_{base}.csv")
        details.to_csv(details_out, index=False)

        if args.plot == 1:
            plot_qc_two_panel(
                years,
                msl,
                ann,
                qc_msl.flags,
                qc_ann.flags,
                base,
                os.path.join(outdir, f"qc_plot_{base}_MSL_ANNMAX.png"),
            )

        notes = " | ".join(
            [
                "MSL:" + ",".join(qc_msl.issues),
                "ANNMAX:" + ",".join(qc_ann.issues),
            ]
        )

        summary_rows.append(
            {
                "file": os.path.basename(path),
                "n": int(len(years)),
                "ok_MSL": bool(qc_msl.ok),
                "ok_ANNMAX": bool(qc_ann.ok),
                "overall_ok": overall_ok,
                "notes": notes,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["overall_ok", "file"], ascending=[True, True]
    )
    summary_out = os.path.join(outdir, "qc_summary.csv")
    summary_df.to_csv(summary_out, index=False)

    print("Wrote:")
    print(" ", summary_out)
    print(" ", os.path.join(outdir, "qc_details_<file>.csv"))
    if args.plot == 1:
        print(" ", os.path.join(outdir, "qc_plot_<file>_MSL_ANNMAX.png"))
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
