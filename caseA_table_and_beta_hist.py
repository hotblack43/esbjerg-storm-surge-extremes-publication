#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
caseA_table_and_beta_hist.py

One script that:
  (1) Filters Case A (class==1) from pp_boot_scores.csv
  (2) Finds the corresponding raw GESLA4 file in ~/GESLA4/DATA/ and parses the first 41 header lines
      to extract station name, country, latitude, longitude.
  (3) Writes:
        - caseA_station_metadata_from_GESLA4.csv
        - caseA_table.tex  (LaTeX longtable, shortened names, fixed-decimal rounding)
        - beta1_caseA_hist.png

Run:
  uv run caseA_table_and_beta_hist.py

Outputs go to:
  ~/WORKSHOP/esbjerg-storm-surge-extremes-publication/FIGURES/GOF_GEV_BOOTSTRAP/

Name shortening:
  - split on '_' or '-'
  - keep first two tokens that contain at least one letter
  - title-case

Rounding / formatting in LaTeX:
  - Lat/Lon: 2 decimals
  - beta1_NS, deltaE, E_S, E_NS: 3 decimals
  - blanks for missing values (no "nan", no 0.000000 noise)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Defaults (match your layout)
# ----------------------------
BASE_DIR = Path.home() / "WORKSHOP/esbjerg-storm-surge-extremes-publication"
DEFAULT_SCORES_CSV = BASE_DIR / "FIGURES" / "GOF_GEV_BOOTSTRAP" / "pp_boot_scores.csv"
DEFAULT_OUT_DIR = BASE_DIR / "FIGURES" / "GOF_GEV_BOOTSTRAP"
DEFAULT_GESLA4_DIR = Path.home() / "GESLA4" / "DATA"
HEADER_LINES = 41


# ----------------------------
# Header parsing helpers
# ----------------------------
def _normalise_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).upper()


def parse_header_lines(lines: List[str]) -> Dict[str, str]:
    """
    Parse GESLA header lines robustly.

    Handles:
      # COUNTRY USA                     (single space)
      # LATITUDE      42.35500000       (multiple spaces)
      # KEY: value
      # KEY=value

    Strategy:
      1) strip leading '#', whitespace
      2) try split on 2+ spaces
      3) else try split on 1+ spaces
      4) else try KEY:VALUE / KEY=VALUE
    """
    d: Dict[str, str] = {}

    for raw in lines:
        line = raw.rstrip("\n").lstrip("#").strip()
        if not line:
            continue

        # Try "KEY<2+ spaces>VALUE"
        parts = re.split(r"\s{2,}", line, maxsplit=1)
        if len(parts) == 2:
            key, val = parts
            d[_normalise_key(key)] = val.strip()
            continue

        # Try "KEY VALUE" (single-space split)
        parts = re.split(r"\s+", line, maxsplit=1)
        if len(parts) == 2:
            key, val = parts
            # key is typically a single token here (e.g. COUNTRY, SITE, etc.)
            # This catches COUNTRY USA, SITE NAME Boston_MA, etc.
            d[_normalise_key(key)] = val.strip()
            continue

        # Fallback: KEY: value or KEY=value
        m = re.match(r"^([^:=]+)[:=]\s*(.+)$", line)
        if m:
            d[_normalise_key(m.group(1))] = m.group(2).strip()

    return d


def pick_first(d: Dict[str, str], keys: List[str]) -> Optional[str]:
    for k in keys:
        kk = _normalise_key(k)
        if kk in d:
            return d[kk]
    return None


def as_float(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s.replace(",", "."))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def read_header(path: Path, n_lines: int = HEADER_LINES) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = [next(f) for _ in range(n_lines)]
    return parse_header_lines(lines)


def find_gesla_file(station_annual: str, gesla_dir: Path) -> Tuple[Optional[Path], str]:
    """
    station_annual example:
      'honolulu-1612340-usa-noaa_annual'
    raw file in GESLA directory usually:
      'honolulu-1612340-usa-noaa'
    """
    raw = station_annual
    if raw.endswith("_annual"):
        raw = raw[: -len("_annual")]

    p = gesla_dir / raw
    if p.exists() and p.is_file():
        return p, "exact"

    hits = sorted([h for h in gesla_dir.glob(raw + "*") if h.is_file()])
    if len(hits) == 1:
        return hits[0], "prefix-unique"
    if len(hits) > 1:
        hits.sort(key=lambda x: (len(x.name), x.name))
        return hits[0], f"prefix-multi({len(hits)})"

    return None, "not-found"


# ----------------------------
# Presentation helpers
# ----------------------------
def shorten_name(name: str) -> str:
    """
    Your requested rule:
      - split on '_' or '-'
      - keep first two tokens that contain at least one letter
      - title-case
    """
    if not name:
        return ""
    raw_tokens = re.split(r"[_-]+", name.strip())

    tokens: List[str] = []
    for t in raw_tokens:
        t = t.strip()
        if not t:
            continue
        if re.search(r"[A-Za-z]", t):
            tokens.append(t)
        if len(tokens) == 2:
            break

    if not tokens:
        return ""
    return " ".join(tokens).title()


def latex_escape_minimal(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def fmt_num(x, ndp: int) -> str:
    """Format number with fixed decimals; blank if missing."""
    try:
        if x is None:
            return ""
        xv = float(x)
        if not np.isfinite(xv):
            return ""
        return f"{xv:.{ndp}f}"
    except Exception:
        return ""


def fd_bins(x: np.ndarray) -> int:
    """Freedman–Diaconis bin count with safety caps."""
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 10
    q75, q25 = np.quantile(x, [0.75, 0.25])
    iqr = q75 - q25
    if iqr <= 0:
        return max(10, int(np.sqrt(n)))
    bw = 2.0 * iqr / (n ** (1.0 / 3.0))
    if bw <= 0:
        return max(10, int(np.sqrt(n)))
    nb = int(np.ceil((x.max() - x.min()) / bw))
    return max(10, min(nb, 80))


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=str, default=str(DEFAULT_SCORES_CSV), help="pp_boot_scores.csv path")
    ap.add_argument("--gesla_dir", type=str, default=str(DEFAULT_GESLA4_DIR), help="GESLA4/DATA directory")
    ap.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of Case A stations (0=all)")
    ap.add_argument("--bins", type=int, default=0, help="Histogram bins (0=Freedman–Diaconis)")
    args = ap.parse_args()

    scores_path = Path(args.scores).expanduser()
    gesla_dir = Path(args.gesla_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not scores_path.exists():
        raise SystemExit(f"Scores CSV not found: {scores_path}")
    if not gesla_dir.exists():
        raise SystemExit(f"GESLA directory not found: {gesla_dir}")

    df = pd.read_csv(scores_path)
    needed = {"station", "filename", "class", "n", "beta1_NS", "deltaE", "E_S", "E_NS", "NS_selected_LRT_5pct"}
    missing = sorted(list(needed - set(df.columns)))
    if missing:
        raise SystemExit(f"Scores CSV missing required columns: {missing}")

    caseA = df[df["class"] == 1].copy()
    if args.limit and args.limit > 0:
        caseA = caseA.head(args.limit)

    print(f"Case A stations: {len(caseA)}")

    rows = []
    not_found = 0

    for _, r in caseA.iterrows():
        station = str(r["station"])
        fp, how = find_gesla_file(station, gesla_dir)

        meta = {
            "station": station,
            "annual_filename": str(r["filename"]),
            "gesla_file": str(fp) if fp else "",
            "gesla_match": how,
            "lat": np.nan,
            "lon": np.nan,
            "country": "",
            "station_name": "",
        }

        if fp is None:
            not_found += 1
        else:
            try:
                h = read_header(fp, HEADER_LINES)

                # GESLA headers commonly use these
                meta["station_name"] = pick_first(h, ["SITE NAME", "STATION NAME", "STATION", "NAME"]) or ""
                meta["country"] = pick_first(h, ["COUNTRY"]) or ""

                meta["lat"] = as_float(pick_first(h, ["LATITUDE", "LAT"])) or np.nan
                meta["lon"] = as_float(pick_first(h, ["LONGITUDE", "LON"])) or np.nan

            except Exception as e:
                meta["gesla_match"] = f"{how}-readfail"
                meta["station_name"] = f"HEADER_READ_FAILED: {e}"

        meta.update({
            "n_annual": int(r["n"]),
            "beta1_NS": float(r["beta1_NS"]),
            "deltaE": float(r["deltaE"]),
            "E_S": float(r["E_S"]),
            "E_NS": float(r["E_NS"]),
            "NS_selected_LRT_5pct": int(r["NS_selected_LRT_5pct"]),
        })

        rows.append(meta)

    out_meta = pd.DataFrame(rows)

    meta_csv = out_dir / "caseA_station_metadata_from_GESLA4.csv"
    out_meta.to_csv(meta_csv, index=False)
    print(f"Wrote: {meta_csv}")
    if not_found:
        print(f"WARNING: {not_found} Case A stations could not be matched in {gesla_dir}")

    # ----------------------------
    # LaTeX table
    # ----------------------------
    tab = out_meta.copy()

    def display_name(row) -> str:
        nm = str(row.get("station_name", "")).strip()
        if nm and not nm.startswith("HEADER_READ_FAILED"):
            return shorten_name(nm)
        return shorten_name(str(row["station"]))

    tab["Station"] = tab.apply(display_name, axis=1).map(latex_escape_minimal)
    tab["Country"] = tab["country"].fillna("").astype(str).map(latex_escape_minimal)

    tab["Lat"] = pd.to_numeric(tab["lat"], errors="coerce").map(lambda v: fmt_num(v, 2))
    tab["Lon"] = pd.to_numeric(tab["lon"], errors="coerce").map(lambda v: fmt_num(v, 2))
    tab["Years"] = tab["n_annual"].astype(int).astype(str)

    tab[r"$\hat{\beta}_1$"] = pd.to_numeric(tab["beta1_NS"], errors="coerce").map(lambda v: fmt_num(v, 3))
    tab[r"$\Delta E$"] = pd.to_numeric(tab["deltaE"], errors="coerce").map(lambda v: fmt_num(v, 3))
    tab[r"$E_S$"] = pd.to_numeric(tab["E_S"], errors="coerce").map(lambda v: fmt_num(v, 3))
    tab[r"$E_{NS}$"] = pd.to_numeric(tab["E_NS"], errors="coerce").map(lambda v: fmt_num(v, 3))

    tab["LRT(5\\%)"] = tab["NS_selected_LRT_5pct"].map({0: "S", 1: "NS"}).astype(str).map(latex_escape_minimal)

    tab = tab[[
        "Station", "Country", "Lat", "Lon", "Years",
        r"$\hat{\beta}_1$", r"$\Delta E$", r"$E_S$", r"$E_{NS}$", "LRT(5\\%)"
    ]].copy()

    # sort by numeric beta descending
    tab["_sortbeta"] = pd.to_numeric(out_meta["beta1_NS"], errors="coerce").values
    tab = tab.sort_values(by="_sortbeta", ascending=False).drop(columns=["_sortbeta"])

    tex_path = out_dir / "caseA_table.tex"
    tex = tab.to_latex(
        index=False,
        escape=False,
        longtable=True,
        caption=(
            "Stations classified as Case~A (non-stationary model improves calibration), "
            "augmented with country/latitude/longitude parsed from the first 41 header lines of the corresponding "
            "GESLA4 files. Station names are shortened for readability. "
            "Here $\\hat{\\beta}_1$ is the estimated mean-sea-level effect on the GEV location parameter, "
            "$\\Delta E = E_S - E_{NS}$ is the reduction in the fraction of PP points outside the 95\\% "
            "parametric-bootstrap envelope when moving from S to NS, and $E_S$ and $E_{NS}$ are the envelope-exceedance "
            "fractions for the two models."
        ),
        label="tab:caseA_stations",
        column_format="p{3.6cm} p{2.0cm} r r r r r r r c",
        na_rep="",
    )
    tex_path.write_text(tex, encoding="utf-8")
    print(f"Wrote: {tex_path}")

    # ----------------------------
    # Histogram of beta1_NS (Case A)
    # ----------------------------
    beta = pd.to_numeric(out_meta["beta1_NS"], errors="coerce").to_numpy()
    beta = beta[np.isfinite(beta)]
    if beta.size == 0:
        print("No finite beta1_NS values for Case A; skipping histogram.")
        return

    bins = args.bins if (args.bins and args.bins > 0) else fd_bins(beta)
    q = np.quantile(beta, [0.05, 0.25, 0.50, 0.75, 0.95])
    frac_pos = float(np.mean(beta > 0))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(beta, bins=bins)
    ax.axvline(0.0, linestyle="--", linewidth=2)
    ax.set_title("Distribution of $\\beta_1$ (Case A: NS improves calibration)")
    ax.set_xlabel(r"$\hat{\beta}_1$ (MSL effect on location)")
    ax.set_ylabel("Count")
    ax.text(
        0.02, 0.98,
        f"n={beta.size}\nmedian={q[2]:.3g}\nIQR=[{q[1]:.3g},{q[3]:.3g}]\nfrac>0={frac_pos:.2f}",
        transform=ax.transAxes,
        ha="left", va="top",
    )
    fig.tight_layout()

    out_png = out_dir / "beta1_caseA_hist.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_png}")

    print("Done.")


if __name__ == "__main__":
    main()

