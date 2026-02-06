#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GESLA_data_checker_4_FAST_RDSFIX.py

FAST checker, but with a *robust* RDS writer that guarantees the resulting
.rds can be read by base R's readRDS().

It writes .rds by calling R itself (Rscript), which avoids compatibility
problems sometimes seen with Python-side RDS writers.

Behaviour preserved:
- WORKDIR: ./
- Reads filenames list: DATA/GESLA_FILENAMES_HOME2.txt
- Cleans: OUTPUT/ONTHEHOUR2/*
- Writes: OUTPUT/ONTHEHOUR2/<station>.rds
- Appends failures to: OUTPUT/ONTHEHOUR2/completeness_log.txt
- Filters qf2==1 and keeps only the most common minute-of-hour
- Completeness definition matches your R code:
    n / ((max_year-min_year) * 365.25 * 24) * 100

Speed preserved:
- Two-pass read: (date,time,qf2) first; full read only for passing stations
- Uses string slicing for year/minute in the cheap pass
- Optional parallel processing with --jobs

Requirements:
- Python: pandas
- R: Rscript in PATH
"""

import os
import re
import glob
import argparse
import tempfile
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd


# ----------------------------
# Configuration (same as R)
# ----------------------------
MINLENGTH_YRS = 55         # year span required
PCT_COMPLETE_REQ = 85      # percentage completeness required

WORKDIR = Path("./").expanduser()
FILENAMES_CSV = WORKDIR / "DATA/GESLA_FILENAMES_HOME2.txt"
OUTDIR = WORKDIR / "OUTPUT/ONTHEHOUR2"
LOGFILE = OUTDIR / "completeness_log.txt"

HEADER_NLINES = 41  # R used readLines(..., n=41) and read.csv(..., skip=41)


# ----------------------------
# Header parsing helpers
# ----------------------------
def get_header(filepath: Path) -> list[str]:
    if not filepath.exists():
        raise FileNotFoundError(f"File does not exist: {filepath}")
    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        header = []
        for _ in range(HEADER_NLINES):
            line = f.readline()
            if line == "":
                break
            header.append(line.rstrip("\n"))
    return header


def _find_line(header: list[str], needle: str) -> str:
    for line in header:
        if needle.lower() in line.lower():
            return line
    raise ValueError(f"No line containing '{needle}' found.")


def _extract_first_float(line: str) -> float:
    m = re.search(r"[-]?\d+\.?\d*", line)
    if not m:
        raise ValueError(f"Numeric value could not be parsed from line: {line}")
    return float(m.group(0))


def get_longitude(header: list[str]) -> float:
    line = _find_line(header, "LONGITUDE")
    return _extract_first_float(line)


def get_latitude(header: list[str]) -> float:
    line = _find_line(header, "LATITUDE")
    return _extract_first_float(line)


def get_country(header: list[str]) -> str:
    line = _find_line(header, "COUNTRY")
    m = re.search(r"\bCOUNTRY\b\s+(.*)$", line, flags=re.IGNORECASE)
    if not m:
        return line.strip()
    return m.group(1).strip()


def get_n_years(header: list[str]) -> float:
    line = _find_line(header, "NUMBER OF YEARS")
    m = re.search(r"\bNUMBER OF YEARS\b\s+([0-9]+(?:\.[0-9]+)?)", line, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not parse number of years from line: {line}")
    return float(m.group(1))


def get_is_coastal(header: list[str]) -> bool:
    try:
        line = _find_line(header, "GAUGE TYPE")
    except ValueError:
        return False
    return re.search(r"\bCoastal\b", line, flags=re.IGNORECASE) is not None


# ----------------------------
# Filenames list loader
# ----------------------------
def load_filenames(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing filenames list: {path}")

    # Try pandas CSV first
    try:
        df = pd.read_csv(path, dtype=str)
        if df.shape[1] >= 1:
            col0_name = str(df.columns[0])
            vals = df.iloc[:, 0].astype(str).tolist()
            if ("/" in col0_name or col0_name.startswith("~")):
                df2 = pd.read_csv(path, header=None, dtype=str)
                return df2.iloc[:, 0].astype(str).tolist()
            return vals
    except Exception:
        pass

    # Fallback: plain text
    out = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s.split(",")[0])
    return out


# ----------------------------
# RDS writing via Rscript (guaranteed readable in R)
# ----------------------------
_R_WRITE_CODE = """
args <- commandArgs(trailingOnly=TRUE)
in_csv <- args[1]
out_rds <- args[2]
d <- read.csv(in_csv, stringsAsFactors=FALSE)
if ("POSIX" %in% names(d)) {
  d$POSIX <- as.POSIXct(d$POSIX, tz="UTC", format="%Y-%m-%d %H:%M:%S")
}
saveRDS(d, file=out_rds)
"""


def write_rds_via_rscript(df: pd.DataFrame, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    dfx = df.copy()
    if "POSIX" in dfx.columns:
        pos = pd.to_datetime(dfx["POSIX"], errors="coerce", utc=True)
        dfx["POSIX"] = pos.dt.strftime("%Y-%m-%d %H:%M:%S")

    with tempfile.TemporaryDirectory(prefix="gesla_rds_") as td:
        td_path = Path(td)
        csv_path = td_path / "tmp.csv"
        dfx.to_csv(csv_path, index=False)

        cmd = ["Rscript", "-e", _R_WRITE_CODE, str(csv_path), str(outpath)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "Rscript failed while writing .rds\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}\n"
            )


# ----------------------------
# Fast station processing
# ----------------------------
def process_one_station(path_str: str) -> tuple[str, str | None]:
    filepath = Path(path_str).expanduser()
    station = filepath.name

    # Header + metadata
    try:
        header = get_header(filepath)
        longitude = get_longitude(header)
        latitude = get_latitude(header)
        country = get_country(header)
        n_years = get_n_years(header)
        is_coastal = get_is_coastal(header)
    except Exception as e:
        return station, f"Station {station} failed header parse: {e}\n"

    _linje = [station, longitude, latitude, country, n_years, is_coastal]

    if not (is_coastal and (n_years >= MINLENGTH_YRS)):
        return station, None

    # PASS A: cheap read
    try:
        df_small = pd.read_csv(
            filepath,
            skiprows=HEADER_NLINES,
            header=None,
            sep=r"\s+",
            engine="python",
            usecols=[0, 1, 4],
            names=["date", "time", "qf2"]
        )
    except Exception as e:
        return station, f"Station {station} failed read_csv (small): {e}\n"

    df_small["qf2"] = pd.to_numeric(df_small["qf2"], errors="coerce")
    df_small = df_small.loc[df_small["qf2"] == 1].copy()
    if df_small.empty:
        return station, f"Station {station} failed completeness: 0.00% (no rows after qf2==1)\n"

    time_str = df_small["time"].astype(str)
    date_str = df_small["date"].astype(str)

    minute = pd.to_numeric(time_str.str.slice(3, 5), errors="coerce")
    df_small["minute"] = minute
    df_small = df_small.loc[df_small["minute"].notna()].copy()
    if df_small.empty:
        return station, f"Station {station} failed completeness: 0.00% (no valid minutes)\n"

    most_common_minute = int(df_small["minute"].value_counts().idxmax())
    df_small = df_small.loc[df_small["minute"] == most_common_minute].copy()

    year = pd.to_numeric(date_str.str.slice(0, 4), errors="coerce")
    df_small["year"] = year
    df_small = df_small.loc[df_small["year"].notna()].copy()
    if df_small.empty:
        return station, f"Station {station} failed completeness: 0.00% (no valid years)\n"

    min_year = int(df_small["year"].min())
    max_year = int(df_small["year"].max())

    denom = (max_year - min_year) * 365.25 * 24.0
    completeness_pct = 0.0 if denom <= 0 else (len(df_small) / denom) * 100.0

    if completeness_pct < PCT_COMPLETE_REQ:
        return station, f"Station {station} failed completeness: {completeness_pct:.2f}%\n"

    # PASS B: full read
    try:
        df = pd.read_csv(
            filepath,
            skiprows=HEADER_NLINES,
            header=None,
            sep=r"\s+",
            engine="python"
        )
    except Exception as e:
        return station, f"Station {station} failed read_csv (full): {e}\n"

    if df.shape[1] < 5:
        return station, f"Station {station} failed read (expected >=5 cols, got {df.shape[1]}).\n"

    df = df.iloc[:, 0:5].copy()
    df.columns = ["date", "time", "obs", "qf1", "qf2"]

    df["qf2"] = pd.to_numeric(df["qf2"], errors="coerce")
    df = df.loc[df["qf2"] == 1].copy()
    if df.empty:
        return station, f"Station {station} failed completeness: 0.00% (no rows after qf2==1 in full read)\n"

    minute_full = pd.to_numeric(df["time"].astype(str).str.slice(3, 5), errors="coerce")
    df = df.loc[minute_full == most_common_minute].copy()
    if df.empty:
        return station, f"Station {station} failed completeness: 0.00% (no rows at mode minute in full read)\n"

    dt = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%Y/%m/%d %H:%M:%S",
        errors="coerce",
        utc=True
    )
    df["POSIX"] = dt
    df = df.loc[df["POSIX"].notna()].copy()
    if df.empty:
        return station, f"Station {station} failed completeness: 0.00% (no valid POSIX after parsing)\n"

    df = df.drop(columns=["date", "time"])

    outpath = OUTDIR / f"{station}.rds"
    try:
        write_rds_via_rscript(df, outpath)
    except Exception as e:
        return station, f"Station {station} failed write_rds: {e}\n"

    return station, None


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="GESLA data checker (fast) writing .rds via Rscript")
    parser.add_argument("--jobs", type=int, default=1, help="Number of worker processes (default 1)")
    parser.add_argument("--quiet", action="store_true", help="Less console output")
    args = parser.parse_args()

    os.chdir(WORKDIR)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Clean old outputs
    for p in glob.glob(str(OUTDIR / "*")):
        try:
            os.remove(p)
        except IsADirectoryError:
            for q in glob.glob(os.path.join(p, "*")):
                try:
                    os.remove(q)
                except Exception:
                    pass
        except FileNotFoundError:
            pass

    filenames = load_filenames(FILENAMES_CSV)
    filenames = [str(s).strip() for s in filenames if str(s).strip()]

    log_lines: list[str] = []

    if args.jobs <= 1:
        for i, name in enumerate(filenames, start=1):
            station, logline = process_one_station(name)
            if (not args.quiet) and (i % 25 == 0):
                print(f"{i}/{len(filenames)} processed... (latest: {station})")
            if logline:
                log_lines.append(logline)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(process_one_station, name): name for name in filenames}
            done = 0
            for fut in as_completed(futs):
                done += 1
                try:
                    station, logline = fut.result()
                except Exception as e:
                    station = Path(futs[fut]).name
                    logline = f"Station {station} failed worker exception: {e}\n"

                if (not args.quiet) and (done % 50 == 0 or done == len(filenames)):
                    print(f"{done}/{len(filenames)} processed... (latest: {station})")
                if logline:
                    log_lines.append(logline)

    if log_lines:
        with LOGFILE.open("a", encoding="utf-8") as logf:
            for line in log_lines:
                logf.write(line)

    print("\f", end="")


if __name__ == "__main__":
    main()
