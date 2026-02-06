#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
from pathlib import Path

import pandas as pd


# ----------------------------
# User settings (edit these)
# ----------------------------
DATA_DIR = Path("/home/pth/GESLA4/DATA/")   # where the GESLA files are kept
OUTPUT_CSV = Path("DATA/gesla_station_index.csv")

HEADER_LINES = 41
HOURS_PER_YEAR = 8760


def extract_last_number(line: str):
    """
    Extract the last numeric token in a line (supports +/-, decimals).
    Returns float or None.
    """
    m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*$", line.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def extract_last_int(line: str):
    m = re.search(r"(\d+)\s*$", line.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def extract_country(line: str):
    """
    Mimic: sub("^.*?\\bCOUNTRY\\s+", "", line)
    """
    m = re.search(r"\bCOUNTRY\s+(.*)$", line)
    if not m:
        return None
    return m.group(1).strip()


def read_header(path: Path, n: int):
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            header = []
            for _ in range(n):
                line = f.readline()
                if line == "":
                    break
                header.append(line.rstrip("\n"))
        return header
    except Exception:
        return None


def read_body_as_dataframe(path: Path, skip_lines: int):
    """
    R uses: read.csv(path, skip=41, header=FALSE, comment.char="#")
    We mimic by reading as CSV with no header, ignoring comment lines starting '#'.
    """
    try:
        df = pd.read_csv(
            path,
            skiprows=skip_lines,
            header=None,
            comment="#",
            engine="python",
        )
        return df
    except Exception:
        return None


def main() -> int:
    if not DATA_DIR.exists():
        print(f"ERROR: DATA_DIR does not exist: {DATA_DIR}", file=sys.stderr)
        return 2

    files = sorted([p for p in DATA_DIR.iterdir() if p.is_file()])
    if not files:
        print(f"ERROR: no files found in: {DATA_DIR}", file=sys.stderr)
        return 2

    station_seen = {}  # station_raw -> count
    rows = []

    # Patterns like your grep("#\\s*LONGITUDE", header, value=TRUE)
    re_lon = re.compile(r"^\#\s*LONGITUDE", re.IGNORECASE)
    re_lat = re.compile(r"^\#\s*LATITUDE", re.IGNORECASE)
    re_country = re.compile(r"^\#\s*COUNTRY", re.IGNORECASE)
    re_years = re.compile(r"^\#\s*NUMBER OF YEARS", re.IGNORECASE)

    for path in files:
        fname = path.name
        print(f"🔍 Processing: {fname}")

        # station_raw = everything before first "-"
        station_raw = re.sub(r"-.*$", "", fname)

        # duplicates -> _v2, _v3...
        if station_raw in station_seen:
            station_seen[station_raw] += 1
            station_id = f"{station_raw}_v{station_seen[station_raw]}"
        else:
            station_seen[station_raw] = 1
            station_id = station_raw

        header = read_header(path, HEADER_LINES)
        if header is None:
            continue

        body = read_body_as_dataframe(path, skip_lines=HEADER_LINES)
        if body is None:
            continue

        lon = None
        lat = None
        country = None
        n_years = 0

        # pick first matching line (like lon_line[1] etc.)
        lon_line = next((h for h in header if re_lon.search(h)), None)
        lat_line = next((h for h in header if re_lat.search(h)), None)
        country_line = next((h for h in header if re_country.search(h)), None)
        years_line = next((h for h in header if re_years.search(h)), None)

        if lon_line:
            lon = extract_last_number(lon_line)
        if lat_line:
            lat = extract_last_number(lat_line)
        if country_line:
            country = extract_country(country_line)
        if years_line:
            n_years_val = extract_last_int(years_line)
            if n_years_val is not None:
                n_years = n_years_val

        nrows = int(body.shape[0])
        if (n_years is not None) and (n_years > 0):
            completeness = round(nrows / (n_years * HOURS_PER_YEAR), 4)
        else:
            completeness = None

        rows.append(
            {
                "filename": fname,
                "station_name": station_id,
                "n_years": n_years,
                "completeness": completeness,
                "longitude": lon,
                "latitude": lat,
                "country": country,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Index saved to: {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

