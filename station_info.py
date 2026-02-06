#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python translation of station_info.Rmd
Purpose: scan GESLA headers and extract longitude/latitude per file.

Behaviour preserved:
- Working directory: ./
- Data directory scanned: ~/GESLA4/DATA/   (same comment: change if needed)
- Reads first 41 header lines per file
- Extracts the first numeric value on the LONGITUDE/LATITUDE lines
- Skips files missing either coordinate
- Writes RDS to: DATA/station_metadata.rds (relative to WORKDIR)

Requires:
- pandas
- pyreadr (for writing .rds)
"""

import os
import re
from pathlib import Path

import pandas as pd


WORKDIR = Path("./").expanduser()
DATA_DIR = Path("~/GESLA4/DATA/").expanduser()  # <-- change this if needed
# DATA_DIR = Path("/dmidata/projects/nckf/earthshine/GESLA4/").expanduser()  # <-- change this if needed
OUT_RDS = WORKDIR / "DATA/station_metadata.rds"
HEADER_NLINES = 41


_NUM_RE = re.compile(r"-?\d+\.\d+|-?\d+")


def read_first_n_lines(path: Path, n: int) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = []
        for _ in range(n):
            line = f.readline()
            if line == "":
                break
            lines.append(line.rstrip("\n"))
    return lines


def extract_first_number(line: str) -> float | None:
    m = _NUM_RE.search(line)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def write_rds(df: pd.DataFrame, outpath: Path) -> None:
    try:
        import pyreadr  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Python cannot write .rds without 'pyreadr'.\n"
            "Install it with:\n"
            "  uv pip install pyreadr\n"
            "or\n"
            "  pip install pyreadr\n"
        ) from e

    outpath.parent.mkdir(parents=True, exist_ok=True)
    pyreadr.write_rds(str(outpath), df)


def main() -> None:
    os.chdir(WORKDIR)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    file_list = sorted([p for p in DATA_DIR.iterdir() if p.is_file()])

    rows = []
    for f in file_list:
        header_lines = read_first_n_lines(f, HEADER_NLINES)

        lon_line = next((ln for ln in header_lines if "LONGITUDE" in ln.upper()), None)
        lat_line = next((ln for ln in header_lines if "LATITUDE" in ln.upper()), None)

        lon = extract_first_number(lon_line) if lon_line else None
        lat = extract_first_number(lat_line) if lat_line else None

        if (lon is not None) and (lat is not None):
            rows.append(
                {
                    "filename": f.name,
                    "longitude": lon,
                    "latitude": lat,
                }
            )
        # else: skip (matches your R code)

    df_coords = pd.DataFrame(rows, columns=["filename", "longitude", "latitude"])

    write_rds(df_coords, OUT_RDS)
    print(f"Wrote {len(df_coords)} rows to: {OUT_RDS}")


if __name__ == "__main__":
    main()
