#!/usr/bin/env python3
"""
plot_gesla4.py

Usage:
  uv run plot_gesla4.py <station_filename>

Example:
  uv run plot_gesla4.py zeelandbrug_noord-zeelbnd-nld-rws

Assumptions:
- Files are in ~/GESLA4/DATA/
- Plain text files with NO extension
- Exactly 41 header lines
- Data rows:
    YYYY/MM/DD HH:MM:SS   <value> <flag1> <flag2>
- Bad data flag value is -99.9999 (omit those rows before plotting)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


HEADER_LINES = 41
DATA_DIR = Path.home() / "GESLA4" / "DATA"
BAD_VALUE = -99.9999


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot raw GESLA4 tide-gauge series (datetime vs value column).")
    p.add_argument("name", help="GESLA4 station filename (no extension), located in ~/GESLA4/DATA/")
    p.add_argument("--max-points", type=int, default=0,
                   help="Optional downsample cap for plotting (0 = no cap). Keeps evenly spaced points.")
    p.add_argument("--show-gaps-hours", type=float, default=6.0,
                   help="Insert NaNs when time gap exceeds this many hours, to visually break the line.")
    p.add_argument("--bad-value", type=float, default=BAD_VALUE,
                   help="Bad data flag value to omit before plotting (default: -99.9999).")
    return p.parse_args()


def try_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def read_gesla_file(path: Path, header_lines: int, bad_value: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      times: numpy array of python datetimes (dtype=object)
      values: numpy float array
    """
    times: list[datetime] = []
    vals: list[float] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(header_lines):
            _ = f.readline()

        for lineno, line in enumerate(f, start=header_lines + 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            parts = s.split()
            if len(parts) < 3:
                continue

            date_s, time_s, val_s = parts[0], parts[1], parts[2]

            try:
                t = datetime.strptime(f"{date_s} {time_s}", "%Y/%m/%d %H:%M:%S")
            except Exception:
                continue

            v = try_float(val_s)

            # Omit bad flag rows and NaNs
            if not np.isfinite(v):
                continue
            if v == bad_value:
                continue

            times.append(t)
            vals.append(v)

    if len(times) == 0:
        raise RuntimeError(f"No valid data rows parsed from {path} after filtering (bad_value={bad_value}).")

    return np.array(times, dtype=object), np.array(vals, dtype=float)


def insert_gap_nans(times: np.ndarray, vals: np.ndarray, gap_hours: float) -> tuple[np.ndarray, np.ndarray]:
    if len(times) < 2:
        return times, vals

    out_t: list[datetime] = [times[0]]
    out_v: list[float] = [vals[0]]

    gap_seconds = gap_hours * 3600.0

    for i in range(1, len(times)):
        dt = (times[i] - times[i - 1]).total_seconds()
        if dt > gap_seconds:
            out_t.append(times[i - 1])
            out_v.append(float("nan"))
        out_t.append(times[i])
        out_v.append(vals[i])

    return np.array(out_t, dtype=object), np.array(out_v, dtype=float)


def downsample_even(times: np.ndarray, vals: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or len(times) <= max_points:
        return times, vals
    idx = np.linspace(0, len(times) - 1, max_points).astype(int)
    return times[idx], vals[idx]


def main() -> int:
    args = parse_args()

    path = DATA_DIR / args.name
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        print("Tip: pass the bare filename (no extension).", file=sys.stderr)
        return 2

    times, vals = read_gesla_file(path, header_lines=HEADER_LINES, bad_value=args.bad_value)

    times2, vals2 = insert_gap_nans(times, vals, gap_hours=args.show_gaps_hours)
    times3, vals3 = downsample_even(times2, vals2, max_points=args.max_points)

    plt.figure(figsize=(18, 6))
    plt.plot(times3, vals3, linewidth=0.6)
    plt.title(f"GESLA4 raw tide-gauge series: {args.name} (bad_value omitted: {args.bad_value})")
    plt.xlabel("Time")
    plt.ylabel("Observed level (3rd column, raw units)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

