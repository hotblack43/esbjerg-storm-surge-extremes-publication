#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find GESLA stations whose header indicates GAUGE TYPE Coastal.
Optionally require a minimum NUMBER OF YEARS (default 55).

Input:
  - A text file with one GESLA file path per line (blank lines ok; lines starting with # ignored)

Outputs:
  - COASTAL_all.txt      : full paths (exactly as in input) where GAUGE TYPE contains "Coastal"
  - COASTAL_55Y.txt      : subset with NUMBER OF YEARS >= min_years
  - NOT_COASTAL.txt      : full paths for remaining or unreadable files
  - QC_summary.csv       : CSV summary for all input files
"""

import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple


def read_header_lines(path: str, max_lines: int = 300) -> List[str]:
    """
    Read "header" lines from a GESLA-like file robustly.
    We stop when we hit a data line that looks like it begins with a year (e.g. 1900)
    or after max_lines.
    """
    lines: List[str] = []
    # GESLA files are plain text; encoding can vary slightly. Use a forgiving decode.
    with open(path, "rb") as f:
        for i, raw in enumerate(f):
            if i >= max_lines:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            lines.append(line)

            # Heuristic: a data line often begins with YYYY or YYYY-MM or similar.
            # If a line starts with 4 digits and later has more digits, we assume data begins.
            if re.match(r"^\s*\d{4}(\D|$)", line):
                # If this was actually a header line with a year in text, this could stop early.
                # But GESLA headers typically have labelled key-value rows, so this works well.
                break

    return lines


def parse_header_kv(lines: List[str]) -> Dict[str, str]:
    """
    Parse key-value pairs from header lines.
    Supports patterns like:
      'GAUGE TYPE Coastal'
      'NUMBER OF YEARS 68'
      'STATION NAME Some Name'
      '# GAUGE TYPE Coastal'
    Returns dict with normalised keys.
    """
    kv: Dict[str, str] = {}

    for line in lines:
        s = line.strip()

        # Remove leading comment marker if present
        if s.startswith("#"):
            s = s.lstrip("#").strip()

        # Many GESLA header lines look like "KEYWORD value..."
        # We'll match a known set of keys, but also allow generic "ALLCAPS words" keys.
        # We keep it conservative to avoid mis-parsing.
        known_keys = [
            "GAUGE TYPE",
            "NUMBER OF YEARS",
            "STATION NAME",
            "COUNTRY",
            "LATITUDE",
            "LONGITUDE",
            "START YEAR",
            "END YEAR",
        ]

        matched = False
        for k in known_keys:
            if s.upper().startswith(k):
                val = s[len(k):].strip()
                kv[k] = val
                matched = True
                break

        if matched:
            continue

        # Generic fallback: lines that look like "SOME KEY  value"
        # Only accept keys that are mostly uppercase letters/spaces and at least 2 words.
        m = re.match(r"^([A-Z][A-Z ]{3,}?)\s+(.+)$", s)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            # Avoid absurdly short keys
            if len(key.split()) >= 2:
                kv.setdefault(key, val)

    return kv


def coerce_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    m = re.search(r"(-?\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def is_coastal(gauge_type_value: Optional[str]) -> bool:
    if gauge_type_value is None:
        return False
    return "COASTAL" in gauge_type_value.strip().upper()


def process_file(path: str) -> Tuple[Dict[str, Optional[str]], Optional[str]]:
    """
    Returns (record, error_message).
    record contains parsed fields; error_message is None if ok.
    """
    rec: Dict[str, Optional[str]] = {
        "input_path": path,
        "exists": None,
        "gauge_type": None,
        "is_coastal": None,
        "number_of_years": None,
        "station_name": None,
    }

    if not os.path.exists(path):
        rec["exists"] = "no"
        return rec, "file_not_found"

    rec["exists"] = "yes"

    try:
        header_lines = read_header_lines(path)
        kv = parse_header_kv(header_lines)

        gt = kv.get("GAUGE TYPE")
        ny = kv.get("NUMBER OF YEARS")
        sn = kv.get("STATION NAME")

        rec["gauge_type"] = gt
        rec["station_name"] = sn

        coastal = is_coastal(gt)
        rec["is_coastal"] = "yes" if coastal else "no"

        ny_i = coerce_int(ny)
        rec["number_of_years"] = str(ny_i) if ny_i is not None else None

        return rec, None
    except Exception as e:
        return rec, f"exception: {type(e).__name__}: {e}"


def read_file_list(list_path: str) -> List[str]:
    paths: List[str] = []
    with open(list_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            paths.append(s)
    return paths


def write_list(out_path: str, lines: List[str]) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--list",
        default="DATA/GESLA_FILENAMES_HOME2.txt",
        help="Text file with one GESLA filename per line",
    )
    ap.add_argument(
        "--min-years",
        type=int,
        default=55,
        help="Minimum NUMBER OF YEARS for the 'COASTAL_XXY' output",
    )
    ap.add_argument(
        "--outdir",
        default=".",
        help="Output directory for result files",
    )
    args = ap.parse_args()

    in_list = args.list
    outdir = args.outdir
    min_years = args.min_years

    os.makedirs(outdir, exist_ok=True)

    paths = read_file_list(in_list)

    coastal_all: List[str] = []
    coastal_long: List[str] = []
    not_coastal: List[str] = []

    summary_rows: List[Dict[str, Optional[str]]] = []

    for p in paths:
        rec, err = process_file(p)
        if err is not None:
            # treat unreadable/missing as NOT_COASTAL but keep in summary
            not_coastal.append(p)
            rec["error"] = err
            summary_rows.append(rec)
            continue

        rec["error"] = None
        summary_rows.append(rec)

        if rec.get("is_coastal") == "yes":
            coastal_all.append(p)
            ny = coerce_int(rec.get("number_of_years"))
            if ny is not None and ny >= min_years:
                coastal_long.append(p)
        else:
            not_coastal.append(p)

    # Write outputs (paths EXACTLY as in the input file)
    out_all = os.path.join(outdir, "COASTAL_all.txt")
    out_long = os.path.join(outdir, f"COASTAL_{min_years}Y.txt")
    out_not = os.path.join(outdir, "NOT_COASTAL.txt")
    out_csv = os.path.join(outdir, "QC_summary.csv")

    write_list(out_all, coastal_all)
    write_list(out_long, coastal_long)
    write_list(out_not, not_coastal)

    # CSV summary
    fieldnames = ["input_path", "exists", "gauge_type", "is_coastal", "number_of_years", "station_name", "error"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            # ensure all fields exist
            row = {k: r.get(k) for k in fieldnames}
            w.writerow(row)

    print(f"Read {len(paths)} paths from: {in_list}")
    print(f"Coastal (all): {len(coastal_all)} -> {out_all}")
    print(f"Coastal (>= {min_years} years): {len(coastal_long)} -> {out_long}")
    print(f"Not coastal / unreadable: {len(not_coastal)} -> {out_not}")
    print(f"Summary CSV: {out_csv}")


if __name__ == "__main__":
    main()

