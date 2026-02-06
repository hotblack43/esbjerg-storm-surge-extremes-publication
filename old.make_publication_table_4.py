#!/usr/bin/env python3
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# Requires: pyreadr
#   pip install pyreadr
# or (uv):
#   uv pip install pyreadr
import pyreadr


def to_title_case_like_r(x: str) -> str:
    """
    Rough analogue of tools::toTitleCase().
    Close for typical station strings.
    """
    x = x.strip()
    if x == "":
        return x
    return " ".join([w[:1].upper() + w[1:].lower() if w else "" for w in x.split(" ")])


def format_value(v, digits: int) -> str:
    """
    Format numeric v to a string with 'digits' decimals.
    Strings unchanged; NA -> empty.
    """
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if pd.isna(v):
        return ""
    if isinstance(v, (np.integer, int)) and digits == 0:
        return str(int(v))
    if isinstance(v, (np.floating, float, np.integer, int)):
        if digits == 0:
            try:
                return str(int(round(float(v))))
            except Exception:
                return str(v)
        fmt = f"{{:.{digits}f}}"
        return fmt.format(float(v))
    return str(v)


def latex_escape(s: str) -> str:
    """
    Escape LaTeX special characters in text.
    Keeps it conservative and safe for filenames/station names.
    """
    if s is None:
        return ""
    s = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def df_to_longtable(
    df: pd.DataFrame,
    digits_by_col: dict,
    caption: str | None = None,
    label: str | None = None,
    align: str | None = None,
) -> str:
    """
    Emit a LaTeX longtable with per-column numeric formatting.
    Produces:
      \\begin{longtable}{...}
      \\caption{...}\\\\
      \\toprule
      header...
      \\midrule
      \\endfirsthead
      ...
      \\end{longtable}

    Uses booktabs rules. (You can remove booktabs if you prefer \\hline style.)
    """
    # format cells
    rows = []
    for _, r in df.iterrows():
        row_cells = []
        for col in df.columns:
            v = r[col]
            if col in digits_by_col:
                v_str = format_value(v, digits_by_col[col])
            else:
                v_str = "" if pd.isna(v) else str(v)
            row_cells.append(latex_escape(v_str))
        rows.append(row_cells)

    # header
    header = [latex_escape(c) for c in df.columns]

    # alignment
    if align is None:
        # default: left for text-like, right for numeric-like (heuristic)
        # To stay close to xtable "all left except numeric", we do:
        # first 2 columns (station_short, country) left; rest right.
        if len(df.columns) >= 2:
            align = "ll" + "r" * (len(df.columns) - 2)
        else:
            align = "l" * len(df.columns)

    lines = []
    lines.append(r"\begin{longtable}{" + align + "}")

    if caption is not None:
        cap_line = r"\caption{" + latex_escape(caption) + r"}"
        if label is not None:
            cap_line += r"\label{" + latex_escape(label) + r"}"
        lines.append(cap_line + r"\\")
    elif label is not None:
        # rare, but allow label without caption
        lines.append(r"\label{" + latex_escape(label) + r"}\\")

    # booktabs header + repeating header on new pages
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")

    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    # footer for continued pages
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(df.columns)) + r"}{r}{\emph{Continued on next page}} \\")
    lines.append(r"\endfoot")

    # last foot
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for row in rows:
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def main():
    # Match your R code's setwd():
    base = Path.home() / "./" 
    os.chdir(base)

    rds_path = Path("./OUTPUT/") / "the_best.rds"
    if not rds_path.exists():
        raise FileNotFoundError(f"Cannot find: {rds_path.resolve()}")

    result = pyreadr.read_r(str(rds_path))
    if len(result.keys()) == 0:
        raise RuntimeError(f"pyreadr read no objects from {rds_path}")
    df = next(iter(result.values()))
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame from RDS, got: {type(df)}")

    df2 = df.copy()

    # ---- Rounding (same as R) ----
    if "longitude" in df2.columns:
        df2["longitude"] = pd.to_numeric(df2["longitude"], errors="coerce").round(2)
    if "latitude" in df2.columns:
        df2["latitude"] = pd.to_numeric(df2["latitude"], errors="coerce").round(2)
    if "beta1" in df2.columns:
        df2["beta1"] = pd.to_numeric(df2["beta1"], errors="coerce").round(1)

    # drop AICbest/BICbest like R
    for col_drop in ["AICbest", "BICbest"]:
        if col_drop in df2.columns:
            df2 = df2.drop(columns=[col_drop])

    # sort by country like R
    if "country" in df2.columns:
        df2 = df2.sort_values(by=["country"], kind="mergesort").reset_index(drop=True)

    df = df2.copy()

    # drop country.1 like R
    if "country.1" in df.columns:
        df = df.drop(columns=["country.1"])

    # move longitude/latitude right after country (same as your R)
    cols = list(df.columns)
    if "country" in cols and "longitude" in cols and "latitude" in cols:
        cols_wo_ll = [c for c in cols if c not in ["longitude", "latitude"]]
        insert_after = cols_wo_ll.index("country") + 1
        new_order = cols_wo_ll[:insert_after] + ["longitude", "latitude"] + cols_wo_ll[insert_after:]
        df = df.loc[:, new_order]

    # handle filename_x -> filename
    if ("filename_x" in df.columns) and ("filename" not in df.columns):
        df = df.rename(columns={"filename_x": "filename"})

    if "filename" not in df.columns:
        raise KeyError("Expected a 'filename' column (or 'filename_x' that can be renamed).")

    # station_short (same regex logic as R)
    def make_station_short(fn: str) -> str:
        if fn is None or (isinstance(fn, float) and np.isnan(fn)):
            return ""
        s = str(fn)
        s = re.sub(r"_annual$", "", s)
        s = re.sub(r"-.*", "", s)
        s = s.replace("_", " ")
        s = to_title_case_like_r(s)
        return s

    df["station_short"] = df["filename"].map(make_station_short)

    # Column selections: EXACTLY as R
    small_cols = ["station_short", "country", "longitude", "latitude", "n_years", "LRT_p", "beta1", "delta_AIC"]
    large_cols = ["station_short", "country", "longitude", "latitude", "n_years", "LRT_p", "beta1", "delta_AIC", "delta_BIC"]

    for c in small_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column for regular table: {c}")
    for c in large_cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column for sideways table: {c}")

    df_small = df.loc[:, small_cols].copy()
    df_large = df.loc[:, large_cols].copy()

    # Digits: keep as per your xtable() calls
    digits_large = {
        "station_short": 0,
        "country": 0,
        "longitude": 2,
        "latitude": 2,
        "n_years": 0,
        "LRT_p": 5,
        "beta1": 1,
        "delta_AIC": 3,
        "delta_BIC": 3,
    }
    digits_small = {
        "station_short": 0,
        "country": 0,
        "longitude": 2,
        "latitude": 2,
        "n_years": 0,
        "LRT_p": 5,
        "beta1": 1,
        "delta_AIC": 1,
    }

    # ---- Produce LONGTABLE outputs ----
    # Regular: longtable
    lt_small = df_to_longtable(
        df_small,
        digits_small,
        caption=None,
        label=None,
        align="llrrrrrr",  # 2 text cols + 6 numeric cols
    )

    # Sideways/multi-page: landscape + longtable
    # (This is the practical way; sidewaystable float won't break.)
    lt_large = df_to_longtable(
        df_large,
        digits_large,
        caption=None,
        label=None,
        align="llrrrrrrr",  # 2 text cols + 7 numeric cols
    )

    out_side = Path("OUTPUT") / "sideways_table.tex"
    out_reg = Path("OUTPUT") / "regular_table.tex"
    out_side.parent.mkdir(parents=True, exist_ok=True)

    out_reg.write_text(
        "\n".join([
            r"\documentclass{article}",
            r"\usepackage{longtable}",
            r"\usepackage{booktabs}",
            r"\begin{document}",
            lt_small.rstrip(),
            r"\end{document}",
            "",
        ]),
        encoding="utf-8",
    )

    out_side.write_text(
        "\n".join([
            r"\documentclass{article}",
            r"\usepackage{pdflscape}",
            r"\usepackage{longtable}",
            r"\usepackage{booktabs}",
            r"\begin{document}",
            r"\begin{landscape}",
            lt_large.rstrip(),
            r"\end{landscape}",
            r"\end{document}",
            "",
        ]),
        encoding="utf-8",
    )

    # ---- Copy figures to FIGURES/THEBEST (same as R) ----
    thebest_dir = Path("FIGURES") / "THEBEST"
    thebest_dir.mkdir(parents=True, exist_ok=True)

    # delete contents
    for p in thebest_dir.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

    suffix = "_3panel_bootstrap_beta1_baseR.png"
    copied = 0
    missing = 0

    for fn in df["filename"].tolist():
        src = Path("FIGURES") / f"{fn}{suffix}"
        if src.exists():
            shutil.copy2(src, thebest_dir / src.name)
            copied += 1
        else:
            missing += 1

    print("Generated outputs in FIGURES/THEBEST and OUTPUT/")
    print(f"Wrote: {out_reg}")
    print(f"Wrote: {out_side}")
    print(f"Copied {copied} figure(s) to {thebest_dir} (missing {missing}).")


if __name__ == "__main__":
    main()

