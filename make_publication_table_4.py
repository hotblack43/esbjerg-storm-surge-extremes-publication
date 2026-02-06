#!/usr/bin/env python3
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr


def to_title_case_like_r(x: str) -> str:
    x = str(x).strip()
    if x == "":
        return x
    # close-enough analogue of tools::toTitleCase for your filenames
    return " ".join([w[:1].upper() + w[1:].lower() if w else "" for w in x.split(" ")])


def format_value(v, digits: int) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    if isinstance(v, (np.integer, int)) and digits == 0:
        return str(int(v))
    if isinstance(v, (np.floating, float, np.integer, int)):
        if digits == 0:
            try:
                return str(int(round(float(v))))
            except Exception:
                return str(v)
        return f"{float(v):.{digits}f}"
    return str(v)


def latex_escape(s: str) -> str:
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
    return "".join(replacements.get(ch, ch) for ch in s)


def df_to_longtable(
    df: pd.DataFrame,
    digits_by_col: dict,
    caption: str | None = None,
    label: str | None = None,
    align: str | None = None,
) -> str:
    rows = []
    for _, r in df.iterrows():
        row_cells = []
        for col in df.columns:
            v = r[col]
            if col in digits_by_col:
                v_str = format_value(v, digits_by_col[col])
            else:
                v_str = "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
            row_cells.append(latex_escape(v_str))
        rows.append(row_cells)

    header = [latex_escape(c) for c in df.columns]

    if align is None:
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
        lines.append(r"\label{" + latex_escape(label) + r"}\\")

    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")

    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(df.columns)) + r"}{r}{\emph{Continued on next page}} \\")
    lines.append(r"\endfoot")

    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for row in rows:
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\end{longtable}")
    return "\n".join(lines) + "\n"


def _series_has_any_real_text(s: pd.Series) -> bool:
    if s is None:
        return False
    ss = s.astype("string")
    ss = ss.fillna("").str.strip()
    return bool((ss != "").any())


def _as_clean_text(x) -> str:
    if x is None:
        return ""
    # pyreadr sometimes yields bytes-like objects
    if isinstance(x, (bytes, bytearray)):
        try:
            x = x.decode("utf-8", errors="replace")
        except Exception:
            x = str(x)
    # robust NA detection
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def make_station_short_from_filename(fn) -> str:
    s = _as_clean_text(fn)
    if s == "":
        return ""
    s = re.sub(r"_annual$", "", s)
    s = re.sub(r"-.*", "", s)
    s = s.replace("_", " ")
    return to_title_case_like_r(s)


def main():
    # YOU requested OUTPUT/ be relative to where you run the code
    base = Path.cwd()

    rds_path = base / "OUTPUT" / "the_best.rds"
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
    for col, digs in [("longitude", 2), ("latitude", 2)]:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce").round(digs)
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

    # ---- Robust station identifier selection ----
    # Prefer already-computed station_short, else station_name/station_id, else filename/filename_x
    station_source = None

    if "station_short" in df.columns and _series_has_any_real_text(df["station_short"]):
        station_source = "station_short"
        df["station_short"] = df["station_short"].map(_as_clean_text)

    elif "station_name" in df.columns and _series_has_any_real_text(df["station_name"]):
        station_source = "station_name"
        df["station_short"] = df["station_name"].map(_as_clean_text)

    elif "station_id" in df.columns and _series_has_any_real_text(df["station_id"]):
        station_source = "station_id"
        df["station_short"] = df["station_id"].map(_as_clean_text)

    else:
        filename_col = None
        if "filename" in df.columns and _series_has_any_real_text(df["filename"]):
            filename_col = "filename"
        elif "filename_x" in df.columns and _series_has_any_real_text(df["filename_x"]):
            filename_col = "filename_x"

        if filename_col is None:
            raise KeyError(
                "Could not find any usable station name column. "
                "Tried: station_short, station_name, station_id, filename, filename_x."
            )

        station_source = filename_col
        df["station_short"] = df[filename_col].map(make_station_short_from_filename)

    # If you want to SEE what it picked, leave this print in:
    print(f"Station column source used: {station_source}")

    # Column selections: EXACTLY as your earlier Python/R intent
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

    lt_small = df_to_longtable(df_small, digits_small, caption=None, label=None, align="llrrrrrr")
    lt_large = df_to_longtable(df_large, digits_large, caption=None, label=None, align="llrrrrrrr")

    out_dir = base / "OUTPUT"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_reg = out_dir / "regular_table.tex"
    out_side = out_dir / "sideways_table.tex"

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

    # ---- Copy figures to FIGURES/THEBEST (same as before) ----
    figures_dir = base / "FIGURES"
    thebest_dir = figures_dir / "THEBEST"
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

    # Use whichever filename column actually exists (for figure names)
    fig_name_col = "filename" if "filename" in df.columns else ("filename_x" if "filename_x" in df.columns else None)
    if fig_name_col is not None:
        for fn in df[fig_name_col].map(_as_clean_text).tolist():
            if fn == "":
                missing += 1
                continue
            src = figures_dir / f"{fn}{suffix}"
            if src.exists():
                shutil.copy2(src, thebest_dir / src.name)
                copied += 1
            else:
                missing += 1

    print(f"Wrote: {out_reg.resolve()}")
    print(f"Wrote: {out_side.resolve()}")
    print(f"Copied {copied} figure(s) to {thebest_dir.resolve()} (missing {missing}).")


if __name__ == "__main__":
    main()

