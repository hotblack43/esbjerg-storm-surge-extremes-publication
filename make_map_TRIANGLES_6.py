#!/usr/bin/env python3
"""
make_map_5.py  (FROM SCRATCH, simplified, PLUS non-satellite closeups)

Inputs (relative to --root):
- DATA/ETOPO1_Bed_g_gmt4.grd
- OUTPUT/the_best.rds     (red layer, marker from numeric slope_MSL)
- OUTPUT/the_goodies.rds  (blue layer, circles)

Marker logic (red):
- slope_MSL > +eps  => '^'
- slope_MSL < -eps  => 'v'
- otherwise or NA   => 'o'

Blue filtering:
- remove any blue dot whose rounded (lon,lat) matches a red TRIANGLE position

Outputs (to FIGURES/):
- map_global_new.png
- map_usa_satellite_new.png
- map_europe_satellite_new.png
- map_europe2_satellite_new.png
- map_europe_closeup_new.png        (non-satellite, PlateCarree)
- map_japan_satellite_new.png
- map_japan_closeup_new.png         (non-satellite, PlateCarree)
- map_australia_satellite_new.png

Diagnostics (to OUTPUT/):
- dropped_red_lonlat_rows.csv
- dropped_blue_lonlat_rows.csv

Run:
  uv run python make_map_5.py
Optional:
  uv run python make_map_5.py --root ~/WORKSHOP/esbjerg-storm-surge-extremes-publication --eps 1e-6
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pyreadr


def die(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def read_first_rds_df(rds_path: Path) -> pd.DataFrame:
    obj = pyreadr.read_r(str(rds_path))
    if len(obj.keys()) == 0:
        die(f"No objects found inside RDS: {rds_path}")
    df = next(iter(obj.values()))
    if not isinstance(df, pd.DataFrame):
        die(f"Expected a DataFrame inside {rds_path}, got {type(df)}")
    return df


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def ensure_filename_column(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """
    Canonical station id column is 'filename'.
    Your the_best.rds uses filename_x; we convert that to 'filename'.
    If both exist, keep 'filename' and drop 'filename_x'.
    """
    df = normalise_columns(df)

    if "filename" not in df.columns and "filename_x" in df.columns:
        df = df.rename(columns={"filename_x": "filename"})

    if "filename" in df.columns and "filename_x" in df.columns:
        df = df.drop(columns=["filename_x"])

    if "filename" not in df.columns:
        die(f"{context}: missing 'filename' (or 'filename_x'). Columns: {list(df.columns)}")

    df["filename"] = df["filename"].astype(str)
    return df


def clean_lonlat(df: pd.DataFrame, nm: str, save_csv_path: Path | None = None) -> pd.DataFrame:
    """
    Coerce longitude/latitude to numeric and drop rows with missing/non-finite lon/lat.
    """
    df = df.copy()

    for col in ("longitude", "latitude"):
        if col not in df.columns:
            die(f"{nm}: needs column '{col}'. Columns: {list(df.columns)}")

    df["_lon_raw"] = df["longitude"]
    df["_lat_raw"] = df["latitude"]

    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")

    bad = (
        df["longitude"].isna()
        | df["latitude"].isna()
        | ~np.isfinite(df["longitude"].to_numpy())
        | ~np.isfinite(df["latitude"].to_numpy())
    )

    dropped = df.loc[bad].copy()
    if dropped.shape[0] > 0:
        cols = [c for c in ["filename", "longitude", "latitude", "_lon_raw", "_lat_raw", "slope_MSL"] if c in dropped.columns]
        print(f"\n🧹 {nm}: dropped {dropped.shape[0]} row(s) with missing/non-finite lon/lat")
        print(dropped[cols].to_string(index=False))
        if save_csv_path is not None:
            save_csv_path.parent.mkdir(parents=True, exist_ok=True)
            dropped.to_csv(save_csv_path, index=False)
            print(f"📝 Wrote dropped rows to: {save_csv_path}\n")

    df_ok = df.loc[~bad].copy()
    df_ok.drop(columns=["_lon_raw", "_lat_raw"], inplace=True, errors="ignore")
    return df_ok


def marker_from_numeric_slope(slope_value, eps: float) -> str:
    """
    slope_MSL numeric -> marker
      > +eps : '^'
      < -eps : 'v'
      otherwise (including NA) : 'o'
    """
    if pd.isna(slope_value):
        return "o"
    try:
        x = float(slope_value)
    except Exception:
        return "o"
    if x > eps:
        return "^"
    if x < -eps:
        return "v"
    return "o"


def coord_key(lon, lat, ndp=4):
    try:
        return f"{round(float(lon), ndp)}_{round(float(lat), ndp)}"
    except Exception:
        return None


def plot_map(
    df_blue: pd.DataFrame,
    df_red: pd.DataFrame,
    lons,
    lats,
    elevation,
    out_dir: Path,
    filename: str,
    lon_min=None,
    lon_max=None,
    lat_min=None,
    lat_max=None,
    satellite_view: bool = False,
    dot_size_red: float = 50,
    dot_size_blue: float = 25,
):
    if satellite_view:
        central_lon = (lon_min + lon_max) / 2 if (lon_min is not None and lon_max is not None) else 0
        central_lat = (lat_min + lat_max) / 2 if (lat_min is not None and lat_max is not None) else 0
        projection = ccrs.NearsidePerspective(
            central_longitude=central_lon,
            central_latitude=central_lat,
            satellite_height=0.7 * 6.371e6,
        )
    else:
        projection = ccrs.PlateCarree() if all(v is not None for v in [lon_min, lon_max, lat_min, lat_max]) else ccrs.Mollweide()

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": projection})

    if (not satellite_view) and isinstance(projection, ccrs.PlateCarree):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    elif not satellite_view:
        ax.set_global()

    skip = 25
    ax.pcolormesh(
        lons[::skip],
        lats[::skip],
        elevation[::skip, ::skip],
        cmap=plt.cm.terrain,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1,
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="black", zorder=2)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", zorder=2)

    # Blue dots
    for _, row in df_blue.iterrows():
        ax.scatter(
            row["longitude"], row["latitude"],
            color="blue", edgecolor="black",
            s=dot_size_blue, marker="o",
            transform=ccrs.PlateCarree(), zorder=4,
        )

    # Red markers; triangles slightly larger
    for _, row in df_red.iterrows():
        marker = row["_marker"]
        size = dot_size_red * (1.3 if marker in ("^", "v") else 1.0)
        ax.scatter(
            row["longitude"], row["latitude"],
            color="red", edgecolor="black",
            s=size, marker=marker,
            transform=ccrs.PlateCarree(), zorder=5,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    outpath = out_dir / filename
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {outpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tide-gauge station maps with triangles from slope_MSL.")
    parser.add_argument(
        "--root",
        default=str(Path(os.path.expanduser("./")).resolve()),
        help="Project root directory (default: ./)",
    )
    parser.add_argument("--etopo", default="/home/pth/WORKSHOP/ESBJERG2/DATA/ETOPO1_Bed_g_gmt4.grd", help="ETOPO grid path relative to root")
    parser.add_argument("--best", default="OUTPUT/the_best.rds", help="the_best.rds relative to root")
    parser.add_argument("--goodies", default="OUTPUT/the_goodies.rds", help="the_goodies.rds relative to root")
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0,
        help="Tolerance for treating slope_MSL as zero (default 0.0). Example: 1e-6",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    etopo_path = (root / args.etopo).resolve()
    best_path = (root / args.best).resolve()
    goodies_path = (root / args.goodies).resolve()

    figures_dir = root / "FIGURES"
    output_dir = root / "OUTPUT"

    dropped_red_csv = output_dir / "dropped_red_lonlat_rows.csv"
    dropped_blue_csv = output_dir / "dropped_blue_lonlat_rows.csv"

    for p in [etopo_path, best_path, goodies_path]:
        if not p.exists():
            die(f"Missing file: {p}")

    best_df = ensure_filename_column(read_first_rds_df(best_path), "best_df (the_best.rds)")
    goodies_df = ensure_filename_column(read_first_rds_df(goodies_path), "goodies_df (the_goodies.rds)")

    if "slope_MSL" not in best_df.columns:
        die(f"the_best.rds has no 'slope_MSL' column. Columns: {list(best_df.columns)}")

    red_df = clean_lonlat(best_df, "red_df (best)", save_csv_path=dropped_red_csv)
    blue_df = clean_lonlat(goodies_df, "blue_df (goodies)", save_csv_path=dropped_blue_csv)

    red_df["_marker"] = red_df["slope_MSL"].apply(lambda v: marker_from_numeric_slope(v, args.eps))
    n_up = int((red_df["_marker"] == "^").sum())
    n_dn = int((red_df["_marker"] == "v").sum())
    n_o = int((red_df["_marker"] == "o").sum())
    print(f"🔺 Marker summary from slope_MSL (eps={args.eps}): up={n_up}, down={n_dn}, circle={n_o}, total={red_df.shape[0]}")

    # Remove blue dots that sit under red TRIANGLES
    red_df["_key"] = [coord_key(lon, lat) for lon, lat in zip(red_df["longitude"], red_df["latitude"])]
    tri_keys = set(red_df.loc[red_df["_marker"].isin(["^", "v"]), "_key"].dropna().tolist())

    blue_df["_key"] = [coord_key(lon, lat) for lon, lat in zip(blue_df["longitude"], blue_df["latitude"])]
    before = blue_df.shape[0]
    blue_df_plot = blue_df[~blue_df["_key"].isin(tri_keys)].copy()
    after = blue_df_plot.shape[0]
    print(f"🧮 Blue filtering: {before - after} blue dot(s) removed under red triangles.")

    etopo = xr.open_dataset(etopo_path)
    lons = etopo["x"].values
    lats = etopo["y"].values
    elevation = etopo["z"].values

    # Maps: global
    plot_map(blue_df_plot, red_df, lons, lats, elevation, figures_dir, "map_global_new.png")

    # Satellite maps (same as your old + our newer ones)
    plot_map(
        blue_df_plot, red_df, lons, lats, elevation, figures_dir,
        "map_usa_satellite_new.png",
        lon_min=-130, lon_max=-60, lat_min=20, lat_max=55,
        satellite_view=True,
    )
    plot_map(
        blue_df_plot, red_df, lons, lats, elevation, figures_dir,
        "map_europe_satellite_new.png",
        lon_min=-10, lon_max=30, lat_min=35, lat_max=70,
        satellite_view=True,
    )
    plot_map(
        blue_df_plot, red_df, lons, lats, elevation, figures_dir,
        "map_europe2_satellite_new.png",
        lon_min=-10, lon_max=30, lat_min=35, lat_max=70,
        satellite_view=True,
        dot_size_red=40,
        dot_size_blue=15,
    )
    plot_map(
        blue_df_plot, red_df, lons, lats, elevation, figures_dir,
        "map_japan_satellite_new.png",
        lon_min=139, lon_max=160, lat_min=32, lat_max=48,
        satellite_view=True,
    )
    plot_map(
        blue_df_plot, red_df, lons, lats, elevation, figures_dir,
        "map_australia_satellite_new.png",
        lon_min=105, lon_max=170, lat_min=-50, lat_max=-10,
        satellite_view=True,
    )

    # Non-satellite closeups (from your old script)
    plot_map(
        blue_df_plot, red_df, lons, lats, elevation, figures_dir,
        "map_europe_closeup_new.png",
        lon_min=-13, lon_max=45, lat_min=34, lat_max=72,
        satellite_view=False,
    )
    plot_map(
        blue_df_plot, red_df, lons, lats, elevation, figures_dir,
        "map_japan_closeup_new.png",
        lon_min=128, lon_max=147, lat_min=29, lat_max=46,
        satellite_view=False,
    )


if __name__ == "__main__":
    main()

