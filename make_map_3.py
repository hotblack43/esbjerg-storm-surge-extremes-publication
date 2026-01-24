import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pyreadr
import numpy as np

# ✅ Paths (unchanged except your own)
#etopo_path = "../DATA/ETOPO1_Bed_g_gmt4.grd"
#etopo_path = "/dmidata/projects/nckf/earthshine/ETOPO1_Bed_g_gmt4.grd"
etopo_path = "/media/pth/new_LaCie_Vol1/ETOPO1_Bed_g_gmt4.grd"
df_stations_rds = "/home/pth/WORKSHOP/ESBJERG2/PUBLICATION/OUTPUT/the_best_merged.rds"     # Red markers (symbol by MSLslope)
df_matched_rds  = "/home/pth/WORKSHOP/ESBJERG2/PUBLICATION/OUTPUT/the_goodies.rds"         # Yellow dots (blue on map)

# ✅ Check required files
for file_path in [etopo_path, df_stations_rds, df_matched_rds]:
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        exit()

# ✅ Read RDS files
red_df = next(iter(pyreadr.read_r(df_stations_rds).items()))[1]
yellow_df = next(iter(pyreadr.read_r(df_matched_rds).items()))[1]

# Ensure required columns exist
for df, nm in [(red_df, "red_df"), (yellow_df, "yellow_df")]:
    for col in ("longitude", "latitude"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in {nm}")

# ✅ Load ETOPO1 topography
etopo = xr.open_dataset(etopo_path)
lons = etopo["x"].values
lats = etopo["y"].values
elevation = etopo["z"].values

# 🔺 Helper to choose marker by MSLslope text
def slope_to_marker(sval):
    """
    Map MSLslope text to a matplotlib marker:
      'pos'  -> '^' (upward triangle)
      'neg'  -> 'v' (downward triangle)
      'zero' -> 'o' (filled circle)
    Any other / missing -> 'o'
    """
    if pd.isna(sval):
        return 'o'
    s = str(sval).strip().lower()
    if 'pos' in s:
        return '^'
    if 'neg' in s:
        return 'v'
    # treat 'zero' and everything else as 'o'
    return 'o'

# 🔑 Coordinate key for matching across dataframes (rounded for robustness)
def coord_key(lon, lat, ndp=4):
    try:
        return f"{round(float(lon), ndp)}_{round(float(lat), ndp)}"
    except Exception:
        return None

# 🧹 Build set of red TRIANGLE station keys (so we can remove those from yellow dots)
use_slope = "MSLslope" in red_df.columns
red_df["_marker"] = red_df["MSLslope"].apply(slope_to_marker) if use_slope else 'o'
red_df["_key"] = [coord_key(lon, lat) for lon, lat in zip(red_df["longitude"], red_df["latitude"])]

triangle_keys = set(red_df.loc[red_df["_marker"].isin(['^', 'v']), "_key"].dropna().tolist())

# Filter yellow: drop any yellow stations that match red triangle coords
yellow_df["_key"] = [coord_key(lon, lat) for lon, lat in zip(yellow_df["longitude"], yellow_df["latitude"])]
before_n = yellow_df.shape[0]
yellow_df_plot = yellow_df[~yellow_df["_key"].isin(triangle_keys)].copy()
after_n = yellow_df_plot.shape[0]
dropped_n = before_n - after_n
print(f"🧮 Yellow filtering: {dropped_n} blue dot(s) removed because a red triangle will be plotted on top.")

# ✅ Map plotting function
def plot_map(df_yellow, df_red, etopo, lon_min=None, lon_max=None, lat_min=None, lat_max=None, filename="map_new.png",
             satellite_view=False, dot_size_red=50, dot_size_yellow=25):
    if satellite_view:
        central_lon = (lon_min + lon_max) / 2 if (lon_min is not None and lon_max is not None) else 0
        central_lat = (lat_min + lat_max) / 2 if (lat_min is not None and lat_max is not None) else 0
        projection = ccrs.NearsidePerspective(
            central_longitude=central_lon,
            central_latitude=central_lat,
            satellite_height=0.7 * 6.371e6
        )
    else:
        if all(v is not None for v in [lon_min, lon_max, lat_min, lat_max]):
            projection = ccrs.PlateCarree()
        else:
            projection = ccrs.Mollweide()

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": projection})

    if not satellite_view and isinstance(projection, ccrs.PlateCarree):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    elif not satellite_view:
        ax.set_global()

    # downsample ETOPO for speed
    skip = 25
    lons_sub = lons[::skip]
    lats_sub = lats[::skip]
    elevation_sub = elevation[::skip, ::skip]

    ax.pcolormesh(
        lons_sub, lats_sub, elevation_sub,
        cmap=plt.cm.terrain, shading="auto", transform=ccrs.PlateCarree()
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, edgecolor="black")

    # 🟡 Yellow points (drawn as blue circles here) — use the pre-filtered df_yellow
    for _, row in df_yellow.iterrows():
        ax.scatter(
            row["longitude"], row["latitude"],
            color="blue", edgecolor="black", s=dot_size_yellow, marker='o',
            transform=ccrs.PlateCarree(), zorder=4
        )

    # 🔴 Red points: already have _marker; triangles slightly larger to cover any residual blue
    for _, row in df_red.iterrows():
        marker = row.get("_marker", 'o')
        size = dot_size_red * (1.3 if marker in ('^', 'v') else 1.0)
        ax.scatter(
            row["longitude"], row["latitude"],
            color="red", edgecolor="black", s=size, marker=marker,
            transform=ccrs.PlateCarree(), zorder=5
        )

    os.makedirs("FIGURES", exist_ok=True)
    plt.savefig(f"FIGURES/{filename}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: FIGURES/{filename}")

# ✅ Example maps (unchanged targets; now with blue-dot filtering & larger triangles)
print("🔴 'the_best' (red_df) coordinates preview:")
print(red_df[["longitude", "latitude"]].dropna().head())

print("\n🟡 'the_goodies' filtered (yellow_df_plot) coordinates preview:")
print(yellow_df_plot[["longitude", "latitude"]].dropna().head())

plot_map(yellow_df_plot, red_df, etopo, filename="map_global_new.png")
#plot_map(yellow_df_plot, red_df, etopo, lon_min=-130, lon_max=-60, lat_min=20, lat_max=55, filename="map_usa_new.png")
#plot_map(yellow_df_plot, red_df, etopo, lon_min=-10, lon_max=30, lat_min=35, lat_max=70, filename="map_europe_new.png")
#plot_map(yellow_df_plot, red_df, etopo, lon_min=125, lon_max=150, lat_min=30, lat_max=50, filename="map_japan_new.png")

plot_map(yellow_df_plot, red_df, etopo, lon_min=-130, lon_max=-60, lat_min=20, lat_max=55, filename="map_usa_satellite_new.png", satellite_view=True)
plot_map(yellow_df_plot, red_df, etopo, lon_min=-10, lon_max=30, lat_min=35, lat_max=70, filename="map_europe_satellite_new.png", satellite_view=True)
plot_map(yellow_df_plot, red_df, etopo, lon_min=-10, lon_max=30, lat_min=35, lat_max=70, filename="map_europe2_satellite_new.png", satellite_view=True, dot_size_red=40, dot_size_yellow=15)
plot_map(yellow_df_plot, red_df, etopo, lon_min=125, lon_max=150, lat_min=30, lat_max=50, filename="map_japan_satellite_new.png", satellite_view=True)
plot_map(yellow_df_plot, red_df, etopo, lon_min=105, lon_max=170, lat_min=-50, lat_max=-10, filename="map_australia_satellite_new.png", satellite_view=True)

