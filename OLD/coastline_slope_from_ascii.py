# coastline_slope_from_ascii.py

import argparse
import sys
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# --- Fixed configuration ---
ETOPO_PATH = "/media/pth/new_LaCie_Vol1/ETOPO1_Bed_g_gmt4.grd"
COASTLINE_FILE = "DATA/europe_coastline_i.txt"
OUTBASE = "coastline_slope"
INNER_RADIUS_KM = 3.0
OUTER_RADIUS_KM = 10.0
ANNULUS_THICKNESS_KM = 1.0

# --- Parse only 4 CLI arguments ---
parser = argparse.ArgumentParser(description="Compute seabed slopes along coastline points from lon/lat list.")
parser.add_argument("--lon-min", type=float, required=True, help="Minimum longitude")
parser.add_argument("--lon-max", type=float, required=True, help="Maximum longitude")
parser.add_argument("--lat-min", type=float, required=True, help="Minimum latitude")
parser.add_argument("--lat-max", type=float, required=True, help="Maximum latitude")

# If no arguments, show help
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    print("\nExample:\n  python coastline_slope_from_ascii.py --lon-min 5 --lon-max 20 --lat-min 50 --lat-max 62")
    sys.exit(1)

args = parser.parse_args()

# --- Load ETOPO ---
print(f"Loading ETOPO1 dataset from: {ETOPO_PATH}")
da = xr.open_dataset(ETOPO_PATH, engine="scipy")["z"]

# --- Load lon/lat ASCII file ---
#print(f"Reading coastline points from: {COASTLINE_FILE}")
#df = pd.read_csv(COASTLINE_FILE, delim_whitespace=True)
# --- Load lon/lat ASCII file ---
print(f"Reading coastline points from: {COASTLINE_FILE}")
df = pd.read_csv(COASTLINE_FILE, delim_whitespace=True, header=None, names=["lon", "lat"])

# --- Filter to bounding box ---
df = df[(df["lon"] >= args.lon_min) & (df["lon"] <= args.lon_max) &
        (df["lat"] >= args.lat_min) & (df["lat"] <= args.lat_max)]
print(f"{len(df)} points inside bounding box")

longitudes = df["lon"].values
latitudes = df["lat"].values

# --- Define slope computation ---
def compute_slope(lat, lon):
    buffer_deg = 0.2
    try:
        local = da.sel(
            x=slice(lon - buffer_deg, lon + buffer_deg),
            y=slice(lat - buffer_deg, lat + buffer_deg)
        )
    except Exception:
        return None, None

    inner_pts = []
    outer_pts = []
    for y in local.y.values:
        for x in local.x.values:
            dist = geodesic((lat, lon), (y, x)).km
            elev = local.sel(x=x, y=y).values.item()
            if np.isnan(elev):
                continue
            if INNER_RADIUS_KM - ANNULUS_THICKNESS_KM <= dist < INNER_RADIUS_KM + ANNULUS_THICKNESS_KM and elev < 0:
                inner_pts.append(elev)
            elif OUTER_RADIUS_KM - ANNULUS_THICKNESS_KM <= dist < OUTER_RADIUS_KM + ANNULUS_THICKNESS_KM and elev < 0:
                outer_pts.append(elev)

    if not inner_pts or not outer_pts:
        return None, None
    d1 = np.median(inner_pts)
    d2 = np.median(outer_pts)
    slope = (d2 - d1) / (OUTER_RADIUS_KM - INNER_RADIUS_KM)
    return slope, (d1, d2)

# --- Compute slopes ---
results = []
for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
#   slope, depths = compute_slope(lat, lon)
#   print(f"{i+1:04d}: lon={lon:.4f}, lat={lat:.4f}, slope={slope}, depths={depths}")
#   results.append({"longitude": lon, "latitude": lat, "slope": slope})
    slope, depths = compute_slope(lat, lon)
    if slope is None or abs(slope) < 1e-6:
        continue  # Skip NaN and (almost) zero slopes
    print(f"{i+1:04d}: lon={lon:.4f}, lat={lat:.4f}, slope={slope}, depths={depths}")
    results.append({"longitude": lon, "latitude": lat, "slope": slope})


# --- Save output ---
df_out = pd.DataFrame(results)
df_out.dropna(inplace=True)
csv_path = f"{OUTBASE}_points.csv"
df_out.to_csv(csv_path, index=False)
print(f"Saved: {csv_path}")

# --- Plot result ---
plt.figure(figsize=(8, 10))
sc = plt.scatter(df_out["longitude"], df_out["latitude"], c=df_out["slope"],
                 cmap="viridis", s=5)
plt.colorbar(sc, label="Seabed Slope (m/km)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Seabed Slopes Along Coastline")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plot_path = f"{OUTBASE}_map.png"
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"Saved: {plot_path}")

