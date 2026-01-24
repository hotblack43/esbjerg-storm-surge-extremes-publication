# coastline_slope_from_ascii.py

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# --- CONFIG ---
etopo_path = "/media/pth/new_LaCie_Vol1/ETOPO1_Bed_g_gmt4.grd"
coastline_file = "DATA/europe_coastline_h.txt"
inner_radius_km = 3
outer_radius_km = 10
annulus_thickness_km = 1

# --- Load ETOPO ---
print("Loading ETOPO1 dataset...")
da = xr.open_dataset(etopo_path, engine="scipy")["z"]

# --- Load coastline coordinates from file ---
print(f"Reading coastline points from: {coastline_file}")
df = pd.read_csv(coastline_file, delim_whitespace=True)
longitudes = df["lon"].values
latitudes = df["lat"].values

print(f"Loaded {len(latitudes)} coastline points")

# --- Define slope computation function ---
def compute_slope(lat, lon):
    buffer_deg = 0.2
    try:
        local = da.sel(
            x=slice(lon - buffer_deg, lon + buffer_deg),
            y=slice(lat - buffer_deg, lat + buffer_deg)
        )
    except Exception:
        return None, None

    inner_points = []
    outer_points = []
    for y in local.y.values:
        for x in local.x.values:
            dist_km = geodesic((lat, lon), (y, x)).km
            elev = local.sel(x=x, y=y).values.item()
            if np.isnan(elev):
                continue
            if inner_radius_km - annulus_thickness_km <= dist_km < inner_radius_km + annulus_thickness_km and elev < 0:
                inner_points.append(elev)
            elif outer_radius_km - annulus_thickness_km <= dist_km < outer_radius_km + annulus_thickness_km and elev < 0:
                outer_points.append(elev)

    if not inner_points or not outer_points:
        return None, None
    d1 = np.median(inner_points)
    d2 = np.median(outer_points)
    slope = (d2 - d1) / (outer_radius_km - inner_radius_km)
    return slope, (d1, d2)

# --- Run slope calculation ---
results = []
for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
    slope, depths = compute_slope(lat, lon)
    print(f"{i+1:04d}: lon={lon:.4f}, lat={lat:.4f}, slope={slope}, depths={depths}")
    results.append({"longitude": lon, "latitude": lat, "slope": slope})

# --- Save to CSV ---
slope_df = pd.DataFrame(results)
slope_df.dropna(inplace=True)
slope_df.to_csv("coastline_slope_points.csv", index=False)
print("Saved to coastline_slope_points.csv")

# --- Plot ---
plt.figure(figsize=(8, 10))
sc = plt.scatter(slope_df["longitude"], slope_df["latitude"], c=slope_df["slope"], cmap="viridis", s=8)
plt.colorbar(sc, label="Seabed Slope (m/km)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Seabed Slopes Along Provided Coastline")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.savefig("coastline_seabed_slopes_from_file.png", dpi=300)
plt.show()
print("Plot saved as coastline_seabed_slopes_from_file.png")

