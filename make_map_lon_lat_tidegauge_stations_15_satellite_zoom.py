import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import pyreadr

# ✅ Updated paths
#etopo_path = "../DATA/ETOPO1_Bed_g_gmt4.grd"
#etopo_path = "/dmidata/projects/nckf/earthshine/ETOPO1_Bed_g_gmt4.grd"
etopo_path = "/home/pth/pCloudDrive/SAFETYHERE/WORKSHOP/ESBJERG2/DATA/ETOPO1_Bed_g_gmt4.grd"
df_stations_rds = "./OUTPUT/the_best.rds"     # Red dots
df_matched_rds = "./OUTPUT/the_goodies.rds"   # Yellow dots

# ✅ Check required files
for file_path in [etopo_path, df_stations_rds, df_matched_rds]:
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        exit()

# ✅ Read RDS files
red_df = next(iter(pyreadr.read_r(df_stations_rds).items()))[1]
yellow_df = next(iter(pyreadr.read_r(df_matched_rds).items()))[1]

# ✅ Print summaries of coordinates
print("🔴 'the_best' (red_df) coordinates:")
print(red_df[["longitude", "latitude"]].dropna())

print("\n🟡 'the_goodies' (yellow_df) coordinates:")
print(yellow_df[["longitude", "latitude"]].dropna())


# ✅ Load ETOPO1 topography
etopo = xr.open_dataset(etopo_path)
lons = etopo["x"].values
lats = etopo["y"].values
elevation = etopo["z"].values

# ✅ Map plotting function
def plot_map(df_yellow, df_red, etopo, lon_min=None, lon_max=None, lat_min=None, lat_max=None, filename="map.png",
             satellite_view=False, dot_size_red=50, dot_size_yellow=25):
    if satellite_view:
        central_lon = (lon_min + lon_max) / 2 if lon_min and lon_max else 0
        central_lat = (lat_min + lat_max) / 2 if lat_min and lat_max else 0
        projection = ccrs.NearsidePerspective(
            central_longitude=central_lon,
            central_latitude=central_lat,
            satellite_height=0.7 * 6.371e6
        )
    else:
        projection = ccrs.PlateCarree() if all(v is not None for v in [lon_min, lon_max, lat_min, lat_max]) else ccrs.Mollweide()

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": projection})

    if not satellite_view and projection == ccrs.PlateCarree():
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    elif not satellite_view:
        ax.set_global()

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

    for _, row in df_yellow.iterrows():
        ax.scatter(row["longitude"], row["latitude"], color="blue", edgecolor="black", s=dot_size_yellow,
                   transform=ccrs.PlateCarree(), zorder=4)

    for _, row in df_red.iterrows():
        ax.scatter(row["longitude"], row["latitude"], color="red", edgecolor="black", s=dot_size_red,
                   transform=ccrs.PlateCarree(), zorder=4)

    os.makedirs("FIGURES", exist_ok=True)
    plt.savefig(f"FIGURES/{filename}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: FIGURES/{filename}")

# ✅ Example maps
plot_map(yellow_df, red_df, etopo, filename="map_global.png")
#plot_map(yellow_df, red_df, etopo, lon_min=-130, lon_max=-60, lat_min=20, lat_max=55, filename="map_usa.png")
#plot_map(yellow_df, red_df, etopo, lon_min=-10, lon_max=30, lat_min=35, lat_max=70, filename="map_europe.png")
#plot_map(yellow_df, red_df, etopo, lon_min=125, lon_max=150, lat_min=30, lat_max=50, filename="map_japan.png")

# ✅ Satellite-style regional maps
plot_map(yellow_df, red_df, etopo, lon_min=-130, lon_max=-60, lat_min=20, lat_max=55, filename="map_usa_satellite.png", satellite_view=True)
plot_map(yellow_df, red_df, etopo, lon_min=-10, lon_max=30, lat_min=35, lat_max=70, filename="map_europe_satellite.png", satellite_view=True)
plot_map(yellow_df, red_df, etopo, lon_min=-10, lon_max=30, lat_min=35, lat_max=70, filename="map_europe2_satellite.png", satellite_view=True , dot_size_red=20, dot_size_yellow=15)
plot_map(yellow_df, red_df, etopo, 
         lon_min=-13, lon_max=45, lat_min=34, lat_max=72,
         filename="map_europe_closeup.png",
         satellite_view=False)
plot_map(yellow_df, red_df, etopo, lon_min=139, lon_max=160, lat_min=32, lat_max=48, filename="map_japan_satellite.png", satellite_view=True)
plot_map(yellow_df, red_df, etopo,
         lon_min=128, lon_max=147, lat_min=29, lat_max=46,
         filename="map_japan_closeup.png",
         satellite_view=False)
plot_map(yellow_df, red_df, etopo, lon_min=105, lon_max=170, lat_min=-50, lat_max=-10, filename="map_australia_satellite.png", satellite_view=True)

