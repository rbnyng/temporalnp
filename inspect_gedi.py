import gedidb as gdb
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

provider = gdb.GEDIProvider(
    storage_type='s3',
    s3_bucket="dog.gedidb.gedi-l2-l4-v002",
    url="https://s3.gfz-potsdam.de"
)

roi_geom = box(-73, 2, -72, 3)
roi = gpd.GeoDataFrame({'geometry': [roi_geom]}, crs="EPSG:4326")

gedi_data = provider.get_data(
    variables=["agbd"],
    query_type="bounding_box",
    geometry=roi,
    start_time="2022-01-01",
    end_time="2022-12-31",
    return_type='xarray'
)

print(f"Retrieved {len(gedi_data.shot_number)} GEDI shots")
print(f"Variables: {list(gedi_data.data_vars)}")

agbd = gedi_data['agbd'].values
lat = gedi_data['latitude'].values
lon = gedi_data['longitude'].values

mask = ~np.isnan(agbd)
agbd = agbd[mask]
lat = lat[mask]
lon = lon[mask]

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(lon, lat, c=agbd, cmap='YlGn', s=1, vmin=0, vmax=200)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('GEDI Aboveground Biomass Density (Mg/ha)')
plt.colorbar(scatter, ax=ax, label='AGBD (Mg/ha)')
plt.tight_layout()

fig.savefig("gedi_agbd_plot.png", dpi=300)
print("Figure saved as 'gedi_agbd_plot.png'")

# statistics
print(f"Mean biomass: {agbd.mean():.2f} Mg/ha")
print(f"Median biomass: {np.median(agbd):.2f} Mg/ha")
print(f"Max biomass: {agbd.max():.2f} Mg/ha")

percentiles = [10, 25, 50, 75, 90, 95, 99]
agbd_percentiles = np.percentile(agbd, percentiles)
for p, value in zip(percentiles, agbd_percentiles):
    print(f"{p}th percentile: {value:.2f} Mg/ha")


fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(agbd, bins=500, alpha=0.7)
ax.axvline(500, color='red', linestyle='--', linewidth=2, label='Cutoff = 500 Mg/ha')

ax.set_xscale('log')

ax.set_xlabel('AGBD (Mg/ha)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of GEDI AGBD Values')
ax.legend()

plt.tight_layout()
fig.savefig("gedi_agbd_distribution_logscale.png", dpi=300)
print("Distribution plot saved as 'gedi_agbd_distribution.png'")

cutoff = 500
agbd_clip = agbd[agbd <= cutoff]

kde = gaussian_kde(agbd_clip)
x_vals = np.linspace(0, cutoff, 2000)
kde_vals = kde(x_vals)

fig, ax = plt.subplots(figsize=(12, 4))

ax.fill_between(x_vals, kde_vals, color='red', alpha=0.5)
ax.plot(x_vals, kde_vals, color='red', linewidth=2)

ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

ax.set_xlim(0, cutoff)
ax.set_ylim(0, max(kde_vals) * 1.05)

ax.set_xlabel("AGBD (Mg/ha)", fontsize=10, fontweight='bold')
ax.set_ylabel("Density", fontsize=10, fontweight='bold')

ax.set_title("AGBD KDE", fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig("gedi_agbd_kde.png", dpi=300)