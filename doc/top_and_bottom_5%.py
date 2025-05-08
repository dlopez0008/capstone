import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


nyc_lat = 40.7128
nyc_lon = 74.0060

file_list = [f"hgt.{year}.nc" for year in range(1980, 2022)]
ts_list = []
for file in file_list:
    ds = xr.open_dataset(file)
    nyc_500hpa = ds['hgt'].sel(lat=nyc_lat, lon=nyc_lon, level=500.0, method='nearest')
    ts_list.append(nyc_500hpa)


hgt_500_nyc = xr.concat(ts_list, dim="time")
df = hgt_500_nyc.to_dataframe().reset_index().set_index("time")
df = df.resample("W").mean()
upper_threshold = np.percentile(df["hgt"], 95)  
lower_threshold = np.percentile(df["hgt"], 5)
extreme_highs = df[df["hgt"] >= upper_threshold]
extreme_lows = df[df["hgt"] <= lower_threshold]

plt.plot(df.index, df["hgt"], label="500 hPa Geopotential Height", color='blue', alpha=0.6)
plt.scatter(extreme_highs.index, extreme_highs["hgt"], color='red', label="Extreme Ridges (Top 5%)", alpha=0.8)
plt.scatter(extreme_lows.index, extreme_lows["hgt"], color='black', label="Extreme Troughs (Bottom 5%)", alpha=0.8)
plt.axhline(upper_threshold, color='red', linestyle='dashed', alpha=0.7, label="95th Percentile")
plt.axhline(lower_threshold, color='black', linestyle='dashed', alpha=0.7, label="5th Percentile")
plt.xlabel("Year")
plt.ylabel("Geopotential Height (m)")
plt.title("Extreme Events in 500 hPa Geopotential Height Over NYC")
plt.legend()
plt.grid()
plt.show()
