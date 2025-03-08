import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


nyc_lat = 40.7128
nyc_lon = -74.0060
file_list = [f"hgt.{year}.nc" for year in range(1979, 2022)]
ts_list = []
for file in file_list:
    ds = xr.open_dataset(file)
    nyc_500hpa = ds['hgt'].sel(lat=nyc_lat, lon=nyc_lon, level=500.0, method='nearest')
    ts_list.append(nyc_500hpa)
hgt_500_nyc = xr.concat(ts_list, dim="time")
df = hgt_500_nyc.to_dataframe().reset_index()
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
season_map = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}
df["season"] = df["month"].map(season_map)

peaks, _ = find_peaks(df["hgt"].values, prominence=10)
troughs, _ = find_peaks(-df["hgt"].values, prominence=10) 
df_peaks = df.iloc[peaks].copy()
df_troughs = df.iloc[troughs].copy()

peak_counts_seasonal = df_peaks.groupby(["year", "season"]).size().unstack()
trough_counts_seasonal = df_troughs.groupby(["year", "season"]).size().unstack()

# using 5 year avg for smoothing
plt.figure(figsize=(12, 6))
peak_counts_seasonal.rolling(window=5, min_periods=1).mean().plot(marker="o", linestyle="-", ax=plt.gca())
plt.title("Seasonal Ridge Frequency Trends (500 hPa Geopotential Height)")
plt.xlabel("Year")
plt.ylabel("Number of Ridges (Peaks)")
plt.legend(title="Season")
plt.grid()
plt.show()


plt.figure(figsize=(12, 6))
trough_counts_seasonal.rolling(window=5, min_periods=1).mean().plot(marker="o", linestyle="-", ax=plt.gca())
plt.title("Seasonal Trough Frequency Trends (500 hPa Geopotential Height)")
plt.xlabel("Year")
plt.ylabel("Number of Troughs (Local Minima)")
plt.legend(title="Season")
plt.grid()
plt.show()
