import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

nyc_lat = 40.7128
nyc_lon = 74.0060
file_list = [f"hgt.{year}.nc" for year in range(1980, 2022)]
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

y = df["hgt"].values  
peaks, _ = find_peaks(y, prominence=10)

df_peaks = df.iloc[peaks].copy()
df_peaks["peak_intensity"] = df_peaks["hgt"]
df_peaks = df_peaks[["time", "hgt", "peak_intensity", "year", "season"]]
peak_counts_seasonal = df_peaks.groupby(["year", "season"]).size().unstack()

plt.figure(figsize=(12, 6))
peak_counts_seasonal.rolling(window=5, min_periods=1).mean().plot(marker="o", linestyle="-", ax=plt.gca())
plt.title("Seasonal Peak Frequency Trends (500 hPa Geopotential Height)")
plt.xlabel("Year")
plt.ylabel("Number of Peaks")
plt.legend(title="Season")
plt.grid()
plt.yticks(np.arange(10, 21, 2)) 
plt.ylim(10, 21)     
plt.show()

peak_intensity_trends = df_peaks.groupby("year")["peak_intensity"].mean()
peak_intensity_smoothed = peak_intensity_trends.rolling(window=5, min_periods=1).mean()

X = np.array(peak_intensity_trends.index).reshape(-1, 1)
y = np.array(peak_intensity_trends.values).reshape(-1, 1)

model = LinearRegression().fit(X, y)
trend_line = model.predict(X)

plt.figure(figsize=(12, 6))
plt.plot(peak_intensity_trends.index, peak_intensity_trends, marker="o", linestyle="-", label="Observed")
plt.plot(peak_intensity_trends.index, peak_intensity_smoothed, linestyle="--", label="Moving Average (5-year)")
plt.plot(peak_intensity_trends.index, trend_line, linestyle="-.", label="Linear Trend", color="red")
plt.title("Peak Intensity Trends (500 hPa Geopotential Height)")
plt.xlabel("Year")
plt.ylabel("Mean Peak Intensity (m)")
plt.legend()
plt.grid()
plt.show()
