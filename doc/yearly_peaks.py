import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression


nyc_lat = 40.7128
nyc_lon = 74.0060

file_list = [f"hgt.{year}.nc" for year in range(1979, 2022)]
ts_list = []

for file in file_list:
    ds = xr.open_dataset(file)
    nyc_500hpa = ds['hgt'].sel(lat=nyc_lat, lon=nyc_lon, level=500.0, method='nearest')
    ts_list.append(nyc_500hpa)


hgt_500_nyc = xr.concat(ts_list, dim="time")
df = hgt_500_nyc.to_dataframe().reset_index().set_index("time")


y = df["hgt"].values  
time = df.index  
peaks, _ = find_peaks(y, height=np.mean(y)) 
peak_dates = time[peaks]
df_peaks = pd.DataFrame({"date": peak_dates})
df_peaks["year"] = df_peaks["date"].dt.year

peak_counts = df_peaks.groupby("year").size().reset_index(name="peak_count")

X = peak_counts["year"].values.reshape(-1, 1)  
y = peak_counts["peak_count"].values

model = LinearRegression()
model.fit(X, y)
trend_line = model.predict(X)  

peak_counts["rolling_avg"] = peak_counts["peak_count"].rolling(window=5, center=True).mean()


plt.figure(figsize=(10, 5))
plt.plot(peak_counts["year"], peak_counts["peak_count"], marker="o", linestyle="-", color="r", label="Annual Peak Count")
plt.plot(peak_counts["year"], trend_line, linestyle="--", color="b", label="Linear Trend")
plt.plot(peak_counts["year"], peak_counts["rolling_avg"], linestyle="-", color="g", label="5-Year Rolling Average")
plt.xlabel("Year")
plt.ylabel("Number of Peaks")
plt.title("Trend in Annual Count of Geopotential Height Peaks (500 hPa, NYC)")
plt.legend()
plt.grid()
plt.show()

slope = model.coef_[0]
print(f"Linear trend slope: {slope:.2f} peaks per year")
