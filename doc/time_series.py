import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

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

# weekly frequency to improve performance
'''I want to change this to daily but it takes super long for me to see results when I make it daily'''
df = df.resample("W").mean()

# for seasonality
fourier = CalendarFourier(freq="A", order=3)
dp = DeterministicProcess(
    index=df.index,  
    constant=True,  
    order=1,    
    seasonal=True,  
    additional_terms=[fourier],  
    drop=True,  
)

X = dp.in_sample()
y = df["hgt"]

# SARIMA model with optimized parameters
model = SARIMAX(y, exog=X, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), enforce_stationarity=False)
result = model.fit()

# Forecast next 5 years (weekly data)
forecast_index = pd.date_range(start=df.index[-1], periods=5 * 52, freq="W")
X_forecast = dp.out_of_sample(steps=5 * 52)

forecast = result.get_forecast(steps=5 * 52, exog=X_forecast)
forecast_mean = forecast.predicted_mean

plt.figure(figsize=(12, 6))
plt.plot(df.index, df["hgt"], label="Observed 500 hPa Geopotential Height", color="b", alpha=0.6)
plt.plot(forecast_index, forecast_mean, label="Forecast (DHR)", color="r", linestyle="dashed")
plt.xlabel("Year")
plt.ylabel("Geopotential Height (m)")
plt.title("Dynamic Harmonic Regression Forecast: 500 hPa Height Over NYC (1979-2021)")
plt.legend()
plt.grid()
plt.show()


print(forecast.summary_frame())



upper_threshold = np.percentile(df["hgt"], 95)
lower_threshold = np.percentile(df["hgt"], 5)
extreme_highs = df[df["hgt"] >= upper_threshold]
extreme_lows = df[df["hgt"] <= lower_threshold]

print(f"Extreme Highs (Top 5% threshold): {upper_threshold:.2f} m")
print(f"Extreme Lows (Bottom 5% threshold): {lower_threshold:.2f} m")
print(f"Total Extreme High Events: {len(extreme_highs)}")
print(f"Total Extreme Low Events: {len(extreme_lows)}")

plt.figure(figsize=(12,6))


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




plt.figure(figsize=(10, 5))
plt.scatter(extreme_counts.index, extreme_counts.values, color='blue', label="Extreme Events")
plt.plot(extreme_counts.index, np.poly1d(np.polyfit(extreme_counts.index, extreme_counts.values, 1))(extreme_counts.index), 
         color='red', linestyle="dashed", label="Trend Line")

plt.xlabel("Year")
plt.ylabel("Number of Extreme Events")
plt.title("Trend of Extreme 500 hPa Height Events Over NYC")
plt.legend()
plt.grid(True)
plt.show()
