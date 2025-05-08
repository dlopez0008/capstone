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
# This was taken from Reference 4(Youtube Channel)

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
