import xarray as xr
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ds = xr.open_dataset("hgt.2010.nc")
print(ds['hgt'])

nyc_500hpa = ds['hgt'].sel(lat=40.7128, lon=-74.0060, method='nearest').isel(level=6)
nyc_500hpa.plot()
plt.title("Geopotential Height at 500 mb for NYC (2010)")
plt.ylabel("Geopotential Height (m)")
plt.xlabel("Time (Days)")
plt.show()


ds_1979 = xr.open_dataset("hgt.1979.nc")
nyc_500mb_1979 = ds_1979['hgt'].sel(lat=40.7128, lon=-74.0060, method='nearest').isel(level=6)
nyc_500mb_1979.plot()
plt.title("Geopotential Height at 500 mb for NYC (1979)")
plt.ylabel("Geopotential Height (m)")
plt.xlabel("Time (Days)")
plt.show()


ds_2024 = xr.open_dataset("hgt.2024.nc")
nyc_500mb_2024 = ds_2024['hgt'].sel(lat=40.7128, lon=-74.0060, method='nearest').isel(level=6)
nyc_500mb_2024.plot()
plt.title("Geopotential Height at 500 mb for NYC (2024)")
plt.ylabel("Geopotential Height (m)")
plt.xlabel("Time (Days)")
plt.show()

# below helped me see what was going on with xarray
'''
import sys
print(sys.executable)
'''

file_names = [
    "hgt.1979.nc", "hgt.1980.nc", "hgt.1981.nc", "hgt.1982.nc", "hgt.1983.nc", 
    "hgt.1984.nc", "hgt.1985.nc","hgt.1986.nc", "hgt.1987.nc", "hgt.1988.nc", 
    "hgt.1989.nc", "hgt.1990.nc", "hgt.1991.nc", "hgt.1992.nc",
    "hgt.1993.nc", "hgt.1994.nc", "hgt.1995.nc", "hgt.1996.nc", "hgt.1997.nc", 
    "hgt.1998.nc", "hgt.1999.nc", "hgt.2000.nc", "hgt.2001.nc", "hgt.2002.nc", 
    "hgt.2003.nc", "hgt.2004.nc", "hgt.2005.nc", "hgt.2006.nc",
    "hgt.2007.nc", "hgt.2008.nc", "hgt.2009.nc", "hgt.2010.nc", 
    "hgt.2011.nc", "hgt.2012.nc", "hgt.2013.nc",
    "hgt.2014.nc", "hgt.2015.nc", "hgt.2016.nc", "hgt.2017.nc", 
    "hgt.2018.nc","hgt.2019.nc", "hgt.2020.nc",
    "hgt.2021.nc", "hgt.2022.nc","hgt.2023.nc", "hgt.2024.nc"
]

# NYC coordinates
nyc_lat = 40.7128
nyc_lon = -74.0060

for file in file_names:
    ds = xr.open_dataset(file)
    nyc_series = ds['hgt'].sel(lat=nyc_lat, lon=nyc_lon, method='nearest').isel(level=6)
    hgt_values = nyc_series.values

    peaks, _ = find_peaks(hgt_values, distance=10)
    troughs, _ = find_peaks(-hgt_values, distance=10)
    
    plt.figure(figsize=(10, 4))
    plt.plot(hgt_values, label="Geopotential Height (500 mb)")
    plt.scatter(peaks, hgt_values[peaks], color='red', label="Ridges (Peaks)")
    plt.scatter(troughs, hgt_values[troughs], color='blue', label="Troughs (Valleys)")
    plt.title(f"Geopotential Height at 500 mb for NYC ({file.split('.')[1]})")
    plt.xlabel("Time (Days)")
    plt.ylabel("Geopotential Height (m)")
    plt.legend()
    plt.show()

    ds.close()

# another way of looking at the "peaks" but I did not use this way
'''
from scipy.ndimage import gaussian_laplace

hgt_500mb_map = ds_2024['hgt'].isel(time=0, level=6)
laplacian = gaussian_laplace(hgt_500mb_map, sigma=1)

plt.figure(figsize=(10, 5))
plt.contourf(hgt_500mb_map.lon, hgt_500mb_map.lat, laplacian, cmap='RdBu_r')
plt.colorbar(label="Laplacian of Geopotential Height")
plt.title("Ridges (Red) & Troughs (Blue) in 500 mb Geopotential Height")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
'''
