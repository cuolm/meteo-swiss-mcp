import asyncio
import json
import logging
import logging.config
import os
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
from dotenv import load_dotenv
from earthkit.data import settings
from geopy.geocoders import Nominatim
from rasterio.crs import CRS
from xarray import DataArray

from meteodatalab import ogd_api
from meteodatalab.operators import regrid, wind


# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Configure logger
def _setup_logger():
    try:
        config_path = Path(__file__).parent.parent / "log_config.json"
        with open(config_path, "rt") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
        logging.getLogger(__name__).info("Logger successfully configured")
    except Exception as e:
        logging.basicConfig(level=logging.ERROR)
        logging.getLogger(__name__).error(f"Failed to configure logger: {e}", exc_info=True)

_setup_logger()
logger = logging.getLogger(__name__)

# Configure caching
# EarthKit cache
CACHE_DIR = Path(__file__).parent.parent / "cache"
EARTHKIT_CACHE_DIR = CACHE_DIR / "EarthKitCache"
settings.set({
    "cache-policy": "user",  # "user" = caches data persistently on disk in the specified directory ("temporary" = not persistent, only RAM)
    "user-cache-directory": EARTHKIT_CACHE_DIR, 
})
# Nominatim geocodecache atomic update
GEOCODE_CACHE_FILE = CACHE_DIR / "nominatim_geocode_cache.json"
def _load_geocode_cache() -> dict:
    """
    Try to load the cache from disk or return an empty one.
    """
    try:
        with open(GEOCODE_CACHE_FILE, "r") as f:
            cache_raw = json.load(f)
            return {k: tuple(v) for k, v in cache_raw.items()}
    except Exception:
        logger.warning("Failed to load geocode cache, using empty.")
        return {}

def _save_geocode_cache(cache):
    """
    Atomically write the cache avoiding corruption on failure by first writing to a temporary file and then replacing the original. 
    """
    tmp_file = GEOCODE_CACHE_FILE.with_suffix('.tmp')
    with open(tmp_file, "w") as f:
        json.dump(cache, f)
    tmp_file.replace(GEOCODE_CACHE_FILE)

# Configure the number of grid points for the bounding box around the location
DistanceOffset = namedtuple('DistanceOffset', ['x', 'y'])
NUM_GRID_POINTS_X = 7
NUM_GRID_POINTS_Y = 7


class MeteoSwissPredictions:
    def __init__(self):
        # With a resolution of 1km we span an area of (7-1)*1km = 6km in x direction and (7-1)*1km = 6km in y direction
        self.distance_offset_degree = self._calc_distance_offset_in_degree(num_grid_points_x=NUM_GRID_POINTS_X, num_grid_points_y=NUM_GRID_POINTS_Y, res_x_km=1, res_y_km=1) 

        # Initielize nominatim geocoder
        self.today_midnight_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        self.nominatim_user_agent = os.environ.get("NOMINATIM_USER_AGENT")
        if not self.nominatim_user_agent:
            logger.error("Nominatim user agent must be specified via the environment variable NOMINATIM_USER_AGENT.")
            raise ValueError(
                "Nominatim user agent must be specified via the environment variable NOMINATIM_USER_AGENT."
            )
        # Initialize geocode cache to reduce geocoding API requests
        self.geocode_cache = _load_geocode_cache()

    def _calc_distance_offset_in_degree(
            self,
            num_grid_points_x: int,
            num_grid_points_y: int,
            res_x_km: float, 
            res_y_km: float
        ) -> Tuple[float, float]:
        """
        Calculate the distance offset from the middle point (e.g. for Zurich: lat=47.3744 lon=8.5410) in degrees for the given grid size and resolution.

        Parameters:
            num_grid_points_x (int): Number of points in the x (longitude) direction.
            num_grid_points_y (int): Number of points in the y (latitude) direction.
            res_x_km (float): Resolution of grid in kilometers along x.
            res_y_km (float): Resolution of grid in kilometers along y.

        Returns:
            Tuple[float, float]: Distance offsets in degrees (x_degree_offset, y_degree_offset).

        See: https://github.com/MeteoSwiss/opendata-nwp-demos/blob/2ddfa03f4e3cfe3e57b0bf1696b62d9c5b61c2ad//computing_num_grid_points_x_num_grid_points_y.md
        """
        if(num_grid_points_x < 1 or num_grid_points_y < 1):
            logger.error("Number of grid points must be at least 1")
            raise ValueError("Number of grid points must be at least 1")
        if(res_x_km < 1 or res_y_km < 1):
            logger.error("Resolution must be at least 1")
            raise ValueError("Resolution must be at least 1")

        # In central europe at 46° latitude, 1° in lattitude direction corresponds to 111.2 km in that direction.
        # 1° of longitude corresponds to a smaller distance at 46° latitude due to the Earth's curvature and can be approximated by 111.2km * cos(46°)
        km_per_degree_x = 111.2 * np.cos(np.radians(46))
        km_per_degree_y = 111.2

        res_x_degree = res_x_km/km_per_degree_x
        res_y_degree = res_y_km/km_per_degree_y

        distance_offset_x_degree = (num_grid_points_x-1)/2 * res_x_degree
        distance_offset_y_degree = (num_grid_points_y-1)/2 * res_y_degree

        return DistanceOffset(x=distance_offset_x_degree, y=distance_offset_y_degree)

    async def _fetch_variable_over_period(
        self,
        location: str,
        variable: str,
        lead_time_start: int,
        lead_time_end: int,
        num_grid_points_x: int = NUM_GRID_POINTS_X,
        num_grid_points_y: int = NUM_GRID_POINTS_Y,
    ) -> DataArray:
        """
        Fetch meteorological data for a given location and variable over a forecast lead time period (UTC).

        Parameters:
            location (str): Name of the location.
            variable (str): Parameter name (e.g., "TOT_PREC", "DURSUN").
            lead_time_start (int): Start hour of the lead time period (0-120).
            lead_time_end (int): End hour of the lead time period (1-120, lead_time_end ≥ lead_time_start).
            num_grid_points_x (int): Grid dimension in the x (longitude) direction.
            num_grid_points_y (int): Grid dimension in the y (latitude) direction.

        Returns:
            xarray.DataArray: Regridded raw data array covering the specified bounding box and lead time period.
        """
        if not (0 <= lead_time_start <= 120):
            logger.error("lead_time_start must be between 0 and 120")
            raise ValueError("lead_time_start must be between 0 and 120")
        if not (1 <= lead_time_end <= 120):
            logger.error("lead_time_end must be between 1 and 120")
            raise ValueError("lead_time_end must be between 1 and 120")
        if lead_time_end <= lead_time_start:
            logger.error("lead_time_end must be greater to lead_time_start")
            raise ValueError("lead_time_end must be greater to lead_time_start")
        
        # Calculate the grid bounding box around the location
        location_lat, location_lon = await self._get_latlon_for_location(location)
        x_min_deg = location_lon - self.distance_offset_degree.x
        x_max_deg = location_lon + self.distance_offset_degree.x
        y_min_deg = location_lat - self.distance_offset_degree.y
        y_max_deg = location_lat + self.distance_offset_degree.y
        
        # Define lead times as timedelta objects to specify forecast hours range
        lead_times = [timedelta(hours=lead_time_start), timedelta(hours=lead_time_end)]

        # Build the API request for given variable and lead time period, using ensemble perturbations (perturbed=True)
        req = ogd_api.Request(
            collection="ogd-forecasting-icon-ch2",
            variable=variable,
            ref_time=self.today_midnight_utc,
            lead_time=lead_times,
            perturbed=True,       
        )

        # Fetch raw weather data array from MeteoSwiss API
        try:
            raw_data = await asyncio.to_thread(ogd_api.get_from_ogd, req)
        except Exception as e:
            logger.exception(f"Error fetching data from MeteoSwiss API: {e}")
            raise RuntimeError(f"Error fetching data from MeteoSwiss API: {e}")

        # Regrid raw data from ICON grid onto regular lon/lat grid
        target_grid = regrid.RegularGrid(CRS.from_epsg(4326), num_grid_points_x, num_grid_points_y, x_min_deg, x_max_deg, y_min_deg, y_max_deg)
        remapped_data = await asyncio.to_thread(regrid.iconremap, raw_data, target_grid)

        return remapped_data
    
    async def _fetch_variable_at_lead_time(
        self,
        location: str,
        variable: str,
        lead_time: int,
        num_grid_points_x: int = NUM_GRID_POINTS_X,
        num_grid_points_y: int = NUM_GRID_POINTS_Y,
    ) -> DataArray:
        """
        Fetch meteorological data for a given location and variable for a specific
        forecast lead time (UTC).

        Parameters:
            location (str): Name of the location.
            variable (str): Parameter name (e.g., "T_2M", "U_10M").
            lead_time (int): Forecast lead time in hours (1-120).
            num_grid_points_x (int): Grid dimension in the x (longitude) direction.
            num_grid_points_y (int): Grid dimension in the y (latitude) direction.

        Returns:
            xarray.DataArray: Regridded raw data array for the specified bounding box and lead time.
        """
        if not (0 <= lead_time <= 120):
            logger.error("lead_time must be between 0 and 120")
            raise ValueError("lead_time must be between 0 and 120")
        
        # Calculate the grid bounding box around the location
        location_lat, location_lon = await self._get_latlon_for_location(location)
        x_min_deg = location_lon - self.distance_offset_degree.x
        x_max_deg = location_lon + self.distance_offset_degree.x
        y_min_deg = location_lat - self.distance_offset_degree.y
        y_max_deg = location_lat + self.distance_offset_degree.y

        # Build the API request for the variable at the specific lead time, using ensemble perturbations (perturbed=True)
        req = ogd_api.Request(
            collection="ogd-forecasting-icon-ch2",
            variable=variable,
            ref_time=self.today_midnight_utc,
            lead_time=timedelta(hours=lead_time),
            perturbed=True,
        )

        # Fetch raw data array from MeteoSwiss API
        try:
            raw_data = await asyncio.to_thread(ogd_api.get_from_ogd, req)
        except Exception as e:
            logger.exception(f"Error fetching data from MeteoSwiss API: {e}")
            raise RuntimeError(f"Error fetching data from MeteoSwiss API: {e}")

        # Regrid raw data from ICON grid onto regular lon/lat grid
        target_grid = regrid.RegularGrid(CRS.from_epsg(4326), num_grid_points_x, num_grid_points_y, x_min_deg, x_max_deg, y_min_deg, y_max_deg)
        remapped_data = await asyncio.to_thread(regrid.iconremap, raw_data, target_grid)

        return remapped_data
    
    async def _get_latlon_for_location(self, location_name: str) -> Tuple[float, float]:
        if location_name in self.geocode_cache:
            lat, lon = self.geocode_cache[location_name]
            return lat, lon        

        geolocator = Nominatim(user_agent=self.nominatim_user_agent)
        try:
            location = await asyncio.to_thread(geolocator.geocode, location_name + ", Switzerland")
        except Exception as e:
            logger.exception(f"Error during gecoding for location {location_name} {e}")
            raise RuntimeError(f"Error during geocoding for location {location_name} {e}")
        if location is None:
            logger.error(f"Could not find location: {location_name}")
            raise ValueError(f"Could not find location: {location_name}")

        # Update geocode cache and write updated cache to disk atomically
        self.geocode_cache[location_name] = (location.latitude, location.longitude)
        _save_geocode_cache(self.geocode_cache)
            
        return (location.latitude, location.longitude)

    async def total_rainfall_for_location(self, location: str, lead_time_start: int, lead_time_end: int) -> float:
        precip_raw = await self._fetch_variable_over_period(
            location=location,
            variable="TOT_PREC",
            lead_time_start=lead_time_start,
            lead_time_end=lead_time_end,
        )
        
        # Difference of the variable values over the lead time period (lead_time_end - lead_time_start) 
        precip_diff = precip_raw.diff(dim="lead_time")

        # Average over ensemble dimensions
        precip_eps_mean = precip_diff.mean(dim="eps")
        # Average over spatial dimensions (y and x) 
        precipation_yx_mean = precip_eps_mean.mean(dim=["y", "x"])
        precip_mm = float(precipation_yx_mean.item())
        return precip_mm 
    
    async def sunshine_hours_for_location(self, location: str, lead_time_start: int, lead_time_end: int) -> float:
        sunshine_raw = await self._fetch_variable_over_period(
            location=location,
            variable="DURSUN",
            lead_time_start=lead_time_start,
            lead_time_end=lead_time_end,
        )

        # Difference of the variable values over the lead time period (lead_time_end - lead_time_start)
        sunshine_diff = sunshine_raw.diff(dim="lead_time")

        # Average over ensemble dimensions
        sunshine_eps_mean = sunshine_diff.mean(dim="eps")
        # Average over spatial dimensions (y and x)
        sunshine_yx_mean= sunshine_eps_mean.mean(dim=["y", "x"])
        sunshine_hours = float(sunshine_yx_mean.item())/ 3600  
        return sunshine_hours

    async def wind_speed_for_location(self, location:str, lead_time: int) -> float:
        wind_speed_raw_u = await self._fetch_variable_at_lead_time(
            location=location,
            variable="U_10M",
            lead_time=lead_time
        )

        wind_speed_raw_v = await self._fetch_variable_at_lead_time(
            location=location,
            variable="V_10M",
            lead_time=lead_time
        )

        wind_speed_raw = np.sqrt(wind_speed_raw_u**2 + wind_speed_raw_v**2)

         # Average over ensemble dimension
        wind_speed_eps_mean = wind_speed_raw.mean(dim="eps")
        # Average over spatial dimensions (y and x)
        wind_speed_yx_mean = wind_speed_eps_mean.mean(dim=["y", "x"])
        wind_speed_mps = float(wind_speed_yx_mean.item())

        return wind_speed_mps

    async def temp_for_location(self, location: str, lead_time: int) -> float:
        temp_raw = await self._fetch_variable_at_lead_time(
            location=location,
            variable="TMAX_2M",
            lead_time=lead_time,
        )
        # Average over ensemble dimensions 
        temp_eps_mean =temp_raw.mean(dim="eps")
        # Avarage over spatial dimensions (y and x)
        temp_yx_mean = temp_eps_mean.mean(dim=["y", "x"])
        temp_K = float(temp_yx_mean.item())
        temp_C = temp_K - 273.15
        return temp_C
    
    async def pressure_msl_for_location(self, location: str, lead_time: int) -> float:
        pressure_msl_raw = await self._fetch_variable_at_lead_time(
            location=location,
            variable="PMSL",
            lead_time=lead_time,
        )
        # Average over ensamble dimensions
        pressure_msl_eps_mean = pressure_msl_raw.mean(dim="eps")
        # Average over spatial dimensions (y and x) 
        pressure_msl_yx_mean = pressure_msl_eps_mean.mean(dim=["y", "x"]) 
        pressure_msl_Pa = float(pressure_msl_yx_mean.item())
        return pressure_msl_Pa

    async def total_cloud_cover_for_location(self, location: str, lead_time: int) -> float:
        cloud_raw = await self._fetch_variable_at_lead_time(
            location=location,
            variable="CLCT",
            lead_time=lead_time,
        )
        # Average over ensemble dimensions
        cloud_eps_mean = cloud_raw.mean(dim="eps")
        # Average over spatial dimensions (y and x) 
        cloud_yx_mean = cloud_eps_mean.mean(dim=["y", "x"])
        cloud_percent = float(cloud_yx_mean.item())
        return cloud_percent

    async def snow_depth_for_location(self, location: str, lead_time: int) -> float:
        snow_raw = await self._fetch_variable_at_lead_time(
            location=location,
            variable="H_SNOW",
            lead_time=lead_time,
        )
        # Average over ensemble dimensions
        snow_eps_mean = snow_raw.mean(dim="eps")
        # Averge over spatial dimensions (y and x) 
        snow_yx_mean = snow_eps_mean.mean(dim=["y", "x"])
        snow_m = float(snow_yx_mean.item())
        return snow_m

    async def total_precipitation_rate_for_location(self, location: str, lead_time: int) -> float:
        precip_raw = await self._fetch_variable_at_lead_time(
            location=location,
            variable="TOT_PR",
            lead_time=lead_time,
        )
        # Average over ensemble dimensions
        precip_eps_mean = precip_raw.mean(dim="eps")
        # Average over spatial dimensions (y and x)
        precip_yx_mean = precip_eps_mean.mean(dim=["y", "x"])
        precip_mm_per_sec = float(precip_yx_mean.item())
        return precip_mm_per_sec
    

# Example usage
async def main():
    meteo_swiss = MeteoSwissPredictions()
    location_name = "Zurich"
    lead_time_start = 0
    lead_time_end = 24
    lead_time = 14
    rainfall_mm = await meteo_swiss.total_rainfall_for_location(location_name, lead_time_start, lead_time_end)
    sunshine_hours = await meteo_swiss.sunshine_hours_for_location(location_name, lead_time_start, lead_time_end)
    temp_C = await meteo_swiss.temp_for_location(location_name, lead_time)
    wind_mps = await meteo_swiss.wind_speed_for_location(location_name, lead_time)
    pressure_msl_Pa = await meteo_swiss.pressure_msl_for_location(location_name, lead_time)
    cloud_cover = await meteo_swiss.total_cloud_cover_for_location(location_name, lead_time) 
    snow_depth = await meteo_swiss.snow_depth_for_location(location_name, lead_time)
    total_precipitation_rate = await meteo_swiss.total_precipitation_rate_for_location(location_name, lead_time)
    logger.info(f"Total rainfall in {location_name} for {lead_time_start} to {lead_time_end} hours: {rainfall_mm:.2f} mm")
    logger.info(f"Mean sunshine in {location_name} for {lead_time_start} to {lead_time_end} hours: {sunshine_hours:.2f} h")
    logger.info(f"Temperature in {location_name} for lead time {lead_time} hours: {temp_C:.2f} C")
    logger.info(f"Wind speed in {location_name} for lead time {lead_time} hours: {wind_mps:.2f} m/s")
    logger.info(f"Pressure in {location_name} for lead time {lead_time} hours: {pressure_msl_Pa:.2f} Pa")
    logger.info(f"Cloud cover in {location_name} for lead time {lead_time} hours: {cloud_cover:.2f} %")
    logger.info(f"Snow depth in {location_name} for lead time {lead_time} hours: {snow_depth:.2f} m")
    logger.info(f"Total precipitation rate in {location_name} for lead time {lead_time} hours: {total_precipitation_rate:.2f} kg/m^2/s")

if __name__ == "__main__":
    asyncio.run(main())