import asyncio
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from platformdirs import user_cache_path

from .localforecast import PICTOGRAM_DESCRIPTIONS, SWISS_TZ, LocalForecast, Point, Series

logger = logging.getLogger(__name__)

# Shared, OS-standard cache location (survives across working directories the server may be launched from).
# Override with SWISS_WEATHER_MCP_CACHE_DIR, e.g. to isolate cache location in tests or Docker.
CACHE_DIR = Path(os.environ.get("SWISS_WEATHER_MCP_CACHE_DIR", user_cache_path("swiss-weather-mcp")))

# Compass points the wind direction in degrees is reported as, clockwise from north
COMPASS_POINTS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")

# Which daily parameter answers which field of the daily forecast
DAILY_PARAMETERS = (
    ("temperature_min_c", "tre200pn"),
    ("temperature_max_c", "tre200px"),
    ("rainfall_mm", "rka150p0"),
    ("rainfall_min_mm", "rreq10p0"),
    ("rainfall_max_mm", "rreq90p0"),
    ("weather", "jp2000d0"),
)


def _utc_hour(moment: datetime) -> datetime:
    """Round a moment down to the UTC hour the forecast rows are keyed by."""
    return moment.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _swiss(moment: datetime) -> str:
    """Render a moment as Swiss local time, for messages a person reads."""
    return f"{moment.astimezone(SWISS_TZ):%Y-%m-%d %H:%M}"


def _valid_at(moment: datetime) -> str:
    """Render a moment as a Swiss local timestamp, for the answer a tool returns."""
    return moment.astimezone(SWISS_TZ).isoformat(timespec="minutes")


def _covered_range(series: Series, parameter: str) -> str:
    """
    Say which times a series actually covers.

    The published window starts at a different hour for each parameter, so this is read off the
    series rather than assumed, which keeps the message right for every tool.
    """
    return (
        f"'{parameter}' covers {_swiss(min(series.values))} to "
        f"{_swiss(max(series.values))} Swiss time."
    )


def _describe(code: int) -> str:
    """Turn a MeteoSwiss pictogram code into the sentence it stands for."""
    return PICTOGRAM_DESCRIPTIONS.get(code, f"unknown weather code {code}")


def _compass_point(degrees: float) -> str:
    """Name the compass point a bearing falls in, such as "SW" for 217 degrees."""
    # The 16 points divide the circle into 22.5 degree sectors, so rounding lands on the nearest
    sector = round(degrees / 22.5) % len(COMPASS_POINTS)
    return COMPASS_POINTS[sector]


class MeteoSwissPredictions:
    def __init__(self, cache_all_locations: bool = False):
        self.forecast = LocalForecast(CACHE_DIR, cache_all_locations=cache_all_locations)

    async def _resolve(self, location: str) -> Point:
        """Resolve a location name or postal code, off the event loop since it may download the table."""
        return await asyncio.to_thread(self.forecast.resolve, location)

    async def _series(self, parameter: str, point: Point) -> Series:
        """Read one parameter for one point, off the event loop since it may download a large file."""
        return await asyncio.to_thread(self.forecast.series, parameter, point)

    def _value_at(self, series: Series, when: datetime, point: Point, parameter: str) -> float:
        """Pick the forecast hour containing the requested time."""
        hour = _utc_hour(when)
        if hour not in series.values:
            raise ValueError(
                f"{_swiss(when)} is outside the forecast for {point.label()}. "
                f"{_covered_range(series, parameter)}"
            )
        return series.values[hour]

    def _sum_between(self, series: Series, start: datetime, end: datetime, point: Point, parameter: str) -> float:
        """Add up the hourly values from start up to, but not including, end."""
        first_hour = _utc_hour(start)
        last_hour = _utc_hour(end)

        total = 0.0
        counted = 0
        for hour, value in series.values.items():
            if first_hour <= hour < last_hour:
                total += value
                counted += 1

        if not counted:
            raise ValueError(
                f"{_swiss(start)} to {_swiss(end)} is outside the forecast for {point.label()}. "
                f"{_covered_range(series, parameter)}"
            )
        return total

    def _result(self, value: Any, unit: str, point: Point, run: datetime, **fields: Any) -> Dict[str, Any]:
        """
        Build the answer a tool returns.

        The resolved point travels with the number because a name maps to many points, and in
        Switzerland the altitude decides the weather as much as the place does.
        """
        result = {"value": value, "unit": unit, "location": point.label(), "altitude_m": point.height_masl}
        result.update(fields)
        result["model_run"] = f"{run:%Y-%m-%dT%H:%M}Z"
        return result

    async def _value_for_location(self, location: str, parameter: str, when: datetime, unit: str) -> Dict[str, Any]:
        """
        Read one parameter at one hour, the shape every point-in-time tool shares.

        Parameters:
            location (str): Location name or postal code.
            parameter (str): MeteoSwiss parameter shortname.
            when (datetime): The forecast hour, timezone aware.
            unit (str): Unit the returned value is expressed in.

        Returns:
            Dict[str, Any]: The value with the resolved point, the hour and the model run.
        """
        point = await self._resolve(location)
        series = await self._series(parameter, point)
        value = self._value_at(series, when, point, parameter)
        return self._result(value, unit, point, series.run, valid_at=_valid_at(when))

    async def temp_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        return await self._value_for_location(location, "tre200h0", when, "°C")

    async def wind_speed_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        return await self._value_for_location(location, "fu3010h0", when, "km/h")

    async def wind_gusts_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        return await self._value_for_location(location, "fu3010h1", when, "km/h")

    async def freezing_level_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        return await self._value_for_location(location, "zprfr0hs", when, "m above sea level")

    async def precipitation_rate_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        return await self._value_for_location(location, "rre150h0", when, "mm/h")

    async def precipitation_probability_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        return await self._value_for_location(location, "rp0003i0", when, "%")

    async def wind_direction_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        result = await self._value_for_location(location, "dkl010h0", when, "degrees")
        result["compass_point"] = _compass_point(result["value"])
        return result

    async def weather_description_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        point = await self._resolve(location)
        series = await self._series("jww003i0", point)
        code = int(self._value_at(series, when, point, "jww003i0"))
        return self._result(
            _describe(code), "description", point, series.run,
            valid_at=_valid_at(when), pictogram_code=code,
        )

    async def total_rainfall_for_location(self, location: str, start: datetime, end: datetime) -> Dict[str, Any]:
        point = await self._resolve(location)
        series = await self._series("rre150h0", point)
        total = self._sum_between(series, start, end, point, "rre150h0")
        return self._result(round(total, 1), "mm", point, series.run,
                            **{"from": _valid_at(start), "to": _valid_at(end)})

    async def sunshine_hours_for_location(self, location: str, start: datetime, end: datetime) -> Dict[str, Any]:
        point = await self._resolve(location)
        series = await self._series("sre000h0", point)
        minutes = self._sum_between(series, start, end, point, "sre000h0")
        return self._result(round(minutes / 60, 1), "h", point, series.run,
                            **{"from": _valid_at(start), "to": _valid_at(end)})

    async def total_cloud_cover_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        point = await self._resolve(location)
        layers = {}
        run = None
        for name, parameter in (("low", "nprolohs"), ("medium", "npromths"), ("high", "nprohihs")):
            series = await self._series(parameter, point)
            layers[name] = self._value_at(series, when, point, parameter)
            run = series.run

        # The layers overlap, so they cannot simply be added. Assuming they are independent, the sky
        # is clear only where all three are clear, which is the standard random overlap estimate.
        clear = (1 - layers["low"]) * (1 - layers["medium"]) * (1 - layers["high"])
        return self._result(
            round((1 - clear) * 100, 1), "%", point, run,
            valid_at=_valid_at(when),
            low_percent=round(layers["low"] * 100, 1),
            medium_percent=round(layers["medium"] * 100, 1),
            high_percent=round(layers["high"] * 100, 1),
        )

    async def _daily_value(
        self, parameter: str, point: Point, day: date
    ) -> Tuple[Optional[float], Optional[datetime]]:
        """
        Read one daily parameter for one calendar day.

        Not every location carries every daily parameter, the regional entries in particular, so a
        missing one is reported as None instead of failing the whole summary.

        Parameters:
            parameter (str): MeteoSwiss parameter shortname (e.g., "tre200px").
            point (Point): The resolved forecast point.
            day (date): The Swiss calendar day wanted.

        Returns:
            Tuple[Optional[float], Optional[datetime]]: The value and the run it came from, or
                (None, None) when MeteoSwiss does not publish it here.
        """
        try:
            series = await self._series(parameter, point)
        except ValueError as error:
            logger.info(f"daily_forecast: no {parameter} for {point.label()}: {error}")
            return None, None

        # Daily rows are stamped at midnight UTC, which is the same calendar day in Switzerland
        for measured_at, value in series.values.items():
            if measured_at.date() == day:
                return value, series.run

        logger.info(f"daily_forecast: {parameter} does not reach {day} for {point.label()}")
        return None, series.run

    async def daily_forecast_for_location(self, location: str, date: datetime) -> Dict[str, Any]:
        """
        Read the whole-day summary for a location.

        The daily parameters answer this far more cheaply than the hourly ones: six daily files are
        about 7 MB together, where the hourly equivalent is about 124 MB.

        Parameters:
            location (str): Location name or postal code.
            date (datetime): Any moment on the day wanted, timezone aware.

        Returns:
            Dict[str, Any]: Minimum and maximum temperature, rainfall with its 10% and 90% range, a
                worded summary, the resolved point and the model run. Fields MeteoSwiss does not
                publish for this location are None.
        """
        point = await self._resolve(location)
        day = date.astimezone(SWISS_TZ).date()

        summary: Dict[str, Any] = {
            "location": point.label(),
            "altitude_m": point.height_masl,
            "date": day.isoformat(),
        }

        run: Optional[datetime] = None
        for field, parameter in DAILY_PARAMETERS:
            value, parameter_run = await self._daily_value(parameter, point, day)
            run = parameter_run or run
            summary[field] = value

        if run is None:
            raise ValueError(f"MeteoSwiss publishes no daily forecast for {point.label()}, try a nearby town.")

        # The pictogram is published as a code, which is only useful once it is spelled out
        if summary["weather"] is not None:
            summary["pictogram_code"] = int(summary["weather"])
            summary["weather"] = _describe(summary["pictogram_code"])

        summary["model_run"] = f"{run:%Y-%m-%dT%H:%M}Z"
        return summary
