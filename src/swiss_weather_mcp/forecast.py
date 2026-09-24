import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional

from . import parameters
from .meteoswiss import SWISS_TZ, ForecastPoint, ForecastSeries, LocalForecastSource

logger = logging.getLogger(__name__)

# Compass points the wind direction in degrees is reported as, clockwise from north
COMPASS_POINTS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")

# The field each cloud layer fills in the answer, lowest first
CLOUD_LAYERS = (
    ("low", parameters.CLOUD_COVER_LOW),
    ("medium", parameters.CLOUD_COVER_MEDIUM),
    ("high", parameters.CLOUD_COVER_HIGH),
)

# The field each daily parameter fills in the daily forecast
DAILY_PARAMETERS = (
    ("temperature_min_c", parameters.TEMPERATURE_DAY_MIN),
    ("temperature_max_c", parameters.TEMPERATURE_DAY_MAX),
    ("rainfall_median_mm", parameters.PRECIPITATION_DAY),
    ("rainfall_10th_percentile_mm", parameters.PRECIPITATION_DAY_Q10),
    ("rainfall_90th_percentile_mm", parameters.PRECIPITATION_DAY_Q90),
    ("weather", parameters.WEATHER_PICTOGRAM_DAY),
)


def _find_closing_stamp(moment: datetime) -> datetime:
    """
    Return the stamp of the row whose hour contains a moment, for averages and sums.

    A row covers the hour before its stamp, so 14:00 reads the row stamped 14:00, and 14:30 the
    row stamped 15:00.
    """
    moment = moment.astimezone(timezone.utc)
    full_hour = moment.replace(minute=0, second=0, microsecond=0)
    if full_hour < moment:
        return full_hour + timedelta(hours=1)
    return full_hour


def _find_nearest_stamp(moment: datetime) -> datetime:
    """Return the full hour closest to a moment, for snapshot values. Half past rounds up."""
    moment = moment.astimezone(timezone.utc) + timedelta(minutes=30)
    return moment.replace(minute=0, second=0, microsecond=0)


def _format_swiss_time(moment: datetime) -> str:
    """Format a moment as ISO 8601 Swiss local time with its UTC offset, such as 2026-09-23T14:00+02:00."""
    return moment.astimezone(SWISS_TZ).isoformat(timespec="minutes")


def _describe_covered_range(series: ForecastSeries) -> str:
    """Say which times a series covers, for an error message."""
    return (
        f"The forecast covers {_format_swiss_time(min(series.values))} to "
        f"{_format_swiss_time(max(series.values))}."
    )


def _describe_pictogram(pictogram_code: int) -> str:
    """Turn a MeteoSwiss pictogram code into the sentence it stands for."""
    return parameters.PICTOGRAM_DESCRIPTIONS.get(pictogram_code, f"unknown weather code {pictogram_code}")


def _find_compass_point(degrees: float) -> str:
    """Find the compass point a bearing falls in, such as "SW" for 217 degrees."""
    # The 16 points divide the circle into 22.5 degree sectors, so rounding lands on the nearest
    sector = round(degrees / 22.5) % len(COMPASS_POINTS)
    return COMPASS_POINTS[sector]


class ForecastService:
    def __init__(self, forecast_source: LocalForecastSource):
        self.forecast_source = forecast_source

    async def _find_point(self, location: str) -> ForecastPoint:
        """Find the forecast point for a location in a worker thread, so a download does not block other requests."""
        return await asyncio.to_thread(self.forecast_source.find_point, location)

    async def _read_series(self, parameter: str, point: ForecastPoint) -> ForecastSeries:
        """Read one parameter for one point in a worker thread, so a download does not block other requests."""
        return await asyncio.to_thread(self.forecast_source.read_series, parameter, point)

    def _read_value_at(self, series: ForecastSeries, moment: datetime, point: ForecastPoint, parameter: str) -> float:
        """
        Return the value at a time: an average or sum from the row whose hour contains the time,
        a snapshot from the row stamped closest to it.
        """
        if parameter in parameters.SNAPSHOTS:
            stamp = _find_nearest_stamp(moment)
        else:
            stamp = _find_closing_stamp(moment)
        if stamp not in series.values:
            raise ValueError(
                f"{_format_swiss_time(moment)} is outside the forecast for {point.display_name}. "
                f"{_describe_covered_range(series)}"
            )
        return series.values[stamp]

    def _sum_between(self, series: ForecastSeries, start_moment: datetime, end_moment: datetime, point: ForecastPoint) -> float:
        """Add up the hourly values from start to end."""
        if end_moment <= start_moment:
            raise ValueError(
                f"The end of the period, {_format_swiss_time(end_moment)}, must be after its start, "
                f"{_format_swiss_time(start_moment)}."
            )

        first_stamp = _find_closing_stamp(start_moment)
        last_stamp = _find_closing_stamp(end_moment)
        # The first hour of the period is the row stamped one hour after its start
        first_hour_stamp = first_stamp + timedelta(hours=1)
        if first_hour_stamp < min(series.values) or last_stamp > max(series.values):
            raise ValueError(
                f"{_format_swiss_time(start_moment)} to {_format_swiss_time(end_moment)} is not fully covered "
                f"by the forecast for {point.display_name}. {_describe_covered_range(series)}"
            )

        total = 0.0
        hours_counted = 0
        for stamp, value in series.values.items():
            # A row covers the hour before its stamp, so the row stamped at start is not in the period
            if first_stamp < stamp <= last_stamp:
                total += value
                hours_counted += 1

        if not hours_counted:
            raise ValueError(
                f"{_format_swiss_time(start_moment)} to {_format_swiss_time(end_moment)} is outside the forecast for {point.display_name}. "
                f"{_describe_covered_range(series)}"
            )
        return total

    def _build_answer(self, value: Any, unit: str, point: ForecastPoint, run_time: datetime, **fields: Any) -> Dict[str, Any]:
        """Build a tool answer: the value and unit, the resolved point, extra fields and the model run."""
        answer = {"value": value, "unit": unit, "location": point.display_name, "altitude_m": point.altitude_m}
        answer.update(fields)
        answer["model_run"] = _format_swiss_time(run_time)
        return answer

    async def _build_hourly_answer(self, location: str, parameter: str, moment: datetime, unit: str) -> Dict[str, Any]:
        """
        Read one parameter for a location at one time.

        Parameters:
            location (str): Location name or postal code.
            parameter (str): MeteoSwiss parameter shortname.
            moment (datetime): The forecast hour, timezone aware.
            unit (str): Unit the returned value is expressed in.

        Returns:
            Dict[str, Any]: The value with the resolved point, the hour and the model run.
        """
        point = await self._find_point(location)
        series = await self._read_series(parameter, point)
        value = self._read_value_at(series, moment, point, parameter)
        return self._build_answer(value, unit, point, series.run_time, valid_at=_format_swiss_time(moment))

    async def read_temperature(self, location: str, moment: datetime) -> Dict[str, Any]:
        return await self._build_hourly_answer(location, parameters.TEMPERATURE, moment, "°C")

    async def read_wind_speed(self, location: str, moment: datetime) -> Dict[str, Any]:
        return await self._build_hourly_answer(location, parameters.WIND_SPEED, moment, "km/h")

    async def read_wind_gusts(self, location: str, moment: datetime) -> Dict[str, Any]:
        return await self._build_hourly_answer(location, parameters.WIND_GUSTS, moment, "km/h")

    async def read_freezing_level(self, location: str, moment: datetime) -> Dict[str, Any]:
        return await self._build_hourly_answer(location, parameters.FREEZING_LEVEL, moment, "m above sea level")

    async def read_precipitation_rate(self, location: str, moment: datetime) -> Dict[str, Any]:
        return await self._build_hourly_answer(location, parameters.PRECIPITATION, moment, "mm/h")

    async def read_precipitation_probability(self, location: str, moment: datetime) -> Dict[str, Any]:
        return await self._build_hourly_answer(location, parameters.PRECIPITATION_PROBABILITY, moment, "%")

    async def read_wind_direction(self, location: str, moment: datetime) -> Dict[str, Any]:
        answer = await self._build_hourly_answer(location, parameters.WIND_DIRECTION, moment, "degrees")
        answer["compass_point"] = _find_compass_point(answer["value"])
        return answer

    async def read_weather_description(self, location: str, moment: datetime) -> Dict[str, Any]:
        point = await self._find_point(location)
        series = await self._read_series(parameters.WEATHER_PICTOGRAM, point)
        pictogram_value = self._read_value_at(series, moment, point, parameters.WEATHER_PICTOGRAM)
        pictogram_code = int(pictogram_value)
        description = _describe_pictogram(pictogram_code)
        return self._build_answer(
            description, "description", point, series.run_time,
            valid_at=_format_swiss_time(moment), pictogram_code=pictogram_code,
        )

    async def read_total_rainfall(self, location: str, start_moment: datetime, end_moment: datetime) -> Dict[str, Any]:
        point = await self._find_point(location)
        series = await self._read_series(parameters.PRECIPITATION, point)
        rainfall_mm = self._sum_between(series, start_moment, end_moment, point)
        return self._build_answer(round(rainfall_mm, 1), "mm", point, series.run_time,
                            **{"from": _format_swiss_time(start_moment), "to": _format_swiss_time(end_moment)})

    async def read_sunshine_hours(self, location: str, start_moment: datetime, end_moment: datetime) -> Dict[str, Any]:
        point = await self._find_point(location)
        series = await self._read_series(parameters.SUNSHINE, point)
        sunshine_minutes = self._sum_between(series, start_moment, end_moment, point)
        return self._build_answer(round(sunshine_minutes / 60, 1), "h", point, series.run_time,
                            **{"from": _format_swiss_time(start_moment), "to": _format_swiss_time(end_moment)})

    async def read_total_cloud_cover(self, location: str, moment: datetime) -> Dict[str, Any]:
        point = await self._find_point(location)
        layers: Dict[str, float] = {}
        for layer, parameter in CLOUD_LAYERS:
            series = await self._read_series(parameter, point)
            layers[layer] = self._read_value_at(series, moment, point, parameter)

        # The layers overlap, so they cannot simply be added. Assuming they are independent, the sky
        # is clear only where all three are clear, which is the standard random overlap estimate.
        clear_sky = (1 - layers["low"]) * (1 - layers["medium"]) * (1 - layers["high"])
        return self._build_answer(
            round((1 - clear_sky) * 100, 1), "%", point, series.run_time,
            valid_at=_format_swiss_time(moment),
            low_percent=round(layers["low"] * 100, 1),
            medium_percent=round(layers["medium"] * 100, 1),
            high_percent=round(layers["high"] * 100, 1),
        )

    async def _read_series_if_published(self, parameter: str, point: ForecastPoint) -> Optional[ForecastSeries]:
        """Read one parameter for one point, or return None when MeteoSwiss does not publish it there."""
        try:
            return await self._read_series(parameter, point)
        except ValueError as error:
            logger.info(f"daily_forecast: no {parameter} for {point.display_name}: {error}")
            return None

    async def read_daily_forecast(self, location: str, day: date) -> Dict[str, Any]:
        """
        Read the whole-day summary for a location.

        Parameters:
            location (str): Location name or postal code.
            day (date): The Swiss calendar day.

        Returns:
            Dict[str, Any]: Minimum and maximum temperature, the median rainfall with its 10th and
                90th percentile, a worded summary, the resolved point and the model run. Fields
                MeteoSwiss does not publish for this location are None.
        """
        point = await self._find_point(location)
        # A daily row is stamped 00:00 on the Swiss calendar day it describes
        day_stamp = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)

        summary: Dict[str, Any] = {
            "location": point.display_name,
            "altitude_m": point.altitude_m,
            "date": day.isoformat(),
        }
        run_time: Optional[datetime] = None
        for field, parameter in DAILY_PARAMETERS:
            series = await self._read_series_if_published(parameter, point)
            if series is None:
                summary[field] = None
                continue
            run_time = series.run_time
            summary[field] = series.values.get(day_stamp)

        values = [summary[field] for field, _ in DAILY_PARAMETERS]
        if run_time is None or all(value is None for value in values):
            raise ValueError(f"MeteoSwiss has no daily forecast for {point.display_name} on {day.isoformat()}.")

        # The pictogram is published as a code, which is only useful once it is spelled out
        if summary["weather"] is not None:
            summary["pictogram_code"] = int(summary["weather"])
            summary["weather"] = _describe_pictogram(summary["pictogram_code"])

        summary["model_run"] = _format_swiss_time(run_time)
        return summary
