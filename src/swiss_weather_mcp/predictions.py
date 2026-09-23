import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from . import parameters
from .localforecast import SWISS_TZ, LocalForecast, Point, Series

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


def _closing_stamp(moment: datetime) -> datetime:
    """
    Return the stamp of the forecast row whose hour contains a moment, for averages and sums.

    MeteoSwiss stamps an hourly average or sum at the end of the hour it covers, so the row stamped
    14:00 is the hour from 13:00 to 14:00. A moment on the full hour therefore reads the row stamped
    at that hour, and 14:30 reads the row stamped 15:00.
    """
    moment = moment.astimezone(timezone.utc)
    full_hour = moment.replace(minute=0, second=0, microsecond=0)
    if full_hour < moment:
        return full_hour + timedelta(hours=1)
    return full_hour


def _nearest_stamp(moment: datetime) -> datetime:
    """
    Return the stamp closest to a moment, for values taken at the moment of their stamp.

    A snapshot such as cloud cover belongs to its stamp alone, so 14:20 is best answered by the
    14:00 value, 20 minutes away, rather than by the 15:00 one. Half past rounds up.
    """
    moment = moment.astimezone(timezone.utc) + timedelta(minutes=30)
    return moment.replace(minute=0, second=0, microsecond=0)


def _swiss_timestamp(moment: datetime) -> str:
    """
    Render a moment as ISO 8601 Swiss local time with its UTC offset, such as 2026-09-23T14:00+02:00.

    Every time the model reads, in an answer or in an error, uses this one form. The offset makes it
    exact, and it changes with the date, +01:00 in winter and +02:00 in summer.
    """
    return moment.astimezone(SWISS_TZ).isoformat(timespec="minutes")


def _covered_range(series: Series, parameter: str) -> str:
    """
    Say which times a series actually covers.

    The published window starts at a different hour for each parameter, so this is read off the
    series rather than assumed, which keeps the message right for every tool.
    """
    return (
        f"'{parameter}' covers {_swiss_timestamp(min(series.values))} to "
        f"{_swiss_timestamp(max(series.values))}."
    )


def _describe(code: int) -> str:
    """Turn a MeteoSwiss pictogram code into the sentence it stands for."""
    return parameters.PICTOGRAM_DESCRIPTIONS.get(code, f"unknown weather code {code}")


def _compass_point(degrees: float) -> str:
    """Name the compass point a bearing falls in, such as "SW" for 217 degrees."""
    # The 16 points divide the circle into 22.5 degree sectors, so rounding lands on the nearest
    sector = round(degrees / 22.5) % len(COMPASS_POINTS)
    return COMPASS_POINTS[sector]


class SwissWeatherPredictions:
    def __init__(self, forecast: LocalForecast):
        """
        Answer weather questions from a forecast data source.

        The source is passed in rather than built here, so the caller decides where its cache
        lives and how much it keeps, and tests can hand in one of their own.
        """
        self.forecast = forecast

    async def _resolve(self, location: str) -> Point:
        """Resolve a location name or postal code, off the event loop since it may download the table."""
        return await asyncio.to_thread(self.forecast.resolve, location)

    async def _series(self, parameter: str, point: Point) -> Series:
        """Read one parameter for one point, off the event loop since it may download a large file."""
        return await asyncio.to_thread(self.forecast.series, parameter, point)

    def _value_at(self, series: Series, when: datetime, point: Point, parameter: str) -> float:
        """
        Pick the forecast row for the requested time.

        An average or sum is read from the row whose hour contains the time, a snapshot from the
        row stamped closest to it.
        """
        if parameter in parameters.SNAPSHOTS:
            stamp = _nearest_stamp(when)
        else:
            stamp = _closing_stamp(when)
        if stamp not in series.values:
            raise ValueError(
                f"{_swiss_timestamp(when)} is outside the forecast for {point.label()}. "
                f"{_covered_range(series, parameter)}"
            )
        return series.values[stamp]

    def _sum_between(self, series: Series, start: datetime, end: datetime, point: Point, parameter: str) -> float:
        """
        Add up the hourly values of the period from start to end.

        Each row covers the hour before its stamp, so the rows stamped after start, up to and
        including end, are exactly the hours of the period.
        """
        first_stamp = _closing_stamp(start)
        last_stamp = _closing_stamp(end)

        total = 0.0
        counted = 0
        for stamp, value in series.values.items():
            if first_stamp < stamp <= last_stamp:
                total += value
                counted += 1

        if not counted:
            raise ValueError(
                f"{_swiss_timestamp(start)} to {_swiss_timestamp(end)} is outside the forecast for {point.label()}. "
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
        result["model_run"] = _swiss_timestamp(run)
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
        return self._result(value, unit, point, series.run, valid_at=_swiss_timestamp(when))

    async def temperature_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        return await self._value_for_location(location, parameters.TEMPERATURE, when, "°C")

    async def wind_speed_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        return await self._value_for_location(location, parameters.WIND_SPEED, when, "km/h")

    async def wind_gusts_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        return await self._value_for_location(location, parameters.WIND_GUSTS, when, "km/h")

    async def freezing_level_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        return await self._value_for_location(location, parameters.FREEZING_LEVEL, when, "m above sea level")

    async def precipitation_rate_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        return await self._value_for_location(location, parameters.PRECIPITATION, when, "mm/h")

    async def precipitation_probability_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        return await self._value_for_location(location, parameters.PRECIPITATION_PROBABILITY, when, "%")

    async def wind_direction_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        result = await self._value_for_location(location, parameters.WIND_DIRECTION, when, "degrees")
        result["compass_point"] = _compass_point(result["value"])
        return result

    async def weather_description_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        point = await self._resolve(location)
        series = await self._series(parameters.WEATHER_PICTOGRAM, point)
        code = int(self._value_at(series, when, point, parameters.WEATHER_PICTOGRAM))
        return self._result(
            _describe(code), "description", point, series.run,
            valid_at=_swiss_timestamp(when), pictogram_code=code,
        )

    async def total_rainfall_for_location(self, location: str, start: datetime, end: datetime) -> Dict[str, Any]:
        point = await self._resolve(location)
        series = await self._series(parameters.PRECIPITATION, point)
        total = self._sum_between(series, start, end, point, parameters.PRECIPITATION)
        return self._result(round(total, 1), "mm", point, series.run,
                            **{"from": _swiss_timestamp(start), "to": _swiss_timestamp(end)})

    async def sunshine_hours_for_location(self, location: str, start: datetime, end: datetime) -> Dict[str, Any]:
        point = await self._resolve(location)
        series = await self._series(parameters.SUNSHINE, point)
        minutes = self._sum_between(series, start, end, point, parameters.SUNSHINE)
        return self._result(round(minutes / 60, 1), "h", point, series.run,
                            **{"from": _swiss_timestamp(start), "to": _swiss_timestamp(end)})

    async def total_cloud_cover_for_location(self, location: str, when: datetime) -> Dict[str, Any]:
        point = await self._resolve(location)
        layers = {}
        run = None
        for name, parameter in CLOUD_LAYERS:
            series = await self._series(parameter, point)
            layers[name] = self._value_at(series, when, point, parameter)
            run = series.run

        # The layers overlap, so they cannot simply be added. Assuming they are independent, the sky
        # is clear only where all three are clear, which is the standard random overlap estimate.
        clear = (1 - layers["low"]) * (1 - layers["medium"]) * (1 - layers["high"])
        return self._result(
            round((1 - clear) * 100, 1), "%", point, run,
            valid_at=_swiss_timestamp(when),
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

    async def daily_forecast_for_location(self, location: str, day: datetime) -> Dict[str, Any]:
        """
        Read the whole-day summary for a location.

        The daily parameters answer this far more cheaply than the hourly ones: six daily files are
        about 7 MB together, where the hourly equivalent is about 124 MB.

        Parameters:
            location (str): Location name or postal code.
            day (datetime): Any moment on the day wanted, timezone aware.

        Returns:
            Dict[str, Any]: Minimum and maximum temperature, the median rainfall with its 10th and
                90th percentile, a worded summary, the resolved point and the model run. Fields
                MeteoSwiss does not publish for this location are None.
        """
        point = await self._resolve(location)
        calendar_day = day.astimezone(SWISS_TZ).date()

        summary: Dict[str, Any] = {
            "location": point.label(),
            "altitude_m": point.height_masl,
            "date": calendar_day.isoformat(),
        }

        run: Optional[datetime] = None
        for field, parameter in DAILY_PARAMETERS:
            value, parameter_run = await self._daily_value(parameter, point, calendar_day)
            run = parameter_run or run
            summary[field] = value

        if run is None:
            raise ValueError(f"MeteoSwiss publishes no daily forecast for {point.label()}, try a nearby town.")

        # The pictogram is published as a code, which is only useful once it is spelled out
        if summary["weather"] is not None:
            summary["pictogram_code"] = int(summary["weather"])
            summary["weather"] = _describe(summary["pictogram_code"])

        summary["model_run"] = _swiss_timestamp(run)
        return summary
