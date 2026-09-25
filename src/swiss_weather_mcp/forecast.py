import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from . import parameters
from .meteoswiss import SWISS_TZ, ForecastPoint, ForecastSeries, LocalForecastSource

logger = logging.getLogger(__name__)

# Compass points the wind direction in degrees is reported as, clockwise from north
COMPASS_POINTS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                  "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
COMPASS_SECTOR_DEGREES = 360 / len(COMPASS_POINTS)

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

# MeteoSwiss publishes today and the next 8 days
MAX_DAYS = 9

RAIN_BLOCK = timedelta(hours=3)
MAX_RAIN_OUTLOOK = timedelta(hours=48)
MAX_HOURLY_FORECAST = timedelta(hours=24)


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


def _check_period_order(start_moment: datetime, end_moment: datetime) -> None:
    """Refuse a period whose end is not after its start."""
    if end_moment <= start_moment:
        raise ValueError(
            f"The end of the period, {_format_swiss_time(end_moment)}, must be after its start, "
            f"{_format_swiss_time(start_moment)}."
        )


def _describe_covered_range(series: ForecastSeries) -> str:
    """Say which times a series covers, for an error message."""
    return (
        f"The forecast covers {_format_swiss_time(min(series.values))} to "
        f"{_format_swiss_time(max(series.values))}."
    )


def _describe_pictogram(pictogram_code: int) -> Tuple[str, Optional[str]]:
    """Turn a MeteoSwiss pictogram code into the sentence it stands for and its emoji."""
    pictogram = parameters.PICTOGRAMS.get(pictogram_code)
    if pictogram is None:
        return f"unknown weather code {pictogram_code}", None
    return pictogram


def _find_compass_point(degrees: float) -> str:
    """Find the compass point a bearing falls in, such as "SW" for 217 degrees."""
    # Rounding picks the nearest point, and the modulo turns a bearing just short of 360 back into N
    sector = round(degrees / COMPASS_SECTOR_DEGREES) % len(COMPASS_POINTS)
    return COMPASS_POINTS[sector]


def _build_day(series_by_field: Dict[str, Optional[ForecastSeries]], day: date) -> Optional[Dict[str, Any]]:
    """Build one day's values from the daily series, or return None when none has a value that day."""
    # A daily row is stamped 00:00 on the Swiss calendar day it describes
    day_stamp = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    day_values: Dict[str, Any] = {}
    for field, series in series_by_field.items():
        day_values[field] = series.values.get(day_stamp) if series is not None else None
    if all(value is None for value in day_values.values()):
        return None

    # The pictogram is published as a code, which is only useful once it is spelled out
    if day_values["weather"] is not None:
        day_values["pictogram_code"] = int(day_values["weather"])
        day_values["weather"], day_values["weather_emoji"] = _describe_pictogram(day_values["pictogram_code"])
    return day_values


def _find_run_time(series_by_field: Dict[str, Optional[ForecastSeries]]) -> Optional[datetime]:
    """Return the model run of the first published series, or None when none is published."""
    for series in series_by_field.values():
        if series is not None:
            return series.run_time
    return None


class ForecastService:
    """Answer weather questions for a location: the value, its unit, the resolved point and the model run."""

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
        _check_period_order(start_moment, end_moment)
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

    async def read_freezing_level(self, location: str, moment: datetime) -> Dict[str, Any]:
        """Read the height of the 0 °C line at a moment."""
        return await self._build_hourly_answer(location, parameters.FREEZING_LEVEL, moment, "m above sea level")

    async def read_wind(self, location: str, moment: datetime) -> Dict[str, Any]:
        """
        Read the mean wind speed, the strongest gust with its 90th percentile and the wind direction
        for the hour up to a moment.
        """
        point = await self._find_point(location)
        speed_series, gust_series, upper_gust_series, direction_series = await asyncio.gather(
            self._read_series(parameters.WIND_SPEED, point),
            self._read_series(parameters.WIND_GUSTS, point),
            self._read_series(parameters.WIND_GUSTS_Q90, point),
            self._read_series(parameters.WIND_DIRECTION, point),
        )

        speed_kmh = self._read_value_at(speed_series, moment, point, parameters.WIND_SPEED)
        gusts_kmh = self._read_value_at(gust_series, moment, point, parameters.WIND_GUSTS)
        upper_gusts_kmh = self._read_value_at(upper_gust_series, moment, point, parameters.WIND_GUSTS_Q90)
        direction_degrees = self._read_value_at(direction_series, moment, point, parameters.WIND_DIRECTION)
        compass_point = _find_compass_point(direction_degrees)
        return {
            "speed_kmh": speed_kmh,
            "gusts_kmh": gusts_kmh,
            "gusts_90th_percentile_kmh": upper_gusts_kmh,
            "direction_degrees": direction_degrees,
            "compass_point": compass_point,
            "location": point.display_name,
            "altitude_m": point.altitude_m,
            "valid_at": _format_swiss_time(moment),
            "model_run": _format_swiss_time(speed_series.run_time),
        }

    async def read_sunshine_hours(self, location: str, start_moment: datetime, end_moment: datetime) -> Dict[str, Any]:
        """Add up the sunshine over a period, in hours."""
        point = await self._find_point(location)
        series = await self._read_series(parameters.SUNSHINE, point)
        sunshine_minutes = self._sum_between(series, start_moment, end_moment, point)
        return self._build_answer(round(sunshine_minutes / 60, 1), "h", point, series.run_time,
                            **{"from": _format_swiss_time(start_moment), "to": _format_swiss_time(end_moment)})

    async def read_total_cloud_cover(self, location: str, moment: datetime) -> Dict[str, Any]:
        """Estimate the total cloud cover at a moment from the three overlapping layers."""
        point = await self._find_point(location)
        series_reads = [self._read_series(parameter, point) for _, parameter in CLOUD_LAYERS]
        all_series = await asyncio.gather(*series_reads)
        layers: Dict[str, float] = {}
        for (layer, parameter), series in zip(CLOUD_LAYERS, all_series):
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

    async def read_hourly_forecast(
        self, location: str, start_moment: datetime, end_moment: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Read the temperature with its 10th and 90th percentile, rain chance and weather for every
        hour from start to end.

        Parameters:
            location (str): Location name or postal code.
            start_moment (datetime): Start of the period, timezone aware.
            end_moment (Optional[datetime]): End of the period, at most MAX_HOURLY_FORECAST after the
                start; None for the one hour starting at start.

        Returns:
            Dict[str, Any]: The resolved point, one row per hour labelled with its start and end, and
                the model run.
        """
        # A row is stamped at the end of its hour, so the first row is the one closing the hour
        # that contains the start; an hour ending exactly at the start lies before the period
        first_stamp = _find_closing_stamp(start_moment)
        if first_stamp == start_moment:
            first_stamp += timedelta(hours=1)

        if end_moment is None:
            last_stamp = first_stamp
            period = _format_swiss_time(start_moment)
        else:
            _check_period_order(start_moment, end_moment)
            if end_moment - start_moment > MAX_HOURLY_FORECAST:
                max_hours = int(MAX_HOURLY_FORECAST.total_seconds() // 3600)
                raise ValueError(f"An hourly forecast covers at most {max_hours} hours; for whole days use daily_forecast.")
            last_stamp = _find_closing_stamp(end_moment)
            period = f"{_format_swiss_time(start_moment)} to {_format_swiss_time(end_moment)}"

        point = await self._find_point(location)
        temperature_series, lower_series, upper_series, chance_series, pictogram_series = await asyncio.gather(
            self._read_series(parameters.TEMPERATURE, point),
            self._read_series(parameters.TEMPERATURE_Q10, point),
            self._read_series(parameters.TEMPERATURE_Q90, point),
            self._read_series(parameters.PRECIPITATION_PROBABILITY, point),
            self._read_series(parameters.WEATHER_PICTOGRAM, point),
        )
        all_series = (temperature_series, lower_series, upper_series, chance_series, pictogram_series)

        hours = []
        stamp = first_stamp
        while stamp <= last_stamp:
            has_values = all(stamp in series.values for series in all_series)
            if not has_values:
                raise ValueError(
                    f"{period} is not fully covered by the forecast for {point.display_name}. "
                    f"{_describe_covered_range(pictogram_series)}"
                )
            pictogram_code = int(pictogram_series.values[stamp])
            description, weather_emoji = _describe_pictogram(pictogram_code)
            hours.append({
                "from": _format_swiss_time(stamp - timedelta(hours=1)),
                "to": _format_swiss_time(stamp),
                "temperature_c": temperature_series.values[stamp],
                "temperature_10th_percentile_c": lower_series.values[stamp],
                "temperature_90th_percentile_c": upper_series.values[stamp],
                "rain_chance_percent": chance_series.values[stamp],
                "weather": description,
                "weather_emoji": weather_emoji,
            })
            stamp += timedelta(hours=1)

        return {
            "location": point.display_name,
            "altitude_m": point.altitude_m,
            "hours": hours,
            "model_run": _format_swiss_time(temperature_series.run_time),
        }

    async def read_rain_outlook(self, location: str, start_moment: datetime, end_moment: datetime) -> Dict[str, Any]:
        """
        Read when and how much it may rain, in 3-hour blocks from start to end.

        Parameters:
            location (str): Location name or postal code.
            start_moment (datetime): Start of the period, timezone aware.
            end_moment (datetime): End of the period, at most MAX_RAIN_OUTLOOK after the start.

        Returns:
            Dict[str, Any]: The resolved point, one row per block with the rain chance, the median
                rainfall and the heaviest hour's 90th percentile, and the model run.
        """
        _check_period_order(start_moment, end_moment)
        if end_moment - start_moment > MAX_RAIN_OUTLOOK:
            max_hours = int(MAX_RAIN_OUTLOOK.total_seconds() // 3600)
            raise ValueError(f"A rain outlook covers at most {max_hours} hours; ask for a shorter period.")

        point = await self._find_point(location)
        chance_series, median_series, upper_series = await asyncio.gather(
            self._read_series(parameters.PRECIPITATION_PROBABILITY, point),
            self._read_series(parameters.PRECIPITATION_3H, point),
            self._read_series(parameters.PRECIPITATION_Q90, point),
        )

        blocks = []
        block_start = _find_closing_stamp(start_moment)
        last_stamp = _find_closing_stamp(end_moment)
        while block_start < last_stamp:
            block_end = block_start + RAIN_BLOCK
            # The 3-hour values sit on the block's end, the hourly ones on each of its three hours
            hour_stamps = [block_end - timedelta(hours=hours_before) for hours_before in (2, 1, 0)]
            if (block_end not in chance_series.values or block_end not in median_series.values
                    or any(stamp not in upper_series.values for stamp in hour_stamps)):
                raise ValueError(
                    f"{_format_swiss_time(start_moment)} to {_format_swiss_time(end_moment)} is not fully covered "
                    f"by the forecast for {point.display_name}. {_describe_covered_range(median_series)}"
                )
            heaviest_hour_mm = max(upper_series.values[stamp] for stamp in hour_stamps)
            blocks.append({
                "from": _format_swiss_time(block_start),
                "to": _format_swiss_time(block_end),
                "rain_chance_percent": chance_series.values[block_end],
                "rainfall_median_mm": round(median_series.values[block_end], 1),
                "heaviest_hour_up_to_mm": round(heaviest_hour_mm, 1),
            })
            block_start = block_end

        return {
            "location": point.display_name,
            "altitude_m": point.altitude_m,
            "blocks": blocks,
            "model_run": _format_swiss_time(median_series.run_time),
        }

    async def _read_series_if_published(self, parameter: str, point: ForecastPoint) -> Optional[ForecastSeries]:
        """Read one parameter for one point, or return None when MeteoSwiss does not publish it there."""
        try:
            return await self._read_series(parameter, point)
        except ValueError as error:
            logger.info(f"No daily {parameter} for {point.display_name}: {error}")
            return None

    async def _read_daily_series(self, point: ForecastPoint) -> Dict[str, Optional[ForecastSeries]]:
        """Read every daily parameter for one point, keyed by answer field; None where it is not published."""
        series_reads = [self._read_series_if_published(parameter, point) for _, parameter in DAILY_PARAMETERS]
        all_series = await asyncio.gather(*series_reads)
        series_by_field: Dict[str, Optional[ForecastSeries]] = {}
        for (field, _), series in zip(DAILY_PARAMETERS, all_series):
            series_by_field[field] = series
        return series_by_field

    async def read_daily_forecast(self, location: str, first_day: date, days: int) -> Dict[str, Any]:
        """
        Read the whole-day forecast for one or several days in a row, one row per day.

        Parameters:
            location (str): Location name or postal code.
            first_day (date): The first Swiss calendar day.
            days (int): How many days, from 1 to MAX_DAYS.

        Returns:
            Dict[str, Any]: The resolved point, one row per day with published values (date,
                weekday, minimum and maximum temperature, rainfall median and 10th and 90th
                percentile, weather in words with its emoji), and the model run. Fields MeteoSwiss
                does not publish for this location are None.
        """
        if not 1 <= days <= MAX_DAYS:
            raise ValueError(f"days must be between 1 and {MAX_DAYS}, not {days}.")

        point = await self._find_point(location)
        series_by_field = await self._read_daily_series(point)
        rows = []
        for offset in range(days):
            day = first_day + timedelta(days=offset)
            day_values = _build_day(series_by_field, day)
            if day_values is None:
                continue
            row: Dict[str, Any] = {"date": day.isoformat(), "weekday": f"{day:%A}"}
            row.update(day_values)
            rows.append(row)

        run_time = _find_run_time(series_by_field)
        if not rows or run_time is None:
            raise ValueError(f"MeteoSwiss has no daily forecast for {point.display_name} from {first_day.isoformat()}.")
        return {
            "location": point.display_name,
            "altitude_m": point.altitude_m,
            "days": rows,
            "model_run": _format_swiss_time(run_time),
        }
