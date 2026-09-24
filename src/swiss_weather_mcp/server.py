import argparse
import functools
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict

import requests
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from platformdirs import user_cache_path

from . import LOG_LEVELS, setup_logging
from .forecast import ForecastService
from .meteoswiss import SWISS_TZ, LocalForecastSource

logger = logging.getLogger(__name__)

# Shared, OS-standard cache location (survives across working directories the server may be launched from).
# Override with SWISS_WEATHER_MCP_CACHE_DIR, e.g. to isolate cache location in tests or Docker.
CACHE_DIR = Path(os.environ.get("SWISS_WEATHER_MCP_CACHE_DIR", user_cache_path("swiss-weather-mcp")))

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="Transport layer for MCP server (default: stdio)",
    )
    parser.add_argument("--host", default="localhost", help="Server host (used only for HTTP)")
    parser.add_argument("--port", type=int, default=8050, help="Server port (used only for HTTP)")
    parser.add_argument(
        "--cache-all-locations",
        action="store_true",
        help="Keep the whole published file (about 31 MB per parameter) instead of only the rows "
             "for the requested location, so questions about further locations need no download",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging level for the server (default: INFO)",
    )
    return parser.parse_args()

def _parse_swiss_time(timestamp: str) -> datetime:
    """
    Read an ISO 8601 timestamp or date. Without an offset it is read as Swiss local time, and an
    explicit offset is kept.

    Parameters:
        timestamp (str): ISO timestamp, e.g. "2026-09-23T14:00", or a date, e.g. "2026-09-23".

    Returns:
        datetime: The same instant, timezone aware.
    """
    try:
        moment = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise ValueError(
            f"'{timestamp}' is not a valid timestamp, use for example '2026-09-23T14:00' or '2026-09-23'"
        ) from error
    return moment if moment.tzinfo else moment.replace(tzinfo=SWISS_TZ)


def _handle_tool_call(tool: Callable[..., Awaitable[Dict[str, Any]]]) -> Callable[..., Awaitable[Dict[str, Any]]]:
    """
    Log each call of a tool, and turn the failures the model can act on into a ToolError.

    A ValueError (a place or time the forecast cannot answer) and a request error (MeteoSwiss out
    of reach) become a ToolError, whose message mcp shows to the model, and are logged as one
    warning line. Every other failure stays a crash, which mcp hides from the model and logs with
    its traceback.

    Parameters:
        tool (Callable): The tool function, called with its arguments as keywords.

    Returns:
        Callable: The same tool, with its calls logged and its failures handled as above.
    """
    @functools.wraps(tool)
    async def run_tool(**arguments: Any) -> Dict[str, Any]:
        try:
            answer = await tool(**arguments)
        except ValueError as error:
            logger.warning(f"{tool.__name__}: {error}")
            raise ToolError(str(error)) from error
        except requests.RequestException as error:
            logger.warning(f"{tool.__name__}: could not reach MeteoSwiss: {error}")
            raise ToolError(f"Could not reach MeteoSwiss, try again later: {error}") from error

        logger.info(f"{tool.__name__}: {arguments} -> {answer}")
        return answer

    return run_tool


class SwissWeatherMCPServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.host = args.host
        self.port = args.port
        self.transport = args.transport
        self.mcp = MCPServer(
            name="swiss_weather_mcp_server",
            instructions="This MCP server provides hourly weather forecast data for Switzerland for up to 9 days ahead.",
            log_level=args.log_level,  # forwarded to uvicorn, which configures its own loggers
        )

        forecast_source = LocalForecastSource(CACHE_DIR, cache_all_locations=args.cache_all_locations)
        self.forecast_service = ForecastService(forecast_source)
        self._register_tools()

    def _register_tools(self) -> None:
        @self.mcp.tool()
        def current_date_and_time() -> str:
            """
            Get the current weekday and Swiss local time.

            Call this first when the question is relative, such as "tomorrow" or "tonight", because
            the forecast tools take a real date rather than an offset. The time is written the way
            the tools read it, ISO 8601 without an offset, so it can be sent back to them as it is.

            Returns:
                str: For example "Today is Wednesday, 2026-09-23T13:02 (Swiss time)".
            """
            now = datetime.now(SWISS_TZ)
            return f"Today is {now:%A}, {now:%Y-%m-%dT%H:%M} (Swiss time)"

        @self.mcp.tool()
        @_handle_tool_call
        async def daily_forecast(location: str, date: str) -> dict:
            """
            Get the whole-day summary for a location: the cheapest way to answer "how is the weather".

            Prefer this over several hourly tools when the question is about a day rather than an
            hour, and always for the rain of a whole day. The values cover a Swiss calendar day,
            00:00 to 24:00 local time. The minimum and maximum are the lowest and highest hourly
            mean temperature of that day, and the weather words describe the daytime.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                    Must be a place MeteoSwiss publishes forecasts for.
                date (str): The Swiss calendar day in ISO 8601, e.g. "2026-09-23". Up to 9 days ahead.

            Returns:
                dict: Minimum and maximum temperature in Celsius; the day's rainfall in millimetres
                    as its median and its 10th and 90th percentile, meaning a 90% chance of at least
                    the 10th-percentile amount and at most the 90th-percentile amount; a worded
                    weather summary, the resolved location with its altitude, and the model run.
                    A value is None when MeteoSwiss does not publish it for that location.

            Examples:
                daily_forecast("Zurich", "2026-09-23")
                daily_forecast("8001", "2026-09-25")
            """
            day = _parse_swiss_time(date).astimezone(SWISS_TZ).date()
            return await self.forecast_service.read_daily_forecast(location, day)

        @self.mcp.tool()
        @_handle_tool_call
        async def weather_description(location: str, when: str) -> dict:
            """
            Get the weather in words for a location at a specific time.

            The description covers the three hours up to the given time, and reads like
            "mostly sunny, some clouds" or "very cloudy, light rain".

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                when (str): Swiss local time in ISO 8601 without offset, e.g. "2026-09-23T14:00". Up to 9 days ahead.

            Returns:
                dict: The description, the MeteoSwiss pictogram code behind it, the resolved
                    location with its altitude, the time it is valid for, and the model run.

            Examples:
                weather_description("Zurich", "2026-09-23T14:00")
                weather_description("Davos", "2026-09-24T08:00")
            """
            moment = _parse_swiss_time(when)
            return await self.forecast_service.read_weather_description(location, moment)

        @self.mcp.tool()
        @_handle_tool_call
        async def temperature(location: str, when: str) -> dict:
            """
            Get the air temperature for a location at a specific time.

            This is the mean over the hour up to that time, 2 metres above ground, so 14:00 means
            13:00 to 14:00. For a day's highest and lowest temperature use daily_forecast instead.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                when (str): Swiss local time in ISO 8601 without offset, e.g. "2026-09-23T14:00". Up to 9 days ahead.

            Returns:
                dict: Temperature in Celsius, the resolved location with its altitude, the time it
                    is valid for, and the model run.

            Examples:
                temperature("Zurich", "2026-09-23T14:00")
                temperature("Zermatt", "2026-09-25T07:00")
            """
            moment = _parse_swiss_time(when)
            return await self.forecast_service.read_temperature(location, moment)

        @self.mcp.tool()
        @_handle_tool_call
        async def total_rainfall(location: str, start: str, end: str) -> dict:
            """
            Get the total rainfall for a location over a period.

            The hourly amounts are added up over exactly the hours from start to end. Each is the
            most likely amount for its hour, so when showers are possible but unlikely in any single
            hour, the sum stays 0 even if the day as a whole is expected to be wet. For the rain of a
            whole day use daily_forecast, and for whether it rains at all precipitation_probability.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                start (str): Swiss local time the period starts, ISO 8601 without offset, e.g. "2026-09-23T06:00".
                end (str): Swiss local time the period ends, ISO 8601 without offset, e.g. "2026-09-23T18:00".

            Returns:
                dict: Rainfall in millimetres, the resolved location with its altitude, the period,
                    and the model run.

            Examples:
                total_rainfall("Zurich", "2026-09-23T00:00", "2026-09-24T00:00")   # the whole day
                total_rainfall("Zurich", "2026-09-23T06:00", "2026-09-23T12:00")   # the morning
            """
            start_moment = _parse_swiss_time(start)
            end_moment = _parse_swiss_time(end)
            return await self.forecast_service.read_total_rainfall(location, start_moment, end_moment)

        @self.mcp.tool()
        @_handle_tool_call
        async def sunshine_hours(location: str, start: str, end: str) -> dict:
            """
            Get the sunshine hours for a location over a period.

            MeteoSwiss publishes sunshine as minutes per hour, which are added up over the period
            and reported as hours.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                start (str): Swiss local time the period starts, ISO 8601 without offset, e.g. "2026-09-23T06:00".
                end (str): Swiss local time the period ends, ISO 8601 without offset, e.g. "2026-09-23T18:00".

            Returns:
                dict: Sunshine in hours, the resolved location with its altitude, the period, and
                    the model run.

            Examples:
                sunshine_hours("Zurich", "2026-09-23T00:00", "2026-09-24T00:00")   # the whole day
                sunshine_hours("Zurich", "2026-09-23T12:00", "2026-09-23T18:00")   # the afternoon
            """
            start_moment = _parse_swiss_time(start)
            end_moment = _parse_swiss_time(end)
            return await self.forecast_service.read_sunshine_hours(location, start_moment, end_moment)

        @self.mcp.tool()
        @_handle_tool_call
        async def precipitation_probability(location: str, when: str) -> dict:
            """
            Get how likely rain is for a location at a specific time.

            The probability covers the three hours up to that time, not a single instant. Use this
            for "will it rain", and total_rainfall for "how much".

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                when (str): Swiss local time in ISO 8601 without offset, e.g. "2026-09-23T14:00". Up to 9 days ahead.

            Returns:
                dict: Probability in percent, the resolved location with its altitude, the time it
                    is valid for, and the model run.

            Examples:
                precipitation_probability("Zurich", "2026-09-23T14:00")
                precipitation_probability("Lugano", "2026-09-24T18:00")
            """
            moment = _parse_swiss_time(when)
            return await self.forecast_service.read_precipitation_probability(location, moment)

        @self.mcp.tool()
        @_handle_tool_call
        async def precipitation_rate(location: str, when: str) -> dict:
            """
            Get how much rain falls at a location during one hour.

            This is the most likely amount in the hour up to that time, so 14:00 means 13:00 to
            14:00. For whether it rains at all use precipitation_probability, and for a whole day
            daily_forecast.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                when (str): Swiss local time in ISO 8601 without offset, e.g. "2026-09-23T14:00". Up to 9 days ahead.

            Returns:
                dict: Rainfall in millimetres per hour, the resolved location with its altitude, the
                    time it is valid for, and the model run.

            Examples:
                precipitation_rate("Zurich", "2026-09-23T14:00")
                precipitation_rate("Lugano", "2026-09-24T18:00")
            """
            moment = _parse_swiss_time(when)
            return await self.forecast_service.read_precipitation_rate(location, moment)

        @self.mcp.tool()
        @_handle_tool_call
        async def wind_speed(location: str, when: str) -> dict:
            """
            Get the wind speed for a location at a specific time.

            This is the mean over the hour up to that time. For the strongest gusts use wind_gusts
            instead, which is what matters for whether the wind is dangerous.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                when (str): Swiss local time in ISO 8601 without offset, e.g. "2026-09-23T14:00". Up to 9 days ahead.

            Returns:
                dict: Wind speed in kilometres per hour, the resolved location with its altitude,
                    the time it is valid for, and the model run.

            Examples:
                wind_speed("Zurich", "2026-09-23T14:00")
                wind_speed("Säntis", "2026-09-24T12:00")
            """
            moment = _parse_swiss_time(when)
            return await self.forecast_service.read_wind_speed(location, moment)

        @self.mcp.tool()
        @_handle_tool_call
        async def wind_gusts(location: str, when: str) -> dict:
            """
            Get the strongest wind gust expected at a location during one hour.

            This is the peak one second gust within the hour up to that time, which is usually
            much higher than the mean wind speed and is what makes wind hazardous.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                when (str): Swiss local time in ISO 8601 without offset, e.g. "2026-09-23T14:00". Up to 9 days ahead.

            Returns:
                dict: Gust speed in kilometres per hour, the resolved location with its altitude,
                    the time it is valid for, and the model run.

            Examples:
                wind_gusts("Zurich", "2026-09-23T14:00")
                wind_gusts("Jungfraujoch", "2026-09-24T12:00")
            """
            moment = _parse_swiss_time(when)
            return await self.forecast_service.read_wind_gusts(location, moment)

        @self.mcp.tool()
        @_handle_tool_call
        async def wind_direction(location: str, when: str) -> dict:
            """
            Get the direction the wind blows from at a location at a specific time.

            Reported as the mean over the hour up to that time, in degrees clockwise from north, so
            0 is a north wind and 180 a south wind.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                when (str): Swiss local time in ISO 8601 without offset, e.g. "2026-09-23T14:00". Up to 9 days ahead.

            Returns:
                dict: Direction in degrees and as a compass point such as "SW", the resolved
                    location with its altitude, the time it is valid for, and the model run.

            Examples:
                wind_direction("Zurich", "2026-09-23T14:00")
                wind_direction("Altdorf", "2026-09-24T12:00")
            """
            moment = _parse_swiss_time(when)
            return await self.forecast_service.read_wind_direction(location, moment)

        @self.mcp.tool()
        @_handle_tool_call
        async def total_cloud_cover(location: str, when: str) -> dict:
            """
            Get how cloudy it is at a location at a specific time.

            MeteoSwiss publishes low, medium and high cloud separately, and they overlap, so the
            total is an estimate that assumes the layers are independent. All three layers are
            returned as well, which tells low fog apart from thin high cloud.

            This reads three files, so it is the most expensive tool. When a number is not needed,
            weather_description answers "how cloudy" more cheaply and in plain words.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                when (str): Swiss local time in ISO 8601 without offset, e.g. "2026-09-23T14:00". Up to 9 days ahead.

            Returns:
                dict: Estimated total cloud cover in percent, the low, medium and high layers in
                    percent, the resolved location with its altitude, the time it is valid for,
                    and the model run.

            Examples:
                total_cloud_cover("Zurich", "2026-09-23T14:00")
                total_cloud_cover("Locarno", "2026-09-24T09:00")
            """
            moment = _parse_swiss_time(when)
            return await self.forecast_service.read_total_cloud_cover(location, moment)

        @self.mcp.tool()
        @_handle_tool_call
        async def freezing_level(location: str, when: str) -> dict:
            """
            Get the height of the 0 degree line at a location at a specific time.

            This is the height where the air is at 0 degrees. Snow usually reaches somewhat below it,
            so it answers questions about snow in the mountains, such as how high a ski area has to be.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                when (str): Swiss local time in ISO 8601 without offset, e.g. "2026-09-23T14:00". Up to 9 days ahead.

            Returns:
                dict: The freezing level in metres above sea level, the resolved location with its
                    altitude, the time it is valid for, and the model run.

            Examples:
                freezing_level("Zermatt", "2026-09-23T14:00")
                freezing_level("Davos", "2026-09-25T06:00")
            """
            moment = _parse_swiss_time(when)
            return await self.forecast_service.read_freezing_level(location, moment)

    def run(self):
        if self.transport == "stdio":
            logger.info("Running server with stdio transport")
            self.mcp.run(transport="stdio")
        elif self.transport == "streamable-http":
            logger.info("Running server with Streamable HTTP transport")
            self.mcp.run(transport="streamable-http", host=self.host, port=self.port, stateless_http=True)
        else:
            raise ValueError(f"Unknown transport: {self.transport}")


def main():
    try:
        args = _parse_args()
        setup_logging(args.log_level)
        server = SwissWeatherMCPServer(args)
        server.run()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down.")
    except Exception:
        logger.exception("Fatal error in MCP server")
        sys.exit(1)

if __name__ == "__main__":
    main()
