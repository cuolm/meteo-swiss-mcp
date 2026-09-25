import argparse
import functools
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

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
    """The MCP server: registers the weather tools and runs them over stdio or streamable HTTP."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.host = args.host
        self.port = args.port
        self.transport = args.transport
        self.mcp = MCPServer(
            name="swiss_weather_mcp_server",
            instructions="This MCP server provides hourly weather forecast data for Switzerland for today and the next 8 days.",
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
        async def daily_forecast(location: str, start_date: str, days: int = 1) -> dict:
            """
            Get the whole-day forecast for a location, for one day or several days in a row: the
            cheapest way to answer "how is the weather" for a day, the weekend or the week.

            Each day covers a Swiss calendar day, 00:00 to 24:00 local time. The minimum and maximum
            are the lowest and highest hourly mean temperature of that day, and the weather words
            describe the daytime. Always use this for the rain of a whole day.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                start_date (str): The first Swiss calendar day in ISO 8601, e.g. "2026-09-23". Today or up to 8 days ahead.
                days (int): How many days in a row, from 1 to 9. Default 1.

            Returns:
                dict: The resolved location with its altitude, one row per day, and the model run.
                    Each row has the date and weekday, the minimum and maximum temperature in
                    Celsius, the day's rainfall in millimetres as its median and its 10th and 90th
                    percentile (in 8 of 10 possible outcomes, the rain lies between the two
                    percentiles), and the weather in words with a matching emoji. A value is None
                    when MeteoSwiss does not publish it for that location.

            Examples:
                daily_forecast("Zurich", "2026-09-23")
                daily_forecast("Lugano", "2026-09-26", 2)   # a weekend
            """
            first_day = _parse_swiss_time(start_date).astimezone(SWISS_TZ).date()
            return await self.forecast_service.read_daily_forecast(location, first_day, days)

        @self.mcp.tool()
        @_handle_tool_call
        async def hourly_forecast(location: str, start: str, end: Optional[str] = None) -> dict:
            """
            Get the weather hour by hour for a location: temperature, chance of rain and the weather
            in words.

            Use this for one hour, such as "how warm is it at 15:00", or part of a day, such as "how
            is the afternoon" or "when is the best time for a walk". Without end it returns the one
            hour starting at start. Each row covers the hour from its "from" to its "to": the
            temperature is the mean of that hour, while the rain chance and the weather cover the 3
            hours up to "to", as MeteoSwiss publishes them only per 3 hours. For whole days use
            daily_forecast, for rain amounts rain_outlook.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                start (str): Swiss local time in ISO 8601 without offset, e.g. "2026-09-23T15:00". Today or up to 8 days ahead.
                end (str): Optional. Swiss local time in ISO 8601 without offset, at most 24 hours after start.

            Returns:
                dict: The resolved location with its altitude, one row per hour (from, to,
                    temperature in Celsius, rain chance in percent, weather in words with a matching
                    emoji), and the model run.

            Examples:
                hourly_forecast("Zurich", "2026-09-23T15:00")                        # one hour
                hourly_forecast("Zurich", "2026-09-23T12:00", "2026-09-23T18:00")    # the afternoon
            """
            start_moment = _parse_swiss_time(start)
            end_moment = _parse_swiss_time(end) if end is not None else None
            return await self.forecast_service.read_hourly_forecast(location, start_moment, end_moment)

        @self.mcp.tool()
        @_handle_tool_call
        async def rain_outlook(location: str, start: str, end: str) -> dict:
            """
            Get when and how much it may rain at a location, in 3-hour blocks over a period.

            Use this for "will it rain this afternoon", "do I need an umbrella" or "when does the
            rain stop". Each block gives the chance of rain, the median rainfall of the 3 hours, and
            how much the wettest hour of the block may bring (its 90th percentile). The median is
            often 0 when showers are possible, so read the chance and the "up to" amount as well.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                start (str): Swiss local time the period starts, ISO 8601 without offset, e.g. "2026-09-23T12:00".
                end (str): Swiss local time the period ends, at most 48 hours after start.

            Returns:
                dict: The resolved location with its altitude, one row per 3-hour block (from, to,
                    rain chance in percent, median rainfall and wettest hour's "up to" amount in
                    millimetres), and the model run.

            Examples:
                rain_outlook("Zurich", "2026-09-23T12:00", "2026-09-24T00:00")
                rain_outlook("Lugano", "2026-09-24T06:00", "2026-09-24T18:00")
            """
            start_moment = _parse_swiss_time(start)
            end_moment = _parse_swiss_time(end)
            return await self.forecast_service.read_rain_outlook(location, start_moment, end_moment)

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
        async def wind(location: str, when: str) -> dict:
            """
            Get the wind at a location at a specific time: mean speed, strongest gust and direction.

            The speed is the mean over the hour up to that time, and the gust the strongest
            one-second gust in that hour, which is what makes wind hazardous. The direction is where
            the wind blows from, in degrees clockwise from north (0 is a north wind, 180 a south
            wind) and as a compass point.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                when (str): Swiss local time in ISO 8601 without offset, e.g. "2026-09-23T14:00". Today or up to 8 days ahead.

            Returns:
                dict: Wind speed and gust in kilometres per hour, the direction in degrees and as a
                    compass point such as "SW", the resolved location with its altitude, the time it
                    is valid for, and the model run.

            Examples:
                wind("Zurich", "2026-09-23T14:00")
                wind("Säntis", "2026-09-24T12:00")
            """
            moment = _parse_swiss_time(when)
            return await self.forecast_service.read_wind(location, moment)

        @self.mcp.tool()
        @_handle_tool_call
        async def total_cloud_cover(location: str, when: str) -> dict:
            """
            Get how cloudy it is at a location at a specific time.

            MeteoSwiss publishes low, medium and high cloud separately, and they overlap, so the
            total is an estimate that assumes the layers are independent. All three layers are
            returned as well, which tells low fog apart from thin high cloud.

            When a number is not needed, hourly_forecast answers "how cloudy" in plain words.

            Args:
                location (str): Location name (e.g., "Zurich") or Swiss postal code (e.g., "8001").
                when (str): Swiss local time in ISO 8601 without offset, e.g. "2026-09-23T14:00". Today or up to 8 days ahead.

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
                when (str): Swiss local time in ISO 8601 without offset, e.g. "2026-09-23T14:00". Today or up to 8 days ahead.

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
        """Serve the tools over the transport given on the command line."""
        if self.transport == "stdio":
            logger.info("Running server with stdio transport")
            self.mcp.run(transport="stdio")
        else:
            logger.info("Running server with Streamable HTTP transport")
            self.mcp.run(transport="streamable-http", host=self.host, port=self.port, stateless_http=True)


def main():
    """Start the MCP server from the command line."""
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
