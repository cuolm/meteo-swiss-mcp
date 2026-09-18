import argparse
import logging
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from . import LOG_LEVELS, setup_logging
from .predictions import MeteoSwissPredictions

# Load environment variables from a .env file found by searching upwards from the current working directory
load_dotenv()

logger = logging.getLogger(__name__)

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
        "--log-level",
        type=str,
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging level for the server (default: INFO)",
    )
    return parser.parse_args()

def _lead_time_swiss_to_utc(lead_time_swiss: int) -> int:
    """
    Convert a lead time expressed in Swiss local hours since Swiss midnight
    into the equivalent lead time in hours since today's 00:00 UTC.

    The forecast API counts lead times from 00:00 UTC (MeteoSwissPredictions passes it
    as ref_time), so the Swiss lead time has to be re-expressed against that anchor.

    Handles DST (daylight saving time) transitions correctly by using timezone-aware datetimes.
    """
    if lead_time_swiss < 0:
        raise ValueError(f"lead_time_swiss must be a non-negative value, got {lead_time_swiss}")

    now_swiss_datetime = datetime.now(ZoneInfo("Europe/Zurich"))

    # Midnight in Swiss local time (today)
    midnight_swiss_datetime = now_swiss_datetime.replace(hour=0, minute=0, second=0, microsecond=0)

    # Target time in Swiss local time
    target_time_swiss_datetime = midnight_swiss_datetime + timedelta(hours=lead_time_swiss)

    # Midnight in UTC (today), the anchor the forecast API counts lead times from
    midnight_utc_datetime = now_swiss_datetime.astimezone(ZoneInfo("UTC")).replace(hour=0, minute=0, second=0, microsecond=0)

    # Target time as a UTC instant
    target_time_utc_datetime = target_time_swiss_datetime.astimezone(ZoneInfo("UTC"))

    # Compute difference in hours
    lead_time_utc = (target_time_utc_datetime - midnight_utc_datetime).total_seconds() // 3600
    return int(lead_time_utc)


class MeteoSwissMCPServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.host = args.host
        self.port = args.port
        self.transport = args.transport
        self.mcp = FastMCP(
            name="meteo_swiss_mcp_server",
            instructions="This MCP server provides hourly weather forecast data for Switzerland for up to 5 days ahead.",
            host=self.host,
            port=self.port,
            stateless_http=True,
        )

        self.meteo = MeteoSwissPredictions()
        self._register_tools()

    def _register_tools(self) -> None:
        @self.mcp.tool()
        def current_date_and_time() -> str:
            """
            Get a human-readable string of the current time, weekday and date.

            Returns:
                str: A string in the format "Today is <weekday> <day>.<month>.<year> <hour>:<minute>:<second>"
            """
            switzerland = ZoneInfo("Europe/Zurich")
            now_in_ch = datetime.now(switzerland)
            formatted_date_time = now_in_ch.strftime("%A %d.%m.%Y %H:%M:%S")
            return f"Today is {formatted_date_time}"

        @self.mcp.tool()
        async def total_rainfall(location: str, lead_time_start_swiss: int, lead_time_end_swiss: int) -> float:
            """
            Get total rainfall for a location and offset period.

            Args:
                location (str): Location name (e.g., "Zurich").
                lead_time_start_swiss (int): Start hour offset from today at 00:00 Swiss local time. Min offset time is 2 hours.
                lead_time_end_swiss (int): End hour offset from today at 00:00 Swiss local time. Max offset time is 121 hours.

            Returns:
                float: Total precipitation accumulation in millimeters for the given period.

            Examples:
                total_rainfall("Zurich", 2, 24)    # Total rainfall today
                total_rainfall("Zurich", 24, 48)   # Total rainfall tomorrow
                total_rainfall("Zurich", 24, 30)   # Total rainfall tonight
                total_rainfall("Zurich", 30, 36)   # Total rainfall tomorrow morning
                total_rainfall("Zurich", 36, 42)   # Total rainfall tomorrow afternoon
                total_rainfall("Zurich", 42, 48)   # Total rainfall tomorrow evening
            """
            try:
                if lead_time_start_swiss >= lead_time_end_swiss:
                    raise ValueError(
                        f"lead_time_start_swiss must be less than lead_time_end_swiss, "
                        f"got lead_time_start_swiss={lead_time_start_swiss}, lead_time_end_swiss={lead_time_end_swiss}"
                    )
                lead_time_start_utc = _lead_time_swiss_to_utc(lead_time_start_swiss)
                lead_time_end_utc = _lead_time_swiss_to_utc(lead_time_end_swiss)
                result = await self.meteo.total_rainfall_for_location(
                    location,
                    lead_time_start_utc,
                    lead_time_end_utc
                )
                logger.info(f"total_rainfall: location={location}, lead_time_start_swiss={lead_time_start_swiss}, lead_time_end_swiss={lead_time_end_swiss}, result={result}")
                return result
            except Exception as e:
                logger.exception(f"Failed to get total rainfall for location '{location}': {e}")
                raise RuntimeError(f"Failed to get total rainfall for location '{location}': {e}") from e

        @self.mcp.tool()
        async def sunshine_hours(location: str, lead_time_start_swiss: int, lead_time_end_swiss: int) -> float:
            """
            Get sunshine hours for a location and offset period.

            Args:
                location (str): Location name (e.g., "Zurich").
                lead_time_start_swiss (int): Start hour offset from today at 00:00 Swiss local time. Min offset time is 2 hours.
                lead_time_end_swiss (int): End hour offset from today at 00:00 Swiss local time. Max offset time is 121 hours.

            Returns:
                float: Predicted sunshine hours for the specified period.

            Examples:
                sunshine_hours("Zurich", 2, 24)    # Total sunshine hours today
                sunshine_hours("Zurich", 24, 48)   # Total sunshine hours tomorrow
                sunshine_hours("Zurich", 30, 36)   # Total sunshine hours tomorrow morning
                sunshine_hours("Zurich", 36, 42)   # Total sunshine hours tomorrow afternoon
                sunshine_hours("Zurich", 42, 48)   # Total sunshine hours tomorrow evening
            """
            try:
                if lead_time_start_swiss >= lead_time_end_swiss:
                    raise ValueError(
                        f"lead_time_start_swiss must be less than lead_time_end_swiss, "
                        f"got lead_time_start_swiss={lead_time_start_swiss}, lead_time_end_swiss={lead_time_end_swiss}"
                    )
                lead_time_start_utc = _lead_time_swiss_to_utc(lead_time_start_swiss)
                lead_time_end_utc = _lead_time_swiss_to_utc(lead_time_end_swiss)
                result = await self.meteo.sunshine_hours_for_location(
                    location,
                    lead_time_start_utc, 
                    lead_time_end_utc 
                )
                logger.info(f"sunshine_hours: location={location}, lead_time_start_swiss={lead_time_start_swiss}, lead_time_end_swiss={lead_time_end_swiss}, result={result}")
                return result
            except Exception as e:
                logger.exception(f"Failed to get sunshine hours for location '{location}': {e}")
                raise RuntimeError(f"Failed to get sunshine hours for location '{location}': {e}") from e

        @self.mcp.tool()
        async def temperature(location: str, lead_time_swiss: int) -> float:
            """
            Get air temperature for a location at a specific offset time.

            Args:
                location (str): Location name (e.g., "Zurich").
                lead_time_swiss (int): Hour offset from today at 00:00 Swiss local time. Min offset time is 2 hours. Max offset time is 121 hours.

            Returns:
                float: Maximum air temperature in Celsius for the given lead time.

            Examples:
                temperature("Zurich", 2)    # Temperature at 02:00 Swiss local time today
                temperature("Zurich", 14)   # Temperature at 14:00 Swiss local time today
                temperature("Zurich", 36)   # Temperature at 12:00 Swiss local time tomorrow
                temperature("Zurich", 113)  # Temperature at 17:00 Swiss local time in 4 days
            """
            try:
                lead_time_utc = _lead_time_swiss_to_utc(lead_time_swiss)
                result = await self.meteo.temp_for_location(location, lead_time_utc)
                logger.info(f"temperature: location={location}, offset={lead_time_swiss}, result={result}")
                return result
            except Exception as e:
                logger.exception(f"Failed to get temperature for location '{location}': {e}")
                raise RuntimeError(f"Failed to get temperature for location '{location}': {e}") from e

        @self.mcp.tool()
        async def wind_speed(location: str, lead_time_swiss: int) -> float:
            """
            Get predicted wind speed for a location at a specific offset time.

            Args:
                location (str): Location name (e.g., "Zurich").
                lead_time_swiss (int): Hour offset from today at 00:00 Swiss local time. Min offset time is 2 hours. Max offset time is 121 hours.

            Returns:
                float: Predicted wind speed in meters per second.

            Examples:
                wind_speed("Zurich", 2)    # Wind speed at 02:00 Swiss local time today
                wind_speed("Zurich", 14)   # Wind speed at 14:00 Swiss local time today
                wind_speed("Zurich", 36)   # Wind speed at 12:00 Swiss local time tomorrow
                wind_speed("Zurich", 113)  # Wind speed at 17:00 Swiss local time in 4 days
            """
            try:
                lead_time_utc = _lead_time_swiss_to_utc(lead_time_swiss)
                result = await self.meteo.wind_speed_for_location(location, lead_time_utc)
                logger.info(f"wind_speed: location={location}, lead_time_swiss={lead_time_swiss}, result={result}")
                return result
            except Exception as e:
                logger.exception(f"Failed to get wind speed for location '{location}': {e}")
                raise RuntimeError(f"Failed to get wind speed for location '{location}': {e}") from e

        @self.mcp.tool()
        async def pressure_msl(location: str, lead_time_swiss: int) -> float:
            """
            Get sea-level pressure for a location at a specific offset time.

            Args:
                location (str): Location name (e.g., "Zurich").
                lead_time_swiss (int): Hour offset from today at 00:00 Swiss local time. Min offset time is 2 hours. Max offset time is 121 hours.

            Returns:
                float: Sea-level pressure in Pascals (Pa).

            Examples:
                pressure_msl("Zurich", 2)    # Pressure at 02:00 Swiss local time today
                pressure_msl("Zurich", 14)   # Pressure at 14:00 Swiss local time today
                pressure_msl("Zurich", 36)   # Pressure at 12:00 Swiss local time tomorrow
                pressure_msl("Zurich", 113)  # Pressure at 17:00 Swiss local time in 4 days
            """
            try:
                lead_time_utc = _lead_time_swiss_to_utc(lead_time_swiss)
                result = await self.meteo.pressure_msl_for_location(location, lead_time_utc) 
                logger.info(f"pressure_msl: location={location}, lead_time_swiss={lead_time_swiss}, result={result}")
                return result  
            except Exception as e:
                logger.exception(f"Failed to get pressure for location '{location}': {e}")
                raise RuntimeError(f"Failed to get pressure for location '{location}': {e}") from e

        @self.mcp.tool()
        async def total_cloud_cover(location: str, lead_time_swiss: int) -> float:
            """
            Get total cloud cover percentage for a location at a specific offset time.

            Args:
                location (str): Location name (e.g., "Zurich").
                lead_time_swiss (int): Hour offset from today at 00:00 Swiss local time. Min offset time is 2 hours. Max offset time is 121 hours.

            Returns:
                float: Total cloud cover percentage.

            Examples:
                total_cloud_cover("Zurich", 2)    # Cloud cover at 02:00 Swiss local time today
                total_cloud_cover("Zurich", 14)   # Cloud cover at 14:00 Swiss local time today
                total_cloud_cover("Zurich", 36)   # Cloud cover at 12:00 Swiss local time tomorrow
                total_cloud_cover("Zurich", 113)  # Cloud cover at 17:00 Swiss local time in 4 days
            """
            try:
                lead_time_utc = _lead_time_swiss_to_utc(lead_time_swiss)
                result = await self.meteo.total_cloud_cover_for_location(location, lead_time_utc) 
                logger.info(f"total_cloud_cover: location={location}, lead_time_swiss={lead_time_swiss}, result={result}")
                return result
            except Exception as e:
                logger.exception(f"Failed to get total cloud cover for location '{location}': {e}")
                raise RuntimeError(f"Failed to get total cloud cover for location '{location}': {e}") from e

        @self.mcp.tool()
        async def snow_depth(location: str, lead_time_swiss: int) -> float:
            """
            Get forecasted snow depth for a location at a specific offset time.

            Args:
                location (str): Location name (e.g., "Zurich").
                lead_time_swiss (int): Hour offset from today at 00:00 Swiss local time. Min offset time is 2 hours. Max offset time is 121 hours.

            Returns:
                float: Forecasted snow depth in meters.

            Examples:
                snow_depth("Zurich", 2)    # Snow depth at 02:00 Swiss local time today
                snow_depth("Zurich", 14)   # Snow depth at 14:00 Swiss local time today
                snow_depth("Zurich", 36)   # Snow depth at 12:00 Swiss local time tomorrow
                snow_depth("Zurich", 113)  # Snow depth at 17:00 Swiss local time in 4 days
            """
            try:
                lead_time_utc = _lead_time_swiss_to_utc(lead_time_swiss)
                result = await self.meteo.snow_depth_for_location(location, lead_time_utc) 
                logger.info(f"snow_depth: location={location}, lead_time_swiss={lead_time_swiss}, result={result}")
                return result
            except Exception as e:
                logger.exception(f"Failed to get snow depth for location '{location}': {e}")
                raise RuntimeError(f"Failed to get snow depth for location '{location}': {e}") from e

        @self.mcp.tool()
        async def precipitation_rate(location: str, lead_time_swiss: int) -> float:
            """
            Get precipitation rate for a location at a specific offset time.

            Args:
                location (str): Location name (e.g., "Zurich").
                lead_time_swiss (int): Hour offset from today at 00:00 Swiss local time. Min offset time is 2 hours. Max offset time is 121 hours.

            Returns:
                float: Precipitation rate in millimeters per second.

            Examples:
                precipitation_rate("Zurich", 2)    # Precipitation rate at 02:00 Swiss local time today
                precipitation_rate("Zurich", 14)   # Precipitation rate at 14:00 Swiss local time today
                precipitation_rate("Zurich", 36)   # Precipitation rate at 12:00 Swiss local time tomorrow
                precipitation_rate("Zurich", 113)  # Precipitation rate at 17:00 Swiss local time in 4 days
            """
            try:
                lead_time_utc = _lead_time_swiss_to_utc(lead_time_swiss)
                result = await self.meteo.total_precipitation_rate_for_location(location, lead_time_utc) 
                logger.info(f"precipitation_rate: location={location}, lead_time_swiss={lead_time_swiss}, result={result}")
                return result
            except Exception as e:
                logger.exception(f"Failed to get precipitation rate for location '{location}': {e}")
                raise RuntimeError(f"Failed to get precipitation rate for location '{location}': {e}") from e

    def run(self):
        if self.transport == "stdio":
            logger.info("Running server with stdio transport")
            self.mcp.run(transport="stdio")
        elif self.transport == "streamable-http":
            logger.info("Running server with Streamable HTTP transport")
            self.mcp.run(transport="streamable-http")
        else:
            raise ValueError(f"Unknown transport: {self.transport}")


def main():
    try:
        args = _parse_args()
        setup_logging(args.log_level)
        server = MeteoSwissMCPServer(args)
        server.run()
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down.")
    except Exception:
        logger.exception("Fatal error in MCP server")
        sys.exit(1)

if __name__ == "__main__":
    main()
