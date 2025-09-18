import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from meteo_swiss_predictions import MeteoSwissPredictions

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Configure logger
def _setup_logger() -> None:
    try:
        config_path = Path(__file__).parent.parent / "log_config.json"
        with open(config_path, "rt") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
        logging.getLogger("meteo_swiss_mcp_server").info("Logger successfully configured")
    except Exception as e:
        logging.basicConfig(level=logging.ERROR) # logging.basicConfig() attaches the root logger to STDERR by default, so no interference with stdio MCP protocol
        logging.getLogger("meteo_swiss_mcp_server").error(f"Failed to configure logger: {e}", exc_info=True)


_setup_logger()
logger = logging.getLogger("meteo_swiss_mcp_server")

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
    return parser.parse_args()

def _lead_time_swiss_to_utc(lead_time_swiss: int) -> float:
    """
    Convert a lead time expressed in Swiss local hours since midnight
    into the equivalent lead time in UTC hours since the same UTC midnight.
    
    Handles DST (daylight saving time) transitions correctly by using timezone-aware datetimes.
    """

    # Midnight in Swiss local time (today)
    midnight_swiss_datetime = datetime.now(ZoneInfo("Europe/Zurich")).replace(hour=0, minute=0, second=0, microsecond=0)

    # Target time in Swiss local time
    target_time_swiss_datetime = midnight_swiss_datetime + timedelta(hours=lead_time_swiss) 

    # Convert both to UTC
    midnight_utc_datetime = midnight_swiss_datetime.astimezone(ZoneInfo("UTC"))
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
                raise RuntimeError(f"Failed to get total rainfall for location '{location}': {e}")

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
                raise RuntimeError(f"Failed to get sunshine hours for location '{location}': {e}")

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
                raise RuntimeError(f"Failed to get temperature for location '{location}': {e}")

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
                raise RuntimeError(f"Failed to get wind speed for location '{location}': {e}")

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
                raise RuntimeError(f"Failed to get pressure for location '{location}': {e}")

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
                raise RuntimeError(f"Failed to get total cloud cover for location '{location}': {e}")

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
                raise RuntimeError(f"Failed to get snow depth for location '{location}': {e}")

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
                raise RuntimeError(f"Failed to get precipitation rate for location '{location}': {e}")

    def run(self):
        if self.transport == "stdio":
            logger.info("Running server with stdio transport")
            self.mcp.run(transport="stdio")
        elif self.transport == "streamable-http":
            logger.info("Running server with Streamable HTTP transport")
            self.mcp.run(transport="streamable-http")
        else:
            logger.exception(f"Unknown transport: {self.transport}")
            raise ValueError(f"Unknown transport: {self.transport}")


def main():
    args = _parse_args()
    server = MeteoSwissMCPServer(args)
    server.run()

if __name__ == "__main__":
    main()
