import argparse
import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest
import requests
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from swiss_weather_mcp.server import SwissWeatherMCPServer, _parse_swiss_time

SWISS_TZ = ZoneInfo("Europe/Zurich")

# Every tool the server offers, with its arguments in order
EXPECTED_TOOLS = {
    "current_date_and_time": [],
    "daily_forecast": ["location", "date"],
    "weather_outlook": ["location", "days"],
    "weather_description": ["location", "when"],
    "hourly_forecast": ["location", "start", "end"],
    "temperature": ["location", "when"],
    "total_rainfall": ["location", "start", "end"],
    "rain_outlook": ["location", "start", "end"],
    "sunshine_hours": ["location", "start", "end"],
    "precipitation_probability": ["location", "when"],
    "precipitation_rate": ["location", "when"],
    "wind_speed": ["location", "when"],
    "wind_gusts": ["location", "when"],
    "wind_direction": ["location", "when"],
    "total_cloud_cover": ["location", "when"],
    "freezing_level": ["location", "when"],
}

TEMPERATURE_CALL = {"location": "Zurich", "when": "2026-09-24T14:00"}


@pytest.fixture
def server_fixture(mocker, tmp_path):
    """Return the MCP server with its weather layer replaced by a mock, so no tool reaches the network."""
    mocker.patch("swiss_weather_mcp.server.CACHE_DIR", tmp_path)
    args = argparse.Namespace(
        host="localhost", port=8050, transport="stdio", cache_all_locations=False, log_level="INFO"
    )
    server = SwissWeatherMCPServer(args)
    server.forecast_service = mocker.AsyncMock()
    return server


# --- _parse_swiss_time ---

def test_parse_swiss_time_without_offset():
    moment = _parse_swiss_time("2026-09-23T14:00")
    assert moment == datetime(2026, 9, 23, 14, tzinfo=SWISS_TZ)


def test_parse_swiss_time_summer():
    # In September Switzerland is on CEST, so 14:00 local is 12:00 UTC
    moment = _parse_swiss_time("2026-09-23T14:00")
    assert moment.astimezone(timezone.utc).hour == 12


def test_parse_swiss_time_winter():
    # In January Switzerland is on CET, so the same local hour is 13:00 UTC
    moment = _parse_swiss_time("2026-01-23T14:00")
    assert moment.astimezone(timezone.utc).hour == 13


def test_parse_swiss_time_keeps_an_explicit_offset():
    moment = _parse_swiss_time("2026-09-23T14:00+00:00")
    assert moment.astimezone(timezone.utc).hour == 14


def test_parse_swiss_time_date_only():
    moment = _parse_swiss_time("2026-09-23")
    assert moment == datetime(2026, 9, 23, 0, 0, tzinfo=SWISS_TZ)


def test_parse_swiss_time_with_a_space():
    assert _parse_swiss_time("2026-09-23 14:00") == _parse_swiss_time("2026-09-23T14:00")


def test_parse_swiss_time_invalid():
    with pytest.raises(ValueError, match="2026-09-23T14:00"):
        _parse_swiss_time("tomorrow afternoon")


# --- _register_tools ---

@pytest.mark.asyncio
async def test_register_tools(server_fixture):
    offered = {}
    for tool in await server_fixture.mcp.list_tools():
        offered[tool.name] = list(tool.input_schema.get("properties", {}))
    assert offered == EXPECTED_TOOLS


# --- _handle_tool_call ---

@pytest.mark.asyncio
async def test_handle_tool_call_returns_the_answer(server_fixture):
    answer = {"value": 19.1, "unit": "°C", "location": "Zürich 8001 (409 m)"}
    server_fixture.forecast_service.read_temperature.return_value = answer

    tool_result = await server_fixture.mcp.call_tool("temperature", TEMPERATURE_CALL)
    assert json.loads(tool_result.content[0].text) == answer


@pytest.mark.asyncio
async def test_handle_tool_call_value_error(server_fixture, caplog):
    server_fixture.forecast_service.read_temperature.side_effect = ValueError("Location 'Tessin' is not one of the places")

    with caplog.at_level(logging.INFO), pytest.raises(ToolError, match="Location 'Tessin' is not one of") as raised:
        await server_fixture.mcp.call_tool("temperature", {"location": "Tessin", "when": "2026-09-24T14:00"})

    assert not isinstance(raised.value, UnexpectedToolError)
    for record in caplog.records:
        assert record.exc_info is None, f"traceback logged by {record.name}"


@pytest.mark.asyncio
async def test_handle_tool_call_invalid_timestamp(server_fixture):
    with pytest.raises(ToolError, match="is not a valid timestamp"):
        await server_fixture.mcp.call_tool("temperature", {"location": "Zurich", "when": "tomorrow"})


@pytest.mark.asyncio
async def test_handle_tool_call_meteoswiss_unreachable(server_fixture):
    server_fixture.forecast_service.read_temperature.side_effect = requests.ConnectionError("connection refused")

    with pytest.raises(ToolError, match="Could not reach MeteoSwiss"):
        await server_fixture.mcp.call_tool("temperature", TEMPERATURE_CALL)


@pytest.mark.asyncio
async def test_handle_tool_call_unexpected_error(server_fixture):
    # A bug is a crash: the SDK logs the traceback and tells the model nothing about the internals
    server_fixture.forecast_service.read_temperature.side_effect = KeyError("internal detail")

    with pytest.raises(UnexpectedToolError) as raised:
        await server_fixture.mcp.call_tool("temperature", TEMPERATURE_CALL)
    assert "internal detail" not in str(raised.value)


# --- current_date_and_time ---

@pytest.mark.asyncio
async def test_current_date_and_time_format(server_fixture):
    # The model builds its next timestamp from this answer, so it must be in the form the tools read
    tool_result = await server_fixture.mcp.call_tool("current_date_and_time", {})
    text = tool_result.content[0].text
    weekday, timestamp = text.removeprefix("Today is ").removesuffix(" (Swiss time)").split(", ")

    moment = _parse_swiss_time(timestamp)
    assert moment.tzinfo is SWISS_TZ, "written without an offset, so read as Swiss time"
    assert moment.strftime("%A") == weekday
