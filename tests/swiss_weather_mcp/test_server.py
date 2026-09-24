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
    "weather_description": ["location", "when"],
    "temperature": ["location", "when"],
    "total_rainfall": ["location", "start", "end"],
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


def test_a_timestamp_without_a_zone_is_read_as_swiss_local_time():
    parsed = _parse_swiss_time("2026-09-23T14:00")
    assert parsed == datetime(2026, 9, 23, 14, tzinfo=SWISS_TZ)


def test_summer_time_is_two_hours_ahead_of_utc():
    # In September Switzerland is on CEST, so 14:00 local is 12:00 UTC
    parsed = _parse_swiss_time("2026-09-23T14:00")
    assert parsed.astimezone(timezone.utc).hour == 12


def test_winter_time_is_one_hour_ahead_of_utc():
    # In January Switzerland is on CET, so the same local hour is 13:00 UTC
    parsed = _parse_swiss_time("2026-01-23T14:00")
    assert parsed.astimezone(timezone.utc).hour == 13


def test_an_explicit_offset_is_honoured_rather_than_overwritten():
    parsed = _parse_swiss_time("2026-09-23T14:00+00:00")
    assert parsed.astimezone(timezone.utc).hour == 14


def test_a_date_on_its_own_is_read_as_that_day_at_midnight():
    parsed = _parse_swiss_time("2026-09-23")
    assert parsed == datetime(2026, 9, 23, 0, 0, tzinfo=SWISS_TZ)


def test_a_space_between_date_and_time_is_accepted():
    assert _parse_swiss_time("2026-09-23 14:00") == _parse_swiss_time("2026-09-23T14:00")


def test_an_unreadable_timestamp_says_what_a_good_one_looks_like():
    with pytest.raises(ValueError, match="2026-09-23T14:00"):
        _parse_swiss_time("tomorrow afternoon")


# ── tools ────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_the_server_offers_every_tool_with_its_arguments(server_fixture):
    offered = {}
    for tool in await server_fixture.mcp.list_tools():
        offered[tool.name] = list(tool.input_schema.get("properties", {}))
    assert offered == EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_a_tool_returns_the_forecast_it_was_given(server_fixture):
    answer = {"value": 19.1, "unit": "°C", "location": "Zürich 8001 (409 m)"}
    server_fixture.forecast_service.temperature_for_location.return_value = answer

    result = await server_fixture.mcp.call_tool("temperature", TEMPERATURE_CALL)
    assert json.loads(result.content[0].text) == answer


@pytest.mark.asyncio
async def test_a_failure_the_caller_can_fix_reaches_the_model_without_a_traceback(server_fixture, caplog):
    server_fixture.forecast_service.temperature_for_location.side_effect = ValueError("Location 'Tessin' is not one of the places")

    with caplog.at_level(logging.INFO), pytest.raises(ToolError, match="Location 'Tessin' is not one of") as raised:
        await server_fixture.mcp.call_tool("temperature", {"location": "Tessin", "when": "2026-09-24T14:00"})

    assert not isinstance(raised.value, UnexpectedToolError)
    for record in caplog.records:
        assert record.exc_info is None, f"traceback logged by {record.name}"


@pytest.mark.asyncio
async def test_a_timestamp_the_model_got_wrong_is_explained_to_it(server_fixture):
    with pytest.raises(ToolError, match="is not a valid timestamp"):
        await server_fixture.mcp.call_tool("temperature", {"location": "Zurich", "when": "tomorrow"})


@pytest.mark.asyncio
async def test_meteoswiss_being_unreachable_is_explained_to_the_model(server_fixture):
    server_fixture.forecast_service.temperature_for_location.side_effect = requests.ConnectionError("connection refused")

    with pytest.raises(ToolError, match="Could not reach MeteoSwiss"):
        await server_fixture.mcp.call_tool("temperature", TEMPERATURE_CALL)


@pytest.mark.asyncio
async def test_an_unexpected_failure_is_hidden_from_the_model(server_fixture):
    # A bug is a crash: the SDK logs the traceback and tells the model nothing about the internals
    server_fixture.forecast_service.temperature_for_location.side_effect = KeyError("internal detail")

    with pytest.raises(UnexpectedToolError) as raised:
        await server_fixture.mcp.call_tool("temperature", TEMPERATURE_CALL)
    assert "internal detail" not in str(raised.value)


@pytest.mark.asyncio
async def test_the_current_time_can_be_sent_straight_back_to_a_tool(server_fixture):
    # The model builds its next timestamp from this answer, so it must be in the form the tools read
    result = await server_fixture.mcp.call_tool("current_date_and_time", {})
    text = result.content[0].text
    weekday, timestamp = text.removeprefix("Today is ").removesuffix(" (Swiss time)").split(", ")

    moment = _parse_swiss_time(timestamp)
    assert moment.tzinfo is SWISS_TZ, "written without an offset, so read as Swiss time"
    assert moment.strftime("%A") == weekday
