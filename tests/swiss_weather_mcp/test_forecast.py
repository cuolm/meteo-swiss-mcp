from datetime import date

import pytest

from fakes import build_swiss_time
from swiss_weather_mcp.forecast import _describe_pictogram, _find_compass_point
from swiss_weather_mcp.parameters import PICTOGRAMS


# --- read_sunshine_hours ---

@pytest.mark.asyncio
async def test_read_sunshine_hours(service_fixture):
    # 08:00 to 15:00 Swiss is 06:00 to 13:00 UTC. The rows stamped 12:00 and 13:00 UTC fall inside:
    # 60 + 45 minutes. The row stamped 06:00 covers 05:00 to 06:00 UTC, before the period starts.
    answer = await service_fixture.read_sunshine_hours(
        "Zurich", build_swiss_time("2026-09-23T08:00"), build_swiss_time("2026-09-23T15:00")
    )
    assert answer["value"] == 1.8
    assert answer["unit"] == "h"
    assert answer["from"] == "2026-09-23T08:00+02:00"
    assert answer["to"] == "2026-09-23T15:00+02:00"


@pytest.mark.asyncio
async def test_read_sunshine_hours_single_hour(service_fixture):
    # 14:00 to 15:00 Swiss is the row stamped 13:00 UTC, 45 minutes. The row stamped 12:00 UTC
    # covers 13:00 to 14:00 Swiss, the hour before the period, and must not be counted.
    answer = await service_fixture.read_sunshine_hours(
        "Zurich", build_swiss_time("2026-09-23T14:00"), build_swiss_time("2026-09-23T15:00")
    )
    assert answer["value"] == 0.8


@pytest.mark.asyncio
async def test_read_sunshine_hours_past_the_end(service_fixture):
    # The last row is stamped 13:00 UTC, 15:00 Swiss, so a period to 16:00 would be summed short
    with pytest.raises(ValueError, match="not fully covered by the forecast"):
        await service_fixture.read_sunshine_hours(
            "Zurich", build_swiss_time("2026-09-23T14:00"), build_swiss_time("2026-09-23T16:00")
        )


@pytest.mark.asyncio
async def test_read_sunshine_hours_reversed_period(service_fixture):
    with pytest.raises(ValueError, match="must be after its start"):
        await service_fixture.read_sunshine_hours(
            "Zurich", build_swiss_time("2026-09-23T15:00"), build_swiss_time("2026-09-23T09:00")
        )


# --- read_total_cloud_cover ---

@pytest.mark.asyncio
async def test_read_total_cloud_cover(service_fixture):
    answer = await service_fixture.read_total_cloud_cover("Zurich", build_swiss_time("2026-09-23T14:00"))

    # Half the sky low and half high: clear only where both are clear, 1 - 0.5 * 0.5 = 75%
    assert answer["value"] == 75.0
    assert (answer["low_percent"], answer["medium_percent"], answer["high_percent"]) == (50.0, 0.0, 50.0)


@pytest.mark.asyncio
async def test_read_total_cloud_cover_nearest_snapshot(service_fixture):
    # Cloud cover is a value at the moment of its stamp, not an average over the hour before, so
    # 14:20 Swiss reads the 14:00 snapshot (12:00 UTC) rather than the one closing that hour
    answer = await service_fixture.read_total_cloud_cover("Zurich", build_swiss_time("2026-09-23T14:20"))
    assert answer["value"] == 75.0


# --- read_daily_forecast ---

@pytest.mark.asyncio
async def test_read_daily_forecast(service_fixture):
    # The fixture publishes daily values for 23 and 24 September only, so 25 September is left out
    answer = await service_fixture.read_daily_forecast("Zurich", date(2026, 9, 23), 3)

    assert [day["date"] for day in answer["days"]] == ["2026-09-23", "2026-09-24"]
    assert [day["weekday"] for day in answer["days"]] == ["Wednesday", "Thursday"]
    assert [day["temperature_max_c"] for day in answer["days"]] == [20.6, 18.4]
    assert answer["location"] == "Zürich 8001 (409 m)"
    assert answer["model_run"] == "2026-09-22T15:00+02:00"


@pytest.mark.asyncio
async def test_read_daily_forecast_one_day(service_fixture):
    answer = await service_fixture.read_daily_forecast("Zurich", date(2026, 9, 24), 1)
    assert [day["date"] for day in answer["days"]] == ["2026-09-24"]


@pytest.mark.asyncio
async def test_read_daily_forecast_missing_parameter(service_fixture):
    # Only tre200px is published in this fixture, the other daily parameters are absent
    answer = await service_fixture.read_daily_forecast("Zurich", date(2026, 9, 23), 1)
    assert answer["days"][0]["rainfall_median_mm"] is None


@pytest.mark.asyncio
async def test_read_daily_forecast_invalid_days(service_fixture):
    for days in (0, 10):
        with pytest.raises(ValueError, match="between 1 and 9"):
            await service_fixture.read_daily_forecast("Zurich", date(2026, 9, 23), days)


@pytest.mark.asyncio
async def test_read_daily_forecast_outside_the_forecast(service_fixture):
    # Every parameter exists, but none reaches these days, so there is nothing to answer with
    with pytest.raises(ValueError, match="no daily forecast"):
        await service_fixture.read_daily_forecast("Zurich", date(2026, 10, 30), 3)


# --- read_wind ---

@pytest.mark.asyncio
async def test_read_wind(service_fixture):
    # 14:00 Swiss is the hour ending at 12:00 UTC
    answer = await service_fixture.read_wind("Zurich", build_swiss_time("2026-09-23T14:00"))

    assert (answer["speed_kmh"], answer["gusts_kmh"]) == (12.5, 38.0)
    assert (answer["direction_degrees"], answer["compass_point"]) == (217.0, "SW")
    assert answer["valid_at"] == "2026-09-23T14:00+02:00"
    assert answer["location"] == "Zürich 8001 (409 m)"


# --- read_hourly_forecast ---

@pytest.mark.asyncio
async def test_read_hourly_forecast(service_fixture):
    # 13:00 to 15:00 Swiss is 11:00 to 13:00 UTC: the hours ending at 12:00 and 13:00 UTC
    answer = await service_fixture.read_hourly_forecast(
        "Zurich", build_swiss_time("2026-09-23T13:00"), build_swiss_time("2026-09-23T15:00")
    )

    assert [(hour["from"], hour["to"]) for hour in answer["hours"]] == [
        ("2026-09-23T13:00+02:00", "2026-09-23T14:00+02:00"),
        ("2026-09-23T14:00+02:00", "2026-09-23T15:00+02:00"),
    ]
    assert [hour["temperature_c"] for hour in answer["hours"]] == [12.0, 14.5]
    assert [hour["rain_chance_percent"] for hour in answer["hours"]] == [10.0, 20.0]
    assert [(hour["weather"], hour["weather_emoji"]) for hour in answer["hours"]] == [
        ("mostly sunny, some clouds", "🌤️"),
        ("partly sunny, thick passing clouds", "⛅"),
    ]
    assert answer["location"] == "Zürich 8001 (409 m)"
    assert answer["model_run"] == "2026-09-22T15:00+02:00"


@pytest.mark.asyncio
async def test_read_hourly_forecast_one_hour(service_fixture):
    # Without an end, the one hour starting at start: 13:00 to 14:00 Swiss, the row stamped 12:00 UTC
    answer = await service_fixture.read_hourly_forecast("Zurich", build_swiss_time("2026-09-23T13:00"))

    assert [(hour["from"], hour["to"]) for hour in answer["hours"]] == [("2026-09-23T13:00+02:00", "2026-09-23T14:00+02:00")]
    assert answer["hours"][0]["temperature_c"] == 12.0


@pytest.mark.asyncio
async def test_read_hourly_forecast_inside_an_hour(service_fixture):
    # 14:30 Swiss lies in the hour 14:00 to 15:00, which is the row stamped 13:00 UTC
    answer = await service_fixture.read_hourly_forecast("Zurich", build_swiss_time("2026-09-23T14:30"))

    assert [(hour["from"], hour["to"]) for hour in answer["hours"]] == [("2026-09-23T14:00+02:00", "2026-09-23T15:00+02:00")]
    assert answer["hours"][0]["temperature_c"] == 14.5


@pytest.mark.asyncio
async def test_read_hourly_forecast_outside_the_forecast(service_fixture):
    # Times are written as the answers write them, with the offset of their own date: 30 October is
    # already winter time, the covered range is still summer time
    with pytest.raises(ValueError, match="is not fully covered by the forecast") as raised:
        await service_fixture.read_hourly_forecast("Zurich", build_swiss_time("2026-10-30T14:00"))

    assert str(raised.value).startswith("2026-10-30T14:00+01:00 is not fully covered by the forecast")
    assert "The forecast covers 2026-09-23T14:00+02:00 to 2026-09-23T15:00+02:00." in str(raised.value)
    assert "jww003i0" not in str(raised.value)  # a parameter code means nothing to the model


@pytest.mark.asyncio
async def test_read_hourly_forecast_past_the_end(service_fixture):
    with pytest.raises(ValueError, match="not fully covered by the forecast"):
        await service_fixture.read_hourly_forecast(
            "Zurich", build_swiss_time("2026-09-23T13:00"), build_swiss_time("2026-09-23T17:00")
        )


@pytest.mark.asyncio
async def test_read_hourly_forecast_too_long(service_fixture):
    with pytest.raises(ValueError, match="at most 24 hours"):
        await service_fixture.read_hourly_forecast(
            "Zurich", build_swiss_time("2026-09-23T13:00"), build_swiss_time("2026-09-24T14:00")
        )


# --- read_rain_outlook ---

@pytest.mark.asyncio
async def test_read_rain_outlook(service_fixture):
    # 14:00 to 20:00 Swiss is 12:00 to 18:00 UTC: two blocks, ending at 15:00 and 18:00 UTC
    answer = await service_fixture.read_rain_outlook(
        "Zurich", build_swiss_time("2026-09-23T14:00"), build_swiss_time("2026-09-23T20:00")
    )

    assert [(block["from"], block["to"]) for block in answer["blocks"]] == [
        ("2026-09-23T14:00+02:00", "2026-09-23T17:00+02:00"),
        ("2026-09-23T17:00+02:00", "2026-09-23T20:00+02:00"),
    ]
    assert [block["rain_chance_percent"] for block in answer["blocks"]] == [40.0, 80.0]
    assert [block["rainfall_median_mm"] for block in answer["blocks"]] == [0.0, 2.4]
    # The heaviest hour of each block, not a sum: quantiles do not add up
    assert [block["heaviest_hour_up_to_mm"] for block in answer["blocks"]] == [1.1, 3.5]
    assert answer["location"] == "Zürich 8001 (409 m)"


@pytest.mark.asyncio
async def test_read_rain_outlook_past_the_end(service_fixture):
    with pytest.raises(ValueError, match="not fully covered by the forecast"):
        await service_fixture.read_rain_outlook(
            "Zurich", build_swiss_time("2026-09-23T14:00"), build_swiss_time("2026-09-23T23:00")
        )


@pytest.mark.asyncio
async def test_read_rain_outlook_too_long(service_fixture):
    with pytest.raises(ValueError, match="at most 48 hours"):
        await service_fixture.read_rain_outlook(
            "Zurich", build_swiss_time("2026-09-23T14:00"), build_swiss_time("2026-09-25T20:00")
        )


@pytest.mark.asyncio
async def test_read_rain_outlook_reversed_period(service_fixture):
    with pytest.raises(ValueError, match="must be after its start"):
        await service_fixture.read_rain_outlook(
            "Zurich", build_swiss_time("2026-09-23T20:00"), build_swiss_time("2026-09-23T14:00")
        )


# --- _describe_pictogram ---

def test_describe_pictogram():
    assert _describe_pictogram(1) == ("sunny", "☀️")
    assert _describe_pictogram(101) == ("clear", "✨")
    assert _describe_pictogram(999) == ("unknown weather code 999", None)


def test_describe_pictogram_every_code_has_words_and_an_emoji():
    for code, (description, emoji) in PICTOGRAMS.items():
        assert description and emoji, code


# --- _find_compass_point ---

def test_find_compass_point():
    # Each of the 16 points covers 22.5 degrees, and a bearing just short of north wraps round to N
    assert [_find_compass_point(degrees) for degrees in (0, 11, 12, 90, 217, 355)] == ["N", "N", "NNE", "E", "SW", "N"]
