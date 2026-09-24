from datetime import date

import pytest

from fakes import build_swiss_time
from swiss_weather_mcp.forecast import _find_compass_point


# ── weather values ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_temperature_reports_the_value_with_the_point_it_resolved(service_fixture):
    result = await service_fixture.read_temperature("Zurich", build_swiss_time("2026-09-23T14:00"))

    assert result["value"] == 12.0  # 14:00 Swiss in September is 12:00 UTC
    assert result["unit"] == "°C"
    assert result["location"] == "Zürich 8001 (409 m)"
    assert result["altitude_m"] == 409.0
    assert result["valid_at"] == "2026-09-23T14:00+02:00"
    assert result["model_run"] == "2026-09-22T15:00+02:00"


@pytest.mark.asyncio
async def test_sunshine_hours_adds_the_minutes_and_reports_hours(service_fixture):
    # 08:00 to 15:00 Swiss is 06:00 to 13:00 UTC. The rows stamped 12:00 and 13:00 UTC fall inside:
    # 60 + 45 minutes. The row stamped 06:00 covers 05:00 to 06:00 UTC, before the period starts.
    result = await service_fixture.read_sunshine_hours(
        "Zurich", build_swiss_time("2026-09-23T08:00"), build_swiss_time("2026-09-23T15:00")
    )
    assert result["value"] == 1.8
    assert result["unit"] == "h"
    assert result["from"] == "2026-09-23T08:00+02:00"
    assert result["to"] == "2026-09-23T15:00+02:00"


@pytest.mark.asyncio
async def test_a_period_past_the_end_of_the_forecast_is_refused(service_fixture):
    # The last row is stamped 13:00 UTC, 15:00 Swiss, so a period to 16:00 would be summed short
    with pytest.raises(ValueError, match="not fully covered by the forecast"):
        await service_fixture.read_sunshine_hours(
            "Zurich", build_swiss_time("2026-09-23T14:00"), build_swiss_time("2026-09-23T16:00")
        )


@pytest.mark.asyncio
async def test_a_period_that_ends_before_it_starts_is_refused(service_fixture):
    with pytest.raises(ValueError, match="must be after its start"):
        await service_fixture.read_sunshine_hours(
            "Zurich", build_swiss_time("2026-09-23T15:00"), build_swiss_time("2026-09-23T09:00")
        )


@pytest.mark.asyncio
async def test_a_window_counts_the_hours_by_the_stamp_at_their_end(service_fixture):
    # 14:00 to 15:00 Swiss is the row stamped 13:00 UTC, 45 minutes. The row stamped 12:00 UTC
    # covers 13:00 to 14:00 Swiss, the hour before the period, and must not be counted.
    result = await service_fixture.read_sunshine_hours(
        "Zurich", build_swiss_time("2026-09-23T14:00"), build_swiss_time("2026-09-23T15:00")
    )
    assert result["value"] == 0.8


@pytest.mark.asyncio
async def test_a_time_inside_an_hour_reads_the_row_that_closes_that_hour(service_fixture):
    # 14:30 Swiss lies in the hour 14:00 to 15:00, which is the row stamped 13:00 UTC
    result = await service_fixture.read_temperature("Zurich", build_swiss_time("2026-09-23T14:30"))
    assert result["value"] == 14.5


@pytest.mark.asyncio
async def test_cloud_cover_combines_the_overlapping_layers(service_fixture):
    result = await service_fixture.read_total_cloud_cover("Zurich", build_swiss_time("2026-09-23T14:00"))

    # Half the sky low and half high: clear only where both are clear, 1 - 0.5 * 0.5 = 75%
    assert result["value"] == 75.0
    assert (result["low_percent"], result["medium_percent"], result["high_percent"]) == (50.0, 0.0, 50.0)


@pytest.mark.asyncio
async def test_cloud_cover_reads_the_nearest_snapshot(service_fixture):
    # Cloud cover is a value at the moment of its stamp, not an average over the hour before, so
    # 14:20 Swiss reads the 14:00 snapshot (12:00 UTC) rather than the one closing that hour
    result = await service_fixture.read_total_cloud_cover("Zurich", build_swiss_time("2026-09-23T14:20"))
    assert result["value"] == 75.0


@pytest.mark.asyncio
async def test_weather_description_turns_the_code_into_words(service_fixture):
    result = await service_fixture.read_weather_description("Zurich", build_swiss_time("2026-09-23T14:00"))

    assert result["value"] == "mostly sunny, some clouds"
    assert result["pictogram_code"] == 2


@pytest.mark.asyncio
async def test_a_time_outside_the_forecast_names_the_range_that_is_covered(service_fixture):
    # Times are written as the answers write them, with the offset of their own date: 30 October is
    # already winter time, the covered range is still summer time
    with pytest.raises(ValueError) as raised:
        await service_fixture.read_temperature("Zurich", build_swiss_time("2026-10-30T14:00"))

    assert str(raised.value).startswith("2026-10-30T14:00+01:00 is outside the forecast")
    assert "The forecast covers 2026-09-23T08:00+02:00 to 2026-09-23T15:00+02:00." in str(raised.value)
    assert "tre200h0" not in str(raised.value)  # a parameter code means nothing to the model


@pytest.mark.asyncio
async def test_daily_forecast_refuses_a_day_outside_the_forecast(service_fixture):
    # Every parameter exists, but none reaches that day, so there is nothing to answer with
    with pytest.raises(ValueError, match="no daily forecast"):
        await service_fixture.read_daily_forecast("Zurich", date(2026, 10, 30))


@pytest.mark.asyncio
async def test_daily_forecast_reports_a_missing_parameter_instead_of_failing(service_fixture):
    # Only tre200px is published in this fixture, the other daily parameters are absent
    result = await service_fixture.read_daily_forecast("Zurich", date(2026, 9, 23))

    assert result["temperature_max_c"] == 20.6
    assert result["rainfall_median_mm"] is None
    assert result["location"] == "Zürich 8001 (409 m)"
    assert result["date"] == "2026-09-23"


def test_a_bearing_turns_into_the_nearest_compass_point():
    # Each of the 16 points covers 22.5 degrees, and a bearing just short of north wraps round to N
    assert [_find_compass_point(degrees) for degrees in (0, 11, 12, 90, 217, 355)] == ["N", "N", "NNE", "E", "SW", "N"]
