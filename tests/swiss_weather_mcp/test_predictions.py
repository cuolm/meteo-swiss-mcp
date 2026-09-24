from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from swiss_weather_mcp.localforecast import LocalForecast
from swiss_weather_mcp.predictions import SwissWeatherPredictions

POINT_TABLE_COLUMNS = (
    "point_id;point_type_id;station_abbr;postal_code;point_name;point_type_de;point_type_fr;"
    "point_type_it;point_type_en;point_height_masl;point_coordinates_lv95_east;"
    "point_coordinates_lv95_north;point_coordinates_wgs84_lat;point_coordinates_wgs84_lon"
)
# Zurich twice so the lowest postal code has to win, Davos with no postal code so the station is
# the only option, and Wallisellen because it is what a near match for "Wallis" would wrongly pick.
POINT_TABLE_ROWS = (
    ("800200", "2", "", "8002", "Zürich", "431.0"),
    ("800100", "2", "", "8001", "Zürich", "409.0"),
    ("26", "1", "DAV", "", "Davos", "1594.0"),
    ("840000", "2", "", "8400", "Wallisellen", "441.0"),
)

RUN_ID = "202609221300"
EARLIER_RUN_ID = "202609221200"


def _build_point_table_csv() -> bytes:
    lines = [POINT_TABLE_COLUMNS]
    for point_id, type_id, abbr, postal_code, name, altitude in POINT_TABLE_ROWS:
        lines.append(
            f"{point_id};{type_id};{abbr};{postal_code};{name};Ort;Lieu;Luogo;Place;{altitude};0;0;47.0;8.0"
        )
    return ("\r\n".join(lines) + "\r\n").encode("latin-1")


def _build_parameter_csv(parameter: str, values: dict, point: str = "800100;2") -> bytes:
    """Build a parameter file the way MeteoSwiss publishes it, header and all locations included."""
    lines = [f"point_id;point_type_id;Date;{parameter}"]
    for stamp, value in values.items():
        lines.append(f"{point};{stamp};{value}")
        lines.append(f"999999;2;{stamp};-1")  # another location, which must be filtered out
    return ("\r\n".join(lines) + "\r\n").encode("latin-1")


class _FakeResponse:
    """Stand in for a streamed requests response."""

    def __init__(self, body: bytes = b"", status_code: int = 200, payload: dict = None):
        self.body = body
        self.status_code = status_code
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected HTTP {self.status_code}")

    def json(self):
        return self.payload

    def iter_content(self, chunk_size):
        # Deliberately split mid line, so the chunk stitching in _write_point_rows is exercised
        for start in range(0, len(self.body), 7):
            yield self.body[start:start + 7]


def _build_stac_item(run_id: str, parameters) -> dict:
    return {
        "assets": {
            f"vnut12.lssw.{run_id}.{parameter}.csv": {"href": f"https://example.test/{run_id}/{parameter}.csv"}
            for parameter in parameters
        }
    }


@pytest.fixture
def forecast_fixture(mocker, tmp_path):
    """
    Return a LocalForecast backed by fake HTTP responses, plus the mock, so tests can count
    downloads and swap in different published data.
    """
    published = {
        "tre200h0": {"202609230600": "6.0", "202609231200": "12.0", "202609231300": "14.5"},
        "sre000h0": {"202609230600": "30", "202609231200": "60", "202609231300": "45"},
        "nprolohs": {"202609231200": "0.50"},
        "npromths": {"202609231200": "0.00"},
        "nprohihs": {"202609231200": "0.50"},
        "jww003i0": {"202609231200": "2"},
        "tre200px": {"202609230000": "20.6"},
    }

    def fake_get(url, **kwargs):
        if url.endswith("meta_point.csv"):
            return _FakeResponse(body=_build_point_table_csv())
        if "/items/" in url:
            return _FakeResponse(payload=_build_stac_item(RUN_ID, published))
        parameter = url.rsplit("/", 1)[-1].removesuffix(".csv")
        return _FakeResponse(body=_build_parameter_csv(parameter, published[parameter]))

    get_mock = mocker.patch("swiss_weather_mcp.localforecast.requests.get", side_effect=fake_get)
    forecast = LocalForecast(tmp_path)
    forecast.get_mock = get_mock
    return forecast


@pytest.fixture
def predictions_fixture(forecast_fixture):
    """Return a SwissWeatherPredictions reading from the fake data source."""
    return SwissWeatherPredictions(forecast_fixture)


def _swiss_time(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp).replace(tzinfo=ZoneInfo("Europe/Zurich"))


# ── location resolution ──────────────────────────────────────────────────────
def test_find_point_matches_a_name_ignoring_case_and_accents(forecast_fixture):
    for spelling in ("Zürich", "zurich", "ZURICH"):
        assert forecast_fixture.find_point(spelling).point_id == "800100"


def test_find_point_picks_the_lowest_postal_code_of_a_city(forecast_fixture):
    # 8001 is the historic centre, 8002 is a different district of the same city
    assert forecast_fixture.find_point("Zürich").postal_code == "8001"


def test_find_point_accepts_a_postal_code(forecast_fixture):
    assert forecast_fixture.find_point("8002").point_id == "800200"


def test_find_point_falls_back_to_the_station_when_there_is_no_postal_code(forecast_fixture):
    point = forecast_fixture.find_point("Davos")
    assert (point.point_id, point.point_type_id) == ("26", "1")


def test_find_point_rejects_a_near_match_instead_of_guessing(forecast_fixture):
    # "Wallis" is the canton Valais, its closest name here is a Zurich suburb 150 km away
    with pytest.raises(ValueError, match="not one of the 4 places"):
        forecast_fixture.find_point("Wallis")


def test_find_point_names_the_location_it_could_not_find(forecast_fixture):
    with pytest.raises(ValueError, match="'Tessin'"):
        forecast_fixture.find_point("Tessin")


# ── runs and caching ─────────────────────────────────────────────────────────
def test_find_latest_run_picks_the_newest_published_run(mocker, tmp_path):
    item = _build_stac_item(EARLIER_RUN_ID, ["tre200h0"])
    item["assets"].update(_build_stac_item(RUN_ID, ["tre200h0"])["assets"])
    mocker.patch("swiss_weather_mcp.localforecast.requests.get", return_value=_FakeResponse(payload=item))

    run_id, _ = LocalForecast(tmp_path).find_latest_run()
    assert run_id == RUN_ID


def test_find_latest_run_falls_back_to_yesterday_while_todays_item_is_empty(mocker, tmp_path):
    # The item for a new day exists before its first run lands, and reports no assets at all
    responses = [_FakeResponse(payload={"assets": {}}), _FakeResponse(payload=_build_stac_item(RUN_ID, ["tre200h0"]))]
    mocker.patch("swiss_weather_mcp.localforecast.requests.get", side_effect=responses)

    run_id, _ = LocalForecast(tmp_path).find_latest_run()
    assert run_id == RUN_ID


def test_find_latest_run_asks_for_the_item_of_the_utc_day(mocker, tmp_path):
    # 22:30 UTC on 23 September is already 24 September in Switzerland
    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 23, 22, 30, tzinfo=timezone.utc).astimezone(tz)

    mocker.patch("swiss_weather_mcp.localforecast.datetime", _FixedDatetime)
    get_mock = mocker.patch(
        "swiss_weather_mcp.localforecast.requests.get",
        return_value=_FakeResponse(payload=_build_stac_item(RUN_ID, ["tre200h0"])),
    )

    LocalForecast(tmp_path).find_latest_run()
    requested_url = get_mock.call_args_list[0].args[0]
    assert requested_url.endswith("/items/20260923-ch")


def test_read_series_reads_every_row_including_the_first(forecast_fixture):
    # A point extract carries no header line, so skipping one would lose the first forecast hour
    series = forecast_fixture.read_series("tre200h0", forecast_fixture.find_point("Zürich"))
    assert len(series.values) == 3
    assert series.values[datetime(2026, 9, 23, 6, tzinfo=timezone.utc)] == 6.0


def test_read_series_keeps_only_the_requested_location(forecast_fixture):
    series = forecast_fixture.read_series("tre200h0", forecast_fixture.find_point("Zürich"))
    assert -1 not in series.values.values()


def test_read_series_reports_the_run_it_came_from(forecast_fixture):
    series = forecast_fixture.read_series("tre200h0", forecast_fixture.find_point("Zürich"))
    assert series.run_time == datetime(2026, 9, 22, 13, tzinfo=timezone.utc)


def test_read_series_does_not_download_the_same_file_twice(forecast_fixture):
    point = forecast_fixture.find_point("Zürich")
    first_series = forecast_fixture.read_series("tre200h0", point)
    downloads = forecast_fixture.get_mock.call_count

    assert forecast_fixture.read_series("tre200h0", point).values == first_series.values
    assert forecast_fixture.get_mock.call_count == downloads


def test_read_series_stores_only_the_point_rows_by_default(forecast_fixture, tmp_path):
    forecast_fixture.read_series("tre200h0", forecast_fixture.find_point("Zürich"))
    stored = list((tmp_path / "runs" / RUN_ID).iterdir())
    assert [path.name for path in stored] == ["tre200h0_800100_2.csv"]
    assert b"999999" not in stored[0].read_bytes()


def test_read_series_stores_every_location_when_asked_to(mocker, tmp_path, forecast_fixture):
    forecast_fixture.cache_all_locations = True
    forecast_fixture.read_series("tre200h0", forecast_fixture.find_point("Zürich"))

    stored = tmp_path / "runs" / RUN_ID / "tre200h0.csv"
    assert stored.exists()
    assert b"999999" in stored.read_bytes()


def test_read_series_reads_the_same_values_from_a_full_file(forecast_fixture, tmp_path):
    point = forecast_fixture.find_point("Zürich")
    from_extract = forecast_fixture.read_series("tre200h0", point).values

    forecast_fixture.cache_all_locations = True
    (tmp_path / "runs" / RUN_ID / "tre200h0_800100_2.csv").unlink()
    assert forecast_fixture.read_series("tre200h0", point).values == from_extract


def test_read_series_drops_runs_older_than_the_previous_one(forecast_fixture, tmp_path):
    for run in ("202609221100", EARLIER_RUN_ID):
        superseded = tmp_path / "runs" / run
        superseded.mkdir(parents=True)
        (superseded / "tre200h0_800100_2.csv").write_bytes(b"stale")

    forecast_fixture.read_series("tre200h0", forecast_fixture.find_point("Zürich"))
    # 12:00 stays, another server process may still be reading it until it sees the 13:00 run
    assert sorted(path.name for path in (tmp_path / "runs").iterdir()) == [EARLIER_RUN_ID, RUN_ID]


def test_read_series_keeps_a_newer_run_written_by_another_process(forecast_fixture, tmp_path):
    # Another server process on the same cache already moved on to a later run
    newer = tmp_path / "runs" / "202609221400"
    newer.mkdir(parents=True)
    (newer / "tre200h0_800100_2.csv").write_bytes(b"newer")

    forecast_fixture.read_series("tre200h0", forecast_fixture.find_point("Zürich"))
    assert sorted(path.name for path in (tmp_path / "runs").iterdir()) == [RUN_ID, "202609221400"]


def test_read_series_leaves_no_partial_file_behind(forecast_fixture, tmp_path):
    forecast_fixture.read_series("tre200h0", forecast_fixture.find_point("Zürich"))
    assert list(tmp_path.rglob("*.tmp")) == []


def test_read_series_explains_when_a_location_has_no_values(forecast_fixture):
    davos = forecast_fixture.find_point("Davos")
    # Regional entries are published for some parameters and not others
    with pytest.raises(ValueError, match="no 'tre200h0' values for Davos"):
        forecast_fixture.read_series("tre200h0", davos)


# ── weather values ───────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_temperature_reports_the_value_with_the_point_it_resolved(predictions_fixture):
    result = await predictions_fixture.temperature_for_location("Zurich", _swiss_time("2026-09-23T14:00"))

    assert result["value"] == 12.0  # 14:00 Swiss in September is 12:00 UTC
    assert result["unit"] == "°C"
    assert result["location"] == "Zürich 8001 (409 m)"
    assert result["altitude_m"] == 409.0
    assert result["valid_at"] == "2026-09-23T14:00+02:00"
    assert result["model_run"] == "2026-09-22T15:00+02:00"


@pytest.mark.asyncio
async def test_sunshine_hours_adds_the_minutes_and_reports_hours(predictions_fixture):
    # 08:00 to 16:00 Swiss is 06:00 to 14:00 UTC. The rows stamped 12:00 and 13:00 UTC fall inside:
    # 60 + 45 minutes. The row stamped 06:00 covers 05:00 to 06:00 UTC, before the period starts.
    result = await predictions_fixture.sunshine_hours_for_location(
        "Zurich", _swiss_time("2026-09-23T08:00"), _swiss_time("2026-09-23T16:00")
    )
    assert result["value"] == 1.8
    assert result["unit"] == "h"
    assert result["from"] == "2026-09-23T08:00+02:00"
    assert result["to"] == "2026-09-23T16:00+02:00"


@pytest.mark.asyncio
async def test_a_window_counts_the_hours_by_the_stamp_at_their_end(predictions_fixture):
    # 14:00 to 15:00 Swiss is the row stamped 13:00 UTC, 45 minutes. The row stamped 12:00 UTC
    # covers 13:00 to 14:00 Swiss, the hour before the period, and must not be counted.
    result = await predictions_fixture.sunshine_hours_for_location(
        "Zurich", _swiss_time("2026-09-23T14:00"), _swiss_time("2026-09-23T15:00")
    )
    assert result["value"] == 0.8


@pytest.mark.asyncio
async def test_a_time_inside_an_hour_reads_the_row_that_closes_that_hour(predictions_fixture):
    # 14:30 Swiss lies in the hour 14:00 to 15:00, which is the row stamped 13:00 UTC
    result = await predictions_fixture.temperature_for_location("Zurich", _swiss_time("2026-09-23T14:30"))
    assert result["value"] == 14.5


@pytest.mark.asyncio
async def test_cloud_cover_combines_the_overlapping_layers(predictions_fixture):
    result = await predictions_fixture.total_cloud_cover_for_location("Zurich", _swiss_time("2026-09-23T14:00"))

    # Half the sky low and half high: clear only where both are clear, 1 - 0.5 * 0.5 = 75%
    assert result["value"] == 75.0
    assert (result["low_percent"], result["medium_percent"], result["high_percent"]) == (50.0, 0.0, 50.0)


@pytest.mark.asyncio
async def test_cloud_cover_reads_the_nearest_snapshot(predictions_fixture):
    # Cloud cover is a value at the moment of its stamp, not an average over the hour before, so
    # 14:20 Swiss reads the 14:00 snapshot (12:00 UTC) rather than the one closing that hour
    result = await predictions_fixture.total_cloud_cover_for_location("Zurich", _swiss_time("2026-09-23T14:20"))
    assert result["value"] == 75.0


@pytest.mark.asyncio
async def test_weather_description_turns_the_code_into_words(predictions_fixture):
    result = await predictions_fixture.weather_description_for_location("Zurich", _swiss_time("2026-09-23T14:00"))

    assert result["value"] == "mostly sunny, some clouds"
    assert result["pictogram_code"] == 2


@pytest.mark.asyncio
async def test_a_time_outside_the_forecast_names_the_range_that_is_covered(predictions_fixture):
    # Times are written as the answers write them, with the offset of their own date: 30 October is
    # already winter time, the covered range is still summer time
    with pytest.raises(ValueError) as raised:
        await predictions_fixture.temperature_for_location("Zurich", _swiss_time("2026-10-30T14:00"))

    assert str(raised.value).startswith("2026-10-30T14:00+01:00 is outside the forecast")
    assert "covers 2026-09-23T08:00+02:00 to 2026-09-23T15:00+02:00" in str(raised.value)


@pytest.mark.asyncio
async def test_daily_forecast_refuses_a_day_outside_the_forecast(predictions_fixture):
    # Every parameter exists, but none reaches that day, so there is nothing to answer with
    with pytest.raises(ValueError, match="no daily forecast"):
        await predictions_fixture.daily_forecast_for_location("Zurich", date(2026, 10, 30))


@pytest.mark.asyncio
async def test_daily_forecast_reports_a_missing_parameter_instead_of_failing(predictions_fixture):
    # Only tre200px is published in this fixture, the other daily parameters are absent
    result = await predictions_fixture.daily_forecast_for_location("Zurich", date(2026, 9, 23))

    assert result["temperature_max_c"] == 20.6
    assert result["rainfall_median_mm"] is None
    assert result["location"] == "Zürich 8001 (409 m)"
    assert result["date"] == "2026-09-23"
