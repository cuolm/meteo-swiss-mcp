from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from meteo_swiss_mcp.server import _lead_time_swiss_to_utc


@pytest.fixture
def freeze_swiss_now_fixture(mocker):
    """
    Return a function that pins datetime.now() inside server.py to a fixed Swiss local instant,
    so the conversion can be tested against a known calendar day.
    """
    def freeze(iso: str) -> None:
        frozen = datetime.fromisoformat(iso).replace(tzinfo=ZoneInfo("Europe/Zurich"))
        datetime_mock = mocker.patch("meteo_swiss_mcp.server.datetime", wraps=datetime)
        datetime_mock.now.return_value = frozen

    return freeze


def test_lead_time_swiss_to_utc_winter(freeze_swiss_now_fixture):
    # In January Switzerland is on CET (UTC+1), so Swiss midnight is one hour before
    # UTC midnight and every lead time shifts down by that one hour.
    freeze_swiss_now_fixture("2026-01-15T09:30:00")

    assert _lead_time_swiss_to_utc(2) == 1
    assert _lead_time_swiss_to_utc(14) == 13
    assert _lead_time_swiss_to_utc(36) == 35


def test_lead_time_swiss_to_utc_summer(freeze_swiss_now_fixture):
    # In July Switzerland is on CEST (UTC+2), so every lead time shifts down by two hours.
    freeze_swiss_now_fixture("2026-07-15T09:30:00")

    assert _lead_time_swiss_to_utc(2) == 0
    assert _lead_time_swiss_to_utc(14) == 12
    assert _lead_time_swiss_to_utc(36) == 34


def test_lead_time_swiss_to_utc_dst_start(freeze_swiss_now_fixture):
    # DST starts 2026-03-29 02:00 Swiss local: the clock jumps straight to 03:00, making this
    # local day 23 hours long. Lead times reaching past the jump lose an extra hour on top of
    # the UTC offset, so 06:00 Swiss is 4 hours after UTC midnight instead of 5.
    freeze_swiss_now_fixture("2026-03-29T09:30:00")

    assert _lead_time_swiss_to_utc(6) == 4
    assert _lead_time_swiss_to_utc(14) == 12
    assert _lead_time_swiss_to_utc(36) == 34


def test_lead_time_swiss_to_utc_dst_end(freeze_swiss_now_fixture):
    # DST ends 2026-10-25 03:00 Swiss local: the clock falls back to 02:00, making this local
    # day 25 hours long. Lead times reaching past the fallback gain an hour back, so 06:00
    # Swiss is 5 hours after UTC midnight instead of 4.
    freeze_swiss_now_fixture("2026-10-25T09:30:00")

    assert _lead_time_swiss_to_utc(6) == 5
    assert _lead_time_swiss_to_utc(14) == 13
    assert _lead_time_swiss_to_utc(36) == 35


def test_lead_time_swiss_to_utc_before_utc_midnight(freeze_swiss_now_fixture):
    # Swiss midnight precedes UTC midnight, so lead times smaller than the UTC offset land
    # before the forecast reference time and go negative. MeteoSwissPredictions rejects those,
    # which is why the tools document a minimum offset of 2 hours.
    freeze_swiss_now_fixture("2026-07-15T09:30:00")

    assert _lead_time_swiss_to_utc(0) == -2
    assert _lead_time_swiss_to_utc(1) == -1


def test_lead_time_swiss_to_utc_returns_int(freeze_swiss_now_fixture):
    freeze_swiss_now_fixture("2026-01-15T09:30:00")

    assert isinstance(_lead_time_swiss_to_utc(14), int)


def test_lead_time_swiss_to_utc_negative_fail(freeze_swiss_now_fixture):
    freeze_swiss_now_fixture("2026-01-15T09:30:00")

    with pytest.raises(ValueError):
        _lead_time_swiss_to_utc(-1)
