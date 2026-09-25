import pytest

from fakes import RUN_ID, FakeResponse, build_parameter_csv, build_point_table_csv, build_stac_item
from swiss_weather_mcp.forecast import ForecastService
from swiss_weather_mcp.meteoswiss import LocalForecastSource


@pytest.fixture
def source_fixture(mocker, tmp_path):
    """
    Return a LocalForecastSource backed by fake HTTP responses, plus the mock, so tests can count
    downloads and swap in different published data.
    """
    published = {
        "tre200h0": {"202609230600": "6.0", "202609231200": "12.0", "202609231300": "14.5"},
        "sre000h0": {"202609230600": "30", "202609231200": "60", "202609231300": "45"},
        "nprolohs": {"202609231200": "0.50"},
        "npromths": {"202609231200": "0.00"},
        "nprohihs": {"202609231200": "0.50"},
        "jww003i0": {"202609231200": "2"},
        "tre200px": {"202609230000": "20.6", "202609240000": "18.4"},
        # Two 3-hour blocks, 12:00 to 15:00 and 15:00 to 18:00 UTC
        "rp0003i0": {"202609231500": "40", "202609231800": "80"},
        "rre003i0": {"202609231500": "0.0", "202609231800": "2.4"},
        "rreq90h0": {"202609231300": "0.2", "202609231400": "1.1", "202609231500": "0.4",
                     "202609231600": "0.9", "202609231700": "3.5", "202609231800": "2.0"},
    }

    def fake_get(url, **kwargs):
        if url.endswith("meta_point.csv"):
            return FakeResponse(body=build_point_table_csv())
        if "/items/" in url:
            return FakeResponse(payload=build_stac_item(RUN_ID, published))
        parameter = url.rsplit("/", 1)[-1].removesuffix(".csv")
        return FakeResponse(body=build_parameter_csv(parameter, published[parameter]))

    get_mock = mocker.patch("swiss_weather_mcp.meteoswiss.requests.get", side_effect=fake_get)
    forecast_source = LocalForecastSource(tmp_path)
    forecast_source.get_mock = get_mock
    return forecast_source


@pytest.fixture
def service_fixture(source_fixture):
    """Return a ForecastService reading from the fake data source."""
    return ForecastService(source_fixture)
