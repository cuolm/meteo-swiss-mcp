# tests/test_meteo_swiss_predictions.py
import pathlib
import sys
import numpy as np
import pytest
import xarray as xr

root_path = pathlib.Path(__file__).parent.parent.absolute()
src_path = str(root_path / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import the module under test
from meteo_swiss_predictions import MeteoSwissPredictions, NUM_GRID_POINTS_X, NUM_GRID_POINTS_Y


@pytest.fixture
def meteo_swiss_fixture(monkeypatch) -> MeteoSwissPredictions:
    """
    Ensure the NOMINATIM_USER_AGENT env var is set so the class can be instantiated.
    Return an instance of MeteoSwissPredictions.
    """
    monkeypatch.setenv("NOMINATIM_USER_AGENT", "pytest-agent")
    meteo_swiss = MeteoSwissPredictions()
    return meteo_swiss 


@pytest.fixture
def raw_data_period_fixture() -> xr.DataArray:
    eps, ref_time, lead_time, y, x = 20, 1, 2, 7, 7

    # Create data where difference along lead_time = 3600 everywhere
    # So, data[:, :, 1, :, :] - data[:, :, 0, :, :] == 3600
    data = np.zeros((eps, ref_time, lead_time, y, x))

    # Set the first lead_time slice to zeros
    data[:, :, 0, :, :] = 0

    # Set the second lead_time slice to 3600 
    data[:, :, 1, :, :] = 3600 

    coords = {
        "eps": np.arange(eps),
        "ref_time": [0],
        "lead_time": np.arange(lead_time),
        "y": np.arange(y),
        "x": np.arange(x)
    }

    dims = ("eps", "ref_time", "lead_time", "y", "x")
    return xr.DataArray(data, coords=coords, dims=dims)


@pytest.fixture
def raw_data_lead_time_fixture() -> xr.DataArray:
    eps, ref_time, lead_time, y, x = 20, 1, 1, 7, 7

    data = np.full((eps, ref_time, lead_time, y, x), 1000)

    coords = {
        "eps": np.arange(eps),
        "ref_time": [0],
        "lead_time": [0],
        "y": np.arange(y),
        "x": np.arange(x)
    }

    dims = ("eps", "ref_time", "lead_time", "y", "x")
    return xr.DataArray(data, coords=coords, dims=dims)


@pytest.mark.asyncio
async def test_calc_distance_offset_in_degree_valid(meteo_swiss_fixture):
    # Call with typical valid values
    result = meteo_swiss_fixture._calc_distance_offset_in_degree(
        num_grid_points_x=7, num_grid_points_y=7, res_x_km=1, res_y_km=1
    )
    
    # Expected km per degree constants and calculation
    km_per_degree_x = 111.2 * np.cos(np.radians(46))
    km_per_degree_y = 111.2
    expected_x = (7 - 1) / 2 * (1 / km_per_degree_x)
    expected_y = (7 - 1) / 2 * (1 / km_per_degree_y)
    
    assert isinstance(result, tuple)
    assert abs(result.x - expected_x) < 1e-6
    assert abs(result.y - expected_y) < 1e-6


@pytest.mark.asyncio
async def test_fetch_variable_over_period(mocker, meteo_swiss_fixture):
    ogd_api_mock = mocker.patch("meteodatalab.ogd_api.get_from_ogd", return_value="dummy_raw_data")
    iconremap_mock = mocker.patch("meteodatalab.operators.regrid.iconremap", return_value="dummy_remapped_data")

    result = await meteo_swiss_fixture._fetch_variable_over_period(
        location="Zurich",
        variable="TOT_PREC",
        lead_time_start=6,
        lead_time_end=16
    )

    ogd_api_mock.assert_called_once()
    iconremap_mock.assert_called_once()
    assert result == "dummy_remapped_data"


@pytest.mark.asyncio
async def test_fetch_variable_at_lead_time(mocker, meteo_swiss_fixture):
    ogd_api_mock = mocker.patch("meteodatalab.ogd_api.get_from_ogd", return_value="dummy_raw_data")
    iconremap_mock = mocker.patch("meteodatalab.operators.regrid.iconremap", return_value="dummy_remapped_data")

    result = await meteo_swiss_fixture._fetch_variable_at_lead_time(
        location="Zurich",
        variable="TOT_PREC",
        lead_time=6,
    )

    ogd_api_mock.assert_called_once()
    iconremap_mock.assert_called_once()
    assert result == "dummy_remapped_data"
    

@pytest.mark.asyncio
async def test_get_latlon_for_location_success(mocker, meteo_swiss_fixture):
    # Replace cache with an empty dict to ensure no cache hits
    meteo_swiss_fixture.geocode_cache = {}
    
    location_mock = mocker.Mock(latitude=47.37, longitude=8.54)
    mocker.patch("geopy.geocoders.Nominatim.geocode", return_value=location_mock)

    lat, lon = await meteo_swiss_fixture._get_latlon_for_location("Zurich")
    
    assert lat == 47.37
    assert lon == 8.54


@pytest.mark.asyncio
async def test_get_latlon_for_location_cache_hit_success(mocker, meteo_swiss_fixture):
    meteo_swiss_fixture.geocode_cache = {
        "Zurich": (47.37, 8.54)
    }

    # Patch geocode to raise error if called (to ensure it is NOT called)
    mocker.patch("geopy.geocoders.Nominatim.geocode", side_effect=Exception("Should not be called"))

    lat, lon = await meteo_swiss_fixture._get_latlon_for_location("Zurich")
    
    assert lat == 47.37
    assert lon == 8.54


@pytest.mark.asyncio
async def test_get_latlon_for_location_fail(mocker, meteo_swiss_fixture):
    mocker.patch("geopy.geocoders.Nominatim.geocode", return_value=None)

    with pytest.raises(ValueError):
        await meteo_swiss_fixture._get_latlon_for_location("InvalidCity")


@pytest.mark.asyncio
async def test_total_rainfall_for_location(mocker, meteo_swiss_fixture, raw_data_period_fixture):
    meteo_swiss_fixture._get_latlon_for_location = mocker.AsyncMock(return_value=(47.37, 8.54))
    meteo_swiss_fixture._fetch_variable_over_period = mocker.AsyncMock(return_value=raw_data_period_fixture)

    result = await meteo_swiss_fixture.total_rainfall_for_location("Zurich", 6, 16)
    
    assert isinstance(result, float)
    assert result == 3600


@pytest.mark.asyncio
async def test_sunshine_hours_for_location(mocker, meteo_swiss_fixture, raw_data_period_fixture):
    meteo_swiss_fixture._get_latlon_for_location = mocker.AsyncMock(return_value=(47.37, 8.54))
    meteo_swiss_fixture._fetch_variable_over_period = mocker.AsyncMock(return_value=raw_data_period_fixture)

    result = await meteo_swiss_fixture.sunshine_hours_for_location("Zurich", 6, 16)
    
    assert isinstance(result, float)
    assert result == 1


@pytest.mark.asyncio
async def test_wind_speed_for_location(mocker, meteo_swiss_fixture, raw_data_lead_time_fixture):
    meteo_swiss_fixture._get_latlon_for_location = mocker.AsyncMock(return_value=(47.37, 8.54))
    meteo_swiss_fixture._fetch_variable_at_lead_time = mocker.AsyncMock(return_value=raw_data_lead_time_fixture)

    result = await meteo_swiss_fixture.wind_speed_for_location("Zurich", 6)

    assert isinstance(result, float)
    assert result == pytest.approx(1414.2, abs=0.05) 


@pytest.mark.asyncio
async def test_temp_for_location(mocker, meteo_swiss_fixture, raw_data_lead_time_fixture):
    meteo_swiss_fixture._get_latlon_for_location = mocker.AsyncMock(return_value=(47.37, 8.54))
    meteo_swiss_fixture._fetch_variable_at_lead_time = mocker.AsyncMock(return_value=raw_data_lead_time_fixture)

    result = await meteo_swiss_fixture.temp_for_location("Zurich", 6)

    assert isinstance(result, float)
    assert result == 1000 - 273.15  # Celsius conversion check


@pytest.mark.asyncio
async def test_pressure_msl_for_location(mocker, meteo_swiss_fixture, raw_data_lead_time_fixture):
    meteo_swiss_fixture._get_latlon_for_location = mocker.AsyncMock(return_value=(47.37, 8.54))
    meteo_swiss_fixture._fetch_variable_at_lead_time = mocker.AsyncMock(return_value=raw_data_lead_time_fixture)

    result = await meteo_swiss_fixture.pressure_msl_for_location("Zurich", 6)

    assert isinstance(result, float)
    assert result == 1000 


@pytest.mark.asyncio
async def test_total_cloud_cover_for_location(mocker, meteo_swiss_fixture, raw_data_lead_time_fixture):
    meteo_swiss_fixture._get_latlon_for_location = mocker.AsyncMock(return_value=(47.37, 8.54))
    meteo_swiss_fixture._fetch_variable_at_lead_time = mocker.AsyncMock(return_value=raw_data_lead_time_fixture)

    result = await meteo_swiss_fixture.total_cloud_cover_for_location("Zurich", 6)

    assert isinstance(result, float)
    assert result == 1000 


@pytest.mark.asyncio
async def test_snow_depth_for_location(mocker, meteo_swiss_fixture, raw_data_lead_time_fixture):
    meteo_swiss_fixture._get_latlon_for_location = mocker.AsyncMock(return_value=(47.37, 8.54))
    meteo_swiss_fixture._fetch_variable_at_lead_time = mocker.AsyncMock(return_value=raw_data_lead_time_fixture)

    result = await meteo_swiss_fixture.snow_depth_for_location("Zurich", 6)

    assert isinstance(result, float)
    assert result == 1000 


@pytest.mark.asyncio
async def test_total_precipitation_rate_for_location(mocker, meteo_swiss_fixture, raw_data_lead_time_fixture):
    meteo_swiss_fixture._get_latlon_for_location = mocker.AsyncMock(return_value=(47.37, 8.54))
    meteo_swiss_fixture._fetch_variable_at_lead_time = mocker.AsyncMock(return_value=raw_data_lead_time_fixture)

    result = await meteo_swiss_fixture.total_precipitation_rate_for_location("Zurich", 6)

    assert isinstance(result, float)
    assert result == 1000 
