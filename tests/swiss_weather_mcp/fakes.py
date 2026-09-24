"""Fake MeteoSwiss data and responses shared by the tests."""
from datetime import datetime
from zoneinfo import ZoneInfo

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


def build_point_table_csv() -> bytes:
    lines = [POINT_TABLE_COLUMNS]
    for point_id, type_id, abbr, postal_code, name, altitude in POINT_TABLE_ROWS:
        lines.append(
            f"{point_id};{type_id};{abbr};{postal_code};{name};Ort;Lieu;Luogo;Place;{altitude};0;0;47.0;8.0"
        )
    return ("\r\n".join(lines) + "\r\n").encode("latin-1")


def build_parameter_csv(parameter: str, values: dict, point: str = "800100;2") -> bytes:
    """Build a parameter file the way MeteoSwiss publishes it, header and all locations included."""
    lines = [f"point_id;point_type_id;Date;{parameter}"]
    for stamp, value in values.items():
        lines.append(f"{point};{stamp};{value}")
        lines.append(f"999999;2;{stamp};-1")  # another location, which must be filtered out
    return ("\r\n".join(lines) + "\r\n").encode("latin-1")


class FakeResponse:
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


def build_stac_item(run_id: str, parameters) -> dict:
    return {
        "assets": {
            f"vnut12.lssw.{run_id}.{parameter}.csv": {"href": f"https://example.test/{run_id}/{parameter}.csv"}
            for parameter in parameters
        }
    }


def build_swiss_time(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp).replace(tzinfo=ZoneInfo("Europe/Zurich"))
