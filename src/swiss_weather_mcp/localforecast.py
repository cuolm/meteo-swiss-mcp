import csv
import logging
import shutil
import unicodedata
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Dict, List, NamedTuple, Optional, Tuple
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

COLLECTION_ID = "ch.meteoschweiz.ogd-local-forecasting"
STAC_BASE_URL = "https://data.geo.admin.ch/api/stac/v1"
POINT_TABLE_URL = f"https://data.geo.admin.ch/{COLLECTION_ID}/ogd-local-forecasting_meta_point.csv"

# The item id is built from the Swiss calendar day, the same anchor the published window uses
SWISS_TZ = ZoneInfo("Europe/Zurich")

# The point table only changes when MeteoSwiss adds a location, so it is refetched rarely
POINT_TABLE_MAX_AGE = timedelta(days=7)
# A run is published every hour, rechecking a few times an hour picks a new one up promptly
RUN_LOOKUP_MAX_AGE = timedelta(minutes=5)
REQUEST_TIMEOUT_SECONDS = 60
DOWNLOAD_CHUNK_BYTES = 1 << 20

# MeteoSwiss pictogram codes, published by jp2000d0 (daily) and jww003i0 (3 hourly). Codes above 100
# are the night variant of the same weather. Taken from the MeteoSwiss icon reference sheet, with its
# "cloudly" spelling corrected because these strings are shown to the user.
PICTOGRAM_DESCRIPTIONS = {
    1: "sunny",
    2: "mostly sunny, some clouds",
    3: "partly sunny, thick passing clouds",
    4: "overcast",
    5: "very cloudy",
    6: "sunny intervals, isolated showers",
    7: "sunny intervals, isolated sleet",
    8: "sunny intervals, snow showers",
    9: "overcast, some rain showers",
    10: "overcast, some sleet",
    11: "overcast, some snow showers",
    12: "sunny intervals, chance of thunderstorms",
    13: "sunny intervals, possible thunderstorms",
    14: "very cloudy, light rain",
    15: "very cloudy, light sleet",
    16: "very cloudy, light snow showers",
    17: "very cloudy, intermittent rain",
    18: "very cloudy, intermittent sleet",
    19: "very cloudy, intermittent snow",
    20: "very overcast with rain",
    21: "very overcast with frequent sleet",
    22: "very overcast with heavy snow",
    23: "very overcast, slight chance of storms",
    24: "very overcast with storms",
    25: "very cloudy, very stormy",
    26: "high clouds",
    27: "stratus",
    28: "fog",
    29: "sunny intervals, scattered showers",
    30: "sunny intervals, scattered snow showers",
    31: "sunny intervals, scattered sleet",
    32: "sunny intervals, some showers",
    33: "short sunny intervals, frequent rain",
    34: "short sunny intervals, frequent snowfalls",
    35: "overcast and dry",
    36: "partly sunny, slightly stormy",
    37: "partly sunny, stormy snow showers",
    38: "overcast, thundery showers",
    39: "overcast, thundery snow showers",
    40: "very cloudy, slightly stormy",
    41: "overcast, slightly stormy",
    42: "very cloudy, thundery snow showers",
    101: "clear",
    102: "slightly overcast",
    103: "heavy cloud formations",
    104: "overcast",
    105: "very cloudy",
    106: "overcast, scattered showers",
    107: "overcast, scattered rain and snow showers",
    108: "overcast, snow showers",
    109: "overcast, some showers",
    110: "overcast, some rain and snow showers",
    111: "overcast, some snow showers",
    112: "slightly stormy",
    113: "storms",
    114: "very cloudy, light rain",
    115: "very cloudy, light rain and snow showers",
    116: "very cloudy, light snowfall",
    117: "very cloudy, intermittent rain",
    118: "very cloudy, intermittant mixed rain and snowfall",
    119: "very cloudy, intermittent snowfall",
    120: "very cloudy, constant rain",
    121: "very cloudy, frequent rain and snowfall",
    122: "very cloudy, heavy snowfall",
    123: "very cloudy, slightly stormy",
    124: "very cloudy, stormy",
    125: "very cloudy, storms",
    126: "high cloud",
    127: "stratus",
    128: "fog",
    129: "slightly overcast, scattered showers",
    130: "slightly overcast, scattered snowfall",
    131: "slightly overcast, rain and snow showers",
    132: "slightly overcast, some showers",
    133: "overcast, frequent snow showers",
    134: "overcast, frequent snow showers",
    135: "overcast and dry",
    136: "slightly overcast, slightly stormy",
    137: "slightly overcast, stormy snow showers",
    138: "overcast, thundery showers",
    139: "overcast, thundery snow showers",
    140: "very cloudy, slightly stormy",
    141: "overcast, slightly stormy",
    142: "very cloudy, thundery snow showers",
}


class Point(NamedTuple):
    """A forecast location as the MeteoSwiss point table describes it."""
    point_id: str
    point_type_id: str
    name: str
    postal_code: str
    height_masl: float

    def label(self) -> str:
        """
        Describe the point well enough that the caller can tell which one was picked.

        A city name maps to many points, so the postal code and altitude are what distinguish
        the village of Zermatt from its mountain station.
        """
        name = f"{self.name} {self.postal_code}" if self.postal_code else self.name
        return f"{name} ({self.height_masl:.0f} m)"

    def prefix(self) -> bytes:
        """
        Return the start every data row of this point has, such as b"800100;2;".

        The download keeps rows by this start and the reader finds them by it, so both must build
        it the same way. Building it here is what guarantees that.
        """
        return f"{self.point_id};{self.point_type_id};".encode()


class Series(NamedTuple):
    """One parameter over the whole forecast window at one point, with the run it came from."""
    run: datetime
    values: Dict[datetime, float]


def _preference(point: Point) -> Tuple[bool, str, int]:
    """
    Order candidates so a location always resolves to the same point.

    Postal code centres come before stations, and the lowest code comes first, which is the
    historic centre of a city. The point id breaks the remaining ties and is compared as a
    number, because "100" sorts before "26" when compared as text.
    """
    return (not point.postal_code, point.postal_code, int(point.point_id))


def _normalise(name: str) -> str:
    """
    Fold case and strip accents so "zurich" finds "Zürich".

    Parameters:
        name (str): Location name as the caller wrote it.

    Returns:
        str: Comparable form of the name.
    """
    folded = unicodedata.normalize("NFKD", name.casefold())
    return "".join(character for character in folded if not unicodedata.combining(character)).strip()


def _parse_stamp(stamp: str) -> datetime:
    """Read a MeteoSwiss timestamp such as "202609231200", which is always UTC."""
    return datetime.strptime(stamp, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def _write_point_rows(response: requests.Response, point: Point, file: BinaryIO) -> None:
    """
    Write the rows of one point from a streamed response.

    Megabyte chunks are split by hand rather than read with iter_lines(), which spends about
    45 seconds per file walking a million lines one at a time.
    """
    prefix = point.prefix()
    remainder = b""
    for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
        lines = (remainder + chunk).split(b"\n")
        remainder = lines.pop()  # the last piece may be half a line
        for line in lines:
            if line.startswith(prefix):
                file.write(line + b"\n")
    if remainder.startswith(prefix):
        file.write(remainder + b"\n")


def _download(url: str, target: Path, point: Optional[Point] = None) -> None:
    """
    Stream a file from MeteoSwiss to disk.

    The file is written under a unique temporary name and moved into place only once complete, so
    an interrupted download is never mistaken for a cached file, and two requests fetching the same
    file at once do not write into each other.

    Parameters:
        url (str): The file to download.
        target (Path): Where the finished file ends up.
        point (Optional[Point]): Keep only this point's rows, or every line when None.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    download = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            with open(download, "wb") as file:
                if point is None:
                    for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                        file.write(chunk)
                else:
                    _write_point_rows(response, point, file)
        download.replace(target)
    finally:
        download.unlink(missing_ok=True)


class LocalForecast:
    """
    Read point forecasts from the MeteoSwiss local forecasting collection.

    MeteoSwiss publishes one CSV per parameter and model run, holding every hour of the forecast
    window for every location. The hourly files are about 31 MB, so they are streamed and discarded,
    and only the rows of the requested point are kept. A new run appears every hour under a new file
    name, so a stored file can never go stale, only be superseded.
    """

    def __init__(self, cache_dir: Path, cache_all_locations: bool = False):
        self.cache_dir = cache_dir
        self.cache_all_locations = cache_all_locations
        self.points: List[Point] = []
        # The newest run, its parameter file URLs, and when that was last asked for
        self.run: Optional[str] = None
        self.run_assets: Dict[str, str] = {}
        self.run_checked_at: Optional[datetime] = None

    def _point_table(self) -> Path:
        """Return the cached point table, downloading it when it is missing or stale."""
        table = self.cache_dir / "ogd-local-forecasting_meta_point.csv"
        if table.exists():
            age = datetime.now() - datetime.fromtimestamp(table.stat().st_mtime)
            if age < POINT_TABLE_MAX_AGE:
                return table

        logger.info("Downloading the MeteoSwiss forecast point table")
        _download(POINT_TABLE_URL, table)
        return table

    def _load_points(self) -> List[Point]:
        """Read the point table into memory once per process."""
        if self.points:
            return self.points

        with open(self._point_table(), newline="", encoding="latin-1") as file:
            for row in csv.DictReader(file, delimiter=";"):
                self.points.append(Point(
                    point_id=row["point_id"],
                    point_type_id=row["point_type_id"],
                    name=row["point_name"],
                    postal_code=row["postal_code"],
                    height_masl=float(row["point_height_masl"]),
                ))
        logger.info(f"Loaded {len(self.points)} forecast locations")
        return self.points

    def resolve(self, location: str) -> Point:
        """
        Find the forecast point for a location name or a Swiss postal code.

        Only exact matches are accepted. Near matching was measured to be actively misleading here:
        "Wallis" is closest to "Wallisellen", a Zurich suburb 150 km from the canton it names, so a
        best guess would answer confidently for the wrong side of the country.

        A city spans several postal code areas, so the lowest code is used, which is the historic
        centre. Locations without a postal code entry fall back to their weather station.

        Parameters:
            location (str): Location name (e.g., "Zurich") or postal code (e.g., "8001").

        Returns:
            Point: The resolved forecast point.
        """
        points = self._load_points()
        wanted = _normalise(location)

        matches = [point for point in points if point.postal_code == wanted]
        if not matches:
            matches = [point for point in points if _normalise(point.name) == wanted]
        if not matches:
            raise ValueError(
                f"Location '{location}' is not one of the {len(points)} places MeteoSwiss publishes "
                f"forecasts for. Check the spelling, or try a nearby town, village or postal code."
            )

        return min(matches, key=_preference)

    def _assets_by_run(self, day: datetime) -> Dict[str, Dict[str, str]]:
        """Return one daily STAC item's assets, grouped by run and keyed by parameter."""
        item_url = f"{STAC_BASE_URL}/collections/{COLLECTION_ID}/items/{day.strftime('%Y%m%d')}-ch"
        response = requests.get(item_url, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 404:
            return {}
        response.raise_for_status()

        by_run: Dict[str, Dict[str, str]] = {}
        for key, asset in response.json().get("assets", {}).items():
            # Asset names look like vnut12.lssw.<run>.<parameter>.csv
            parts = key.split(".")
            if len(parts) == 5:
                by_run.setdefault(parts[2], {})[parts[3]] = asset["href"]
        return by_run

    def latest_run(self) -> Tuple[str, Dict[str, str]]:
        """
        Find the newest published run and the parameter files it holds.

        Tomorrow's item already exists but stays empty until its first run lands, hence the fall
        back to the previous day. The answer is held briefly so that a tool needing six parameters
        does not ask the API six times.

        Returns:
            Tuple[str, Dict[str, str]]: The run stamp (YYYYMMDDHHMM) and its parameter file URLs.
        """
        now = datetime.now(timezone.utc)
        if self.run_checked_at and now - self.run_checked_at < RUN_LOOKUP_MAX_AGE:
            return self.run, self.run_assets

        today = datetime.now(SWISS_TZ)
        for day in (today, today - timedelta(days=1)):
            by_run = self._assets_by_run(day)
            if by_run:
                # The stamp is fixed width and zero padded, so the newest run is the largest string
                self.run = max(by_run)
                self.run_assets = by_run[self.run]
                self.run_checked_at = now
                return self.run, self.run_assets

        raise RuntimeError(
            "The MeteoSwiss local forecasting collection published no run for today or yesterday"
        )

    def _drop_superseded_runs(self, current_run: str) -> None:
        """
        Remove the folders of runs that newer ones have replaced, keeping the run just before this
        one.

        Several server processes can share this cache, because with the stdio transport every
        client session starts its own. Each remembers the newest run for up to RUN_LOOKUP_MAX_AGE,
        so one process can still be reading the previous run while another has moved on. Deleting
        that folder would pull the file out from under it. Runs are published about an hour apart,
        measured at 57 to 62 minutes over two days, so a process is never more than one run behind
        and keeping the previous run is enough. For the same reason, newer runs are never touched.

        It runs after a successful download rather than on a timer. The run stamps are fixed width,
        so comparing them as text compares them in time.
        """
        older_runs = []
        for folder in (self.cache_dir / "runs").iterdir():
            if folder.is_dir() and folder.name < current_run:
                older_runs.append(folder)

        older_runs.sort()

        # The last of the older runs is the previous one, which another process may still read
        for folder in older_runs[:-1]:
            logger.info(f"Dropping superseded run {folder.name}")
            shutil.rmtree(folder, ignore_errors=True)

    def _cached_file(self, parameter: str, point: Point, run: str, url: str) -> Path:
        """
        Return the file holding this parameter for this run, fetching it if it is not there yet.

        A run already on disk may have been stored either way round, so a full file is used when
        one exists and a point extract otherwise. Only the current run's folder is looked in,
        which makes a superseded file unreadable rather than merely unwanted.
        """
        run_dir = self.cache_dir / "runs" / run
        full_file = run_dir / f"{parameter}.csv"
        point_file = run_dir / f"{parameter}_{point.point_id}_{point.point_type_id}.csv"

        if full_file.exists():
            return full_file
        if point_file.exists():
            return point_file

        # The published file holds every location and runs to tens of megabytes. By default only
        # this point's rows are kept, a few kilobytes, and the rest is discarded while streaming.
        logger.info(f"Reading {parameter} for {point.label()} from run {run}")
        if self.cache_all_locations:
            _download(url, full_file)
            target = full_file
        else:
            _download(url, point_file, point=point)
            target = point_file

        self._drop_superseded_runs(run)
        return target

    def _read_values(self, path: Path, point: Point) -> Dict[datetime, float]:
        """
        Read one point's values out of a cached file.

        No header is skipped: a point extract has none, and a header line can never start with
        the point prefix, so the same scan serves a full file and an extract alike.
        """
        prefix = point.prefix()
        values: Dict[datetime, float] = {}

        with open(path, "rb") as file:
            for line in file:
                if not line.startswith(prefix):
                    continue
                _, _, stamp, value = line.decode("latin-1").strip().split(";")
                try:
                    values[_parse_stamp(stamp)] = float(value)
                except ValueError:
                    continue  # gaps are published as empty fields

        return values

    def series(self, parameter: str, point: Point) -> Series:
        """
        Read one parameter over the whole forecast window at one point.

        Parameters:
            parameter (str): MeteoSwiss parameter shortname (e.g., "tre200h0", "fu3010h0").
            point (Point): The resolved forecast point.

        Returns:
            Series: The run the values came from, and the values keyed by UTC timestamp.
        """
        run, assets = self.latest_run()
        if parameter not in assets:
            raise ValueError(
                f"Run {run} does not publish parameter '{parameter}', it has: {', '.join(sorted(assets))}"
            )

        values = self._read_values(self._cached_file(parameter, point, run, assets[parameter]), point)
        if not values:
            raise ValueError(
                f"MeteoSwiss publishes no '{parameter}' values for {point.label()}. Some entries, "
                f"such as the regional ones, only carry part of the forecast, try a nearby town."
            )

        return Series(run=_parse_stamp(run), values=values)
