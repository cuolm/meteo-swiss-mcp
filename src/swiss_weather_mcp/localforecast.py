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


class Point(NamedTuple):
    """A forecast location as the MeteoSwiss point table describes it."""
    point_id: str
    point_type_id: str
    name: str
    postal_code: str
    altitude_m: float

    @property
    def label(self) -> str:
        """Name the point with its postal code and altitude, such as "Zermatt 3920 (1610 m)"."""
        name = f"{self.name} {self.postal_code}" if self.postal_code else self.name
        return f"{name} ({self.altitude_m:.0f} m)"

    @property
    def row_prefix(self) -> bytes:
        """The start every data row of this point has, such as b"800100;2;"."""
        return f"{self.point_id};{self.point_type_id};".encode()


class Series(NamedTuple):
    """One parameter over the whole forecast window at one point, with the run it came from."""
    run_time: datetime
    values: Dict[datetime, float]


def _by_preference(point: Point) -> Tuple[bool, str, int]:
    """
    Sort key for points with the same name: postal code centres before stations, then the
    lowest postal code, which is the historic centre of a city, then the lowest point id.
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
    """Write the rows of one point from a streamed response to a file."""
    prefix = point.row_prefix
    # Chunks are split by hand, iter_lines() takes about 45 seconds for the million lines of a file
    remainder = b""
    for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
        lines = (remainder + chunk).split(b"\n")
        remainder = lines.pop()  # the last piece may be half a line
        for line in lines:
            if line.startswith(prefix):
                file.write(line + b"\n")
    if remainder.startswith(prefix):
        file.write(remainder + b"\n")


def _download(url: str, target: Path, only_point: Optional[Point] = None) -> None:
    """
    Stream a file from MeteoSwiss to disk.

    Parameters:
        url (str): The file to download.
        target (Path): Where the finished file ends up.
        only_point (Optional[Point]): Keep only this point's rows, or every line when None.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    # A unique name, moved into place only when complete, so an interrupted download is never
    # taken for a cached file, and two requests for the same file do not write into each other
    partial_file = target.with_name(f"{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            with open(partial_file, "wb") as file:
                if only_point is None:
                    for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                        file.write(chunk)
                else:
                    _write_point_rows(response, only_point, file)
        partial_file.replace(target)
    finally:
        partial_file.unlink(missing_ok=True)


class LocalForecast:
    """Read point forecasts from the MeteoSwiss local forecasting collection, cached per model run."""

    def __init__(self, cache_dir: Path, cache_all_locations: bool = False):
        self.cache_dir = cache_dir
        self.cache_all_locations = cache_all_locations
        self.points: List[Point] = []
        # The newest run, its parameter file URLs, and when that was last asked for
        self.run_id: Optional[str] = None
        self.run_assets: Dict[str, str] = {}
        self.run_checked_at: Optional[datetime] = None

    def _ensure_point_table(self) -> Path:
        """Return the cached point table, downloading it when it is missing or stale."""
        table = self.cache_dir / "ogd-local-forecasting_meta_point.csv"
        if table.exists():
            age = datetime.now(timezone.utc) - datetime.fromtimestamp(table.stat().st_mtime, timezone.utc)
            if age < POINT_TABLE_MAX_AGE:
                return table

        logger.info("Downloading the MeteoSwiss forecast point table")
        _download(POINT_TABLE_URL, table)
        return table

    def _load_points(self) -> List[Point]:
        """Read the point table into memory once per process."""
        if self.points:
            return self.points

        with open(self._ensure_point_table(), newline="", encoding="latin-1") as file:
            for row in csv.DictReader(file, delimiter=";"):
                self.points.append(Point(
                    point_id=row["point_id"],
                    point_type_id=row["point_type_id"],
                    name=row["point_name"],
                    postal_code=row["postal_code"],
                    altitude_m=float(row["point_height_masl"]),
                ))
        logger.info(f"Loaded {len(self.points)} forecast locations")
        return self.points

    def find_point(self, location: str) -> Point:
        """
        Find the forecast point for a location name or a Swiss postal code.

        Case and accents are ignored, otherwise only exact matches count. When several points
        match, the one with the lowest postal code is used, and a weather station otherwise.

        Parameters:
            location (str): Location name (e.g., "Zurich") or postal code (e.g., "8001").

        Returns:
            Point: The resolved forecast point.
        """
        points = self._load_points()
        wanted_name = _normalise(location)

        # Exact matches only: the nearest name to "Wallis" is "Wallisellen", 150 km from the canton
        matches = [point for point in points if point.postal_code == wanted_name]
        if not matches:
            matches = [point for point in points if _normalise(point.name) == wanted_name]
        if not matches:
            raise ValueError(
                f"Location '{location}' is not one of the {len(points)} places MeteoSwiss publishes "
                f"forecasts for. Check the spelling, or try a nearby town, village or postal code."
            )

        return min(matches, key=_by_preference)

    def _fetch_assets_by_run(self, day: datetime) -> Dict[str, Dict[str, str]]:
        """Return one daily STAC item's assets, grouped by run and keyed by parameter."""
        item_url = f"{STAC_BASE_URL}/collections/{COLLECTION_ID}/items/{day.strftime('%Y%m%d')}-ch"
        response = requests.get(item_url, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code == 404:
            return {}
        response.raise_for_status()

        assets_by_run: Dict[str, Dict[str, str]] = {}
        for key, asset in response.json().get("assets", {}).items():
            # Asset names look like vnut12.lssw.<run>.<parameter>.csv
            parts = key.split(".")
            if len(parts) == 5:
                assets_by_run.setdefault(parts[2], {})[parts[3]] = asset["href"]
        return assets_by_run

    def find_latest_run(self) -> Tuple[str, Dict[str, str]]:
        """
        Find the newest published run and the parameter files it holds, reusing the answer for
        RUN_LOOKUP_MAX_AGE.

        Returns:
            Tuple[str, Dict[str, str]]: The run stamp (YYYYMMDDHHMM) and its parameter file URLs.
        """
        now = datetime.now(timezone.utc)
        if self.run_checked_at and now - self.run_checked_at < RUN_LOOKUP_MAX_AGE:
            return self.run_id, self.run_assets

        # A day's item exists before its first run lands, so just after midnight it can be empty
        today = datetime.now(SWISS_TZ)
        for day in (today, today - timedelta(days=1)):
            assets_by_run = self._fetch_assets_by_run(day)
            if assets_by_run:
                # The stamp is fixed width and zero padded, so the newest run is the largest string
                self.run_id = max(assets_by_run)
                self.run_assets = assets_by_run[self.run_id]
                self.run_checked_at = now
                return self.run_id, self.run_assets

        raise RuntimeError(
            "The MeteoSwiss local forecasting collection published no run for today or yesterday"
        )

    def _drop_superseded_runs(self, current_run_id: str) -> None:
        """Remove the folders of runs older than the current one, except the run just before it."""
        # Run IDs are fixed width, so comparing them as text compares them in time
        older_runs = []
        for folder in (self.cache_dir / "runs").iterdir():
            if folder.is_dir() and folder.name < current_run_id:
                older_runs.append(folder)

        older_runs.sort()

        # The last of the older runs is the previous one. With stdio every client session starts
        # its own server process, and one may still read that run for up to RUN_LOOKUP_MAX_AGE.
        # Runs are about an hour apart, so no process is more than one run behind.
        for folder in older_runs[:-1]:
            logger.info(f"Dropping superseded run {folder.name}")
            shutil.rmtree(folder, ignore_errors=True)

    def _get_cached_file(self, parameter: str, point: Point, run_id: str, url: str) -> Path:
        """Return the cached file with this parameter for this point and run, downloading it if missing."""
        # Named by the MeteoSwiss run ID, which is UTC: Swiss time repeats an hour in October
        run_dir = self.cache_dir / "runs" / run_id
        full_file = run_dir / f"{parameter}.csv"
        point_file = run_dir / f"{parameter}_{point.point_id}_{point.point_type_id}.csv"

        # A full file from an earlier download serves every point
        if full_file.exists():
            return full_file
        if point_file.exists():
            return point_file

        # The published file holds every location and runs to tens of megabytes. By default only
        # this point's rows are kept, a few kilobytes, and the rest is discarded while streaming.
        logger.info(f"Reading {parameter} for {point.label} from run {run_id}")
        if self.cache_all_locations:
            _download(url, full_file)
            target = full_file
        else:
            _download(url, point_file, only_point=point)
            target = point_file

        self._drop_superseded_runs(run_id)
        return target

    def _read_point_values(self, path: Path, point: Point) -> Dict[datetime, float]:
        """Read one point's values from a cached file, keyed by UTC timestamp."""
        prefix = point.row_prefix
        values: Dict[datetime, float] = {}

        with open(path, "rb") as file:
            for line in file:
                # This also skips the header of a full file, a point extract has none
                if not line.startswith(prefix):
                    continue
                _, _, stamp, value = line.decode("latin-1").strip().split(";")
                try:
                    values[_parse_stamp(stamp)] = float(value)
                except ValueError:
                    continue  # gaps are published as empty fields

        return values

    def read_series(self, parameter: str, point: Point) -> Series:
        """
        Read one parameter over the whole forecast window at one point.

        Parameters:
            parameter (str): MeteoSwiss parameter shortname (e.g., "tre200h0", "fu3010h0").
            point (Point): The resolved forecast point.

        Returns:
            Series: The run the values came from, and the values keyed by UTC timestamp.
        """
        run_id, assets = self.find_latest_run()
        if parameter not in assets:
            # The run and the published codes help whoever finds out what MeteoSwiss changed, but mean
            # nothing to the model reading the error, so they go to the log only
            logger.warning(f"Run {run_id} does not publish '{parameter}', it has: {', '.join(sorted(assets))}")
            raise ValueError(f"MeteoSwiss's newest forecast does not include '{parameter}'.")

        values = self._read_point_values(self._get_cached_file(parameter, point, run_id, assets[parameter]), point)
        if not values:
            raise ValueError(
                f"MeteoSwiss publishes no '{parameter}' values for {point.label}. Some entries, "
                f"such as the regional ones, only carry part of the forecast, try a nearby town."
            )

        return Series(run_time=_parse_stamp(run_id), values=values)
