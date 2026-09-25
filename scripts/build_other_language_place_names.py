"""
Build src/swiss_weather_mcp/other_language_place_names.csv: names of Swiss places in other
languages, each mapped to the MeteoSwiss point name, so that "Genf" or "Geneva" finds "Genève".

Every name comes from one of the Wikipedia lists in SOURCES; nothing is written by hand. A row is
kept only when one name of a place is a MeteoSwiss point name and the other name is not.

Run from the project root: uv run python scripts/build_other_language_place_names.py
"""
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests

from swiss_weather_mcp.meteoswiss import POINT_TABLE_URL

OUTPUT_FILE = Path("src/swiss_weather_mcp/other_language_place_names.csv")
LANGUAGES = ("de", "fr", "it", "rm", "en")
REQUEST_TIMEOUT_SECONDS = 60
USER_AGENT = "swiss-weather-mcp place name builder (https://github.com/cuolm/swiss-weather-mcp)"

GERMAN_LIST = ("de", "Liste deutscher Bezeichnungen von Schweizer Orten")
FRENCH_LIST = ("de", "Liste französischer Bezeichnungen von Schweizer Orten")
ROMANSH_LIST = ("de", "Liste rätoromanischer Bezeichnungen von Schweizer Orten")
ENGLISH_LIST = ("en", "List of places in Switzerland")
SOURCES = (GERMAN_LIST, FRENCH_LIST, ROMANSH_LIST, ENGLISH_LIST)

# The German list marks the language of each place: F(rench), I(talian), R(omansh)
LIST_LANGUAGES = {"F": "fr", "I": "it", "R": "rm"}

# A place and all the names one source gives it, each with its language
PlaceNames = List[Tuple[str, str]]


def fetch_wikitext(site: str, title: str) -> str:
    """Fetch the wiki source of one Wikipedia page."""
    response = requests.get(
        f"https://{site}.wikipedia.org/w/api.php",
        params={"action": "parse", "page": title, "prop": "wikitext", "format": "json", "formatversion": 2},
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["parse"]["wikitext"]


def page_url(site: str, title: str) -> str:
    """Return the address of a Wikipedia page, such as https://de.wikipedia.org/wiki/Liste_..."""
    return f"https://{site}.wikipedia.org/wiki/{title.replace(' ', '_')}"


def clean_markup(text: str) -> str:
    """Turn wiki markup into plain text: links keep their label, audio templates their spoken name."""
    text = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", text)
    text = re.sub(r"\{\{Audio\|[^|}]*\|([^}]*)\}\}", r"\1", text)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = text.replace("&nbsp;", " ")
    return " ".join(text.split())


def clean_name(text: str) -> str:
    """Turn one name in wiki markup into plain text, without remarks in parentheses."""
    # Remarks can sit inside a template, so they go after the markup
    return re.sub(r"\([^)]*\)", "", clean_markup(text)).strip(" ;.")


def split_names(text: str) -> List[str]:
    """Split a list of names at commas and "oder", dropping remarks in parentheses and historic names."""
    # Remarks in parentheses can contain commas, so they go before the split
    text = clean_name(text)
    names = []
    for piece in re.split(r",| oder ", text):
        # The German list writes historic names, hardly known today, in italics
        if "''" in piece:
            continue
        name = piece.strip(" ;.")
        if name:
            names.append(name)
    return names


def read_german_list(wikitext: str) -> List[PlaceNames]:
    """Read lines like "* Sion (F): [[Sitten]]": a place, its language, then its German names."""
    places = []
    for line in wikitext.splitlines():
        match = re.match(r"^\*\s*(?P<place>.+?)\s*\((?P<marker>[FIR])[^)]*\)\s*:\s*(?P<german>.+)$", line)
        if not match:
            continue
        place = clean_name(match["place"])
        names = [(place, LIST_LANGUAGES[match["marker"]])]
        names.extend((name, "de") for name in split_names(match["german"]))
        places.append(names)
    return places


def read_french_list(wikitext: str) -> List[PlaceNames]:
    """Read table rows like "| Bâle || [[Basel]]": the French name, then the German one."""
    places = []
    for line in wikitext.splitlines():
        if not line.startswith("|") or "||" not in line:
            continue
        french, german = line[1:].split("||", 1)
        french_names = split_names(french)
        german_names = split_names(german)
        if french_names and german_names:
            places.append([(name, "fr") for name in french_names] + [(german_names[0], "de")])
    return places


def read_romansh_list(wikitext: str) -> List[PlaceNames]:
    """Read lines like "* Cuira (Rumantsch Grischun): [[Chur]]": Romansh names, then the usual name."""
    places = []
    for line in wikitext.splitlines():
        if not line.startswith("*") or ":" not in line:
            continue
        romansh, usual = line[1:].split(":", 1)
        romansh_names = split_names(romansh)
        # "[[Bern]], auch [[Kanton Bern]]": the place comes first, the rest are remarks
        usual_names = split_names(usual.split(", auch")[0])
        if romansh_names and usual_names:
            places.append([(name, "rm") for name in romansh_names] + [(usual_names[0], "de")])
    return places


def read_english_list(wikitext: str) -> List[PlaceNames]:
    """Read table rows: the English name first, "de: Luzern, fr: Lucerne, it: Lucerna" last."""
    places = []
    for line in wikitext.splitlines():
        if not line.startswith("|") or "||" not in line:
            continue
        columns = line[1:].split("||")
        english = clean_name(columns[0])
        names = [(english, "en")]
        for language, name in re.findall(r"\b([a-z]{2})\s*:\s*([^,]+)", columns[-1]):
            if language in LANGUAGES:
                names.append((clean_name(name), language))
        if len(names) > 1:
            places.append(names)
    return places


def normalise(name: str) -> str:
    """Fold case and strip accents, the same way the server compares names."""
    folded = unicodedata.normalize("NFKD", name.casefold())
    return "".join(character for character in folded if not unicodedata.combining(character)).strip()


def read_point_names() -> Dict[str, str]:
    """Return the MeteoSwiss point names, keyed by their normalised form."""
    response = requests.get(POINT_TABLE_URL, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    rows = csv.DictReader(response.content.decode("latin-1").splitlines(), delimiter=";")
    point_names: Dict[str, str] = {}
    for row in rows:
        point_names.setdefault(normalise(row["point_name"]), row["point_name"])
    return point_names


def build_rows(point_names: Dict[str, str]) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, int]]:
    """Match every place of every source against the point names; return the rows and a report."""
    readers = {GERMAN_LIST: read_german_list, FRENCH_LIST: read_french_list,
               ROMANSH_LIST: read_romansh_list, ENGLISH_LIST: read_english_list}
    report: Dict[str, int] = defaultdict(int)
    point_names_by_other_language_place_name: Dict[str, Set[str]] = defaultdict(set)
    rows = {}
    for source in SOURCES:
        site, title = source
        source_label = page_url(site, title)
        places = readers[source](fetch_wikitext(site, title))
        report[f"places on {source_label}"] = len(places)
        for names in places:
            points = {point_names[normalise(name)] for name, _ in names if normalise(name) in point_names}
            if len(points) != 1:
                report["skipped: no MeteoSwiss point" if not points else "skipped: names of two different points"] += 1
                continue
            point_name = points.pop()
            for other_language_place_name, language in names:
                # A name that is itself a point name already works, and must never be redirected
                if normalise(other_language_place_name) in point_names:
                    continue
                point_names_by_other_language_place_name[normalise(other_language_place_name)].add(point_name)
                rows.setdefault(
                    (point_name, normalise(other_language_place_name)),
                    (other_language_place_name, language, point_name, source_label),
                )

    conflicting = {name for name, targets in point_names_by_other_language_place_name.items() if len(targets) > 1}
    report["dropped: name points to two places"] = len(conflicting)
    # Sorted by point name, so the names of one place stay together
    kept = [row for (_, name), row in sorted(rows.items()) if name not in conflicting]
    for row in kept:
        report[f"rows from {row[3]}"] += 1
    return kept, report


def write_other_language_place_names(rows: List[Tuple[str, str, str, str]]) -> None:
    """Write the rows with a header naming the sources and their licence."""
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as file:
        file.write("# Generated by scripts/build_other_language_place_names.py. Do not edit by hand.\n")
        file.write("# Names from Wikipedia (CC BY-SA 4.0):\n")
        for site, title in SOURCES:
            file.write(f"#   {page_url(site, title)}\n")
        writer = csv.writer(file, delimiter=";", lineterminator="\n")
        writer.writerow(("other_language_place_name", "language", "meteoswiss_point_name", "source"))
        writer.writerows(rows)


def main() -> None:
    point_names = read_point_names()
    rows, report = build_rows(point_names)
    write_other_language_place_names(rows)
    for line, count in report.items():
        print(f"{count:5}  {line}")
    print(f"{len(rows):5}  rows written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
