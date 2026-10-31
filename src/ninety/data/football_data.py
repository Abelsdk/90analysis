"""football-data.co.uk: results, match stats, referees and odds for domestic leagues."""

import io

import pandas as pd

from ninety.data.cache import fetch_text
from ninety.paths import RAW_DIR

BASE_URL = "https://www.football-data.co.uk/mmz4281"
CACHE_DIR = RAW_DIR / "football_data"

# Our country code -> football-data's top-division file code.
TOP_DIVISIONS = {"ENG": "E0", "ESP": "SP1", "GER": "D1", "ITA": "I1", "FRA": "F1"}

COLUMNS = {
    "Date": "date",
    "HomeTeam": "home",
    "AwayTeam": "away",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
}


def season_code(season: str) -> str:
    """'2023-24' -> '2324', the folder name football-data uses."""
    start, end = season.split("-")
    return start[-2:] + end[-2:]


def parse_matches(text: str) -> pd.DataFrame:
    """Keep the result columns, drop blank trailing rows, parse day-first dates."""
    frame = pd.read_csv(io.StringIO(text), usecols=lambda name: name in COLUMNS)
    frame = frame.rename(columns=COLUMNS).dropna(subset=["home", "away", "result"])
    frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, format="mixed")
    frame["home_goals"] = frame["home_goals"].astype(int)
    frame["away_goals"] = frame["away_goals"].astype(int)
    return frame.reset_index(drop=True)


def load_season(country: str, season: str, *, finished: bool = True) -> pd.DataFrame:
    """All played matches of one top-division season.

    A finished season never changes, so its file is cached forever; the current one for 12 hours.
    """
    division = TOP_DIVISIONS[country]
    code = season_code(season)
    text = fetch_text(
        f"{BASE_URL}/{code}/{division}.csv",
        CACHE_DIR / code / f"{division}.csv",
        max_age_hours=None if finished else 12,
    )
    frame = parse_matches(text)
    frame.insert(0, "season", season)
    frame.insert(0, "country", country)
    return frame
