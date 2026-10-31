"""ClubElo ratings (api.clubelo.com): one Elo number per club per day."""

import io
from datetime import date

import pandas as pd

from ninety.data.cache import fetch_text
from ninety.paths import RAW_DIR

BASE_URL = "http://api.clubelo.com"
CACHE_DIR = RAW_DIR / "clubelo"


def parse_ratings(text: str) -> pd.DataFrame:
    """Parse ClubElo's CSV (Rank,Club,Country,Level,Elo,From,To) into tidy columns."""
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [column.lower() for column in frame.columns]
    frame["from"] = pd.to_datetime(frame["from"])
    frame["to"] = pd.to_datetime(frame["to"])
    return frame[["club", "country", "level", "elo", "from", "to"]]


def snapshot(on: date) -> pd.DataFrame:
    """Every club's rating on one day."""
    is_past = on < date.today()
    text = fetch_text(
        f"{BASE_URL}/{on.isoformat()}",
        CACHE_DIR / "snapshot" / f"{on.isoformat()}.csv",
        max_age_hours=None if is_past else 6,
    )
    return parse_ratings(text)


def club_history(club: str) -> pd.DataFrame:
    """One club's full rating history, one row per period between matches."""
    slug = club.replace(" ", "")
    text = fetch_text(
        f"{BASE_URL}/{slug}",
        CACHE_DIR / "club" / f"{slug}.csv",
        max_age_hours=24,
    )
    return parse_ratings(text)


def rating_before(history: pd.DataFrame, kickoff: pd.Timestamp) -> float:
    """The club's rating going into a match: the period covering the day before kickoff."""
    day_before = kickoff.normalize() - pd.Timedelta(days=1)
    rows = history[(history["from"] <= day_before) & (history["to"] >= day_before)]
    if rows.empty:
        return float("nan")
    return float(rows["elo"].iloc[0])
