"""Training table for the Elo draw model: past league matches + both clubs' pre-match Elo."""

import difflib
from datetime import date

import pandas as pd

from ninety.data import clubelo, football_data
from ninety.paths import REFERENCE_DIR

ALIASES_CSV = REFERENCE_DIR / "clubelo_aliases.csv"


def season_midpoint(season: str) -> date:
    """Mid-January of the season: every club in that top division has a ClubElo row by then."""
    return date(int(season.split("-")[0]) + 1, 1, 15)


def match_names(names, candidates, aliases: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Map football-data names to ClubElo names: exact, then alias table, then close spelling.

    Returns the mapping and the names that matched nothing (those need a row in the alias table).
    """
    candidates = list(candidates)
    mapping, unmatched = {}, []
    for name in sorted(set(names)):
        if name in candidates:
            mapping[name] = name
        elif name in aliases:
            mapping[name] = aliases[name]
        elif close := difflib.get_close_matches(name, candidates, n=1, cutoff=0.8):
            mapping[name] = close[0]
        else:
            unmatched.append(name)
    return mapping, unmatched


def build(seasons: list[str], countries=tuple(football_data.TOP_DIVISIONS)) -> pd.DataFrame:
    """One row per match with elo_home, elo_away and the result (H/D/A)."""
    aliases = dict(pd.read_csv(ALIASES_CSV).itertuples(index=False, name=None))
    matches = pd.concat(
        [football_data.load_season(c, s) for c in countries for s in seasons], ignore_index=True
    )

    clubs_by_country: dict[str, set[str]] = {}
    for season in seasons:
        snap = clubelo.snapshot(season_midpoint(season))
        for country, clubs in snap.groupby("country")["club"]:
            clubs_by_country.setdefault(country, set()).update(clubs)

    mapping, unmatched = {}, []
    for country, group in matches.groupby("country"):
        names = pd.concat([group["home"], group["away"]])
        found, missing = match_names(names, clubs_by_country.get(country, ()), aliases)
        mapping.update(found)
        unmatched += missing
    if unmatched:
        raise ValueError(f"No ClubElo name for: {unmatched}. Add them to {ALIASES_CSV.name}.")

    histories = {club: clubelo.club_history(club) for club in sorted(set(mapping.values()))}

    def elo(team: str, kickoff: pd.Timestamp) -> float:
        return clubelo.rating_before(histories[mapping[team]], kickoff)

    matches["elo_home"] = [elo(t, d) for t, d in zip(matches["home"], matches["date"], strict=True)]
    matches["elo_away"] = [elo(t, d) for t, d in zip(matches["away"], matches["date"], strict=True)]
    return matches.dropna(subset=["elo_home", "elo_away"]).reset_index(drop=True)
