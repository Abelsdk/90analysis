"""Data validation: the hand-entered files we commit must be internally consistent."""

import pandas as pd

from ninety.paths import FIXTURES_DIR
from ninety.teams import load_teams

MD4 = FIXTURES_DIR / "ucl" / "2026-27" / "md4.csv"


def test_team_ids_and_clubelo_names_are_unique():
    teams = load_teams()
    assert teams["team_id"].is_unique
    assert teams["clubelo_name"].is_unique
    assert (teams["clubelo_name"] != "").all()


def test_md4_has_18_matches_and_36_different_teams():
    fixtures = pd.read_csv(MD4)
    assert len(fixtures) == 18
    teams = pd.concat([fixtures["home_id"], fixtures["away_id"]])
    assert teams.is_unique and len(teams) == 36


def test_every_md4_team_is_in_the_mapping_table():
    fixtures = pd.read_csv(MD4)
    known = set(load_teams()["team_id"])
    assert set(fixtures["home_id"]) | set(fixtures["away_id"]) <= known


def test_md4_ids_and_kickoffs_are_consistent():
    fixtures = pd.read_csv(MD4)
    kickoff = pd.to_datetime(fixtures["kickoff_utc"], utc=True)
    expected_ids = (
        kickoff.dt.strftime("%Y-%m-%d") + "_" + fixtures["home_id"] + "_" + fixtures["away_id"]
    )
    assert (fixtures["match_id"] == expected_ids).all()
    assert kickoff.dt.strftime("%H:%M").isin(["17:45", "20:00"]).all()  # 18:45 / 21:00 CET
