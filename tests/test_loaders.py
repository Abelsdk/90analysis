"""Loader tests run on small saved samples, never the network."""

import pandas as pd
import pytest

from ninety.data import cache, clubelo, football_data

CLUBELO_SAMPLE = """Rank,Club,Country,Level,Elo,From,To
1,Bayern,GER,1,2010.5,2026-10-20,2026-10-24
None,Bayern,GER,1,2003.1,2026-10-25,2026-10-31
"""

FOOTBALL_DATA_SAMPLE = """﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Referee,B365H
E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H,A Taylor,1.3
E0,16/08/25,12:30,Aston Villa,Newcastle,0,0,D,S Hooper,2.4
,,,,,,,,,
"""


def test_clubelo_parse():
    frame = clubelo.parse_ratings(CLUBELO_SAMPLE)
    assert list(frame.columns) == ["club", "country", "level", "elo", "from", "to"]
    assert frame["from"].dtype.kind == "M"


def test_clubelo_rating_before_uses_the_day_before_kickoff():
    history = clubelo.parse_ratings(CLUBELO_SAMPLE)
    # Kickoff on the 25th: the rating on the 24th applies, not the post-match update.
    assert clubelo.rating_before(history, pd.Timestamp("2026-10-25 20:00")) == 2010.5
    assert clubelo.rating_before(history, pd.Timestamp("2026-10-27")) == 2003.1


def test_football_data_parse_keeps_results_and_drops_blank_rows():
    frame = football_data.parse_matches(FOOTBALL_DATA_SAMPLE.lstrip("﻿"))
    assert list(frame.columns) == ["date", "home", "away", "home_goals", "away_goals", "result"]
    assert len(frame) == 2
    assert frame["date"].tolist() == [pd.Timestamp("2025-08-15"), pd.Timestamp("2025-08-16")]


def test_season_code():
    assert football_data.season_code("2023-24") == "2324"


class FakeResponse:
    def __init__(self, status, body=b""):
        self.status_code, self.content = status, body
        self.ok = 200 <= status < 300


def test_cache_downloads_once(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return FakeResponse(200, b"a,b\n1,2\n")

    monkeypatch.setattr(cache.requests, "get", fake_get)
    path = tmp_path / "x.csv"
    assert cache.fetch_text("http://example.test/x", path) == "a,b\n1,2\n"
    assert cache.fetch_text("http://example.test/x", path) == "a,b\n1,2\n"
    assert len(calls) == 1


def test_cache_raises_when_the_source_is_down(tmp_path, monkeypatch):
    monkeypatch.setattr(cache.requests, "get", lambda url, **kw: FakeResponse(502))
    monkeypatch.setattr(cache.time, "sleep", lambda s: None)
    with pytest.raises(cache.SourceUnavailable, match="HTTP 502"):
        cache.fetch_text("http://example.test/x", tmp_path / "x.csv", retries=2)
