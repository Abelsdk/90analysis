from ninety.models.elo_training import match_names


def test_match_names_exact_alias_and_close_spelling():
    candidates = ["Liverpool", "Man City", "Atletico", "Wolfsburg"]
    mapping, unmatched = match_names(
        ["Liverpool", "Ath Madrid", "Wolfsburgg", "Mystery FC"],
        candidates,
        aliases={"Ath Madrid": "Atletico"},
    )
    assert mapping == {
        "Liverpool": "Liverpool",
        "Ath Madrid": "Atletico",
        "Wolfsburgg": "Wolfsburg",
    }
    assert unmatched == ["Mystery FC"]
