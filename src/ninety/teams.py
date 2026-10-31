"""The team mapping table: one canonical team_id per club, one column per source's spelling."""

import pandas as pd

from ninety.paths import REFERENCE_DIR

TEAMS_CSV = REFERENCE_DIR / "teams.csv"


def load_teams() -> pd.DataFrame:
    return pd.read_csv(TEAMS_CSV, keep_default_na=False)


def clubelo_names() -> dict[str, str]:
    """team_id -> ClubElo club name."""
    teams = load_teams()
    return dict(zip(teams["team_id"], teams["clubelo_name"], strict=True))
