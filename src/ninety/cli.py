"""The `ninety` command."""

import argparse
from datetime import UTC, date, datetime

import pandas as pd

from ninety import __version__
from ninety.paths import MODELS_DIR, PREDICTIONS_DIR, REPO_ROOT

ELO_MODEL_PATH = MODELS_DIR / "elo.json"
TRAIN_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2017, 2024)]  # 2017-18 .. 2023-24
TEST_SEASONS = ["2024-25", "2025-26"]


def fit_elo(_args) -> None:
    from ninety.evaluation.metrics import log_loss
    from ninety.models import elo, elo_training

    train = elo_training.build(TRAIN_SEASONS)
    test = elo_training.build(TEST_SEASONS)
    model = elo.fit(train["elo_home"], train["elo_away"], train["result"])
    flat = elo.fixed_draw_rate_model(model.home_advantage, (train["result"] == "D").mean())

    print(f"train: {len(train)} matches ({TRAIN_SEASONS[0]} to {TRAIN_SEASONS[-1]})")
    print(f"test:  {len(test)} matches ({', '.join(TEST_SEASONS)})")
    print(f"home advantage: {model.home_advantage:.1f} Elo points")
    print(f"draw curve: sigmoid({model.draw_intercept:.3f} + {model.draw_slope:.3f} * |gap|/100)")
    for name, m in [("gap-dependent draws", model), ("fixed draw rate", flat)]:
        probs = m.predict(test["elo_home"], test["elo_away"])
        print(f"test log loss, {name}: {log_loss(*probs, test['result']):.4f}")

    model.save(ELO_MODEL_PATH)
    print(f"saved {ELO_MODEL_PATH.relative_to(REPO_ROOT)}")


def predict_elo(args) -> None:
    from ninety.data import clubelo
    from ninety.models.elo import EloModel
    from ninety.teams import clubelo_names

    model = EloModel.load(ELO_MODEL_PATH)
    fixtures = pd.read_csv(args.fixtures)
    ratings = clubelo.snapshot(date.fromisoformat(args.ratings_date)).set_index("club")["elo"]
    names = clubelo_names()
    missing = sorted(
        {names[t] for t in fixtures[["home_id", "away_id"]].stack()} - set(ratings.index)
    )
    if missing:
        raise SystemExit(f"Not in the ClubElo snapshot: {missing}. Fix reference/teams.csv.")

    elo_home = fixtures["home_id"].map(names).map(ratings)
    elo_away = fixtures["away_id"].map(names).map(ratings)
    p_home, p_draw, p_away = model.predict(elo_home, elo_away)
    out = fixtures[["match_id", "kickoff_utc"]].assign(
        model="elo",
        p_home=p_home.round(4),
        p_draw=p_draw.round(4),
        p_away=p_away.round(4),
        elo_home=elo_home.round(1),
        elo_away=elo_away.round(1),
        ratings_date=args.ratings_date,
        created_at_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    row = fixtures.iloc[0]
    path = (
        PREDICTIONS_DIR
        / row["competition"].lower()
        / row["season"]
        / f"md{row['matchday']}"
        / "elo.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out[["match_id", "p_home", "p_draw", "p_away"]].to_string(index=False))
    print(f"saved {path.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ninety")
    parser.add_argument("--version", action="version", version=f"ninety {__version__}")
    commands = parser.add_subparsers(dest="command")

    commands.add_parser("fit-elo", help="fit home advantage and the draw curve").set_defaults(
        run=fit_elo
    )
    predict = commands.add_parser("predict-elo", help="Elo win/draw/loss for a fixtures file")
    predict.add_argument("fixtures")
    predict.add_argument("--ratings-date", default=date.today().isoformat())
    predict.set_defaults(run=predict_elo)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return
    args.run(args)
