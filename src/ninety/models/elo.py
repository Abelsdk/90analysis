"""Elo ratings -> win/draw/loss probabilities.

1. The rating gap (plus home advantage) gives an expected score E = P(win) + P(draw) / 2.
2. A draw model, fitted on past league seasons, gives P(draw) from the size of the gap.
3. P(win) = E - P(draw) / 2 and P(loss) = 1 - E - P(draw) / 2.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ELO_SCALE = 400.0
GAP_UNIT = 100.0  # the draw model sees the gap in units of 100 Elo points


def expected_score(gap):
    """Elo's logistic curve: expected points share for the side that is `gap` points stronger."""
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(gap, dtype=float) / ELO_SCALE))


def fit_home_advantage(elo_diff, home_points, *, low=-200.0, high=300.0) -> float:
    """The Elo bonus H that makes average predicted home points match what really happened.

    `home_points` is 1 for a home win, 0.5 for a draw, 0 for a home loss. Mean expected score
    rises with H, so bisection finds the single H where the two averages agree.
    """
    elo_diff = np.asarray(elo_diff, dtype=float)
    target = float(np.mean(home_points))
    for _ in range(60):
        mid = (low + high) / 2
        if expected_score(elo_diff + mid).mean() < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


@dataclass(frozen=True)
class EloModel:
    """Home advantage plus a logistic draw curve.

    P(draw) = sigmoid(intercept + slope * |gap| / 100)
    """

    home_advantage: float
    draw_intercept: float
    draw_slope: float

    def gap(self, elo_home, elo_away, neutral=False):
        bonus = 0.0 if neutral else self.home_advantage
        return np.asarray(elo_home, dtype=float) - np.asarray(elo_away, dtype=float) + bonus

    def draw_probability(self, gap):
        z = self.draw_intercept + self.draw_slope * np.abs(gap) / GAP_UNIT
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, elo_home, elo_away, neutral=False):
        """Return (p_home, p_draw, p_away) arrays that are each >= 0 and sum to 1."""
        gap = self.gap(elo_home, elo_away, neutral)
        e = expected_score(gap)
        # Cap the draw so neither win probability goes negative in huge mismatches.
        p_draw = np.minimum(self.draw_probability(gap), 2 * np.minimum(e, 1 - e))
        p_home = e - p_draw / 2
        p_away = 1 - e - p_draw / 2
        return p_home, p_draw, p_away

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "EloModel":
        return cls(**json.loads(path.read_text()))


def fit(elo_home, elo_away, result) -> EloModel:
    """Fit home advantage, then the draw curve, from past matches.

    `result` holds 'H', 'D' or 'A' for each match.
    """
    result = np.asarray(result)
    elo_diff = np.asarray(elo_home, dtype=float) - np.asarray(elo_away, dtype=float)
    home_points = np.select([result == "H", result == "D"], [1.0, 0.5], 0.0)
    home_advantage = fit_home_advantage(elo_diff, home_points)

    abs_gap = np.abs(elo_diff + home_advantage).reshape(-1, 1) / GAP_UNIT
    draw_curve = LogisticRegression(C=np.inf).fit(abs_gap, result == "D")
    return EloModel(
        home_advantage=float(home_advantage),
        draw_intercept=float(draw_curve.intercept_[0]),
        draw_slope=float(draw_curve.coef_[0, 0]),
    )


def fixed_draw_rate_model(home_advantage: float, draw_rate: float) -> EloModel:
    """The simple alternative: the same draw probability for every match (slope 0)."""
    return EloModel(home_advantage, float(np.log(draw_rate / (1 - draw_rate))), 0.0)
