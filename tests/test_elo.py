import numpy as np
import pytest

from ninety.models import elo

MODEL = elo.EloModel(home_advantage=60.0, draw_intercept=-1.0, draw_slope=-0.3)


# Unit tests: known answers worked out by hand.
def test_expected_score_even_gap_is_half():
    assert elo.expected_score(0) == pytest.approx(0.5)


def test_expected_score_200_points():
    # 1 / (1 + 10^-0.5) = 0.7597
    assert elo.expected_score(200) == pytest.approx(0.7597, abs=1e-4)


# Sanity tests: rules that must hold for any input.
def test_expected_score_is_symmetric():
    gaps = np.array([-500, -120, 0, 35, 400])
    np.testing.assert_allclose(elo.expected_score(gaps) + elo.expected_score(-gaps), 1.0)


@pytest.mark.parametrize("gap", [-1500, -400, -50, 0, 50, 400, 1500])
def test_probabilities_are_valid(gap):
    p_home, p_draw, p_away = MODEL.predict(1800 + gap, 1800)
    for p in (p_home, p_draw, p_away):
        assert 0 <= p <= 1
    assert p_home + p_draw + p_away == pytest.approx(1.0)


def test_probabilities_keep_the_expected_score():
    p_home, p_draw, _ = MODEL.predict(1900, 1750)
    assert p_home + p_draw / 2 == pytest.approx(elo.expected_score(150 + 60))


def test_equal_teams_on_neutral_ground_are_a_coin_flip():
    p_home, _, p_away = MODEL.predict(1800, 1800, neutral=True)
    assert p_home == pytest.approx(p_away)


def test_draws_get_rarer_as_the_gap_grows():
    assert MODEL.draw_probability(0) > MODEL.draw_probability(300)


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "elo.json"
    MODEL.save(path)
    assert elo.EloModel.load(path) == MODEL


# Fitting tests: simulate matches from a known model, check the fit recovers it.
def simulate(model, n, seed=0):
    rng = np.random.default_rng(seed)
    elo_home = rng.normal(1750, 120, n)
    elo_away = rng.normal(1750, 120, n)
    p_home, p_draw, _ = model.predict(elo_home, elo_away)
    u = rng.random(n)
    result = np.where(u < p_home, "H", np.where(u < p_home + p_draw, "D", "A"))
    return elo_home, elo_away, result


def test_fit_recovers_known_parameters():
    true = elo.EloModel(home_advantage=70.0, draw_intercept=-0.9, draw_slope=-0.25)
    fitted = elo.fit(*simulate(true, 60_000))
    assert fitted.home_advantage == pytest.approx(70, abs=8)
    assert fitted.draw_intercept == pytest.approx(-0.9, abs=0.08)
    assert fitted.draw_slope == pytest.approx(-0.25, abs=0.06)


def test_fixed_draw_rate_model_gives_the_same_draw_everywhere():
    flat = elo.fixed_draw_rate_model(60.0, 0.25)
    assert flat.draw_probability(0) == pytest.approx(0.25)
    assert flat.draw_probability(400) == pytest.approx(0.25)
