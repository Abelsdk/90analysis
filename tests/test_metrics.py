import math

import pytest

from ninety.evaluation.metrics import log_loss


def test_log_loss_hand_computed():
    # Match 1 home win given 0.5, match 2 draw given 0.25: mean of -ln(0.5) and -ln(0.25).
    loss = log_loss([0.5, 0.5], [0.3, 0.25], [0.2, 0.25], ["H", "D"])
    assert loss == pytest.approx((math.log(2) + math.log(4)) / 2)


def test_perfect_prediction_scores_zero():
    assert log_loss([1.0], [0.0], [0.0], ["H"]) == pytest.approx(0.0)


def test_zero_probability_on_the_result_is_finite():
    assert math.isfinite(log_loss([1.0], [0.0], [0.0], ["A"]))
