"""Proper scoring rules for win/draw/loss probabilities. Lower is better for all of them."""

import numpy as np

OUTCOMES = ("H", "D", "A")


def _one_hot(result) -> np.ndarray:
    result = np.asarray(result)
    return np.stack([result == outcome for outcome in OUTCOMES], axis=1).astype(float)


def log_loss(p_home, p_draw, p_away, result, eps: float = 1e-15) -> float:
    """Mean of -ln(probability given to what actually happened)."""
    probs = np.stack([p_home, p_draw, p_away], axis=1)
    picked = (probs * _one_hot(result)).sum(axis=1)
    return float(-np.log(np.clip(picked, eps, 1)).mean())
