import numpy as np
import pandas as pd
import pytest

from src.features.market import normalize_odds
from src.models.baselines import EloLogitBaseline, FrequencyBaseline


def test_normalize_odds_removes_margin():
    probs, margin = normalize_odds(np.array([[2.0, 3.5, 4.0]]))
    assert probs.sum() == pytest.approx(1.0)
    assert margin[0] > 0


def test_frequency_baseline():
    y = np.array([0, 0, 0, 1, 2])
    m = FrequencyBaseline().fit(pd.DataFrame(index=range(5)), y)
    p = m.predict_proba(pd.DataFrame(index=range(3)))
    assert p.shape == (3, 3) and p[0] == pytest.approx([0.6, 0.2, 0.2])


def test_elo_logit_is_monotone_and_normalised():
    rng = np.random.default_rng(0)
    diff = rng.normal(0, 150, 4000)
    p_home = 1 / (1 + np.exp(-diff / 200))
    y = np.where(rng.random(4000) < p_home, 0, np.where(rng.random(4000) < 0.4, 1, 2))
    m = EloLogitBaseline().fit(pd.DataFrame({"elo_diff": diff}), y)
    p = m.predict_proba(pd.DataFrame({"elo_diff": [-300.0, 0.0, 300.0]}))
    assert p.sum(axis=1) == pytest.approx([1, 1, 1])
    assert p[0, 0] < p[1, 0] < p[2, 0]
