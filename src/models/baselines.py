"""Baselines : fréquences, Elo seul, cotes du marché."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.market import pick_market_probs
from src.models.base import ProbModel


class FrequencyBaseline(ProbModel):
    """Prédit toujours les fréquences H/D/A observées sur le jeu d'entraînement."""

    name = "frequencies"

    def fit(self, X, y):
        counts = np.bincount(np.asarray(y).astype(int), minlength=3).astype(float)
        self.p_ = counts / counts.sum()
        return self

    def predict_proba(self, X):
        return np.tile(self.p_, (len(X), 1))


class EloLogitBaseline(ProbModel):
    """Régression logistique multinomiale sur la seule différence Elo (avantage terrain inclus)."""

    name = "elo_logit"

    def __init__(self, feature: str = "elo_diff", C: float = 1.0):
        self.feature = feature
        self.C = C

    def fit(self, X, y):
        self.clf_ = LogisticRegression(C=self.C, max_iter=1000)
        self.clf_.fit(X[[self.feature]].to_numpy() / 100.0, np.asarray(y).astype(int))
        return self

    def predict_proba(self, X):
        raw = self.clf_.predict_proba(X[[self.feature]].to_numpy() / 100.0)
        out = np.zeros((len(X), 3))
        for j, cls in enumerate(self.clf_.classes_):    # aligne même si une classe manque
            out[:, int(cls)] = raw[:, j]
        return out


class MarketBaseline(ProbModel):
    """Probabilités implicites des cotes (marge retirée). Pas d'entraînement."""

    def __init__(self, prefixes: list[str], name: str = "market"):
        self.prefixes = prefixes
        self.name = name

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return pick_market_probs(X, self.prefixes)   # NaN si cotes absentes
