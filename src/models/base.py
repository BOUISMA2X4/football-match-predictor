"""Interface commune : tout modèle (baseline, Dixon-Coles, LightGBM...) la respecte."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ProbModel(ABC):
    """fit(X, y) puis predict_proba(X) -> tableau (n, 3) : colonnes H, D, A."""

    name: str = "model"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> ProbModel:
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...
