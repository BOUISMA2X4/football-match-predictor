"""Découpage walk-forward par saison : on entraîne sur le passé, on teste sur le futur."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def walk_forward_splits(df: pd.DataFrame, season_col: str = "season",
                        min_train_seasons: int = 3,
                        max_train_seasons: int | None = None
                        ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Génère (saison_test, indices_train, indices_test) avec des positions (iloc)."""
    seasons = sorted(df[season_col].unique())
    if len(seasons) <= min_train_seasons:
        raise ValueError(
            f"{len(seasons)} saison(s) disponible(s) : il en faut plus de {min_train_seasons}.")
    values = df[season_col].to_numpy()
    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        if max_train_seasons is not None:
            train_seasons = train_seasons[-max_train_seasons:]
        yield (int(seasons[i]),
               np.flatnonzero(np.isin(values, train_seasons)),
               np.flatnonzero(values == seasons[i]))
