"""Probabilités implicites des cotes, sans la marge du bookmaker."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_odds(odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cotes (n,3) -> (probabilités normalisées, marge). Normalisation proportionnelle.

    pi_k = 1/o_k ; marge = sum(pi) - 1 ; p_k = pi_k / sum(pi).
    """
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 1.0 / odds
    valid = np.isfinite(raw).all(axis=1) & (odds > 1.0).all(axis=1)
    total = raw.sum(axis=1)
    probs = np.full(odds.shape, np.nan)
    probs[valid] = raw[valid] / total[valid, None]
    margin = np.where(valid, total - 1.0, np.nan)
    return probs, margin


def pick_market_probs(df: pd.DataFrame, prefixes: list[str]) -> np.ndarray:
    """Pour chaque match, utilise le premier jeu de cotes disponible dans `prefixes`.

    Un préfixe 'PSC' correspond aux colonnes PSCH, PSCD, PSCA. NaN si aucun jeu complet.
    """
    out = np.full((len(df), 3), np.nan)
    for prefix in prefixes:
        cols = [f"{prefix}H", f"{prefix}D", f"{prefix}A"]
        if not all(c in df.columns for c in cols):
            continue
        probs, _ = normalize_odds(df[cols].to_numpy(dtype=float))
        missing = np.isnan(out).any(axis=1)
        out[missing] = probs[missing]
    return out
