"""Métriques pour des probabilités 1X2. Convention des colonnes : 0=H, 1=D, 2=A."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-15
N_CLASSES = 3


def _check(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y).astype(int)
    if p.ndim != 2 or p.shape[1] != N_CLASSES:
        raise ValueError(f"p doit avoir la forme (n, {N_CLASSES}), reçu {p.shape}")
    if len(p) != len(y):
        raise ValueError("p et y n'ont pas la même longueur")
    if not np.isfinite(p).all():
        raise ValueError("p contient des NaN ou des valeurs infinies")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("les lignes de p ne somment pas à 1")
    if ((y < 0) | (y >= N_CLASSES)).any():
        raise ValueError("y doit contenir des entiers 0, 1 ou 2")
    return p, y


def onehot(y: np.ndarray) -> np.ndarray:
    return np.eye(N_CLASSES)[np.asarray(y).astype(int)]


def rps_per_match(p, y) -> np.ndarray:
    """Ranked Probability Score par match (r=3 classes ordonnées H < D < A)."""
    p, y = _check(p, y)
    cum_p = np.cumsum(p, axis=1)[:, :-1]
    cum_y = np.cumsum(onehot(y), axis=1)[:, :-1]
    return ((cum_p - cum_y) ** 2).sum(axis=1) / (N_CLASSES - 1)


def logloss_per_match(p, y) -> np.ndarray:
    p, y = _check(p, y)
    return -np.log(np.clip(p[np.arange(len(y)), y], EPS, 1.0))


def brier_per_match(p, y) -> np.ndarray:
    """Brier multiclasse : somme des carrés des écarts sur les 3 classes."""
    p, y = _check(p, y)
    return ((p - onehot(y)) ** 2).sum(axis=1)


def rps(p, y) -> float:
    return float(rps_per_match(p, y).mean())


def log_loss(p, y) -> float:
    return float(logloss_per_match(p, y).mean())


def brier(p, y) -> float:
    return float(brier_per_match(p, y).mean())


def accuracy(p, y) -> float:
    p, y = _check(p, y)
    return float((p.argmax(axis=1) == y).mean())


def ece(p, y, n_bins: int = 10) -> float:
    """Expected Calibration Error (confiance de la classe prédite, bins de même largeur)."""
    p, y = _check(p, y)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def reliability_table(p, y, cls: int = 0, n_bins: int = 10) -> pd.DataFrame:
    """Table de calibration pour UNE classe : probabilité prédite vs fréquence observée."""
    p, y = _check(p, y)
    q = p[:, cls]
    hit = (y == cls).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(q, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(mask.sum()),
                     "mean_pred": float(q[mask].mean()) if mask.any() else np.nan,
                     "observed": float(hit[mask].mean()) if mask.any() else np.nan})
    return pd.DataFrame(rows)


def summary(p, y, n_bins: int = 10) -> dict[str, float]:
    return {"n": int(len(y)), "rps": rps(p, y), "log_loss": log_loss(p, y),
            "brier": brier(p, y), "ece": ece(p, y, n_bins), "accuracy": accuracy(p, y)}


def paired_bootstrap_diff(loss_a, loss_b, n_boot: int = 2000, seed: int = 42,
                          alpha: float = 0.05) -> dict[str, float]:
    """IC bootstrap de mean(loss_a - loss_b), par match. Négatif = A meilleur que B."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi)}
