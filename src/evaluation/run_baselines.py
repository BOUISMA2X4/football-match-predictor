"""Compare les baselines en walk-forward : fréquences, Elo, marché (ouverture, clôture)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import data_dir, load_config
from src.data.columns import ODDS_COLS
from src.evaluation import metrics as M
from src.evaluation.walkforward import walk_forward_splits
from src.features.market import pick_market_probs
from src.models.base import ProbModel
from src.models.baselines import EloLogitBaseline, FrequencyBaseline, MarketBaseline

DEFAULT_CFG = {
    "walk_forward": {"min_train_seasons": 3, "max_train_seasons": None},
    "metrics": {"ece_bins": 10},
    "bootstrap": {"n_boot": 2000, "seed": 42},
    "market": {"closing_prefixes": ["PSC", "AvgC"], "opening_prefixes": ["B365"]},
}
BENCHMARK = "market_closing"


def make_baselines(cfg: dict) -> list[ProbModel]:
    return [
        FrequencyBaseline(),
        EloLogitBaseline(),
        MarketBaseline(cfg["market"]["opening_prefixes"], name="market_opening"),
        MarketBaseline(cfg["market"]["closing_prefixes"], name=BENCHMARK),
    ]


def load_dataset() -> pd.DataFrame:
    gold = pd.read_parquet(data_dir() / "gold" / "features.parquet")
    silver = pd.read_parquet(data_dir() / "silver" / "matches.parquet")
    odds = silver[["match_id"] + [c for c in ODDS_COLS if c in silver.columns]]
    return gold.merge(odds, on="match_id", how="left", validate="one_to_one")


def run_walk_forward(df: pd.DataFrame, models: list[ProbModel], cfg: dict) -> pd.DataFrame:
    """Prédictions hors-échantillon de chaque modèle, une ligne par match de test."""
    wf = cfg["walk_forward"]
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    parts = []
    for _test_season, tr, te in walk_forward_splits(
            df, min_train_seasons=wf["min_train_seasons"],
            max_train_seasons=wf["max_train_seasons"]):
        train, test = df.iloc[tr], df.iloc[te]
        assert train["date"].max() < test["date"].min(), "fuite temporelle train/test"
        out = test[["match_id", "season", "date", "y"]].copy()
        for model in models:
            model.fit(train, train["y"].to_numpy())
            p = model.predict_proba(test)
            for k, col in enumerate(["H", "D", "A"]):
                out[f"{model.name}__{col}"] = p[:, k]
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def _probs(preds: pd.DataFrame, name: str) -> np.ndarray:
    return preds[[f"{name}__H", f"{name}__D", f"{name}__A"]].to_numpy()


def evaluate(preds: pd.DataFrame, names: list[str], cfg: dict
             ) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    """Évalue sur le sous-ensemble COMMUN (tous les modèles ont une prédiction valide)."""
    ok = np.ones(len(preds), dtype=bool)
    for n in names:
        ok &= np.isfinite(_probs(preds, n)).all(axis=1)
    common = preds[ok].reset_index(drop=True)
    if common.empty:
        raise ValueError("Aucun match commun : vérifiez la couverture des cotes.")
    y = common["y"].to_numpy()
    bins = cfg["metrics"]["ece_bins"]

    overall = pd.DataFrame({n: M.summary(_probs(common, n), y, bins) for n in names}).T
    overall["n"] = overall["n"].astype(int)
    by_season = pd.DataFrame({
        n: {int(s): M.rps(_probs(g, n), g["y"].to_numpy()) for s, g in common.groupby("season")}
        for n in names})
    by_season.index.name = "season"

    boot = {}
    if BENCHMARK in names:
        b = M.rps_per_match(_probs(common, BENCHMARK), y)
        for n in names:
            if n != BENCHMARK:
                boot[n] = M.paired_bootstrap_diff(
                    M.rps_per_match(_probs(common, n), y), b,
                    n_boot=cfg["bootstrap"]["n_boot"], seed=cfg["bootstrap"]["seed"])
    calib = M.reliability_table(_probs(common, BENCHMARK if BENCHMARK in names else names[0]),
                                y, cls=0, n_bins=bins)
    return overall, by_season, boot, common, calib


def odds_coverage(df: pd.DataFrame, cfg: dict) -> pd.Series:
    probs = pick_market_probs(df, cfg["market"]["closing_prefixes"])
    covered = ~np.isnan(probs).any(axis=1)
    return pd.Series(covered, index=df.index).groupby(df["season"]).mean().round(2)


def main() -> None:
    user = load_config("evaluation")
    cfg = {k: {**DEFAULT_CFG[k], **user.get(k, {})} for k in DEFAULT_CFG}
    df = load_dataset()

    print("Couverture des cotes de clôture par saison (part des matchs) :")
    print(odds_coverage(df, cfg).to_string(), "\n")

    models = make_baselines(cfg)
    names = [m.name for m in models]
    preds = run_walk_forward(df, models, cfg)
    overall, by_season, boot, common, calib = evaluate(preds, names, cfg)

    print(f"Évaluation walk-forward sur {len(common)} matchs communs "
          f"(saisons {int(common['season'].min())}-{int(common['season'].max())})\n")
    print(overall.round(4).to_string(), "\n")
    print("RPS par saison :")
    print(by_season.round(4).to_string(), "\n")
    print(f"Écart de RPS vs {BENCHMARK} (IC95 % bootstrap ; positif = moins bon que le marché) :")
    for n, r in boot.items():
        print(f"  {n:16s} {r['mean_diff']:+.4f}  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")

    out = data_dir() / "reports"
    out.mkdir(parents=True, exist_ok=True)
    overall.to_csv(out / "baselines_overall.csv")
    by_season.to_csv(out / "baselines_by_season.csv")
    calib.to_csv(out / "calibration_market_home.csv", index=False)
    preds.to_parquet(out / "predictions_baselines.parquet", index=False)
    print(f"\nRapports écrits dans {out}")


if __name__ == "__main__":
    main()
