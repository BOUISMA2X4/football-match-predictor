# football-match-predictor

Prédiction probabiliste de matchs de football, intégrée dans une pipeline MLOps.

## Objectif

Pour un match futur précis (équipe à domicile, équipe à l'extérieur, date, ligue),
le système renvoie :

- la probabilité de victoire, de nul et de défaite de chaque équipe,
- les buts attendus,
- un niveau de confiance,
- les données qui justifient la prédiction (forme récente, domicile/extérieur,
  confrontations directes).

Le but est d'aider un utilisateur à décider avec des informations transparentes.
Les probabilités sont des estimations : aucun résultat n'est garanti.

## Principes

- Probabilités calibrées, évaluées avec RPS, log-loss, Brier et ECE.
- Aucune fuite de données : pour un match de date t, seules les données antérieures à t
  sont utilisées.
- Pipeline modulaire et reproductible (code, données et modèles versionnés).

## Données

- Production : football-data.co.uk (résultats, cotes, statistiques de match).
- Validation académique : Open International Soccer Database.

Vérifiez les conditions d'utilisation de chaque source avant de collecter.

## Démarrage rapide

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
pytest
```

## Structure

```
configs/     configuration YAML (ligues, saisons, modèle, seuils)
dags/        DAGs Airflow (fins, appellent src/)
src/         data, features, models, evaluation, serving, monitoring, streaming
tests/       tests unitaires, de données et anti-fuite
infra/       docker-compose, configuration Kafka et Airflow
data/        raw / silver / gold (non versionné dans Git)
notebooks/   exploration uniquement
```

## Feuille de route

0. Environnement
1. Données et features (Elo, forme, H2H)
2. Évaluation et baselines
3. Modèles, calibration et gate de promotion
4. API FastAPI
5. Airflow
6. Lakehouse (MinIO) et qualité des données
7. Kafka
8. Monitoring et CI/CD

## Avertissement

Ce projet est à visée éducative et analytique. Il ne garantit aucun gain.
Les paris peuvent entraîner des pertes d'argent et une dépendance. Vérifiez la
réglementation applicable dans votre pays avant tout usage lié aux paris.