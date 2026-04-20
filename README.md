# IPL 2026 Winner Prediction

An end-to-end machine learning project that predicts IPL match winners from pre-match information and simulates the full 2026 season to estimate title probabilities.

This project is built around a realistic sports-analytics workflow:
- ingest messy historical cricket data
- engineer pre-match team, venue, and player-strength features
- train and validate a classifier using time-aware splits
- simulate an entire tournament thousands of times

## Why this is a strong ML project

This repo is more than a notebook experiment. It includes:
- real-world data preparation from ball-by-ball IPL records
- feature engineering from match history, venue context, likely XIs, and squad strength
- leakage-aware training and rolling time-series validation
- probability-based tournament simulation with playoffs and NRR-style standings
- reproducible scripts, Git history, CI, and a project structure suitable for extension

## Current model quality

Latest validated metrics from [artifacts/training_metrics.csv](C:\Users\aarav\Desktop\IPL\artifacts\training_metrics.csv) and [artifacts/training_cv_metrics.csv](C:\Users\aarav\Desktop\IPL\artifacts\training_cv_metrics.csv):

- 2025 holdout accuracy: `70.00%`
- 2025 holdout ROC-AUC: `0.8630`
- 2025 holdout log-loss: `0.5145`
- walk-forward CV mean accuracy: `68.05%`
- walk-forward CV mean ROC-AUC: `0.8022`
- walk-forward CV mean log-loss: `0.5638`

Those cross-validation numbers are the more honest measure of expected performance because they test the model season by season rather than on a single held-out year.

## Current 2026 forecast

Latest simulation output from [artifacts/title_odds_2026.csv](C:\Users\aarav\Desktop\IPL\artifacts\title_odds_2026.csv):

| Team | Title Probability |
|---|---:|
| Lucknow Super Giants | 0.50 |
| Sunrisers Hyderabad | 0.18 |
| Chennai Super Kings | 0.14 |
| Gujarat Titans | 0.07 |
| Delhi Capitals | 0.06 |
| Rajasthan Royals | 0.05 |
| Mumbai Indians | 0.00 |
| Kolkata Knight Riders | 0.00 |
| Punjab Kings | 0.00 |
| Royal Challengers Bengaluru | 0.00 |

Note: the checked-in file above comes from a 100-simulation smoke run, so the odds are intentionally coarse. For a presentation-grade forecast, rerun with `2000+` simulations.

## How the system works

### 1. Historical data ingestion

The raw ball-by-ball source in [IPL.csv](C:\Users\aarav\Desktop\IPL.csv) is converted into match-level training data via [prepare_ball_by_ball_data.py](C:\Users\aarav\Desktop\IPL\prepare_ball_by_ball_data.py).

This step:
- normalizes team naming across seasons
- derives match-level winners, venues, scores, and metadata
- preserves enough structure for pre-match feature generation

### 2. Player-strength generation

[prepare_player_strength_features.py](C:\Users\aarav\Desktop\IPL\prepare_player_strength_features.py) creates optional player-derived team features from the same ball-by-ball source.

It currently includes:
- batting strength
- bowling strength
- powerplay batting strength
- death bowling strength
- middle-over batting strength
- middle-over bowling strength
- batting depth
- bowling depth

### 3. Pre-match feature engineering

[src/ipl_predictor/features.py](C:\Users\aarav\Desktop\IPL\src\ipl_predictor\features.py) builds features that are available before a match starts.

Feature groups include:
- recent form
- all-time and recent-season team win rates
- venue-specific win rates
- team-specific venue scoring rates
- batting-first and chasing style tendencies
- recent scoring and conceding trends
- Elo strength differential
- head-to-head differential
- player-strength and phase-strength differentials
- squad depth signals

### 4. Model training

[train.py](C:\Users\aarav\Desktop\IPL\train.py) trains a pre-match classifier and evaluates it using:
- a latest-season holdout
- rolling season-by-season walk-forward validation

The current model is an AdaBoost classifier with held-out sigmoid calibration. The calibration is fit on a late slice of the most recent training season rather than by mixing mirrored rows across CV folds.

### 5. Tournament simulation

[simulate_2026.py](C:\Users\aarav\Desktop\IPL\simulate_2026.py) loads the trained model, applies 2026 squad/lineup priors, and simulates the entire IPL season.

[src/ipl_predictor/simulation.py](C:\Users\aarav\Desktop\IPL\src\ipl_predictor\simulation.py) handles:
- league-stage simulation
- NRR-style table tracking
- deterministic tie-break logic with head-to-head support
- playoff progression
- score estimation for NRR and standings updates

## 2026-specific inputs

This project does not rely only on old franchise history. It also includes 2026 context:

- official fixtures in [data/raw/fixtures_2026.csv](C:\Users\aarav\Desktop\IPL\data\raw\fixtures_2026.csv)
- 2026 teams in [data/raw/teams_2026.csv](C:\Users\aarav\Desktop\IPL\data\raw\teams_2026.csv)
- official squad extraction in [data/raw/team_squads_2026.csv](C:\Users\aarav\Desktop\IPL\data\raw\team_squads_2026.csv)
- likely XI data in [data/raw/likely_lineups_2026.csv](C:\Users\aarav\Desktop\IPL\data\raw\likely_lineups_2026.csv)
- priors in [data/raw/team_priors_2026.csv](C:\Users\aarav\Desktop\IPL\data\raw\team_priors_2026.csv)
- latest team strength snapshot in [data/raw/team_player_strengths_latest.csv](C:\Users\aarav\Desktop\IPL\data\raw\team_player_strengths_latest.csv)

## Project structure

```text
IPL/
|-- data/
|   `-- raw/
|       |-- fixtures_2026.csv
|       |-- historical_matches.csv
|       |-- likely_lineups_2026.csv
|       |-- match_player_strengths.csv
|       |-- teams_2026.csv
|       |-- team_overview_2026.csv
|       |-- team_player_strengths_latest.csv
|       |-- team_priors_2026.csv
|       `-- team_squads_2026.csv
|-- artifacts/
|   |-- feature_columns.joblib
|   |-- last_simulation_table.csv
|   |-- match_winner_model.joblib
|   |-- title_odds_2026.csv
|   |-- training_cv_metrics.csv
|   `-- training_metrics.csv
|-- src/
|   `-- ipl_predictor/
|       |-- config.py
|       |-- data.py
|       |-- features.py
|       |-- model.py
|       `-- simulation.py
|-- .github/workflows/ci.yml
|-- prepare_ball_by_ball_data.py
|-- prepare_fixture_schedule.py
|-- prepare_player_strength_features.py
|-- prepare_team_priors.py
|-- prepare_team_squads.py
|-- simulate_2026.py
|-- train.py
`-- README.md
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python prepare_fixture_schedule.py --source C:\Users\aarav\Desktop\1774525332894_TATA_IPL_2026-Schedule.pdf
python prepare_team_squads.py
python prepare_ball_by_ball_data.py --source C:\Users\aarav\Desktop\IPL.csv
python prepare_player_strength_features.py --source C:\Users\aarav\Desktop\IPL.csv
python prepare_team_priors.py --source C:\Users\aarav\Desktop\IPL.csv
python train.py
python simulate_2026.py --n-simulations 5000
```

## Reproducible outputs

Main outputs:
- [artifacts/match_winner_model.joblib](C:\Users\aarav\Desktop\IPL\artifacts\match_winner_model.joblib)
- [artifacts/feature_columns.joblib](C:\Users\aarav\Desktop\IPL\artifacts\feature_columns.joblib)
- [artifacts/training_metrics.csv](C:\Users\aarav\Desktop\IPL\artifacts\training_metrics.csv)
- [artifacts/training_cv_metrics.csv](C:\Users\aarav\Desktop\IPL\artifacts\training_cv_metrics.csv)
- [artifacts/title_odds_2026.csv](C:\Users\aarav\Desktop\IPL\artifacts\title_odds_2026.csv)
- [artifacts/last_simulation_table.csv](C:\Users\aarav\Desktop\IPL\artifacts\last_simulation_table.csv)

## Limitations

This is a strong project, but it is still a forecasting system with assumptions.

Current limitations:
- playoff venues are assumed, not officially released
- likely XIs are manually curated rather than confirmed team sheets
- score simulation is realistic enough for NRR, but not ball-by-ball
- injuries, late squad changes, and in-season form shocks are not modeled explicitly
- forecast outputs can vary if you materially change priors or lineups

## Why this stands out in a portfolio

This project demonstrates:
- applied machine learning
- feature engineering on messy sports data
- time-aware evaluation
- probabilistic forecasting
- simulation modeling
- reproducible ML engineering
- GitHub workflow and CI discipline

For a portfolio, resume, or interview project, that is a very strong combination.
