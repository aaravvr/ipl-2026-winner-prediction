# IPL 2026 Winner Prediction

This project trains a machine learning model on historical IPL match results and then simulates the 2026 season to estimate each team's title probability.

## What this project does

- Trains a match-level winner prediction model from historical IPL data.
- Builds pre-match features such as recent form, venue history, head-to-head record, and Elo-style team strength.
- Simulates the 2026 league stage and playoffs thousands of times.
- Produces title probabilities for every team.
- Evaluates the model on a time-aware holdout split instead of a random split.

## Project structure

```text
IPL/
|-- data/
|   `-- raw/
|       |-- historical_matches.csv
|       |-- fixtures_2026.csv
|       `-- teams_2026.csv
|-- src/
|   `-- ipl_predictor/
|       |-- __init__.py
|       |-- config.py
|       |-- data.py
|       |-- features.py
|       |-- model.py
|       `-- simulation.py
|-- prepare_ball_by_ball_data.py
|-- prepare_player_strength_features.py
|-- train.py
|-- simulate_2026.py
`-- requirements.txt
```

## Expected data files

### `data/raw/historical_matches.csv`

Required columns:

- `season`
- `date`
- `team_1`
- `team_2`
- `venue`
- `toss_winner`
- `toss_decision`
- `winner`

Optional but recommended columns:

- `team_1_score`
- `team_2_score`
- `team_1_wickets_lost`
- `team_2_wickets_lost`
- `city`
- `win_outcome`
- `result_type`

### `data/raw/fixtures_2026.csv`

Required columns:

- `match_id`
- `date`
- `team_1`
- `team_2`
- `venue`

This file should contain only league-stage fixtures. The playoffs are generated automatically from the simulated league table using the standard IPL top-4 format.

### `data/raw/teams_2026.csv`

Required columns:

- `team`

### Optional player-strength files

- `data/raw/match_player_strengths.csv`
- `data/raw/team_player_strengths_latest.csv`

These are generated from the ball-by-ball source and let the model incorporate pre-match player-derived batting and bowling strength at the team level.

## Quick start

1. Create a virtual environment and install dependencies.
2. Convert the ball-by-ball source into match-level history.
3. Train the model.
4. Simulate the 2026 tournament.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python prepare_ball_by_ball_data.py --source C:\Users\aarav\Desktop\IPL.csv
python prepare_player_strength_features.py --source C:\Users\aarav\Desktop\IPL.csv
python train.py
python simulate_2026.py --n-simulations 5000
```

## Outputs

- `artifacts/match_winner_model.joblib`: trained sklearn pipeline
- `artifacts/feature_columns.joblib`: metadata about model inputs
- `artifacts/training_metrics.csv`: evaluation summary for the latest training run
- `artifacts/title_odds_2026.csv`: estimated title probabilities
- `artifacts/last_simulation_table.csv`: most recent simulated league table summary

## Notes

- The sample data included here is only for structure and smoke testing. It is not enough for a meaningful forecast.
- This repo can now ingest the detailed ball-by-ball source at `C:\Users\aarav\Desktop\IPL.csv` and aggregate it into match history automatically.
- It can also derive optional player-based team strength features from the same source before training.
- The current setup is team-level. The next major upgrade would be player-level strength, expected XIs, injuries, auction outcomes, and venue-adjusted batting or bowling features.
