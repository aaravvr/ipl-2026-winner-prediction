from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

HISTORICAL_MATCHES_PATH = DATA_DIR / "historical_matches.csv"
FIXTURES_2026_PATH = DATA_DIR / "fixtures_2026.csv"
TEAMS_2026_PATH = DATA_DIR / "teams_2026.csv"
BALL_BY_BALL_SOURCE_PATH = ROOT_DIR.parent / "IPL.csv"

MODEL_PATH = ARTIFACTS_DIR / "match_winner_model.joblib"
FEATURE_METADATA_PATH = ARTIFACTS_DIR / "feature_columns.joblib"
TITLE_ODDS_PATH = ARTIFACTS_DIR / "title_odds_2026.csv"
LAST_SIM_TABLE_PATH = ARTIFACTS_DIR / "last_simulation_table.csv"
TRAINING_METRICS_PATH = ARTIFACTS_DIR / "training_metrics.csv"
