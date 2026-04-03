from __future__ import annotations

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_model_pipeline() -> Pipeline:
    categorical_features = ["team_1", "team_2", "venue"]
    numeric_features = [
        "team_1_recent_win_rate",
        "team_2_recent_win_rate",
        "recent_win_rate_diff",
        "team_1_overall_win_rate",
        "team_2_overall_win_rate",
        "overall_win_rate_diff",
        "team_1_venue_win_rate",
        "team_2_venue_win_rate",
        "venue_win_rate_diff",
        "team_1_avg_runs_scored",
        "team_2_avg_runs_scored",
        "avg_runs_scored_diff",
        "team_1_avg_runs_conceded",
        "team_2_avg_runs_conceded",
        "avg_runs_conceded_diff",
        "team_1_recent_margin",
        "team_2_recent_margin",
        "recent_margin_diff",
        "venue_avg_innings_score",
        "team_1_h2h_win_rate",
        "team_2_h2h_win_rate",
        "h2h_win_rate_diff",
        "team_1_elo",
        "team_2_elo",
        "elo_diff",
        "team_1_expected_score",
        "team_2_expected_score",
        "team_1_player_batting_strength",
        "team_2_player_batting_strength",
        "team_1_player_bowling_strength",
        "team_2_player_bowling_strength",
        "player_batting_strength_diff",
        "player_bowling_strength_diff",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ]
    )

    model = AdaBoostClassifier(
        n_estimators=150,
        learning_rate=0.2,
        random_state=42,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def evaluate_model(model: Pipeline, x_test, y_test) -> dict[str, float]:
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities) if len(set(y_test)) > 1 else float("nan"),
        "log_loss": log_loss(y_test, probabilities, labels=[0, 1]),
    }


def save_model(model: Pipeline, path, feature_metadata_path, feature_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    joblib.dump(feature_columns, feature_metadata_path)


def load_model(path):
    return joblib.load(path)
