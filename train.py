from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipl_predictor.config import FEATURE_METADATA_PATH, HISTORICAL_MATCHES_PATH, MODEL_PATH, TRAINING_CV_METRICS_PATH, TRAINING_METRICS_PATH
from ipl_predictor.config import MATCH_PLAYER_STRENGTHS_PATH
from ipl_predictor.data import load_historical_matches, load_optional_match_player_strengths
from ipl_predictor.features import build_training_frame
from ipl_predictor.model import build_model_pipeline, evaluate_model, save_model


def augment_training_data(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    train_frame = x_train.copy()
    train_frame["target"] = y_train.to_numpy()
    swapped = train_frame.copy()

    swap_pairs = [
        ("team_1", "team_2"),
        ("team_1_recent_win_rate", "team_2_recent_win_rate"),
        ("team_1_overall_win_rate", "team_2_overall_win_rate"),
        ("team_1_venue_win_rate", "team_2_venue_win_rate"),
        ("team_1_avg_runs_scored", "team_2_avg_runs_scored"),
        ("team_1_avg_runs_conceded", "team_2_avg_runs_conceded"),
        ("team_1_recent_margin", "team_2_recent_margin"),
        ("team_1_h2h_win_rate", "team_2_h2h_win_rate"),
        ("team_1_elo", "team_2_elo"),
        ("team_1_expected_score", "team_2_expected_score"),
        ("team_1_player_batting_strength", "team_2_player_batting_strength"),
        ("team_1_player_bowling_strength", "team_2_player_bowling_strength"),
    ]
    for left, right in swap_pairs:
        if left in train_frame.columns and right in train_frame.columns:
            swapped[left] = train_frame[right].to_numpy()
            swapped[right] = train_frame[left].to_numpy()

    invert_columns = [
        "recent_win_rate_diff",
        "overall_win_rate_diff",
        "venue_win_rate_diff",
        "avg_runs_scored_diff",
        "avg_runs_conceded_diff",
        "recent_margin_diff",
        "h2h_win_rate_diff",
        "elo_diff",
        "player_batting_strength_diff",
        "player_bowling_strength_diff",
    ]
    for column in invert_columns:
        if column in train_frame.columns:
            swapped[column] = -train_frame[column].to_numpy()

    if "team_1_won_toss" in train_frame.columns:
        swapped["team_1_won_toss"] = 1 - train_frame["team_1_won_toss"].to_numpy()
    if "team_1_bats_first" in train_frame.columns:
        swapped["team_1_bats_first"] = 1 - train_frame["team_1_bats_first"].to_numpy()
    if "toss_winner" in train_frame.columns:
        swapped["toss_winner"] = train_frame.apply(
            lambda row: row["team_2"]
            if row["toss_winner"] == row["team_1"]
            else (row["team_1"] if row["toss_winner"] == row["team_2"] else row["toss_winner"]),
            axis=1,
        )

    swapped["target"] = 1 - train_frame["target"].to_numpy()
    augmented = pd.concat([train_frame, swapped], ignore_index=True)
    return augmented.drop(columns=["target"]), augmented["target"]


def time_based_split(training_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    ordered = training_frame.reset_index(drop=True)
    if ordered["season"].nunique() > 1:
        test_season = ordered["season"].max()
        train_frame = ordered[ordered["season"] < test_season]
        test_frame = ordered[ordered["season"] == test_season]
        if not train_frame.empty and not test_frame.empty:
            x_train = train_frame.drop(columns=["target"])
            y_train = train_frame["target"]
            x_test = test_frame.drop(columns=["target"])
            y_test = test_frame["target"]
            return x_train, x_test, y_train, y_test

    split_index = max(1, int(len(ordered) * 0.8))
    train_frame = ordered.iloc[:split_index]
    test_frame = ordered.iloc[split_index:]
    x_train = train_frame.drop(columns=["target"])
    y_train = train_frame["target"]
    x_test = test_frame.drop(columns=["target"])
    y_test = test_frame["target"]
    return x_train, x_test, y_train, y_test


def evaluate_time_series_cv(training_frame: pd.DataFrame, min_train_seasons: int = 5) -> pd.DataFrame:
    ordered = training_frame.sort_values(["season", "date"] if "date" in training_frame.columns else ["season"]).reset_index(drop=True)
    seasons = sorted(ordered["season"].unique())
    rows: list[dict[str, float | int | str]] = []

    for season in seasons:
        prior_seasons = [value for value in seasons if value < season]
        if len(prior_seasons) < min_train_seasons:
            continue
        train_frame = ordered[ordered["season"] < season]
        test_frame = ordered[ordered["season"] == season]
        if train_frame.empty or test_frame.empty:
            continue

        x_train = train_frame.drop(columns=["target"])
        y_train = train_frame["target"]
        x_test = test_frame.drop(columns=["target"])
        y_test = test_frame["target"]

        raw_train_rows = len(x_train)
        x_train_augmented, y_train_augmented = augment_training_data(x_train, y_train)
        model = build_model_pipeline()
        model.fit(x_train_augmented, y_train_augmented)
        metrics = evaluate_model(model, x_test, y_test)
        rows.append(
            {
                "test_season": int(season),
                "train_rows": raw_train_rows,
                "augmented_train_rows": len(x_train_augmented),
                "test_rows": len(x_test),
                **metrics,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["test_season", "train_rows", "augmented_train_rows", "test_rows", "accuracy", "roc_auc", "log_loss"])

    fold_metrics = pd.DataFrame(rows)
    summary = {
        "test_season": "mean",
        "train_rows": float(fold_metrics["train_rows"].mean()),
        "augmented_train_rows": float(fold_metrics["augmented_train_rows"].mean()),
        "test_rows": float(fold_metrics["test_rows"].mean()),
        "accuracy": float(fold_metrics["accuracy"].mean()),
        "roc_auc": float(fold_metrics["roc_auc"].mean()),
        "log_loss": float(fold_metrics["log_loss"].mean()),
    }
    return pd.concat([fold_metrics, pd.DataFrame([summary])], ignore_index=True)


def main() -> None:
    matches = load_historical_matches(HISTORICAL_MATCHES_PATH)
    match_player_strengths = load_optional_match_player_strengths(MATCH_PLAYER_STRENGTHS_PATH)
    if match_player_strengths is not None and "match_id" in matches.columns:
        matches = matches.merge(match_player_strengths, on="match_id", how="left")
    training_frame = build_training_frame(matches)

    if len(training_frame) < 20:
        print("Warning: sample data is very small. Replace it with full IPL history for meaningful predictions.")

    x_train, x_test, y_train, y_test = time_based_split(training_frame)
    raw_train_rows = len(x_train)
    x_train, y_train = augment_training_data(x_train, y_train)
    augmented_train_rows = len(x_train)

    model = build_model_pipeline()
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)
    cv_metrics = evaluate_time_series_cv(training_frame)
    save_model(model, MODEL_PATH, FEATURE_METADATA_PATH, x_train.columns.tolist())
    metrics_frame = pd.DataFrame(
        [
            {
                "train_rows": raw_train_rows,
                "augmented_train_rows": augmented_train_rows,
                "test_rows": len(x_test),
                "train_seasons": ",".join(map(str, sorted(x_train["season"].unique()))),
                "test_seasons": ",".join(map(str, sorted(x_test["season"].unique()))),
                **metrics,
            }
        ]
    )
    TRAINING_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(TRAINING_METRICS_PATH, index=False)
    cv_metrics.to_csv(TRAINING_CV_METRICS_PATH, index=False)

    print("Training complete.")
    print(pd.Series(metrics).round(4).to_string())
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved training metrics to: {TRAINING_METRICS_PATH}")
    print(f"Saved time-series CV metrics to: {TRAINING_CV_METRICS_PATH}")


if __name__ == "__main__":
    main()
