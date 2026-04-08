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
from ipl_predictor.model import build_model_pipeline, build_prefit_calibrated_model, evaluate_model, save_model


def augment_training_data(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    train_frame = x_train.copy()
    train_frame["target"] = y_train.to_numpy()
    swapped = train_frame.copy()

    swap_pairs = [
        ("team_1", "team_2"),
        ("team_1_recent_win_rate", "team_2_recent_win_rate"),
        ("team_1_overall_win_rate", "team_2_overall_win_rate"),
        ("team_1_recent_season_win_rate", "team_2_recent_season_win_rate"),
        ("team_1_venue_win_rate", "team_2_venue_win_rate"),
        ("team_1_avg_runs_scored", "team_2_avg_runs_scored"),
        ("team_1_venue_avg_runs_scored", "team_2_venue_avg_runs_scored"),
        ("team_1_avg_runs_conceded", "team_2_avg_runs_conceded"),
        ("team_1_recent_margin", "team_2_recent_margin"),
        ("team_1_batting_first_win_rate", "team_2_batting_first_win_rate"),
        ("team_1_chasing_win_rate", "team_2_chasing_win_rate"),
        ("team_1_player_batting_strength", "team_2_player_batting_strength"),
        ("team_1_player_bowling_strength", "team_2_player_bowling_strength"),
        ("team_1_powerplay_batting_strength", "team_2_powerplay_batting_strength"),
        ("team_1_death_bowling_strength", "team_2_death_bowling_strength"),
        ("team_1_middle_batting_strength", "team_2_middle_batting_strength"),
        ("team_1_middle_bowling_strength", "team_2_middle_bowling_strength"),
        ("team_1_batting_depth", "team_2_batting_depth"),
        ("team_1_bowling_depth", "team_2_bowling_depth"),
    ]
    for left, right in swap_pairs:
        if left in train_frame.columns and right in train_frame.columns:
            swapped[left] = train_frame[right].to_numpy()
            swapped[right] = train_frame[left].to_numpy()

    invert_columns = [
        "recent_win_rate_diff",
        "overall_win_rate_diff",
        "recent_season_win_rate_diff",
        "venue_win_rate_diff",
        "avg_runs_scored_diff",
        "venue_avg_runs_scored_diff",
        "avg_runs_conceded_diff",
        "recent_margin_diff",
        "batting_first_win_rate_diff",
        "chasing_win_rate_diff",
        "h2h_win_rate_diff",
        "elo_diff",
        "player_batting_strength_diff",
        "player_bowling_strength_diff",
        "powerplay_batting_strength_diff",
        "death_bowling_strength_diff",
        "middle_batting_strength_diff",
        "middle_bowling_strength_diff",
        "batting_depth_diff",
        "bowling_depth_diff",
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


def calibration_split(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    calibration_fraction: float = 0.40,
    min_calibration_matches: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_frame = x_train.copy()
    train_frame["target"] = y_train.to_numpy()
    seasons = sorted(train_frame["season"].unique()) if "season" in train_frame.columns else []
    if len(seasons) > 1:
        calibration_season = seasons[-1]
        calibration_candidates = train_frame[train_frame["season"] == calibration_season]
        calibration_size = max(min_calibration_matches, int(len(calibration_candidates) * calibration_fraction))
        calibration_size = min(calibration_size, max(len(calibration_candidates) - 1, 0))
        if calibration_size > 0:
            calibration_frame = calibration_candidates.tail(calibration_size)
            fit_frame = train_frame.drop(index=calibration_frame.index)
        else:
            fit_frame = train_frame
            calibration_frame = train_frame.iloc[0:0]
        if (
            not fit_frame.empty
            and not calibration_frame.empty
            and len(calibration_frame["target"].unique()) > 1
        ):
            return (
                fit_frame.drop(columns=["target"]),
                calibration_frame.drop(columns=["target"]),
                fit_frame["target"],
                calibration_frame["target"],
            )

    split_index = max(1, int(len(train_frame) * 0.85))
    fit_frame = train_frame.iloc[:split_index]
    calibration_frame = train_frame.iloc[split_index:]
    if calibration_frame.empty or len(calibration_frame["target"].unique()) <= 1:
        return x_train, pd.DataFrame(columns=x_train.columns), y_train, pd.Series(dtype=y_train.dtype)
    return fit_frame.drop(columns=["target"]), calibration_frame.drop(columns=["target"]), fit_frame["target"], calibration_frame["target"]


def fit_match_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 150,
    learning_rate: float = 0.2,
    calibration_method: str | None = "sigmoid",
):
    x_fit, x_calibration, y_fit, y_calibration = calibration_split(x_train, y_train)
    x_fit_augmented, y_fit_augmented = augment_training_data(x_fit, y_fit)
    model = build_model_pipeline(n_estimators=n_estimators, learning_rate=learning_rate, calibrated=False)
    model.fit(x_fit_augmented, y_fit_augmented)
    if calibration_method and not x_calibration.empty and len(y_calibration.unique()) > 1:
        model = build_prefit_calibrated_model(model, x_calibration, y_calibration, method=calibration_method)
    return model, len(x_fit_augmented), len(x_calibration)


def evaluate_time_series_cv(training_frame: pd.DataFrame, min_train_seasons: int = 5) -> pd.DataFrame:
    # build_training_frame returns rows in chronological match order, which is
    # what matters for leakage-safe rolling season evaluation.
    ordered = training_frame.reset_index(drop=True)
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
        model, augmented_train_rows, calibration_rows = fit_match_model(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        rows.append(
            {
                "test_season": int(season),
                "train_rows": raw_train_rows,
                "augmented_train_rows": augmented_train_rows,
                "calibration_rows": calibration_rows,
                "test_rows": len(x_test),
                **metrics,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["test_season", "train_rows", "augmented_train_rows", "calibration_rows", "test_rows", "accuracy", "roc_auc", "log_loss"])

    fold_metrics = pd.DataFrame(rows)
    summary = {
        "test_season": "mean",
        "train_rows": float(fold_metrics["train_rows"].mean()),
        "augmented_train_rows": float(fold_metrics["augmented_train_rows"].mean()),
        "calibration_rows": float(fold_metrics["calibration_rows"].mean()),
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
    model, augmented_train_rows, calibration_rows = fit_match_model(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)
    cv_metrics = evaluate_time_series_cv(training_frame)
    save_model(model, MODEL_PATH, FEATURE_METADATA_PATH, x_train.columns.tolist())
    metrics_frame = pd.DataFrame(
        [
            {
                "train_rows": raw_train_rows,
                "augmented_train_rows": augmented_train_rows,
                "calibration_rows": calibration_rows,
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
