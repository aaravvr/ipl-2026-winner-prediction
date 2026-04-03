from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipl_predictor.config import FEATURE_METADATA_PATH, HISTORICAL_MATCHES_PATH, MODEL_PATH, TRAINING_METRICS_PATH
from ipl_predictor.data import load_historical_matches
from ipl_predictor.features import build_training_frame
from ipl_predictor.model import build_model_pipeline, evaluate_model, save_model


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


def main() -> None:
    matches = load_historical_matches(HISTORICAL_MATCHES_PATH)
    training_frame = build_training_frame(matches)

    if len(training_frame) < 20:
        print("Warning: sample data is very small. Replace it with full IPL history for meaningful predictions.")

    x_train, x_test, y_train, y_test = time_based_split(training_frame)

    model = build_model_pipeline()
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)
    save_model(model, MODEL_PATH, FEATURE_METADATA_PATH, x_train.columns.tolist())
    metrics_frame = pd.DataFrame(
        [
            {
                "train_rows": len(x_train),
                "test_rows": len(x_test),
                "train_seasons": ",".join(map(str, sorted(x_train["season"].unique()))),
                "test_seasons": ",".join(map(str, sorted(x_test["season"].unique()))),
                **metrics,
            }
        ]
    )
    TRAINING_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(TRAINING_METRICS_PATH, index=False)

    print("Training complete.")
    print(pd.Series(metrics).round(4).to_string())
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved training metrics to: {TRAINING_METRICS_PATH}")


if __name__ == "__main__":
    main()
