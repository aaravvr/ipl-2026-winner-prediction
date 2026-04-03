from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipl_predictor.config import BALL_BY_BALL_SOURCE_PATH, HISTORICAL_MATCHES_PATH
from ipl_predictor.data import normalize_team_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ball-by-ball IPL data into match-level historical records.")
    parser.add_argument("--source", type=Path, default=BALL_BY_BALL_SOURCE_PATH, help="Path to the raw IPL ball-by-ball CSV")
    parser.add_argument("--output", type=Path, default=HISTORICAL_MATCHES_PATH, help="Output path for aggregated historical matches CSV")
    return parser.parse_args()


def load_ball_by_ball(path: Path) -> pd.DataFrame:
    usecols = [
        "match_id",
        "date",
        "season",
        "venue",
        "city",
        "innings",
        "batting_team",
        "bowling_team",
        "runs_total",
        "team_wicket",
        "match_won_by",
        "win_outcome",
        "toss_winner",
        "toss_decision",
        "result_type",
        "method",
        "gender",
        "event_name",
        "match_type",
    ]
    df = pd.read_csv(path, engine="python", on_bad_lines="warn", usecols=usecols)
    df = normalize_team_columns(df, ["batting_team", "bowling_team", "match_won_by", "toss_winner"])
    df = df[(df["event_name"] == "Indian Premier League") & (df["match_type"] == "T20") & (df["gender"] == "male")]
    return df


def build_historical_matches(df: pd.DataFrame) -> pd.DataFrame:
    innings_summary = (
        df.groupby(["match_id", "innings"], dropna=False)
        .agg(
            batting_team=("batting_team", "first"),
            bowling_team=("bowling_team", "first"),
            innings_runs=("runs_total", "sum"),
            innings_wickets=("team_wicket", "max"),
        )
        .reset_index()
    )

    match_meta = (
        df.groupby("match_id", dropna=False)
        .agg(
            date=("date", "first"),
            season=("season", "first"),
            venue=("venue", "first"),
            city=("city", "first"),
            toss_winner=("toss_winner", "first"),
            toss_decision=("toss_decision", "first"),
            winner=("match_won_by", "first"),
            win_outcome=("win_outcome", "first"),
            result_type=("result_type", "first"),
            method=("method", "first"),
        )
        .reset_index()
    )

    first_innings = (
        innings_summary[innings_summary["innings"] == 1]
        .rename(
            columns={
                "batting_team": "team_1",
                "bowling_team": "team_2",
                "innings_runs": "team_1_score",
                "innings_wickets": "team_1_wickets_lost",
            }
        )
        .drop(columns=["innings"])
    )
    second_innings = (
        innings_summary[innings_summary["innings"] == 2]
        .rename(
            columns={
                "batting_team": "team_2",
                "bowling_team": "team_1",
                "innings_runs": "team_2_score",
                "innings_wickets": "team_2_wickets_lost",
            }
        )
        .drop(columns=["innings"])
    )

    matches = match_meta.merge(first_innings, on="match_id", how="inner").merge(
        second_innings[["match_id", "team_2_score", "team_2_wickets_lost"]],
        on="match_id",
        how="inner",
    )

    matches = matches[matches["result_type"].fillna("").str.lower().ne("no result")].copy()
    matches = matches[matches["winner"].notna()].copy()
    matches["date"] = pd.to_datetime(matches["date"])
    # Match date is more reliable than the raw season field for this dataset.
    matches["season"] = matches["date"].dt.year.astype(int)
    matches["team_1_won"] = (matches["winner"] == matches["team_1"]).astype(int)
    matches["margin_runs"] = (matches["team_1_score"] - matches["team_2_score"]).where(matches["winner"] == matches["team_1"], 0)
    matches["margin_wickets"] = (10 - matches["team_2_wickets_lost"]).where(matches["winner"] == matches["team_2"], 0)
    matches = matches.sort_values(["date", "season", "match_id"]).reset_index(drop=True)
    return matches


def main() -> None:
    args = parse_args()
    df = load_ball_by_ball(args.source)
    matches = build_historical_matches(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(args.output, index=False)
    print(f"Prepared {len(matches)} historical matches from {matches['season'].nunique()} seasons.")
    print(f"Saved aggregated match history to: {args.output}")


if __name__ == "__main__":
    main()
