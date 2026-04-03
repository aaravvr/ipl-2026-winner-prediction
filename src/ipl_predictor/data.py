from __future__ import annotations

from pathlib import Path

import pandas as pd


TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}


def normalize_team_name(team: object) -> object:
    if pd.isna(team):
        return team
    team_name = str(team).strip()
    if not team_name or team_name == "Unknown":
        return pd.NA
    return TEAM_ALIASES.get(team_name, team_name)


def normalize_team_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for column in columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(normalize_team_name)
    return normalized


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_historical_matches(path: Path) -> pd.DataFrame:
    required_columns = {
        "season",
        "date",
        "team_1",
        "team_2",
        "venue",
        "toss_winner",
        "toss_decision",
        "winner",
    }
    df = load_csv(path)
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Historical matches file is missing columns: {sorted(missing)}")

    df = normalize_team_columns(df, ["team_1", "team_2", "toss_winner", "winner"])
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = (
        df["season"]
        .astype(str)
        .str.extract(r"(\d{4})", expand=False)
        .fillna(df["date"].dt.year.astype(str))
        .astype(int)
    )
    df = df.dropna(subset=["team_1", "team_2", "winner"])
    df = df.sort_values(["date", "season"]).reset_index(drop=True)
    return df


def load_fixtures(path: Path) -> pd.DataFrame:
    required_columns = {"match_id", "date", "team_1", "team_2", "venue"}
    df = load_csv(path)
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Fixtures file is missing columns: {sorted(missing)}")

    df = normalize_team_columns(df, ["team_1", "team_2"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["team_1", "team_2"])
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    return df


def load_teams(path: Path) -> list[str]:
    df = load_csv(path)
    if "team" not in df.columns:
        raise ValueError("Teams file must include a 'team' column")
    df = normalize_team_columns(df, ["team"])
    return sorted(df["team"].dropna().astype(str).unique().tolist())
