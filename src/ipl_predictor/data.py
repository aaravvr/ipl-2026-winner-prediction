from __future__ import annotations

from pathlib import Path

import pandas as pd


TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

VENUE_ALIASES = {
    "Ahmedabad": "Narendra Modi Stadium",
    "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium",
    "Bengaluru": "M Chinnaswamy Stadium",
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium",
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "M. Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "Chennai": "MA Chidambaram Stadium",
    "M. A. Chidambaram Stadium": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chennai": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium",
    "Delhi": "Arun Jaitley Stadium",
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Dharamshala": "Himachal Pradesh Cricket Association Stadium",
    "Himachal Pradesh Cricket Association Stadium, Dharamshala": "Himachal Pradesh Cricket Association Stadium",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium",
    "Guwahati": "Barsapara Cricket Stadium",
    "ACA Stadium, Guwahati": "Barsapara Cricket Stadium",
    "Barsapara Cricket Stadium, Guwahati": "Barsapara Cricket Stadium",
    "Hyderabad": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Hyderabad": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi Intl. Cricket Stadium": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
    "Jaipur": "Sawai Mansingh Stadium",
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
    "Kolkata": "Eden Gardens",
    "Eden Gardens, Kolkata": "Eden Gardens",
    "Lucknow": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    "BRSABV Ekana Cricket Stadium": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
    "Mumbai": "Wankhede Stadium",
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",
    "New Chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium",
    "New International Cricket Stadium": "Maharaja Yadavindra Singh International Cricket Stadium",
    "New International Cricket Stadium, New Chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur": "Maharaja Yadavindra Singh International Cricket Stadium",
    "Raipur": "Shaheed Veer Narayan Singh International Stadium",
    "Shaheed Veer Narayan Singh International Cricket Stadium, Raipur": "Shaheed Veer Narayan Singh International Stadium",
}

PLAYER_ALIASES = {
    "Lungisani Ngidi": "Lungi Ngidi",
    "L Ngidi": "Lungi Ngidi",
    "N Tilak Varma": "Tilak Varma",
    "Shahbaz Ahamad": "Shahbaz Ahmad",
    "Shahbaz Ahmed": "Shahbaz Ahmad",
    "T. Natarajan": "T Natarajan",
    "Mohammed Shami": "Mohammad Shami",
    "Nithish Kumar Reddy": "Nitish Kumar Reddy",
    "Vaibhav Suryavanshi": "Vaibhav Sooryavanshi",
    "V Suryavanshi": "Vaibhav Sooryavanshi",
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


def normalize_venue_name(venue: object) -> object:
    if pd.isna(venue):
        return venue
    venue_name = " ".join(str(venue).replace(".", ". ").split()).strip()
    if not venue_name or venue_name == "Unknown":
        return pd.NA
    return VENUE_ALIASES.get(venue_name, venue_name)


def normalize_venue_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for column in columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(normalize_venue_name)
    return normalized


def normalize_player_name(player: object) -> object:
    if pd.isna(player):
        return player
    player_name = " ".join(str(player).replace(".", " ").split()).strip()
    if not player_name or player_name == "Unknown":
        return pd.NA
    return PLAYER_ALIASES.get(player_name, player_name)


def normalize_player_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = df.copy()
    for column in columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(normalize_player_name)
    return normalized


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
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
    df = normalize_venue_columns(df, ["venue"])
    df["date"] = pd.to_datetime(df["date"])
    # IPL seasons are contained within a single calendar year, and the raw season
    # field in this dataset is occasionally off by one. Match date is more reliable.
    df["season"] = df["date"].dt.year.astype(int)
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
    df = normalize_venue_columns(df, ["venue"])
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


def load_optional_match_player_strengths(path: Path) -> pd.DataFrame | None:
    df = load_optional_csv(path)
    if df is None:
        return None
    required_columns = {
        "match_id",
        "team_1_batting_strength",
        "team_2_batting_strength",
        "team_1_bowling_strength",
        "team_2_bowling_strength",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Match player strengths file is missing columns: {sorted(missing)}")
    return df


def load_optional_team_player_strengths(path: Path) -> dict[str, dict[str, float]]:
    df = load_optional_csv(path)
    if df is None:
        return {}
    required_columns = {"team", "batting_strength", "bowling_strength"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Team player strengths file is missing columns: {sorted(missing)}")
    df = normalize_team_columns(df, ["team"])
    df = df.dropna(subset=["team"])
    return {
        str(row.team): {
            "batting_strength": float(row.batting_strength),
            "bowling_strength": float(row.bowling_strength),
        }
        for row in df.itertuples(index=False)
    }


def load_optional_team_priors(path: Path) -> dict[str, dict[str, float]]:
    df = load_optional_csv(path)
    if df is None:
        return {}
    required_columns = {"team", "prior_rating", "batting_bonus", "bowling_bonus"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Team priors file is missing columns: {sorted(missing)}")
    df = normalize_team_columns(df, ["team"])
    df = df.dropna(subset=["team"])
    return {
        str(row.team): {
            "prior_rating": float(row.prior_rating),
            "batting_bonus": float(row.batting_bonus),
            "bowling_bonus": float(row.bowling_bonus),
        }
        for row in df.itertuples(index=False)
    }
