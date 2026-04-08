from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipl_predictor.config import (
    BALL_BY_BALL_SOURCE_PATH,
    HISTORICAL_MATCHES_PATH,
    MATCH_PLAYER_STRENGTHS_PATH,
    TEAM_PLAYER_STRENGTHS_PATH,
    TEAMS_2026_PATH,
)
from ipl_predictor.data import load_teams, normalize_team_columns


NON_BOWLER_WICKETS = {"run out", "retired hurt", "retired out", "obstructing the field"}
DEFAULT_BATTING_STRENGTH = 35.0
DEFAULT_BOWLING_STRENGTH = 18.0
RECENT_FORM_WINDOW = 8
RECENT_FORM_WEIGHT = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build optional player-strength features from ball-by-ball IPL data.")
    parser.add_argument("--source", type=Path, default=BALL_BY_BALL_SOURCE_PATH, help="Path to the raw IPL ball-by-ball CSV")
    parser.add_argument(
        "--historical",
        type=Path,
        default=HISTORICAL_MATCHES_PATH,
        help="Path to the aggregated historical match CSV",
    )
    parser.add_argument(
        "--match-output",
        type=Path,
        default=MATCH_PLAYER_STRENGTHS_PATH,
        help="Output path for match-level player strength features",
    )
    parser.add_argument(
        "--team-output",
        type=Path,
        default=TEAM_PLAYER_STRENGTHS_PATH,
        help="Output path for latest team player strength values",
    )
    parser.add_argument(
        "--teams",
        type=Path,
        default=TEAMS_2026_PATH,
        help="Optional current teams CSV used to filter latest team strengths",
    )
    return parser.parse_args()


def load_ball_by_ball(path: Path) -> pd.DataFrame:
    usecols = [
        "match_id",
        "date",
        "event_name",
        "match_type",
        "gender",
        "batting_team",
        "bowling_team",
        "batter",
        "bowler",
        "over",
        "runs_batter",
        "runs_bowler",
        "valid_ball",
        "wicket_kind",
    ]
    df = pd.read_csv(path, engine="python", on_bad_lines="warn", usecols=usecols)
    df = normalize_team_columns(df, ["batting_team", "bowling_team"])
    df = df[(df["event_name"] == "Indian Premier League") & (df["match_type"] == "T20") & (df["gender"] == "male")].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def batting_rating(stats: dict[str, float]) -> float:
    innings = max(float(stats["innings"]), 1.0)
    balls = max(float(stats["balls"]), 1.0)
    average = float(stats["runs"]) / innings
    strike_rate = 100.0 * float(stats["runs"]) / balls
    return 0.55 * average + 0.15 * strike_rate


def bowling_rating(stats: dict[str, float]) -> float:
    matches = max(float(stats["matches"]), 1.0)
    balls = max(float(stats["balls"]), 1.0)
    economy = 6.0 * float(stats["runs"]) / balls
    wickets_per_match = float(stats["wickets"]) / matches
    return 14.0 * wickets_per_match + max(0.0, 10.0 - economy)


def batting_match_rating(runs: float, balls: float) -> float:
    balls = max(float(balls), 1.0)
    return 0.55 * float(runs) + 0.15 * (100.0 * float(runs) / balls)


def bowling_match_rating(runs: float, balls: float, wickets: float) -> float:
    balls = max(float(balls), 1.0)
    economy = 6.0 * float(runs) / balls
    return 14.0 * float(wickets) + max(0.0, 10.0 - economy)


def powerplay_batting_rating(stats: dict[str, float]) -> float:
    balls = max(float(stats.get("powerplay_balls", 0.0)), 1.0)
    runs = float(stats.get("powerplay_runs", 0.0))
    if balls < 12.0:
        return batting_rating(stats)
    strike_rate = 100.0 * runs / balls
    runs_per_innings = runs / max(float(stats.get("innings", 1.0)), 1.0)
    return 0.17 * strike_rate + 0.60 * runs_per_innings


def death_bowling_rating(stats: dict[str, float]) -> float:
    balls = max(float(stats.get("death_balls", 0.0)), 1.0)
    runs = float(stats.get("death_runs", 0.0))
    wickets = float(stats.get("death_wickets", 0.0))
    if balls < 12.0:
        return bowling_rating(stats)
    economy = 6.0 * runs / balls
    wickets_per_match = wickets / max(float(stats.get("matches", 1.0)), 1.0)
    return 45.0 * wickets_per_match + max(0.0, 13.0 - economy)


def middle_batting_rating(stats: dict[str, float]) -> float:
    balls = max(float(stats.get("middle_balls", 0.0)), 1.0)
    runs = float(stats.get("middle_runs", 0.0))
    if balls < 12.0:
        return batting_rating(stats)
    strike_rate = 100.0 * runs / balls
    runs_per_innings = runs / max(float(stats.get("innings", 1.0)), 1.0)
    return 0.16 * strike_rate + 0.55 * runs_per_innings


def middle_bowling_rating(stats: dict[str, float]) -> float:
    balls = max(float(stats.get("middle_balls", 0.0)), 1.0)
    runs = float(stats.get("middle_runs", 0.0))
    wickets = float(stats.get("middle_wickets", 0.0))
    if balls < 12.0:
        return bowling_rating(stats)
    economy = 6.0 * runs / balls
    wickets_per_match = wickets / max(float(stats.get("matches", 1.0)), 1.0)
    return 40.0 * wickets_per_match + max(0.0, 10.5 - economy)


def team_depth(
    players: list[str],
    player_stats: dict[str, dict[str, float]],
    rating_fn,
    default_strength: float,
    top_n: int,
) -> float:
    ratings = []
    for player in players:
        stats = player_stats.get(player)
        ratings.append(rating_fn(stats) if stats else default_strength)
    if len(ratings) < 3:
        return 0.5
    ratings = sorted(ratings, reverse=True)[:top_n]
    total = sum(max(rating, 0.0) for rating in ratings)
    if total <= 0:
        return 0.5
    top_two_share = sum(max(rating, 0.0) for rating in ratings[:2]) / total
    return 1.0 - top_two_share


def team_strength(
    players: list[str],
    player_stats: dict[str, dict[str, float]],
    recent_form: dict[str, list[float]],
    rating_fn,
    default_strength: float,
    top_n: int,
    recent_weight: float = RECENT_FORM_WEIGHT,
) -> float:
    ratings = []
    for player in players:
        stats = player_stats.get(player)
        if not stats:
            ratings.append(default_strength)
            continue
        career_rating = rating_fn(stats)
        recent_values = recent_form.get(player, [])
        if recent_values:
            recent_rating = sum(recent_values) / len(recent_values)
            ratings.append((1.0 - recent_weight) * career_rating + recent_weight * recent_rating)
        else:
            ratings.append(career_rating)
    if not ratings:
        return default_strength
    ratings.sort(reverse=True)
    return sum(ratings[:top_n]) / min(len(ratings), top_n)


def build_match_player_strengths(ball_by_ball: pd.DataFrame, historical_matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    batter_appearances = (
        ball_by_ball[["match_id", "date", "batting_team", "batter"]]
        .dropna(subset=["batting_team", "batter"])
        .drop_duplicates()
    )
    bowler_appearances = (
        ball_by_ball[["match_id", "date", "bowling_team", "bowler"]]
        .dropna(subset=["bowling_team", "bowler"])
        .drop_duplicates()
    )

    batting_updates = (
        ball_by_ball.groupby(["match_id", "batting_team", "batter"], dropna=False)
        .agg(runs=("runs_batter", "sum"), balls=("valid_ball", "sum"))
        .reset_index()
    )
    bowling_base = ball_by_ball.copy()
    bowling_base["is_bowler_wicket"] = (
        bowling_base["wicket_kind"].notna()
        & ~bowling_base["wicket_kind"].astype(str).str.lower().isin(NON_BOWLER_WICKETS)
    ).astype(int)
    bowling_updates = (
        bowling_base.groupby(["match_id", "bowling_team", "bowler"], dropna=False)
        .agg(runs=("runs_bowler", "sum"), balls=("valid_ball", "sum"), wickets=("is_bowler_wicket", "sum"))
        .reset_index()
    )
    powerplay_updates = (
        ball_by_ball[ball_by_ball["over"] < 6]
        .groupby(["match_id", "batting_team", "batter"], dropna=False)
        .agg(runs=("runs_batter", "sum"), balls=("valid_ball", "sum"))
        .reset_index()
    )
    death_bowling_updates = (
        bowling_base[bowling_base["over"] >= 16]
        .groupby(["match_id", "bowling_team", "bowler"], dropna=False)
        .agg(runs=("runs_bowler", "sum"), balls=("valid_ball", "sum"), wickets=("is_bowler_wicket", "sum"))
        .reset_index()
    )
    middle_updates = (
        ball_by_ball[(ball_by_ball["over"] >= 6) & (ball_by_ball["over"] < 16)]
        .groupby(["match_id", "batting_team", "batter"], dropna=False)
        .agg(runs=("runs_batter", "sum"), balls=("valid_ball", "sum"))
        .reset_index()
    )
    middle_bowling_updates = (
        bowling_base[(bowling_base["over"] >= 6) & (bowling_base["over"] < 16)]
        .groupby(["match_id", "bowling_team", "bowler"], dropna=False)
        .agg(runs=("runs_bowler", "sum"), balls=("valid_ball", "sum"), wickets=("is_bowler_wicket", "sum"))
        .reset_index()
    )

    batting_updates_by_match = {
        match_id: frame for match_id, frame in batting_updates.groupby("match_id", sort=False)
    }
    bowling_updates_by_match = {
        match_id: frame for match_id, frame in bowling_updates.groupby("match_id", sort=False)
    }
    powerplay_updates_by_match = {
        match_id: frame for match_id, frame in powerplay_updates.groupby("match_id", sort=False)
    }
    death_bowling_updates_by_match = {
        match_id: frame for match_id, frame in death_bowling_updates.groupby("match_id", sort=False)
    }
    middle_updates_by_match = {
        match_id: frame for match_id, frame in middle_updates.groupby("match_id", sort=False)
    }
    middle_bowling_updates_by_match = {
        match_id: frame for match_id, frame in middle_bowling_updates.groupby("match_id", sort=False)
    }
    batters_by_match_team = {
        (match_id, team): sorted(frame["batter"].astype(str).unique().tolist())
        for (match_id, team), frame in batter_appearances.groupby(["match_id", "batting_team"], sort=False)
    }
    bowlers_by_match_team = {
        (match_id, team): sorted(frame["bowler"].astype(str).unique().tolist())
        for (match_id, team), frame in bowler_appearances.groupby(["match_id", "bowling_team"], sort=False)
    }

    batting_player_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"runs": 0.0, "balls": 0.0, "innings": 0.0, "powerplay_runs": 0.0, "powerplay_balls": 0.0, "middle_runs": 0.0, "middle_balls": 0.0}
    )
    bowling_player_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "runs": 0.0,
            "balls": 0.0,
            "wickets": 0.0,
            "matches": 0.0,
            "death_runs": 0.0,
            "death_balls": 0.0,
            "death_wickets": 0.0,
            "middle_runs": 0.0,
            "middle_balls": 0.0,
            "middle_wickets": 0.0,
        }
    )
    batting_recent_form: dict[str, list[float]] = defaultdict(lambda: [])
    bowling_recent_form: dict[str, list[float]] = defaultdict(lambda: [])
    latest_team_strengths: dict[str, dict[str, float]] = {}
    rows: list[dict[str, float | int]] = []

    for match in historical_matches.sort_values(["date", "match_id"]).itertuples(index=False):
        team_1_batters = batters_by_match_team.get((match.match_id, match.team_1), [])
        team_2_batters = batters_by_match_team.get((match.match_id, match.team_2), [])
        team_1_bowlers = bowlers_by_match_team.get((match.match_id, match.team_1), [])
        team_2_bowlers = bowlers_by_match_team.get((match.match_id, match.team_2), [])

        team_1_batting_strength = team_strength(
            team_1_batters,
            batting_player_stats,
            batting_recent_form,
            batting_rating,
            DEFAULT_BATTING_STRENGTH,
            top_n=6,
        )
        team_2_batting_strength = team_strength(
            team_2_batters,
            batting_player_stats,
            batting_recent_form,
            batting_rating,
            DEFAULT_BATTING_STRENGTH,
            top_n=6,
        )
        team_1_bowling_strength = team_strength(
            team_1_bowlers,
            bowling_player_stats,
            bowling_recent_form,
            bowling_rating,
            DEFAULT_BOWLING_STRENGTH,
            top_n=5,
        )
        team_2_bowling_strength = team_strength(
            team_2_bowlers,
            bowling_player_stats,
            bowling_recent_form,
            bowling_rating,
            DEFAULT_BOWLING_STRENGTH,
            top_n=5,
        )
        team_1_powerplay_batting_strength = team_strength(
            team_1_batters,
            batting_player_stats,
            defaultdict(lambda: []),
            powerplay_batting_rating,
            DEFAULT_BATTING_STRENGTH,
            top_n=4,
            recent_weight=0.0,
        )
        team_2_powerplay_batting_strength = team_strength(
            team_2_batters,
            batting_player_stats,
            defaultdict(lambda: []),
            powerplay_batting_rating,
            DEFAULT_BATTING_STRENGTH,
            top_n=4,
            recent_weight=0.0,
        )
        team_1_death_bowling_strength = team_strength(
            team_1_bowlers,
            bowling_player_stats,
            defaultdict(lambda: []),
            death_bowling_rating,
            DEFAULT_BOWLING_STRENGTH,
            top_n=4,
            recent_weight=0.0,
        )
        team_2_death_bowling_strength = team_strength(
            team_2_bowlers,
            bowling_player_stats,
            defaultdict(lambda: []),
            death_bowling_rating,
            DEFAULT_BOWLING_STRENGTH,
            top_n=4,
            recent_weight=0.0,
        )
        team_1_middle_batting_strength = team_strength(
            team_1_batters,
            batting_player_stats,
            defaultdict(lambda: []),
            middle_batting_rating,
            DEFAULT_BATTING_STRENGTH,
            top_n=5,
            recent_weight=0.0,
        )
        team_2_middle_batting_strength = team_strength(
            team_2_batters,
            batting_player_stats,
            defaultdict(lambda: []),
            middle_batting_rating,
            DEFAULT_BATTING_STRENGTH,
            top_n=5,
            recent_weight=0.0,
        )
        team_1_middle_bowling_strength = team_strength(
            team_1_bowlers,
            bowling_player_stats,
            defaultdict(lambda: []),
            middle_bowling_rating,
            DEFAULT_BOWLING_STRENGTH,
            top_n=5,
            recent_weight=0.0,
        )
        team_2_middle_bowling_strength = team_strength(
            team_2_bowlers,
            bowling_player_stats,
            defaultdict(lambda: []),
            middle_bowling_rating,
            DEFAULT_BOWLING_STRENGTH,
            top_n=5,
            recent_weight=0.0,
        )
        team_1_batting_depth = team_depth(team_1_batters, batting_player_stats, batting_rating, DEFAULT_BATTING_STRENGTH, top_n=7)
        team_2_batting_depth = team_depth(team_2_batters, batting_player_stats, batting_rating, DEFAULT_BATTING_STRENGTH, top_n=7)
        team_1_bowling_depth = team_depth(team_1_bowlers, bowling_player_stats, bowling_rating, DEFAULT_BOWLING_STRENGTH, top_n=6)
        team_2_bowling_depth = team_depth(team_2_bowlers, bowling_player_stats, bowling_rating, DEFAULT_BOWLING_STRENGTH, top_n=6)

        rows.append(
            {
                "match_id": int(match.match_id),
                "team_1_batting_strength": team_1_batting_strength,
                "team_2_batting_strength": team_2_batting_strength,
                "team_1_bowling_strength": team_1_bowling_strength,
                "team_2_bowling_strength": team_2_bowling_strength,
                "team_1_powerplay_batting_strength": team_1_powerplay_batting_strength,
                "team_2_powerplay_batting_strength": team_2_powerplay_batting_strength,
                "team_1_death_bowling_strength": team_1_death_bowling_strength,
                "team_2_death_bowling_strength": team_2_death_bowling_strength,
                "team_1_middle_batting_strength": team_1_middle_batting_strength,
                "team_2_middle_batting_strength": team_2_middle_batting_strength,
                "team_1_middle_bowling_strength": team_1_middle_bowling_strength,
                "team_2_middle_bowling_strength": team_2_middle_bowling_strength,
                "team_1_batting_depth": team_1_batting_depth,
                "team_2_batting_depth": team_2_batting_depth,
                "team_1_bowling_depth": team_1_bowling_depth,
                "team_2_bowling_depth": team_2_bowling_depth,
                "batting_strength_diff": team_1_batting_strength - team_2_batting_strength,
                "bowling_strength_diff": team_1_bowling_strength - team_2_bowling_strength,
                "powerplay_batting_strength_diff": team_1_powerplay_batting_strength - team_2_powerplay_batting_strength,
                "death_bowling_strength_diff": team_1_death_bowling_strength - team_2_death_bowling_strength,
                "middle_batting_strength_diff": team_1_middle_batting_strength - team_2_middle_batting_strength,
                "middle_bowling_strength_diff": team_1_middle_bowling_strength - team_2_middle_bowling_strength,
                "batting_depth_diff": team_1_batting_depth - team_2_batting_depth,
                "bowling_depth_diff": team_1_bowling_depth - team_2_bowling_depth,
            }
        )

        latest_team_strengths[match.team_1] = {
            "batting_strength": team_1_batting_strength,
            "bowling_strength": team_1_bowling_strength,
            "powerplay_batting_strength": team_1_powerplay_batting_strength,
            "death_bowling_strength": team_1_death_bowling_strength,
            "middle_batting_strength": team_1_middle_batting_strength,
            "middle_bowling_strength": team_1_middle_bowling_strength,
            "batting_depth": team_1_batting_depth,
            "bowling_depth": team_1_bowling_depth,
        }
        latest_team_strengths[match.team_2] = {
            "batting_strength": team_2_batting_strength,
            "bowling_strength": team_2_bowling_strength,
            "powerplay_batting_strength": team_2_powerplay_batting_strength,
            "death_bowling_strength": team_2_death_bowling_strength,
            "middle_batting_strength": team_2_middle_batting_strength,
            "middle_bowling_strength": team_2_middle_bowling_strength,
            "batting_depth": team_2_batting_depth,
            "bowling_depth": team_2_bowling_depth,
        }

        for update in batting_updates_by_match.get(match.match_id, pd.DataFrame()).itertuples(index=False):
            player = str(update.batter)
            recent = batting_recent_form[player]
            recent.append(batting_match_rating(update.runs, update.balls))
            if len(recent) > RECENT_FORM_WINDOW:
                del recent[0]
            batting_player_stats[player]["runs"] += float(update.runs)
            batting_player_stats[player]["balls"] += float(update.balls)
            batting_player_stats[player]["innings"] += 1.0
        for update in powerplay_updates_by_match.get(match.match_id, pd.DataFrame()).itertuples(index=False):
            player = str(update.batter)
            batting_player_stats[player]["powerplay_runs"] += float(update.runs)
            batting_player_stats[player]["powerplay_balls"] += float(update.balls)
        for update in middle_updates_by_match.get(match.match_id, pd.DataFrame()).itertuples(index=False):
            player = str(update.batter)
            batting_player_stats[player]["middle_runs"] += float(update.runs)
            batting_player_stats[player]["middle_balls"] += float(update.balls)

        for update in bowling_updates_by_match.get(match.match_id, pd.DataFrame()).itertuples(index=False):
            player = str(update.bowler)
            recent = bowling_recent_form[player]
            recent.append(bowling_match_rating(update.runs, update.balls, update.wickets))
            if len(recent) > RECENT_FORM_WINDOW:
                del recent[0]
            bowling_player_stats[player]["runs"] += float(update.runs)
            bowling_player_stats[player]["balls"] += float(update.balls)
            bowling_player_stats[player]["wickets"] += float(update.wickets)
            bowling_player_stats[player]["matches"] += 1.0
        for update in death_bowling_updates_by_match.get(match.match_id, pd.DataFrame()).itertuples(index=False):
            player = str(update.bowler)
            bowling_player_stats[player]["death_runs"] += float(update.runs)
            bowling_player_stats[player]["death_balls"] += float(update.balls)
            bowling_player_stats[player]["death_wickets"] += float(update.wickets)
        for update in middle_bowling_updates_by_match.get(match.match_id, pd.DataFrame()).itertuples(index=False):
            player = str(update.bowler)
            bowling_player_stats[player]["middle_runs"] += float(update.runs)
            bowling_player_stats[player]["middle_balls"] += float(update.balls)
            bowling_player_stats[player]["middle_wickets"] += float(update.wickets)

    match_strengths = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)
    latest_strengths = (
        pd.DataFrame(
            [{"team": team, **strengths} for team, strengths in latest_team_strengths.items()]
        )
        .sort_values("team")
        .reset_index(drop=True)
    )
    return match_strengths, latest_strengths


def main() -> None:
    args = parse_args()
    historical_matches = pd.read_csv(args.historical)
    historical_matches = normalize_team_columns(historical_matches, ["team_1", "team_2"])
    historical_matches["date"] = pd.to_datetime(historical_matches["date"])
    historical_matches = historical_matches.dropna(subset=["match_id", "team_1", "team_2"])
    ball_by_ball = load_ball_by_ball(args.source)
    match_strengths, latest_strengths = build_match_player_strengths(ball_by_ball, historical_matches)
    current_teams = set(load_teams(args.teams)) if args.teams.exists() else set()
    if current_teams:
        latest_strengths = latest_strengths[latest_strengths["team"].isin(current_teams)].reset_index(drop=True)

    args.match_output.parent.mkdir(parents=True, exist_ok=True)
    args.team_output.parent.mkdir(parents=True, exist_ok=True)
    match_strengths.to_csv(args.match_output, index=False)
    latest_strengths.to_csv(args.team_output, index=False)

    print(f"Saved match-level player strengths to: {args.match_output}")
    print(f"Saved latest team strengths to: {args.team_output}")


if __name__ == "__main__":
    main()
