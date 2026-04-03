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
    LIKELY_LINEUPS_2026_PATH,
    TEAM_OVERVIEW_2026_PATH,
    TEAM_PRIORS_2026_PATH,
    TEAM_SQUADS_2026_PATH,
)
from ipl_predictor.data import normalize_team_columns
from prepare_player_strength_features import (
    DEFAULT_BATTING_STRENGTH,
    DEFAULT_BOWLING_STRENGTH,
    NON_BOWLER_WICKETS,
    batting_rating,
    bowling_rating,
)


ROLE_WEIGHTS = {"Batters": 1.0, "All Rounders": 0.85, "Bowlers": 0.3}
BOWLING_ROLE_WEIGHTS = {"Batters": 0.1, "All Rounders": 0.8, "Bowlers": 1.0}
LINEUP_STATUS_WEIGHTS = {"starting_xi": 1.0, "impact_player": 0.65, "replaced_player": 0.45}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate editable 2026 team priors from official squads and IPL player history.")
    parser.add_argument("--source", type=Path, default=BALL_BY_BALL_SOURCE_PATH, help="Ball-by-ball IPL source CSV")
    parser.add_argument("--squads", type=Path, default=TEAM_SQUADS_2026_PATH, help="Official 2026 squads CSV")
    parser.add_argument("--overview", type=Path, default=TEAM_OVERVIEW_2026_PATH, help="Official 2026 team overview CSV")
    parser.add_argument("--lineups", type=Path, default=LIKELY_LINEUPS_2026_PATH, help="Optional likely XI and impact-player CSV")
    parser.add_argument("--output", type=Path, default=TEAM_PRIORS_2026_PATH, help="Output priors CSV")
    return parser.parse_args()


def load_ball_by_ball(path: Path) -> pd.DataFrame:
    usecols = [
        "event_name",
        "match_type",
        "gender",
        "batter",
        "bowler",
        "runs_batter",
        "runs_bowler",
        "valid_ball",
        "wicket_kind",
    ]
    df = pd.read_csv(path, engine="python", on_bad_lines="warn", usecols=usecols)
    df = df[(df["event_name"] == "Indian Premier League") & (df["match_type"] == "T20") & (df["gender"] == "male")].copy()
    return df


def build_player_stats(ball_by_ball: pd.DataFrame) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    batting_updates = (
        ball_by_ball.groupby("batter", dropna=False)
        .agg(runs=("runs_batter", "sum"), balls=("valid_ball", "sum"), innings=("batter", "size"))
        .reset_index()
    )
    bowling_base = ball_by_ball.copy()
    bowling_base["is_bowler_wicket"] = (
        bowling_base["wicket_kind"].notna()
        & ~bowling_base["wicket_kind"].astype(str).str.lower().isin(NON_BOWLER_WICKETS)
    ).astype(int)
    bowling_updates = (
        bowling_base.groupby("bowler", dropna=False)
        .agg(runs=("runs_bowler", "sum"), balls=("valid_ball", "sum"), wickets=("is_bowler_wicket", "sum"), matches=("bowler", "size"))
        .reset_index()
    )

    batting_stats = {
        str(row.batter): {"runs": float(row.runs), "balls": float(row.balls), "innings": float(row.innings)}
        for row in batting_updates.itertuples(index=False)
        if pd.notna(row.batter)
    }
    bowling_stats = {
        str(row.bowler): {"runs": float(row.runs), "balls": float(row.balls), "wickets": float(row.wickets), "matches": float(row.matches)}
        for row in bowling_updates.itertuples(index=False)
        if pd.notna(row.bowler)
    }
    return batting_stats, bowling_stats


def squad_metric(group: pd.DataFrame, batting_stats: dict[str, dict[str, float]], bowling_stats: dict[str, dict[str, float]]) -> tuple[float, float]:
    batting_scores = []
    bowling_scores = []
    for row in group.itertuples(index=False):
        batting_base = DEFAULT_BATTING_STRENGTH
        bowling_base = DEFAULT_BOWLING_STRENGTH
        if row.player in batting_stats:
            batting_base = batting_rating(batting_stats[row.player])
        if row.player in bowling_stats:
            bowling_base = bowling_rating(bowling_stats[row.player])
        lineup_weight = LINEUP_STATUS_WEIGHTS.get(getattr(row, "lineup_status", "starting_xi"), 1.0)
        batting_scores.append(batting_base * ROLE_WEIGHTS.get(row.role_group, 0.5) * lineup_weight)
        bowling_scores.append(bowling_base * BOWLING_ROLE_WEIGHTS.get(row.role_group, 0.5) * lineup_weight)

    batting_scores.sort(reverse=True)
    bowling_scores.sort(reverse=True)
    batting_strength = sum(batting_scores[:7]) / min(len(batting_scores), 7)
    bowling_strength = sum(bowling_scores[:6]) / min(len(bowling_scores), 6)
    return batting_strength, bowling_strength


def scale_bonus(value: float, center: float, spread: float) -> float:
    scaled = round((value - center) / spread)
    return float(max(-2, min(10, scaled + 5)))


def main() -> None:
    args = parse_args()
    squads = pd.read_csv(args.squads)
    overview = pd.read_csv(args.overview)
    squads = normalize_team_columns(squads, ["team"])
    overview = normalize_team_columns(overview, ["team"])
    if args.lineups.exists():
        squads = pd.read_csv(args.lineups)
        squads = normalize_team_columns(squads, ["team"])
    else:
        squads = squads.copy()
        squads["lineup_status"] = "starting_xi"
    ball_by_ball = load_ball_by_ball(args.source)
    batting_stats, bowling_stats = build_player_stats(ball_by_ball)

    rows = []
    for team, group in squads.groupby("team", sort=True):
        batting_strength, bowling_strength = squad_metric(group, batting_stats, bowling_stats)
        prior_rating = scale_bonus((batting_strength + bowling_strength) / 2.0, center=25.0, spread=2.5)
        batting_bonus = scale_bonus(batting_strength, center=28.0, spread=2.0)
        bowling_bonus = scale_bonus(bowling_strength, center=21.0, spread=1.5)
        team_meta = overview[overview["team"] == team].iloc[0]
        rows.append(
            {
                "team": team,
                "prior_rating": prior_rating,
                "batting_bonus": batting_bonus,
                "bowling_bonus": bowling_bonus,
                "captain": team_meta["captain"],
                "coach": team_meta["coach"],
                "venue": team_meta["venue"],
                "derived_batting_strength": round(batting_strength, 3),
                "derived_bowling_strength": round(bowling_strength, 3),
                "notes": "Auto-generated from official squads and historical IPL player performance",
            }
        )

    priors = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    priors.to_csv(args.output, index=False)
    print(f"Saved team priors to: {args.output}")


if __name__ == "__main__":
    main()
