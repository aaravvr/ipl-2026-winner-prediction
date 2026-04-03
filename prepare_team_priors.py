from __future__ import annotations

import argparse
import sys
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
from ipl_predictor.data import normalize_player_columns, normalize_player_name, normalize_team_columns
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
PLAYER_HISTORY_ALIASES = {
    "Mohammad Shami": "Mohammed Shami",
    "Nitish Kumar Reddy": "Nithish Kumar Reddy",
    "Prasidh Krishna": "M Prasidh Krishna",
    "Sai Sudharsan": "B Sai Sudharsan",
    "Shahbaz Ahmad": "Shahbaz Ahmed",
    "Varun Chakaravarthy": "CV Varun",
    "Vyshak Vijaykumar": "Vijaykumar Vyshak",
}


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


def player_candidate_score(stats: dict[str, float]) -> float:
    return sum(float(value) for value in stats.values())


def canonical_player_keys(player: str) -> list[str]:
    normalized_player = str(normalize_player_name(player))
    parts = normalized_player.split()
    if not parts:
        return []

    surname = parts[-1].lower()
    initials = "".join(part[0].lower() for part in parts[:-1] if part)
    first_initial = parts[0][0].lower()
    keys = [f"{surname}|{first_initial}"]
    if initials:
        keys.insert(0, f"{surname}|{initials}")
    return keys


def build_player_indices(stats_lookup: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    indices: dict[str, list[str]] = {}
    for player in stats_lookup:
        for key in canonical_player_keys(player):
            indices.setdefault(key, []).append(player)
    return indices


def resolve_player_name(
    player: str,
    stats_lookup: dict[str, dict[str, float]],
    player_indices: dict[str, list[str]],
) -> str | None:
    normalized_player = str(normalize_player_name(player))
    if normalized_player in stats_lookup:
        return normalized_player

    alias = PLAYER_HISTORY_ALIASES.get(normalized_player)
    if alias and alias in stats_lookup:
        return alias

    candidates: list[str] = []
    for key in canonical_player_keys(normalized_player):
        candidates.extend(player_indices.get(key, []))
        if candidates:
            break

    if candidates:
        unique_candidates = sorted(set(candidates))
        if len(unique_candidates) == 1:
            return unique_candidates[0]
        return max(unique_candidates, key=lambda candidate: player_candidate_score(stats_lookup[candidate]))
    return None


def squad_metric(
    group: pd.DataFrame,
    batting_stats: dict[str, dict[str, float]],
    bowling_stats: dict[str, dict[str, float]],
    batting_indices: dict[str, list[str]],
    bowling_indices: dict[str, list[str]],
) -> tuple[float, float]:
    batting_scores = []
    bowling_scores = []
    for row in group.itertuples(index=False):
        batting_base = DEFAULT_BATTING_STRENGTH
        bowling_base = DEFAULT_BOWLING_STRENGTH
        batting_name = resolve_player_name(row.player, batting_stats, batting_indices)
        bowling_name = resolve_player_name(row.player, bowling_stats, bowling_indices)
        if batting_name:
            batting_base = batting_rating(batting_stats[batting_name])
        if bowling_name:
            bowling_base = bowling_rating(bowling_stats[bowling_name])
        lineup_weight = LINEUP_STATUS_WEIGHTS.get(getattr(row, "lineup_status", "starting_xi"), 1.0)
        batting_scores.append(batting_base * ROLE_WEIGHTS.get(row.role_group, 0.5) * lineup_weight)
        bowling_scores.append(bowling_base * BOWLING_ROLE_WEIGHTS.get(row.role_group, 0.5) * lineup_weight)

    batting_scores.sort(reverse=True)
    bowling_scores.sort(reverse=True)
    batting_strength = sum(batting_scores[:7]) / min(len(batting_scores), 7)
    bowling_strength = sum(bowling_scores[:6]) / min(len(bowling_scores), 6)
    return batting_strength, bowling_strength


def scale_from_rank(values: list[float], low: int, high: int) -> list[float]:
    series = pd.Series(values, dtype=float)
    if series.empty:
        return []
    if series.nunique(dropna=False) <= 1:
        midpoint = round((low + high) / 2)
        return [float(midpoint)] * len(series)
    ranks = series.rank(method="average", pct=True)
    scaled = (low + (high - low) * ranks).round().clip(lower=low, upper=high)
    return scaled.astype(float).tolist()


def main() -> None:
    args = parse_args()
    squads = pd.read_csv(args.squads)
    overview = pd.read_csv(args.overview)
    squads = normalize_team_columns(squads, ["team"])
    overview = normalize_team_columns(overview, ["team"])
    squads = normalize_player_columns(squads, ["player"])
    if args.lineups.exists():
        squads = pd.read_csv(args.lineups)
        squads = normalize_team_columns(squads, ["team"])
        squads = normalize_player_columns(squads, ["player"])
    else:
        squads = squads.copy()
        squads["lineup_status"] = "starting_xi"
    ball_by_ball = load_ball_by_ball(args.source)
    batting_stats, bowling_stats = build_player_stats(ball_by_ball)
    batting_indices = build_player_indices(batting_stats)
    bowling_indices = build_player_indices(bowling_stats)

    rows = []
    for team, group in squads.groupby("team", sort=True):
        batting_strength, bowling_strength = squad_metric(group, batting_stats, bowling_stats, batting_indices, bowling_indices)
        team_meta = overview[overview["team"] == team].iloc[0]
        rows.append(
            {
                "team": team,
                "captain": team_meta["captain"],
                "coach": team_meta["coach"],
                "venue": team_meta["venue"],
                "derived_batting_strength": batting_strength,
                "derived_bowling_strength": bowling_strength,
                "notes": "Auto-generated from official squads and historical IPL player performance",
            }
        )

    priors = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)
    priors["prior_rating"] = scale_from_rank(
        ((priors["derived_batting_strength"] + priors["derived_bowling_strength"]) / 2.0).tolist(),
        low=-2,
        high=3,
    )
    priors["batting_bonus"] = scale_from_rank(priors["derived_batting_strength"].tolist(), low=-2, high=4)
    priors["bowling_bonus"] = scale_from_rank(priors["derived_bowling_strength"].tolist(), low=-2, high=4)
    priors["derived_batting_strength"] = priors["derived_batting_strength"].round(3)
    priors["derived_bowling_strength"] = priors["derived_bowling_strength"].round(3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    priors.to_csv(args.output, index=False)
    print(f"Saved team priors to: {args.output}")


if __name__ == "__main__":
    main()
