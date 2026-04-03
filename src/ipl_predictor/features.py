from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

import pandas as pd


def _safe_rate(wins: int, total: int, default: float = 0.5) -> float:
    if total <= 0:
        return default
    return wins / total


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def build_training_frame(
    matches: pd.DataFrame,
    recent_window: int = 5,
    elo_k_factor: float = 24.0,
    base_elo: float = 1500.0,
) -> pd.DataFrame:
    team_results: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=recent_window))
    head_to_head: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    team_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    venue_totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    batting_totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    bowling_totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    elo_ratings: dict[str, float] = defaultdict(lambda: base_elo)
    rows: list[dict] = []

    for match in matches.sort_values(["date", "season"]).itertuples(index=False):
        team_1 = match.team_1
        team_2 = match.team_2
        winner = match.winner
        team_1_total_wins, team_1_total_matches = team_totals[team_1]
        team_2_total_wins, team_2_total_matches = team_totals[team_2]
        team_1_venue_wins, team_1_venue_matches = venue_totals[(team_1, match.venue)]
        team_2_venue_wins, team_2_venue_matches = venue_totals[(team_2, match.venue)]
        team_1_runs_scored, team_1_batting_matches = batting_totals[team_1]
        team_2_runs_scored, team_2_batting_matches = batting_totals[team_2]
        team_1_runs_conceded, team_1_bowling_matches = bowling_totals[team_1]
        team_2_runs_conceded, team_2_bowling_matches = bowling_totals[team_2]

        team_1_recent = list(team_results[team_1])
        team_2_recent = list(team_results[team_2])

        h2h_key = tuple(sorted((team_1, team_2)))
        h2h_wins = head_to_head[h2h_key]
        if team_1 <= team_2:
            team_1_h2h_wins, team_2_h2h_wins = h2h_wins
        else:
            team_2_h2h_wins, team_1_h2h_wins = h2h_wins
        h2h_total = team_1_h2h_wins + team_2_h2h_wins

        team_1_elo = elo_ratings[team_1]
        team_2_elo = elo_ratings[team_2]
        team_1_expected = _expected_score(team_1_elo, team_2_elo)
        team_2_expected = 1.0 - team_1_expected

        rows.append(
            {
                "season": match.season,
                "team_1": team_1,
                "team_2": team_2,
                "venue": match.venue,
                "toss_winner": match.toss_winner,
                "toss_decision": match.toss_decision,
                "team_1_recent_win_rate": _safe_rate(sum(team_1_recent), len(team_1_recent)),
                "team_2_recent_win_rate": _safe_rate(sum(team_2_recent), len(team_2_recent)),
                "team_1_overall_win_rate": _safe_rate(team_1_total_wins, team_1_total_matches),
                "team_2_overall_win_rate": _safe_rate(team_2_total_wins, team_2_total_matches),
                "team_1_venue_win_rate": _safe_rate(team_1_venue_wins, team_1_venue_matches),
                "team_2_venue_win_rate": _safe_rate(team_2_venue_wins, team_2_venue_matches),
                "team_1_avg_runs_scored": _safe_rate(team_1_runs_scored, team_1_batting_matches, default=160.0),
                "team_2_avg_runs_scored": _safe_rate(team_2_runs_scored, team_2_batting_matches, default=160.0),
                "team_1_avg_runs_conceded": _safe_rate(team_1_runs_conceded, team_1_bowling_matches, default=160.0),
                "team_2_avg_runs_conceded": _safe_rate(team_2_runs_conceded, team_2_bowling_matches, default=160.0),
                "team_1_h2h_win_rate": _safe_rate(team_1_h2h_wins, h2h_total),
                "team_2_h2h_win_rate": _safe_rate(team_2_h2h_wins, h2h_total),
                "team_1_elo": team_1_elo,
                "team_2_elo": team_2_elo,
                "elo_diff": team_1_elo - team_2_elo,
                "team_1_expected_score": team_1_expected,
                "team_2_expected_score": team_2_expected,
                "target": 1 if winner == team_1 else 0,
            }
        )

        team_1_won = int(winner == team_1)
        team_2_won = int(winner == team_2)
        team_1_score = float(getattr(match, "team_1_score", np.nan))
        team_2_score = float(getattr(match, "team_2_score", np.nan))
        team_totals[team_1][0] += team_1_won
        team_totals[team_1][1] += 1
        team_totals[team_2][0] += team_2_won
        team_totals[team_2][1] += 1
        venue_totals[(team_1, match.venue)][0] += team_1_won
        venue_totals[(team_1, match.venue)][1] += 1
        venue_totals[(team_2, match.venue)][0] += team_2_won
        venue_totals[(team_2, match.venue)][1] += 1
        if not np.isnan(team_1_score):
            batting_totals[team_1][0] += team_1_score
            batting_totals[team_1][1] += 1
            bowling_totals[team_2][0] += team_1_score
            bowling_totals[team_2][1] += 1
        if not np.isnan(team_2_score):
            batting_totals[team_2][0] += team_2_score
            batting_totals[team_2][1] += 1
            bowling_totals[team_1][0] += team_2_score
            bowling_totals[team_1][1] += 1
        team_results[team_1].append(team_1_won)
        team_results[team_2].append(team_2_won)

        if team_1 <= team_2:
            head_to_head[h2h_key][0] += team_1_won
            head_to_head[h2h_key][1] += team_2_won
        else:
            head_to_head[h2h_key][0] += team_2_won
            head_to_head[h2h_key][1] += team_1_won

        elo_ratings[team_1] += elo_k_factor * (team_1_won - team_1_expected)
        elo_ratings[team_2] += elo_k_factor * (team_2_won - team_2_expected)

    return pd.DataFrame(rows)


def initialize_state(
    matches: pd.DataFrame,
    recent_window: int = 5,
    elo_k_factor: float = 24.0,
    base_elo: float = 1500.0,
) -> dict:
    team_results: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=recent_window))
    head_to_head: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    team_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    venue_totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    batting_totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    bowling_totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    elo_ratings: dict[str, float] = defaultdict(lambda: base_elo)

    for match in matches.sort_values(["date", "season"]).itertuples(index=False):
        team_1 = match.team_1
        team_2 = match.team_2
        winner = match.winner

        team_1_won = int(winner == team_1)
        team_2_won = int(winner == team_2)
        team_1_score = float(getattr(match, "team_1_score", np.nan))
        team_2_score = float(getattr(match, "team_2_score", np.nan))

        team_totals[team_1][0] += team_1_won
        team_totals[team_1][1] += 1
        team_totals[team_2][0] += team_2_won
        team_totals[team_2][1] += 1
        venue_totals[(team_1, match.venue)][0] += team_1_won
        venue_totals[(team_1, match.venue)][1] += 1
        venue_totals[(team_2, match.venue)][0] += team_2_won
        venue_totals[(team_2, match.venue)][1] += 1
        if not np.isnan(team_1_score):
            batting_totals[team_1][0] += team_1_score
            batting_totals[team_1][1] += 1
            bowling_totals[team_2][0] += team_1_score
            bowling_totals[team_2][1] += 1
        if not np.isnan(team_2_score):
            batting_totals[team_2][0] += team_2_score
            batting_totals[team_2][1] += 1
            bowling_totals[team_1][0] += team_2_score
            bowling_totals[team_1][1] += 1
        team_results[team_1].append(team_1_won)
        team_results[team_2].append(team_2_won)

        key = tuple(sorted((team_1, team_2)))
        if team_1 <= team_2:
            head_to_head[key][0] += team_1_won
            head_to_head[key][1] += team_2_won
        else:
            head_to_head[key][0] += team_2_won
            head_to_head[key][1] += team_1_won

        team_1_elo = elo_ratings[team_1]
        team_2_elo = elo_ratings[team_2]
        team_1_expected = _expected_score(team_1_elo, team_2_elo)
        team_2_expected = 1.0 - team_1_expected
        elo_ratings[team_1] += elo_k_factor * (team_1_won - team_1_expected)
        elo_ratings[team_2] += elo_k_factor * (team_2_won - team_2_expected)

    return {
        "team_results": team_results,
        "head_to_head": head_to_head,
        "team_totals": team_totals,
        "venue_totals": venue_totals,
        "batting_totals": batting_totals,
        "bowling_totals": bowling_totals,
        "elo_ratings": elo_ratings,
        "recent_window": recent_window,
        "elo_k_factor": elo_k_factor,
        "base_elo": base_elo,
    }


def make_match_features(match_row: pd.Series, state: dict, season: int = 2026) -> pd.DataFrame:
    team_1 = match_row["team_1"]
    team_2 = match_row["team_2"]
    key = tuple(sorted((team_1, team_2)))
    h2h_wins = state["head_to_head"].setdefault(key, [0, 0])
    if team_1 <= team_2:
        team_1_h2h_wins, team_2_h2h_wins = h2h_wins
    else:
        team_2_h2h_wins, team_1_h2h_wins = h2h_wins
    h2h_total = team_1_h2h_wins + team_2_h2h_wins

    team_1_recent = list(state["team_results"][team_1])
    team_2_recent = list(state["team_results"][team_2])
    team_1_total_wins, team_1_total_matches = state["team_totals"][team_1]
    team_2_total_wins, team_2_total_matches = state["team_totals"][team_2]
    team_1_venue_wins, team_1_venue_matches = state["venue_totals"][(team_1, match_row["venue"])]
    team_2_venue_wins, team_2_venue_matches = state["venue_totals"][(team_2, match_row["venue"])]
    team_1_runs_scored, team_1_batting_matches = state["batting_totals"][team_1]
    team_2_runs_scored, team_2_batting_matches = state["batting_totals"][team_2]
    team_1_runs_conceded, team_1_bowling_matches = state["bowling_totals"][team_1]
    team_2_runs_conceded, team_2_bowling_matches = state["bowling_totals"][team_2]
    team_1_elo = state["elo_ratings"][team_1]
    team_2_elo = state["elo_ratings"][team_2]
    team_1_expected = _expected_score(team_1_elo, team_2_elo)
    team_2_expected = 1.0 - team_1_expected

    toss_winner = match_row.get("toss_winner", team_1)
    toss_decision = match_row.get("toss_decision", "field")

    return pd.DataFrame(
        [
            {
                "season": season,
                "team_1": team_1,
                "team_2": team_2,
                "venue": match_row["venue"],
                "toss_winner": toss_winner,
                "toss_decision": toss_decision,
                "team_1_recent_win_rate": _safe_rate(sum(team_1_recent), len(team_1_recent)),
                "team_2_recent_win_rate": _safe_rate(sum(team_2_recent), len(team_2_recent)),
                "team_1_overall_win_rate": _safe_rate(team_1_total_wins, team_1_total_matches),
                "team_2_overall_win_rate": _safe_rate(team_2_total_wins, team_2_total_matches),
                "team_1_venue_win_rate": _safe_rate(team_1_venue_wins, team_1_venue_matches),
                "team_2_venue_win_rate": _safe_rate(team_2_venue_wins, team_2_venue_matches),
                "team_1_avg_runs_scored": _safe_rate(team_1_runs_scored, team_1_batting_matches, default=160.0),
                "team_2_avg_runs_scored": _safe_rate(team_2_runs_scored, team_2_batting_matches, default=160.0),
                "team_1_avg_runs_conceded": _safe_rate(team_1_runs_conceded, team_1_bowling_matches, default=160.0),
                "team_2_avg_runs_conceded": _safe_rate(team_2_runs_conceded, team_2_bowling_matches, default=160.0),
                "team_1_h2h_win_rate": _safe_rate(team_1_h2h_wins, h2h_total),
                "team_2_h2h_win_rate": _safe_rate(team_2_h2h_wins, h2h_total),
                "team_1_elo": team_1_elo,
                "team_2_elo": team_2_elo,
                "elo_diff": team_1_elo - team_2_elo,
                "team_1_expected_score": team_1_expected,
                "team_2_expected_score": team_2_expected,
            }
        ]
    )


def update_state_after_match(team_1: str, team_2: str, winner: str, state: dict) -> None:
    team_1_won = int(winner == team_1)
    team_2_won = int(winner == team_2)

    state["team_results"][team_1].append(team_1_won)
    state["team_results"][team_2].append(team_2_won)
    state["team_totals"][team_1][0] += team_1_won
    state["team_totals"][team_1][1] += 1
    state["team_totals"][team_2][0] += team_2_won
    state["team_totals"][team_2][1] += 1
    team_1_score = state.get("current_team_1_score")
    team_2_score = state.get("current_team_2_score")
    if team_1_score is not None:
        state["batting_totals"][team_1][0] += team_1_score
        state["batting_totals"][team_1][1] += 1
        state["bowling_totals"][team_2][0] += team_1_score
        state["bowling_totals"][team_2][1] += 1
    if team_2_score is not None:
        state["batting_totals"][team_2][0] += team_2_score
        state["batting_totals"][team_2][1] += 1
        state["bowling_totals"][team_1][0] += team_2_score
        state["bowling_totals"][team_1][1] += 1

    key = tuple(sorted((team_1, team_2)))
    state["head_to_head"].setdefault(key, [0, 0])
    if team_1 <= team_2:
        state["head_to_head"][key][0] += team_1_won
        state["head_to_head"][key][1] += team_2_won
    else:
        state["head_to_head"][key][0] += team_2_won
        state["head_to_head"][key][1] += team_1_won

    venue = state.get("current_venue")
    if venue is not None:
        state["venue_totals"][(team_1, venue)][0] += team_1_won
        state["venue_totals"][(team_1, venue)][1] += 1
        state["venue_totals"][(team_2, venue)][0] += team_2_won
        state["venue_totals"][(team_2, venue)][1] += 1

    team_1_elo = state["elo_ratings"][team_1]
    team_2_elo = state["elo_ratings"][team_2]
    team_1_expected = _expected_score(team_1_elo, team_2_elo)
    team_2_expected = 1.0 - team_1_expected
    state["elo_ratings"][team_1] += state["elo_k_factor"] * (team_1_won - team_1_expected)
    state["elo_ratings"][team_2] += state["elo_k_factor"] * (team_2_won - team_2_expected)


def prepare_simulation_state(initial_state: dict) -> dict:
    return {
        "team_results": defaultdict(
            lambda: deque(maxlen=initial_state["recent_window"]),
            {team: initial_state["team_results"][team].copy() for team in initial_state["team_results"]},
        ),
        "head_to_head": {key: value[:] for key, value in initial_state["head_to_head"].items()},
        "team_totals": defaultdict(lambda: [0, 0], {team: value[:] for team, value in initial_state["team_totals"].items()}),
        "venue_totals": defaultdict(
            lambda: [0, 0],
            {key: value[:] for key, value in initial_state["venue_totals"].items()},
        ),
        "batting_totals": defaultdict(
            lambda: [0.0, 0.0],
            {team: value[:] for team, value in initial_state["batting_totals"].items()},
        ),
        "bowling_totals": defaultdict(
            lambda: [0.0, 0.0],
            {team: value[:] for team, value in initial_state["bowling_totals"].items()},
        ),
        "elo_ratings": defaultdict(
            lambda: initial_state["base_elo"],
            dict(initial_state["elo_ratings"]),
        ),
        "recent_window": initial_state["recent_window"],
        "elo_k_factor": initial_state["elo_k_factor"],
        "base_elo": initial_state["base_elo"],
        "current_venue": None,
        "current_team_1_score": None,
        "current_team_2_score": None,
    }
