from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

import pandas as pd


def _safe_rate(wins: int, total: int, default: float = 0.5) -> float:
    if total <= 0:
        return default
    return wins / total


def _safe_mean(total: float, count: int, default: float = 0.0) -> float:
    if count <= 0:
        return default
    return total / count


def _deque_mean(values: deque[float] | list[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(sum(values)) / len(values)


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _season_window_rate(records: list[tuple[int, int]], current_season: int, seasons: int = 3) -> float:
    lower_bound = current_season - seasons + 1
    recent_records = [result for season, result in records if lower_bound <= season <= current_season]
    return _safe_rate(sum(recent_records), len(recent_records))


def _is_reliable_scorecard(match) -> bool:
    method = str(getattr(match, "method", "") or "").strip().upper()
    return method != "D/L"


def build_training_frame(
    matches: pd.DataFrame,
    recent_window: int = 10,
    elo_k_factor: float = 24.0,
    base_elo: float = 1500.0,
) -> pd.DataFrame:
    team_results: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=recent_window))
    team_margins: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=recent_window))
    head_to_head: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    team_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    venue_totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    venue_scoring: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    venue_team_batting_totals: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    venue_batting_first_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    team_batting_first_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    team_chasing_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    team_season_results: dict[str, list[tuple[int, int]]] = defaultdict(lambda: [])
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
        team_1_venue_runs_scored, team_1_venue_batting_matches = venue_team_batting_totals[(team_1, match.venue)]
        team_2_venue_runs_scored, team_2_venue_batting_matches = venue_team_batting_totals[(team_2, match.venue)]
        team_1_runs_conceded, team_1_bowling_matches = bowling_totals[team_1]
        team_2_runs_conceded, team_2_bowling_matches = bowling_totals[team_2]
        venue_batting_first_wins, venue_batting_first_matches = venue_batting_first_totals[match.venue]
        team_1_batting_first_wins, team_1_batting_first_matches = team_batting_first_totals[team_1]
        team_2_batting_first_wins, team_2_batting_first_matches = team_batting_first_totals[team_2]
        team_1_chasing_wins, team_1_chasing_matches = team_chasing_totals[team_1]
        team_2_chasing_wins, team_2_chasing_matches = team_chasing_totals[team_2]
        team_1_recent_season_win_rate = _season_window_rate(team_season_results[team_1], int(match.season))
        team_2_recent_season_win_rate = _season_window_rate(team_season_results[team_2], int(match.season))

        team_1_recent = list(team_results[team_1])
        team_2_recent = list(team_results[team_2])
        team_1_recent_win_rate = _safe_rate(sum(team_1_recent), len(team_1_recent))
        team_2_recent_win_rate = _safe_rate(sum(team_2_recent), len(team_2_recent))
        team_1_recent_margin = _deque_mean(team_margins[team_1], default=0.0)
        team_2_recent_margin = _deque_mean(team_margins[team_2], default=0.0)
        venue_runs_total, venue_innings = venue_scoring[match.venue]

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
                "team_1_recent_win_rate": team_1_recent_win_rate,
                "team_2_recent_win_rate": team_2_recent_win_rate,
                "recent_win_rate_diff": team_1_recent_win_rate - team_2_recent_win_rate,
                "team_1_overall_win_rate": _safe_rate(team_1_total_wins, team_1_total_matches),
                "team_2_overall_win_rate": _safe_rate(team_2_total_wins, team_2_total_matches),
                "overall_win_rate_diff": _safe_rate(team_1_total_wins, team_1_total_matches) - _safe_rate(team_2_total_wins, team_2_total_matches),
                "team_1_recent_season_win_rate": team_1_recent_season_win_rate,
                "team_2_recent_season_win_rate": team_2_recent_season_win_rate,
                "recent_season_win_rate_diff": team_1_recent_season_win_rate - team_2_recent_season_win_rate,
                "team_1_venue_win_rate": _safe_rate(team_1_venue_wins, team_1_venue_matches),
                "team_2_venue_win_rate": _safe_rate(team_2_venue_wins, team_2_venue_matches),
                "venue_win_rate_diff": _safe_rate(team_1_venue_wins, team_1_venue_matches) - _safe_rate(team_2_venue_wins, team_2_venue_matches),
                "team_1_avg_runs_scored": _safe_rate(team_1_runs_scored, team_1_batting_matches, default=160.0),
                "team_2_avg_runs_scored": _safe_rate(team_2_runs_scored, team_2_batting_matches, default=160.0),
                "avg_runs_scored_diff": _safe_rate(team_1_runs_scored, team_1_batting_matches, default=160.0) - _safe_rate(team_2_runs_scored, team_2_batting_matches, default=160.0),
                "team_1_venue_avg_runs_scored": _safe_rate(team_1_venue_runs_scored, team_1_venue_batting_matches, default=160.0),
                "team_2_venue_avg_runs_scored": _safe_rate(team_2_venue_runs_scored, team_2_venue_batting_matches, default=160.0),
                "venue_avg_runs_scored_diff": _safe_rate(team_1_venue_runs_scored, team_1_venue_batting_matches, default=160.0)
                - _safe_rate(team_2_venue_runs_scored, team_2_venue_batting_matches, default=160.0),
                "team_1_avg_runs_conceded": _safe_rate(team_1_runs_conceded, team_1_bowling_matches, default=160.0),
                "team_2_avg_runs_conceded": _safe_rate(team_2_runs_conceded, team_2_bowling_matches, default=160.0),
                "avg_runs_conceded_diff": _safe_rate(team_1_runs_conceded, team_1_bowling_matches, default=160.0) - _safe_rate(team_2_runs_conceded, team_2_bowling_matches, default=160.0),
                "team_1_recent_margin": team_1_recent_margin,
                "team_2_recent_margin": team_2_recent_margin,
                "recent_margin_diff": team_1_recent_margin - team_2_recent_margin,
                "venue_avg_innings_score": _safe_mean(venue_runs_total, int(venue_innings), default=160.0),
                "venue_batting_first_win_rate": _safe_rate(venue_batting_first_wins, venue_batting_first_matches),
                "team_1_batting_first_win_rate": _safe_rate(team_1_batting_first_wins, team_1_batting_first_matches),
                "team_2_batting_first_win_rate": _safe_rate(team_2_batting_first_wins, team_2_batting_first_matches),
                "batting_first_win_rate_diff": _safe_rate(team_1_batting_first_wins, team_1_batting_first_matches)
                - _safe_rate(team_2_batting_first_wins, team_2_batting_first_matches),
                "team_1_chasing_win_rate": _safe_rate(team_1_chasing_wins, team_1_chasing_matches),
                "team_2_chasing_win_rate": _safe_rate(team_2_chasing_wins, team_2_chasing_matches),
                "chasing_win_rate_diff": _safe_rate(team_1_chasing_wins, team_1_chasing_matches) - _safe_rate(team_2_chasing_wins, team_2_chasing_matches),
                "h2h_win_rate_diff": _safe_rate(team_1_h2h_wins, h2h_total) - _safe_rate(team_2_h2h_wins, h2h_total),
                "team_1_elo": team_1_elo,
                "team_2_elo": team_2_elo,
                "elo_diff": team_1_elo - team_2_elo,
                "team_1_expected_score": team_1_expected,
                "team_2_expected_score": team_2_expected,
                "team_1_player_batting_strength": float(getattr(match, "team_1_batting_strength", np.nan)),
                "team_2_player_batting_strength": float(getattr(match, "team_2_batting_strength", np.nan)),
                "team_1_player_bowling_strength": float(getattr(match, "team_1_bowling_strength", np.nan)),
                "team_2_player_bowling_strength": float(getattr(match, "team_2_bowling_strength", np.nan)),
                "player_batting_strength_diff": float(getattr(match, "batting_strength_diff", np.nan)),
                "player_bowling_strength_diff": float(getattr(match, "bowling_strength_diff", np.nan)),
                "team_1_powerplay_batting_strength": float(getattr(match, "team_1_powerplay_batting_strength", np.nan)),
                "team_2_powerplay_batting_strength": float(getattr(match, "team_2_powerplay_batting_strength", np.nan)),
                "team_1_death_bowling_strength": float(getattr(match, "team_1_death_bowling_strength", np.nan)),
                "team_2_death_bowling_strength": float(getattr(match, "team_2_death_bowling_strength", np.nan)),
                "team_1_middle_batting_strength": float(getattr(match, "team_1_middle_batting_strength", np.nan)),
                "team_2_middle_batting_strength": float(getattr(match, "team_2_middle_batting_strength", np.nan)),
                "team_1_middle_bowling_strength": float(getattr(match, "team_1_middle_bowling_strength", np.nan)),
                "team_2_middle_bowling_strength": float(getattr(match, "team_2_middle_bowling_strength", np.nan)),
                "team_1_batting_depth": float(getattr(match, "team_1_batting_depth", np.nan)),
                "team_2_batting_depth": float(getattr(match, "team_2_batting_depth", np.nan)),
                "team_1_bowling_depth": float(getattr(match, "team_1_bowling_depth", np.nan)),
                "team_2_bowling_depth": float(getattr(match, "team_2_bowling_depth", np.nan)),
                "powerplay_batting_strength_diff": float(getattr(match, "powerplay_batting_strength_diff", np.nan)),
                "death_bowling_strength_diff": float(getattr(match, "death_bowling_strength_diff", np.nan)),
                "middle_batting_strength_diff": float(getattr(match, "middle_batting_strength_diff", np.nan)),
                "middle_bowling_strength_diff": float(getattr(match, "middle_bowling_strength_diff", np.nan)),
                "batting_depth_diff": float(getattr(match, "batting_depth_diff", np.nan)),
                "bowling_depth_diff": float(getattr(match, "bowling_depth_diff", np.nan)),
                "target": 1 if winner == team_1 else 0,
            }
        )

        team_1_won = int(winner == team_1)
        team_2_won = int(winner == team_2)
        team_1_score = float(getattr(match, "team_1_score", np.nan))
        team_2_score = float(getattr(match, "team_2_score", np.nan))
        score_reliable = _is_reliable_scorecard(match)
        team_totals[team_1][0] += team_1_won
        team_totals[team_1][1] += 1
        team_totals[team_2][0] += team_2_won
        team_totals[team_2][1] += 1
        venue_totals[(team_1, match.venue)][0] += team_1_won
        venue_totals[(team_1, match.venue)][1] += 1
        venue_totals[(team_2, match.venue)][0] += team_2_won
        venue_totals[(team_2, match.venue)][1] += 1
        if score_reliable and not np.isnan(team_1_score):
            batting_totals[team_1][0] += team_1_score
            batting_totals[team_1][1] += 1
            venue_team_batting_totals[(team_1, match.venue)][0] += team_1_score
            venue_team_batting_totals[(team_1, match.venue)][1] += 1
            bowling_totals[team_2][0] += team_1_score
            bowling_totals[team_2][1] += 1
            venue_scoring[match.venue][0] += team_1_score
            venue_scoring[match.venue][1] += 1
        if score_reliable and not np.isnan(team_2_score):
            batting_totals[team_2][0] += team_2_score
            batting_totals[team_2][1] += 1
            venue_team_batting_totals[(team_2, match.venue)][0] += team_2_score
            venue_team_batting_totals[(team_2, match.venue)][1] += 1
            bowling_totals[team_1][0] += team_2_score
            bowling_totals[team_1][1] += 1
            venue_scoring[match.venue][0] += team_2_score
            venue_scoring[match.venue][1] += 1
        if score_reliable and not np.isnan(team_1_score) and not np.isnan(team_2_score):
            margin = team_1_score - team_2_score
            team_margins[team_1].append(margin)
            team_margins[team_2].append(-margin)
        if not pd.isna(getattr(match, "team_1_batted_first", pd.NA)):
            team_1_batted_first = bool(getattr(match, "team_1_batted_first"))
            batting_first_won = (team_1_batted_first and team_1_won) or (not team_1_batted_first and team_2_won)
            venue_batting_first_totals[match.venue][0] += int(batting_first_won)
            venue_batting_first_totals[match.venue][1] += 1
            if team_1_batted_first:
                team_batting_first_totals[team_1][0] += team_1_won
                team_batting_first_totals[team_1][1] += 1
                team_chasing_totals[team_2][0] += team_2_won
                team_chasing_totals[team_2][1] += 1
            else:
                team_chasing_totals[team_1][0] += team_1_won
                team_chasing_totals[team_1][1] += 1
                team_batting_first_totals[team_2][0] += team_2_won
                team_batting_first_totals[team_2][1] += 1
        team_results[team_1].append(team_1_won)
        team_results[team_2].append(team_2_won)
        team_season_results[team_1].append((int(match.season), team_1_won))
        team_season_results[team_2].append((int(match.season), team_2_won))

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
    recent_window: int = 10,
    elo_k_factor: float = 24.0,
    base_elo: float = 1500.0,
) -> dict:
    team_results: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=recent_window))
    team_margins: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=recent_window))
    head_to_head: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    team_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    venue_totals: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    venue_scoring: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    venue_team_batting_totals: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    venue_batting_first_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    team_batting_first_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    team_chasing_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    team_season_results: dict[str, list[tuple[int, int]]] = defaultdict(lambda: [])
    batting_totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    bowling_totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    reliable_scores: list[float] = []
    elo_ratings: dict[str, float] = defaultdict(lambda: base_elo)
    player_team_strengths: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "batting_strength": 35.0,
            "bowling_strength": 18.0,
            "powerplay_batting_strength": 35.0,
            "death_bowling_strength": 18.0,
            "middle_batting_strength": 35.0,
            "middle_bowling_strength": 18.0,
            "batting_depth": 0.5,
            "bowling_depth": 0.5,
        }
    )

    for match in matches.sort_values(["date", "season"]).itertuples(index=False):
        team_1 = match.team_1
        team_2 = match.team_2
        winner = match.winner

        team_1_won = int(winner == team_1)
        team_2_won = int(winner == team_2)
        team_1_score = float(getattr(match, "team_1_score", np.nan))
        team_2_score = float(getattr(match, "team_2_score", np.nan))
        score_reliable = _is_reliable_scorecard(match)

        team_totals[team_1][0] += team_1_won
        team_totals[team_1][1] += 1
        team_totals[team_2][0] += team_2_won
        team_totals[team_2][1] += 1
        venue_totals[(team_1, match.venue)][0] += team_1_won
        venue_totals[(team_1, match.venue)][1] += 1
        venue_totals[(team_2, match.venue)][0] += team_2_won
        venue_totals[(team_2, match.venue)][1] += 1
        if score_reliable and not np.isnan(team_1_score):
            reliable_scores.append(team_1_score)
            batting_totals[team_1][0] += team_1_score
            batting_totals[team_1][1] += 1
            venue_team_batting_totals[(team_1, match.venue)][0] += team_1_score
            venue_team_batting_totals[(team_1, match.venue)][1] += 1
            bowling_totals[team_2][0] += team_1_score
            bowling_totals[team_2][1] += 1
            venue_scoring[match.venue][0] += team_1_score
            venue_scoring[match.venue][1] += 1
        if score_reliable and not np.isnan(team_2_score):
            reliable_scores.append(team_2_score)
            batting_totals[team_2][0] += team_2_score
            batting_totals[team_2][1] += 1
            venue_team_batting_totals[(team_2, match.venue)][0] += team_2_score
            venue_team_batting_totals[(team_2, match.venue)][1] += 1
            bowling_totals[team_1][0] += team_2_score
            bowling_totals[team_1][1] += 1
            venue_scoring[match.venue][0] += team_2_score
            venue_scoring[match.venue][1] += 1
        if score_reliable and not np.isnan(team_1_score) and not np.isnan(team_2_score):
            margin = team_1_score - team_2_score
            team_margins[team_1].append(margin)
            team_margins[team_2].append(-margin)
        if not pd.isna(getattr(match, "team_1_batted_first", pd.NA)):
            team_1_batted_first = bool(getattr(match, "team_1_batted_first"))
            batting_first_won = (team_1_batted_first and team_1_won) or (not team_1_batted_first and team_2_won)
            venue_batting_first_totals[match.venue][0] += int(batting_first_won)
            venue_batting_first_totals[match.venue][1] += 1
            if team_1_batted_first:
                team_batting_first_totals[team_1][0] += team_1_won
                team_batting_first_totals[team_1][1] += 1
                team_chasing_totals[team_2][0] += team_2_won
                team_chasing_totals[team_2][1] += 1
            else:
                team_chasing_totals[team_1][0] += team_1_won
                team_chasing_totals[team_1][1] += 1
                team_batting_first_totals[team_2][0] += team_2_won
                team_batting_first_totals[team_2][1] += 1
        team_results[team_1].append(team_1_won)
        team_results[team_2].append(team_2_won)
        team_season_results[team_1].append((int(match.season), team_1_won))
        team_season_results[team_2].append((int(match.season), team_2_won))

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
        "team_margins": team_margins,
        "head_to_head": head_to_head,
        "team_totals": team_totals,
        "venue_totals": venue_totals,
        "venue_scoring": venue_scoring,
        "venue_team_batting_totals": venue_team_batting_totals,
        "venue_batting_first_totals": venue_batting_first_totals,
        "team_batting_first_totals": team_batting_first_totals,
        "team_chasing_totals": team_chasing_totals,
        "team_season_results": team_season_results,
        "batting_totals": batting_totals,
        "bowling_totals": bowling_totals,
        "elo_ratings": elo_ratings,
        "player_team_strengths": player_team_strengths,
        "score_context": {
            "league_score_baseline": float(np.mean(reliable_scores)) if reliable_scores else 157.0,
            "innings_score_std": float(np.std(reliable_scores, ddof=1)) if len(reliable_scores) > 1 else 18.0,
        },
        "recent_window": recent_window,
        "elo_k_factor": elo_k_factor,
        "base_elo": base_elo,
    }


def make_match_features(match_row: pd.Series, state: dict, season: int = 2026) -> pd.DataFrame:
    team_1 = match_row["team_1"]
    team_2 = match_row["team_2"]
    match_season = int(pd.to_datetime(match_row["date"]).year) if "date" in match_row and not pd.isna(match_row["date"]) else season
    key = tuple(sorted((team_1, team_2)))
    h2h_wins = state["head_to_head"].setdefault(key, [0, 0])
    if team_1 <= team_2:
        team_1_h2h_wins, team_2_h2h_wins = h2h_wins
    else:
        team_2_h2h_wins, team_1_h2h_wins = h2h_wins
    h2h_total = team_1_h2h_wins + team_2_h2h_wins

    team_1_recent = list(state["team_results"][team_1])
    team_2_recent = list(state["team_results"][team_2])
    team_1_recent_win_rate = _safe_rate(sum(team_1_recent), len(team_1_recent))
    team_2_recent_win_rate = _safe_rate(sum(team_2_recent), len(team_2_recent))
    team_1_recent_margin = _deque_mean(state["team_margins"][team_1], default=0.0)
    team_2_recent_margin = _deque_mean(state["team_margins"][team_2], default=0.0)
    team_1_total_wins, team_1_total_matches = state["team_totals"][team_1]
    team_2_total_wins, team_2_total_matches = state["team_totals"][team_2]
    team_1_venue_wins, team_1_venue_matches = state["venue_totals"][(team_1, match_row["venue"])]
    team_2_venue_wins, team_2_venue_matches = state["venue_totals"][(team_2, match_row["venue"])]
    team_1_runs_scored, team_1_batting_matches = state["batting_totals"][team_1]
    team_2_runs_scored, team_2_batting_matches = state["batting_totals"][team_2]
    team_1_venue_runs_scored, team_1_venue_batting_matches = state["venue_team_batting_totals"][(team_1, match_row["venue"])]
    team_2_venue_runs_scored, team_2_venue_batting_matches = state["venue_team_batting_totals"][(team_2, match_row["venue"])]
    team_1_runs_conceded, team_1_bowling_matches = state["bowling_totals"][team_1]
    team_2_runs_conceded, team_2_bowling_matches = state["bowling_totals"][team_2]
    venue_runs_total, venue_innings = state["venue_scoring"][match_row["venue"]]
    venue_batting_first_wins, venue_batting_first_matches = state["venue_batting_first_totals"][match_row["venue"]]
    team_1_batting_first_wins, team_1_batting_first_matches = state["team_batting_first_totals"][team_1]
    team_2_batting_first_wins, team_2_batting_first_matches = state["team_batting_first_totals"][team_2]
    team_1_chasing_wins, team_1_chasing_matches = state["team_chasing_totals"][team_1]
    team_2_chasing_wins, team_2_chasing_matches = state["team_chasing_totals"][team_2]
    team_1_recent_season_win_rate = _season_window_rate(state["team_season_results"][team_1], match_season)
    team_2_recent_season_win_rate = _season_window_rate(state["team_season_results"][team_2], match_season)
    team_1_elo = state["elo_ratings"][team_1]
    team_2_elo = state["elo_ratings"][team_2]
    team_1_expected = _expected_score(team_1_elo, team_2_elo)
    team_2_expected = 1.0 - team_1_expected
    team_1_player_strengths = state["player_team_strengths"][team_1]
    team_2_player_strengths = state["player_team_strengths"][team_2]
    return pd.DataFrame(
        [
            {
                "season": match_season,
                "team_1": team_1,
                "team_2": team_2,
                "venue": match_row["venue"],
                "team_1_recent_win_rate": team_1_recent_win_rate,
                "team_2_recent_win_rate": team_2_recent_win_rate,
                "recent_win_rate_diff": team_1_recent_win_rate - team_2_recent_win_rate,
                "team_1_overall_win_rate": _safe_rate(team_1_total_wins, team_1_total_matches),
                "team_2_overall_win_rate": _safe_rate(team_2_total_wins, team_2_total_matches),
                "overall_win_rate_diff": _safe_rate(team_1_total_wins, team_1_total_matches) - _safe_rate(team_2_total_wins, team_2_total_matches),
                "team_1_recent_season_win_rate": team_1_recent_season_win_rate,
                "team_2_recent_season_win_rate": team_2_recent_season_win_rate,
                "recent_season_win_rate_diff": team_1_recent_season_win_rate - team_2_recent_season_win_rate,
                "team_1_venue_win_rate": _safe_rate(team_1_venue_wins, team_1_venue_matches),
                "team_2_venue_win_rate": _safe_rate(team_2_venue_wins, team_2_venue_matches),
                "venue_win_rate_diff": _safe_rate(team_1_venue_wins, team_1_venue_matches) - _safe_rate(team_2_venue_wins, team_2_venue_matches),
                "team_1_avg_runs_scored": _safe_rate(team_1_runs_scored, team_1_batting_matches, default=160.0),
                "team_2_avg_runs_scored": _safe_rate(team_2_runs_scored, team_2_batting_matches, default=160.0),
                "avg_runs_scored_diff": _safe_rate(team_1_runs_scored, team_1_batting_matches, default=160.0) - _safe_rate(team_2_runs_scored, team_2_batting_matches, default=160.0),
                "team_1_venue_avg_runs_scored": _safe_rate(team_1_venue_runs_scored, team_1_venue_batting_matches, default=160.0),
                "team_2_venue_avg_runs_scored": _safe_rate(team_2_venue_runs_scored, team_2_venue_batting_matches, default=160.0),
                "venue_avg_runs_scored_diff": _safe_rate(team_1_venue_runs_scored, team_1_venue_batting_matches, default=160.0)
                - _safe_rate(team_2_venue_runs_scored, team_2_venue_batting_matches, default=160.0),
                "team_1_avg_runs_conceded": _safe_rate(team_1_runs_conceded, team_1_bowling_matches, default=160.0),
                "team_2_avg_runs_conceded": _safe_rate(team_2_runs_conceded, team_2_bowling_matches, default=160.0),
                "avg_runs_conceded_diff": _safe_rate(team_1_runs_conceded, team_1_bowling_matches, default=160.0) - _safe_rate(team_2_runs_conceded, team_2_bowling_matches, default=160.0),
                "team_1_recent_margin": team_1_recent_margin,
                "team_2_recent_margin": team_2_recent_margin,
                "recent_margin_diff": team_1_recent_margin - team_2_recent_margin,
                "venue_avg_innings_score": _safe_mean(venue_runs_total, int(venue_innings), default=160.0),
                "venue_batting_first_win_rate": _safe_rate(venue_batting_first_wins, venue_batting_first_matches),
                "team_1_batting_first_win_rate": _safe_rate(team_1_batting_first_wins, team_1_batting_first_matches),
                "team_2_batting_first_win_rate": _safe_rate(team_2_batting_first_wins, team_2_batting_first_matches),
                "batting_first_win_rate_diff": _safe_rate(team_1_batting_first_wins, team_1_batting_first_matches)
                - _safe_rate(team_2_batting_first_wins, team_2_batting_first_matches),
                "team_1_chasing_win_rate": _safe_rate(team_1_chasing_wins, team_1_chasing_matches),
                "team_2_chasing_win_rate": _safe_rate(team_2_chasing_wins, team_2_chasing_matches),
                "chasing_win_rate_diff": _safe_rate(team_1_chasing_wins, team_1_chasing_matches) - _safe_rate(team_2_chasing_wins, team_2_chasing_matches),
                "h2h_win_rate_diff": _safe_rate(team_1_h2h_wins, h2h_total) - _safe_rate(team_2_h2h_wins, h2h_total),
                "team_1_elo": team_1_elo,
                "team_2_elo": team_2_elo,
                "elo_diff": team_1_elo - team_2_elo,
                "team_1_expected_score": team_1_expected,
                "team_2_expected_score": team_2_expected,
                "team_1_player_batting_strength": team_1_player_strengths["batting_strength"],
                "team_2_player_batting_strength": team_2_player_strengths["batting_strength"],
                "team_1_player_bowling_strength": team_1_player_strengths["bowling_strength"],
                "team_2_player_bowling_strength": team_2_player_strengths["bowling_strength"],
                "player_batting_strength_diff": team_1_player_strengths["batting_strength"] - team_2_player_strengths["batting_strength"],
                "player_bowling_strength_diff": team_1_player_strengths["bowling_strength"] - team_2_player_strengths["bowling_strength"],
                "team_1_powerplay_batting_strength": team_1_player_strengths["powerplay_batting_strength"],
                "team_2_powerplay_batting_strength": team_2_player_strengths["powerplay_batting_strength"],
                "team_1_death_bowling_strength": team_1_player_strengths["death_bowling_strength"],
                "team_2_death_bowling_strength": team_2_player_strengths["death_bowling_strength"],
                "team_1_middle_batting_strength": team_1_player_strengths["middle_batting_strength"],
                "team_2_middle_batting_strength": team_2_player_strengths["middle_batting_strength"],
                "team_1_middle_bowling_strength": team_1_player_strengths["middle_bowling_strength"],
                "team_2_middle_bowling_strength": team_2_player_strengths["middle_bowling_strength"],
                "team_1_batting_depth": team_1_player_strengths["batting_depth"],
                "team_2_batting_depth": team_2_player_strengths["batting_depth"],
                "team_1_bowling_depth": team_1_player_strengths["bowling_depth"],
                "team_2_bowling_depth": team_2_player_strengths["bowling_depth"],
                "powerplay_batting_strength_diff": team_1_player_strengths["powerplay_batting_strength"] - team_2_player_strengths["powerplay_batting_strength"],
                "death_bowling_strength_diff": team_1_player_strengths["death_bowling_strength"] - team_2_player_strengths["death_bowling_strength"],
                "middle_batting_strength_diff": team_1_player_strengths["middle_batting_strength"] - team_2_player_strengths["middle_batting_strength"],
                "middle_bowling_strength_diff": team_1_player_strengths["middle_bowling_strength"] - team_2_player_strengths["middle_bowling_strength"],
                "batting_depth_diff": team_1_player_strengths["batting_depth"] - team_2_player_strengths["batting_depth"],
                "bowling_depth_diff": team_1_player_strengths["bowling_depth"] - team_2_player_strengths["bowling_depth"],
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
        if state.get("current_venue") is not None:
            state["venue_team_batting_totals"][(team_1, state["current_venue"])][0] += team_1_score
            state["venue_team_batting_totals"][(team_1, state["current_venue"])][1] += 1
        state["bowling_totals"][team_2][0] += team_1_score
        state["bowling_totals"][team_2][1] += 1
        if state.get("current_venue") is not None:
            state["venue_scoring"][state["current_venue"]][0] += team_1_score
            state["venue_scoring"][state["current_venue"]][1] += 1
    if team_2_score is not None:
        state["batting_totals"][team_2][0] += team_2_score
        state["batting_totals"][team_2][1] += 1
        if state.get("current_venue") is not None:
            state["venue_team_batting_totals"][(team_2, state["current_venue"])][0] += team_2_score
            state["venue_team_batting_totals"][(team_2, state["current_venue"])][1] += 1
        state["bowling_totals"][team_1][0] += team_2_score
        state["bowling_totals"][team_1][1] += 1
        if state.get("current_venue") is not None:
            state["venue_scoring"][state["current_venue"]][0] += team_2_score
            state["venue_scoring"][state["current_venue"]][1] += 1
    if team_1_score is not None and team_2_score is not None:
        margin = team_1_score - team_2_score
        state["team_margins"][team_1].append(margin)
        state["team_margins"][team_2].append(-margin)
    current_season = int(state.get("current_season", 2026))
    state["team_season_results"][team_1].append((current_season, team_1_won))
    state["team_season_results"][team_2].append((current_season, team_2_won))

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
        "team_margins": defaultdict(
            lambda: deque(maxlen=initial_state["recent_window"]),
            {team: initial_state["team_margins"][team].copy() for team in initial_state["team_margins"]},
        ),
        "head_to_head": {key: value[:] for key, value in initial_state["head_to_head"].items()},
        "team_totals": defaultdict(lambda: [0, 0], {team: value[:] for team, value in initial_state["team_totals"].items()}),
        "venue_totals": defaultdict(
            lambda: [0, 0],
            {key: value[:] for key, value in initial_state["venue_totals"].items()},
        ),
        "venue_scoring": defaultdict(
            lambda: [0.0, 0.0],
            {venue: value[:] for venue, value in initial_state["venue_scoring"].items()},
        ),
        "venue_team_batting_totals": defaultdict(
            lambda: [0.0, 0.0],
            {key: value[:] for key, value in initial_state["venue_team_batting_totals"].items()},
        ),
        "venue_batting_first_totals": defaultdict(
            lambda: [0, 0],
            {venue: value[:] for venue, value in initial_state["venue_batting_first_totals"].items()},
        ),
        "team_batting_first_totals": defaultdict(
            lambda: [0, 0],
            {team: value[:] for team, value in initial_state["team_batting_first_totals"].items()},
        ),
        "team_chasing_totals": defaultdict(
            lambda: [0, 0],
            {team: value[:] for team, value in initial_state["team_chasing_totals"].items()},
        ),
        "team_season_results": defaultdict(
            lambda: [],
            {team: value[:] for team, value in initial_state["team_season_results"].items()},
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
        "player_team_strengths": defaultdict(
            lambda: {
                "batting_strength": 35.0,
                "bowling_strength": 18.0,
                "powerplay_batting_strength": 35.0,
                "death_bowling_strength": 18.0,
                "middle_batting_strength": 35.0,
                "middle_bowling_strength": 18.0,
                "batting_depth": 0.5,
                "bowling_depth": 0.5,
            },
            {team: values.copy() for team, values in initial_state["player_team_strengths"].items()},
        ),
        "score_context": dict(initial_state.get("score_context", {})),
        "recent_window": initial_state["recent_window"],
        "elo_k_factor": initial_state["elo_k_factor"],
        "base_elo": initial_state["base_elo"],
        "current_venue": None,
        "current_season": None,
        "current_team_1_score": None,
        "current_team_2_score": None,
    }
