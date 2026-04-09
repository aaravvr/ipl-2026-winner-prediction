from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .features import make_match_features, prepare_simulation_state, update_state_after_match


PLAYOFF_VENUES = {
    "qualifier_1": "Narendra Modi Stadium",
    "eliminator": "Eden Gardens",
    "qualifier_2": "Wankhede Stadium",
    "final": "Narendra Modi Stadium",
}

LEAGUE_SCORE_BASELINE = 157.0
LEAGUE_BATTING_STRENGTH = 33.0
LEAGUE_BOWLING_STRENGTH = 17.0
INNINGS_SCORE_STD = 18.0
MIN_INNINGS_SCORE = 70.0
MAX_INNINGS_SCORE = 260.0


def _initialize_table(teams: list[str]) -> pd.DataFrame:
    table = pd.DataFrame({"team": teams})
    table["played"] = 0
    table["wins"] = 0
    table["losses"] = 0
    table["points"] = 0
    table["runs_for"] = 0.0
    table["overs_faced"] = 0.0
    table["runs_against"] = 0.0
    table["overs_bowled"] = 0.0
    table["nrr"] = 0.0
    return table.set_index("team")


def _sample_margin(win_probability: float, rng: np.random.Generator) -> float:
    closeness = abs(win_probability - 0.5)
    median_margin = np.interp(closeness, [0.0, 0.3], [4.0, 32.0])
    sigma = 1.0 if closeness < 0.12 else 0.8
    sampled = rng.lognormal(mean=np.log(max(median_margin, 1.0)), sigma=sigma)
    return float(np.clip(sampled, 1.0, 120.0))


def _score_context(state: dict) -> tuple[float, float]:
    context = state.get("score_context", {})
    baseline = float(context.get("league_score_baseline", LEAGUE_SCORE_BASELINE))
    innings_std = float(context.get("innings_score_std", INNINGS_SCORE_STD))
    return baseline, float(np.clip(innings_std, 12.0, 28.0))


def _strength_context(state: dict) -> tuple[float, float]:
    strengths = list(state["player_team_strengths"].values())
    if not strengths:
        return LEAGUE_BATTING_STRENGTH, LEAGUE_BOWLING_STRENGTH
    batting = np.mean([float(value.get("batting_strength", LEAGUE_BATTING_STRENGTH)) for value in strengths])
    bowling = np.mean([float(value.get("bowling_strength", LEAGUE_BOWLING_STRENGTH)) for value in strengths])
    return float(batting), float(bowling)


def _estimate_scores(
    features: pd.DataFrame,
    state: dict,
    winner: str,
    team_1: str,
    team_2: str,
    win_probability: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    row = features.iloc[0]
    venue_base = float(row["venue_avg_innings_score"])
    league_score_baseline, innings_score_std = _score_context(state)
    league_batting_strength, league_bowling_strength = _strength_context(state)

    team_1_expected = (
        venue_base
        + 0.45 * (float(row["team_1_avg_runs_scored"]) - venue_base)
        + 0.30 * (float(row["team_2_avg_runs_conceded"]) - venue_base)
        + 1.8 * (float(row["team_1_player_batting_strength"]) - league_batting_strength)
        - 1.8 * (float(row["team_2_player_bowling_strength"]) - league_bowling_strength)
        + 14.0 * (float(row["team_1_expected_score"]) - 0.5)
    )
    team_2_expected = (
        venue_base
        + 0.45 * (float(row["team_2_avg_runs_scored"]) - venue_base)
        + 0.30 * (float(row["team_1_avg_runs_conceded"]) - venue_base)
        + 1.8 * (float(row["team_2_player_batting_strength"]) - league_batting_strength)
        - 1.8 * (float(row["team_1_player_bowling_strength"]) - league_bowling_strength)
        + 14.0 * (float(row["team_2_expected_score"]) - 0.5)
    )
    total_bias = 0.35 * ((team_1_expected + team_2_expected) / 2.0 - league_score_baseline)
    team_1_score = np.clip(team_1_expected + total_bias + rng.normal(0.0, innings_score_std), MIN_INNINGS_SCORE, MAX_INNINGS_SCORE)
    team_2_score = np.clip(team_2_expected + total_bias + rng.normal(0.0, innings_score_std), MIN_INNINGS_SCORE, MAX_INNINGS_SCORE)
    margin = _sample_margin(win_probability, rng)
    if winner == team_1 and team_1_score <= team_2_score:
        team_2_score = min(team_2_score, MAX_INNINGS_SCORE - margin)
        team_1_score = team_2_score + margin
    elif winner == team_2 and team_2_score <= team_1_score:
        team_1_score = min(team_1_score, MAX_INNINGS_SCORE - margin)
        team_2_score = team_1_score + margin
    team_1_score = float(np.clip(team_1_score, MIN_INNINGS_SCORE, MAX_INNINGS_SCORE))
    team_2_score = float(np.clip(team_2_score, MIN_INNINGS_SCORE, MAX_INNINGS_SCORE))
    return round(team_1_score, 1), round(team_2_score, 1)


def _update_nrr(table: pd.DataFrame, team_1: str, team_2: str, team_1_score: float, team_2_score: float) -> None:
    overs = 20.0
    table.loc[team_1, "runs_for"] += team_1_score
    table.loc[team_1, "overs_faced"] += overs
    table.loc[team_1, "runs_against"] += team_2_score
    table.loc[team_1, "overs_bowled"] += overs
    table.loc[team_2, "runs_for"] += team_2_score
    table.loc[team_2, "overs_faced"] += overs
    table.loc[team_2, "runs_against"] += team_1_score
    table.loc[team_2, "overs_bowled"] += overs


def _refresh_nrr(table: pd.DataFrame) -> pd.DataFrame:
    refreshed = table.copy()
    batting_rate = refreshed["runs_for"] / refreshed["overs_faced"].replace(0, np.nan)
    bowling_rate = refreshed["runs_against"] / refreshed["overs_bowled"].replace(0, np.nan)
    refreshed["nrr"] = (batting_rate - bowling_rate).fillna(0.0)
    return refreshed


def _record_result(table: pd.DataFrame, team_1: str, team_2: str, winner: str, team_1_score: float, team_2_score: float) -> None:
    loser = team_2 if winner == team_1 else team_1
    table.loc[team_1, "played"] += 1
    table.loc[team_2, "played"] += 1
    table.loc[winner, "wins"] += 1
    table.loc[loser, "losses"] += 1
    table.loc[winner, "points"] += 2
    _update_nrr(table, team_1, team_2, team_1_score, team_2_score)


def _head_to_head_tiebreak(team_a: str, team_b: str, state: dict) -> int:
    key = tuple(sorted((team_a, team_b)))
    wins = state["head_to_head"].get(key, [0, 0])
    if team_a <= team_b:
        team_a_wins, team_b_wins = wins
    else:
        team_b_wins, team_a_wins = wins
    if team_a_wins != team_b_wins:
        return -1 if team_a_wins > team_b_wins else 1
    if team_a != team_b:
        return -1 if team_a < team_b else 1
    return 0


def _mini_league_metrics(teams: list[str], state: dict) -> dict[str, tuple[int, float]]:
    metrics: dict[str, tuple[int, float]] = {}
    for team in teams:
        mini_wins = 0
        mini_played = 0
        for opponent in teams:
            if team == opponent:
                continue
            key = tuple(sorted((team, opponent)))
            wins = state["head_to_head"].get(key, [0, 0])
            if team <= opponent:
                team_wins, opponent_wins = wins
            else:
                opponent_wins, team_wins = wins
            mini_wins += team_wins
            mini_played += team_wins + opponent_wins
        metrics[team] = (mini_wins, (mini_wins / mini_played) if mini_played else 0.5)
    return metrics


def _sort_table(table: pd.DataFrame, state: dict) -> pd.DataFrame:
    sorted_table = _refresh_nrr(table).reset_index()
    sorted_table = sorted_table.sort_values(
        by=["points", "nrr", "wins", "team"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = sorted_table.groupby(["points", "nrr", "wins"], sort=False, dropna=False)
    ordered_groups: list[pd.DataFrame] = []
    for _, group in grouped:
        if len(group) == 2:
            teams = group["team"].tolist()
            if _head_to_head_tiebreak(teams[0], teams[1], state) > 0:
                group = group.iloc[::-1].reset_index(drop=True)
        elif len(group) > 2:
            metrics = _mini_league_metrics(group["team"].tolist(), state)
            group = group.assign(
                mini_wins=group["team"].map(lambda team: metrics[team][0]),
                mini_win_rate=group["team"].map(lambda team: metrics[team][1]),
            ).sort_values(
                by=["mini_wins", "mini_win_rate", "team"],
                ascending=[False, False, True],
                kind="mergesort",
            ).drop(columns=["mini_wins", "mini_win_rate"]).reset_index(drop=True)
        ordered_groups.append(group)

    return pd.concat(ordered_groups, ignore_index=True).set_index("team")


def _team_1_bats_first_probability(features: pd.DataFrame) -> float:
    row = features.iloc[0]
    team_1_style = float(row["team_1_batting_first_win_rate"]) - float(row["team_1_chasing_win_rate"])
    team_2_style = float(row["team_2_batting_first_win_rate"]) - float(row["team_2_chasing_win_rate"])
    style_edge = np.clip(team_1_style - team_2_style, -1.0, 1.0)
    venue_bias = float(row["venue_batting_first_win_rate"]) - 0.5
    probability = 0.5 + 0.22 * style_edge + 0.08 * venue_bias
    return float(np.clip(probability, 0.15, 0.85))


def _simulate_match(model, match_row: pd.Series, state: dict, rng: np.random.Generator) -> tuple[str, float, float, float]:
    features = make_match_features(match_row, state)
    win_probability = float(model.predict_proba(features)[:, 1][0])
    winner = match_row["team_1"] if rng.random() < win_probability else match_row["team_2"]
    team_1_batted_first = rng.random() < _team_1_bats_first_probability(features)
    current_batted_first = "team_1" if team_1_batted_first else "team_2"
    team_1_score, team_2_score = _estimate_scores(
        features,
        state,
        winner,
        match_row["team_1"],
        match_row["team_2"],
        win_probability,
        rng,
    )
    state["current_venue"] = match_row["venue"]
    state["current_season"] = int(pd.to_datetime(match_row["date"]).year) if "date" in match_row and not pd.isna(match_row["date"]) else 2026
    state["current_team_1_score"] = team_1_score
    state["current_team_2_score"] = team_2_score
    state["current_batted_first"] = current_batted_first
    update_state_after_match(match_row["team_1"], match_row["team_2"], winner, state)
    state["current_venue"] = None
    state["current_season"] = None
    state["current_team_1_score"] = None
    state["current_team_2_score"] = None
    state["current_batted_first"] = None
    return winner, win_probability, team_1_score, team_2_score


def _playoff_match(model, team_1: str, team_2: str, state: dict, rng: np.random.Generator, venue: str) -> str:
    # Playoff rows intentionally omit date, so make_match_features falls back to
    # the simulation season when building season-window features.
    row = pd.Series(
        {
            "team_1": team_1,
            "team_2": team_2,
            "venue": venue,
        }
    )
    winner, _, _, _ = _simulate_match(model, row, state, rng)
    return winner


def simulate_tournament(model, fixtures: pd.DataFrame, teams: list[str], initial_state: dict, n_simulations: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    champions = Counter()
    latest_table = None

    for simulation_index in range(n_simulations):
        state = prepare_simulation_state(initial_state)
        table = _initialize_table(teams)

        for match in fixtures.itertuples(index=False):
            match_row = pd.Series(match._asdict())
            winner, _, team_1_score, team_2_score = _simulate_match(model, match_row, state, rng)
            _record_result(table, match.team_1, match.team_2, winner, team_1_score, team_2_score)

        sorted_table = _sort_table(table, state)
        top_four = sorted_table.head(4).index.tolist()
        q1_winner = _playoff_match(model, top_four[0], top_four[1], state, rng, PLAYOFF_VENUES["qualifier_1"])
        q1_loser = top_four[1] if q1_winner == top_four[0] else top_four[0]
        eliminator_winner = _playoff_match(model, top_four[2], top_four[3], state, rng, PLAYOFF_VENUES["eliminator"])
        q2_winner = _playoff_match(model, q1_loser, eliminator_winner, state, rng, PLAYOFF_VENUES["qualifier_2"])
        champion = _playoff_match(model, q1_winner, q2_winner, state, rng, PLAYOFF_VENUES["final"])
        champions[champion] += 1

        if simulation_index == n_simulations - 1:
            latest_table = sorted_table.reset_index()

    odds = pd.DataFrame(
        [{"team": team, "title_probability": champions[team] / n_simulations} for team in teams]
    ).sort_values("title_probability", ascending=False)

    return odds.reset_index(drop=True), latest_table
