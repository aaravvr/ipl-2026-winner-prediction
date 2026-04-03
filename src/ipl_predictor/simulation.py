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


def _estimate_scores(features: pd.DataFrame, winner: str, team_1: str, team_2: str, rng: np.random.Generator) -> tuple[float, float]:
    row = features.iloc[0]
    venue_base = float(row["venue_avg_innings_score"])
    league_score_baseline = 157.0
    league_batting_strength = 33.0
    league_bowling_strength = 17.0

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
    margin_bias = abs(float(row["team_1_expected_score"]) - 0.5)
    margin = max(4.0, 8.0 + 35.0 * margin_bias + rng.normal(0.0, 6.0))
    team_1_score = min(260.0, max(110.0, team_1_expected + rng.normal(0.0, 9.0)))
    team_2_score = min(260.0, max(110.0, team_2_expected + rng.normal(0.0, 9.0)))
    if winner == team_1 and team_1_score <= team_2_score:
        team_1_score = team_2_score + margin
    elif winner == team_2 and team_2_score <= team_1_score:
        team_2_score = team_1_score + margin
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


def _sort_table(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sorted_table = _refresh_nrr(table)
    sorted_table["tie_breaker"] = rng.random(len(sorted_table))
    sorted_table = sorted_table.sort_values(
        by=["points", "nrr", "wins", "tie_breaker"],
        ascending=[False, False, False, False],
    )
    return sorted_table.drop(columns=["tie_breaker"])


def _simulate_match(model, match_row: pd.Series, state: dict, rng: np.random.Generator) -> tuple[str, float, float, float]:
    features = make_match_features(match_row, state)
    win_probability = float(model.predict_proba(features)[:, 1][0])
    winner = match_row["team_1"] if rng.random() < win_probability else match_row["team_2"]
    team_1_score, team_2_score = _estimate_scores(features, winner, match_row["team_1"], match_row["team_2"], rng)
    state["current_venue"] = match_row["venue"]
    state["current_team_1_score"] = team_1_score
    state["current_team_2_score"] = team_2_score
    update_state_after_match(match_row["team_1"], match_row["team_2"], winner, state)
    state["current_venue"] = None
    state["current_team_1_score"] = None
    state["current_team_2_score"] = None
    return winner, win_probability, team_1_score, team_2_score


def _playoff_match(model, team_1: str, team_2: str, state: dict, rng: np.random.Generator, venue: str) -> str:
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

        sorted_table = _sort_table(table, rng)
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
