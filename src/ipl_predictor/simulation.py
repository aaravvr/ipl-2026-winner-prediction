from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .features import make_match_features, prepare_simulation_state, update_state_after_match


def _initialize_table(teams: list[str]) -> pd.DataFrame:
    table = pd.DataFrame({"team": teams})
    table["played"] = 0
    table["wins"] = 0
    table["losses"] = 0
    table["points"] = 0
    return table.set_index("team")


def _record_result(table: pd.DataFrame, team_1: str, team_2: str, winner: str) -> None:
    loser = team_2 if winner == team_1 else team_1
    table.loc[team_1, "played"] += 1
    table.loc[team_2, "played"] += 1
    table.loc[winner, "wins"] += 1
    table.loc[loser, "losses"] += 1
    table.loc[winner, "points"] += 2


def _sort_table(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    sorted_table = table.copy()
    sorted_table["tie_breaker"] = rng.random(len(sorted_table))
    sorted_table = sorted_table.sort_values(
        by=["points", "wins", "tie_breaker"],
        ascending=[False, False, False],
    )
    return sorted_table.drop(columns=["tie_breaker"])


def _simulate_match(model, match_row: pd.Series, state: dict, rng: np.random.Generator) -> tuple[str, float]:
    features = make_match_features(match_row, state)
    win_probability = float(model.predict_proba(features)[:, 1][0])
    winner = match_row["team_1"] if rng.random() < win_probability else match_row["team_2"]
    state["current_venue"] = match_row["venue"]
    state["current_team_1_score"] = float(match_row.get("team_1_score", np.nan)) if pd.notna(match_row.get("team_1_score", np.nan)) else None
    state["current_team_2_score"] = float(match_row.get("team_2_score", np.nan)) if pd.notna(match_row.get("team_2_score", np.nan)) else None
    update_state_after_match(match_row["team_1"], match_row["team_2"], winner, state)
    state["current_venue"] = None
    state["current_team_1_score"] = None
    state["current_team_2_score"] = None
    return winner, win_probability


def _playoff_match(model, team_1: str, team_2: str, state: dict, rng: np.random.Generator, venue: str) -> str:
    row = pd.Series(
        {
            "team_1": team_1,
            "team_2": team_2,
            "venue": venue,
        }
    )
    winner, _ = _simulate_match(model, row, state, rng)
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
            winner, _ = _simulate_match(model, match_row, state, rng)
            _record_result(table, match.team_1, match.team_2, winner)

        sorted_table = _sort_table(table, rng)
        top_four = sorted_table.head(4).index.tolist()
        q1_winner = _playoff_match(model, top_four[0], top_four[1], state, rng, "Playoff Venue")
        q1_loser = top_four[1] if q1_winner == top_four[0] else top_four[0]
        eliminator_winner = _playoff_match(model, top_four[2], top_four[3], state, rng, "Playoff Venue")
        q2_winner = _playoff_match(model, q1_loser, eliminator_winner, state, rng, "Playoff Venue")
        champion = _playoff_match(model, q1_winner, q2_winner, state, rng, "Final Venue")
        champions[champion] += 1

        if simulation_index == n_simulations - 1:
            latest_table = sorted_table.reset_index()

    odds = pd.DataFrame(
        [{"team": team, "title_probability": champions[team] / n_simulations} for team in teams]
    ).sort_values("title_probability", ascending=False)

    return odds.reset_index(drop=True), latest_table
