from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipl_predictor.config import (
    FIXTURES_2026_PATH,
    HISTORICAL_MATCHES_PATH,
    LAST_SIM_TABLE_PATH,
    MATCH_PLAYER_STRENGTHS_PATH,
    MODEL_PATH,
    TEAM_PRIORS_2026_PATH,
    TEAM_PLAYER_STRENGTHS_PATH,
    TEAMS_2026_PATH,
    TITLE_ODDS_PATH,
)
from ipl_predictor.data import (
    load_fixtures,
    load_historical_matches,
    load_optional_match_player_strengths,
    load_optional_team_priors,
    load_optional_team_player_strengths,
    load_teams,
)
from ipl_predictor.features import initialize_state
from ipl_predictor.model import load_model
from ipl_predictor.simulation import simulate_tournament


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the IPL 2026 season.")
    parser.add_argument("--n-simulations", type=int, default=5000, help="Number of tournament simulations")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = load_model(MODEL_PATH)
    historical_matches = load_historical_matches(HISTORICAL_MATCHES_PATH)
    match_player_strengths = load_optional_match_player_strengths(MATCH_PLAYER_STRENGTHS_PATH)
    if match_player_strengths is not None and "match_id" in historical_matches.columns:
        historical_matches = historical_matches.merge(match_player_strengths, on="match_id", how="left")
    fixtures = load_fixtures(FIXTURES_2026_PATH)
    teams = load_teams(TEAMS_2026_PATH)
    state = initialize_state(historical_matches)
    latest_team_strengths = load_optional_team_player_strengths(TEAM_PLAYER_STRENGTHS_PATH)
    for team, strengths in latest_team_strengths.items():
        state["player_team_strengths"][team] = strengths
    latest_team_priors = load_optional_team_priors(TEAM_PRIORS_2026_PATH)
    for team, priors in latest_team_priors.items():
        state["team_priors"][team] = priors

    odds, latest_table = simulate_tournament(
        model=model,
        fixtures=fixtures,
        teams=teams,
        initial_state=state,
        n_simulations=args.n_simulations,
    )

    TITLE_ODDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    odds.to_csv(TITLE_ODDS_PATH, index=False)
    latest_table.to_csv(LAST_SIM_TABLE_PATH, index=False)

    print("Simulation complete.")
    print(odds.to_string(index=False))
    print(f"Saved title odds to: {TITLE_ODDS_PATH}")


if __name__ == "__main__":
    main()
