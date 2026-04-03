from __future__ import annotations

import argparse
import re
from html import unescape
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from src.ipl_predictor.config import TEAM_OVERVIEW_2026_PATH, TEAM_SQUADS_2026_PATH


TEAM_URLS = {
    "Chennai Super Kings": "https://www.iplt20.com/teams/chennai-super-kings",
    "Delhi Capitals": "https://www.iplt20.com/teams/delhi-capitals",
    "Gujarat Titans": "https://www.iplt20.com/teams/gujarat-titans",
    "Kolkata Knight Riders": "https://www.iplt20.com/teams/kolkata-knight-riders",
    "Lucknow Super Giants": "https://www.iplt20.com/teams/lucknow-super-giants",
    "Mumbai Indians": "https://www.iplt20.com/teams/mumbai-indians",
    "Punjab Kings": "https://www.iplt20.com/teams/punjab-kings",
    "Rajasthan Royals": "https://www.iplt20.com/teams/rajasthan-royals",
    "Royal Challengers Bengaluru": "https://www.iplt20.com/teams/royal-challengers-bengaluru",
    "Sunrisers Hyderabad": "https://www.iplt20.com/teams/sunrisers-hyderabad",
}

ROLE_SECTIONS = ["Batters", "All Rounders", "Bowlers"]
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract IPL 2026 squads from official IPLT20 team pages.")
    parser.add_argument("--overview-output", type=Path, default=TEAM_OVERVIEW_2026_PATH)
    parser.add_argument("--squads-output", type=Path, default=TEAM_SQUADS_2026_PATH)
    return parser.parse_args()


def fetch_html(url: str) -> str:
    request = Request(url, headers=HEADERS)
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def extract_field(html: str, label: str) -> str:
    pattern = rf"<p><span>{re.escape(label)}</span>\s*<b>-</b>\s*(.*?)</p>"
    match = re.search(pattern, html, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find {label!r} in team page")
    value = re.sub(r"<.*?>", "", match.group(1))
    return unescape(value).strip()


def extract_players(html: str, role_group: str) -> list[str]:
    start_match = re.search(
        rf'<h2 class="what-r-u-text mb-0">\s*{re.escape(role_group)}\s*</h2>',
        html,
        flags=re.DOTALL,
    )
    if not start_match:
        return []

    next_positions = []
    for other_role in ROLE_SECTIONS:
        if other_role == role_group:
            continue
        other_match = re.search(
            rf'<h2 class="what-r-u-text mb-0">\s*{re.escape(other_role)}\s*</h2>',
            html[start_match.end() :],
            flags=re.DOTALL,
        )
        if other_match:
            next_positions.append(start_match.end() + other_match.start())

    end_index = min(next_positions) if next_positions else len(html)
    block = html[start_match.end() : end_index]
    players = re.findall(r'data-player_name="([^"]+)"', block)
    cleaned = []
    seen = set()
    for player in players:
        name = unescape(player).strip()
        if name and name not in seen:
            cleaned.append(name)
            seen.add(name)
    return cleaned


def main() -> None:
    args = parse_args()

    overview_rows: list[dict[str, str]] = []
    squad_rows: list[dict[str, str]] = []

    for team, url in TEAM_URLS.items():
        html = fetch_html(url)
        captain = extract_field(html, "Captain")
        coach = extract_field(html, "Coach")
        venue = extract_field(html, "Venue")

        overview_rows.append(
            {
                "team": team,
                "captain": captain,
                "coach": coach,
                "venue": venue,
                "source_url": url,
            }
        )

        for role_group in ROLE_SECTIONS:
            for player in extract_players(html, role_group):
                squad_rows.append(
                    {
                        "team": team,
                        "role_group": role_group,
                        "player": player,
                        "source_url": url,
                    }
                )

    overview = pd.DataFrame(overview_rows).sort_values("team").reset_index(drop=True)
    squads = pd.DataFrame(squad_rows).sort_values(["team", "role_group", "player"]).reset_index(drop=True)

    args.overview_output.parent.mkdir(parents=True, exist_ok=True)
    args.squads_output.parent.mkdir(parents=True, exist_ok=True)
    overview.to_csv(args.overview_output, index=False)
    squads.to_csv(args.squads_output, index=False)

    print(f"Saved team overview to: {args.overview_output}")
    print(f"Saved team squads to: {args.squads_output}")


if __name__ == "__main__":
    main()
