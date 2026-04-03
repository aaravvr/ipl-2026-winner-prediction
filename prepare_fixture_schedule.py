from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ipl_predictor.config import FIXTURES_2026_PATH


TIME_PATTERN = re.compile(r"^\d{1,2}:\d{2}\s+[AP]M$")
DAY_PATTERN = re.compile(r"^[A-Za-z]{3,4}$")
DATE_PATTERN = re.compile(r"^\d{2}-[A-Z]{3}-\d{2}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the IPL 2026 fixture schedule from the official PDF.")
    parser.add_argument("--source", type=Path, required=True, help="Path to the IPL 2026 schedule PDF")
    parser.add_argument("--output", type=Path, default=FIXTURES_2026_PATH, help="Output CSV path")
    return parser.parse_args()


def extract_structured_lines(path: Path) -> list[dict[str, list[str]]]:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        lines = [line.strip() for line in (page.extract_text() or "").splitlines() if line.strip()]
        start_index = next((idx for idx, value in enumerate(lines) if TIME_PATTERN.match(value)), None)
        if start_index is None:
            raise ValueError("Could not detect start times in the PDF")

        block_size = 0
        while start_index + block_size < len(lines) and TIME_PATTERN.match(lines[start_index + block_size]):
            block_size += 1
        if block_size == 0:
            raise ValueError("Could not detect fixture rows in the PDF")

        starts = lines[start_index : start_index + block_size]

        day_index = start_index + block_size
        days = lines[day_index : day_index + block_size]
        date_index = day_index + block_size
        dates = lines[date_index : date_index + block_size]
        match_number_index = date_index + block_size
        match_numbers = lines[match_number_index : match_number_index + block_size]

        away_start = start_index - block_size
        venues_start = away_start - block_size
        home_start = venues_start - block_size
        if home_start < 0:
            raise ValueError("Unexpected PDF layout while extracting teams and venues")

        home_teams = lines[home_start : home_start + block_size]
        venues = lines[venues_start : venues_start + block_size]
        away_teams = lines[away_start : away_start + block_size]

        pages.append(
            {
                "match_numbers": match_numbers,
                "dates": dates,
                "days": days,
                "starts": starts,
                "home_teams": home_teams,
                "away_teams": away_teams,
                "venues": venues,
            }
        )
    return pages


def build_fixtures(path: Path) -> pd.DataFrame:
    pages = extract_structured_lines(path)
    rows: list[dict[str, str | int]] = []
    for page in pages:
        for match_id, date, day, start, away_team, home_team, venue in zip(
            page["match_numbers"],
            page["dates"],
            page["days"],
            page["starts"],
            page["home_teams"],
            page["away_teams"],
            page["venues"],
            strict=True,
        ):
            rows.append(
                {
                    "match_id": int(match_id),
                    "date": pd.to_datetime(date, format="%d-%b-%y").date().isoformat(),
                    "day": "Sat" if day == "Satt" else day,
                    "start_time_local": start,
                    "team_1": home_team,
                    "team_2": away_team,
                    "venue": venue,
                }
            )

    fixtures = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)
    return fixtures


def main() -> None:
    args = parse_args()
    fixtures = build_fixtures(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(args.output, index=False)
    print(f"Saved {len(fixtures)} fixtures to: {args.output}")


if __name__ == "__main__":
    main()
