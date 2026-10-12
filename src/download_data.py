"""Download historical Premier League match data from football-data.co.uk.

Each season is a CSV at:
    https://www.football-data.co.uk/mmz4281/<SEASON>/E0.csv
where <SEASON> is e.g. "2324" for the 2023-24 season and E0 = Premier League.

The files include full-time results plus per-match stats (shots, shots on
target, corners, fouls, cards) and bookmaker odds - good raw material for an
ML classifier.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Season codes: "2021" = 2020-21 ... "2627" = 2026-27 (current season).
SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526", "2627"]


def download_season(season: str) -> Path | None:
    url = BASE_URL.format(season=season)
    dest = RAW_DIR / f"E0_{season}.csv"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! {season}: failed ({exc})")
        return None
    dest.write_bytes(resp.content)
    kb = len(resp.content) / 1024
    print(f"  + {season}: {kb:,.0f} KB -> {dest.name}")
    return dest


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(SEASONS)} seasons into {RAW_DIR}")
    ok = 0
    for season in SEASONS:
        if download_season(season) is not None:
            ok += 1
        time.sleep(0.5)  # be polite to the server
    print(f"Done: {ok}/{len(SEASONS)} seasons downloaded.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
