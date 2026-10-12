"""Load the downloaded Premier League CSVs and explore them.

Prints a first look at the data: how many matches, what columns are useful,
missing-data check, and some sanity stats (home advantage, outcome balance).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# The columns we actually care about for a predictor. football-data.co.uk has
# ~100+ columns (mostly odds from many bookmakers); these are the core ones.
CORE_COLS = {
    "Date": "match date",
    "HomeTeam": "home team",
    "AwayTeam": "away team",
    "FTHG": "full-time home goals",
    "FTAG": "full-time away goals",
    "FTR": "full-time result (H/D/A)",
    "HS": "home shots",
    "AS": "away shots",
    "HST": "home shots on target",
    "AST": "away shots on target",
    "HC": "home corners",
    "AC": "away corners",
    "HF": "home fouls",
    "AF": "away fouls",
    "HY": "home yellow cards",
    "AY": "away yellow cards",
    "HR": "home red cards",
    "AR": "away red cards",
    # Market odds (Bet365) - a strong baseline/feature
    "B365H": "bet365 home-win odds",
    "B365D": "bet365 draw odds",
    "B365A": "bet365 away-win odds",
}


def load_all() -> pd.DataFrame:
    frames = []
    for csv in sorted(RAW_DIR.glob("E0_*.csv")):
        df = pd.read_csv(csv, encoding="latin-1")
        df["season_file"] = csv.stem.replace("E0_", "")
        frames.append(df)
        print(f"  loaded {csv.name}: {df.shape[0]} rows x {df.shape[1]} cols")
    if not frames:
        raise SystemExit("No CSVs found. Run download_data.py first.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print("=== Loading seasons ===")
    df = load_all()
    print(f"\nCombined: {df.shape[0]} matches across {df['season_file'].nunique()} seasons")

    present = [c for c in CORE_COLS if c in df.columns]
    missing = [c for c in CORE_COLS if c not in df.columns]
    print(f"\n=== Core columns present ({len(present)}/{len(CORE_COLS)}) ===")
    for c in present:
        print(f"  {c:8s} - {CORE_COLS[c]}")
    if missing:
        print(f"  (not found: {', '.join(missing)})")

    core = df[present].copy()

    print("\n=== Missing values in core columns ===")
    na = core.isna().sum()
    na = na[na > 0]
    print(na.to_string() if not na.empty else "  none - clean!")

    print("\n=== Outcome balance (full-time result) ===")
    if "FTR" in core:
        counts = core["FTR"].value_counts()
        total = counts.sum()
        label = {"H": "Home win", "D": "Draw", "A": "Away win"}
        for k in ["H", "D", "A"]:
            if k in counts:
                print(f"  {label[k]:9s}: {counts[k]:4d}  ({counts[k] / total:5.1%})")

    print("\n=== Home advantage (avg goals) ===")
    if {"FTHG", "FTAG"}.issubset(core.columns):
        print(f"  avg home goals: {core['FTHG'].mean():.2f}")
        print(f"  avg away goals: {core['FTAG'].mean():.2f}")

    print("\n=== Sample of recent matches ===")
    show = [c for c in ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"] if c in core]
    print(core[show].tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
