"""Build pre-match features for every match from the raw season CSVs.

The golden rule: every feature must be knowable BEFORE kickoff. We therefore
derive each team's features from their PREVIOUS matches only, using rolling
windows that are shifted so the current match is never included.

Output: data/processed/matches_features.csv — one row per match with the
target (FTR) plus numeric pre-match features for the home and away teams.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from elo import add_elo, add_xg_elo

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "matches_features.csv"

ROLL = 5  # rolling window: last N matches of form

# Understat uses full club names; football-data.co.uk uses short ones.
# Map Understat -> football-data so we can join on team names.
XG_TEAM_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom",
}

# Raw columns we read from each CSV.
READ_COLS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST", "B365H", "B365D", "B365A",
]


def load_matches() -> pd.DataFrame:
    frames = []
    for csv in sorted(RAW_DIR.glob("E0_*.csv")):
        raw = pd.read_csv(csv, encoding="latin-1")
        cols = [c for c in READ_COLS if c in raw.columns]
        df = raw[cols].copy()
        df["season"] = csv.stem.replace("E0_", "")
        frames.append(df)
    matches = pd.concat(frames, ignore_index=True)
    matches["Date"] = pd.to_datetime(matches["Date"], dayfirst=True, format="mixed")
    matches = matches.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTR"])
    matches = matches.sort_values("Date").reset_index(drop=True)
    matches["match_id"] = matches.index
    return matches


# Opta per-match stats file (team names -> football-data convention).
POSS_FILE = RAW_DIR.parent.parent / "_all_events_stats.xlsx"
POSS_TEAM_MAP = {"Man Utd": "Man United", "Spurs": "Tottenham", "Sheff Utd": "Sheffield United"}


def _season_code(label: str) -> str:
    """'2020-21' -> '2021' to match football-data season codes."""
    start = int(label[:4])
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def load_possession() -> pd.DataFrame:
    """Per-match home/away Opta stats: possession %, big chances created/missed."""
    cols = ["possessionPercentage", "bigChanceCreated", "bigChanceMissed"]
    df = pd.read_excel(POSS_FILE, usecols=["season", "matchId", "team", "venue"] + cols)
    df["team"] = df["team"].replace(POSS_TEAM_MAP)
    df["season"] = df["season"].map(_season_code)
    # Big chances left blank mean none recorded in that match.
    df[["bigChanceCreated", "bigChanceMissed"]] = df[["bigChanceCreated", "bigChanceMissed"]].fillna(0)
    ren = {"possessionPercentage": "poss", "bigChanceCreated": "bcc", "bigChanceMissed": "bcm"}
    home = df[df["venue"] == "Home"].rename(columns={"team": "HomeTeam",
            **{k: f"home_{v}" for k, v in ren.items()}})
    away = df[df["venue"] == "Away"].rename(columns={"team": "AwayTeam",
            **{k: f"away_{v}" for k, v in ren.items()}})
    merged = home[["season", "matchId", "HomeTeam", "home_poss", "home_bcc", "home_bcm"]].merge(
        away[["matchId", "AwayTeam", "away_poss", "away_bcc", "away_bcm"]], on="matchId")
    return merged.drop(columns="matchId")


def merge_possession(matches: pd.DataFrame) -> pd.DataFrame:
    poss = load_possession()
    merged = matches.merge(poss, on=["season", "HomeTeam", "AwayTeam"], how="left")
    missing = merged["home_poss"].isna().sum()
    if missing:
        print(f"  note: {missing}/{len(merged)} matches had no Opta match")
    return merged


def load_xg() -> pd.DataFrame:
    """Load Understat match xG and align team names to football-data."""
    path = RAW_DIR / "understat_xg.csv"
    xg = pd.read_csv(path, header=None,
                     names=["season", "HomeTeam", "AwayTeam", "home_xg", "away_xg"],
                     dtype={"season": str})
    for col in ("HomeTeam", "AwayTeam"):
        xg[col] = xg[col].replace(XG_TEAM_MAP)
    return xg


def merge_xg(matches: pd.DataFrame) -> pd.DataFrame:
    """Attach home_xg / away_xg to each match; report any that don't join."""
    xg = load_xg()
    merged = matches.merge(xg, on=["season", "HomeTeam", "AwayTeam"], how="left")
    missing = merged["home_xg"].isna().sum()
    if missing:
        print(f"  note: {missing}/{len(merged)} matches had no xG match (name/fixture gaps)")
    return merged


def to_team_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Reshape to one row per team per match (home + away rows stacked)."""
    pts = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}
    home = pd.DataFrame({
        "match_id": matches["match_id"],
        "date": matches["Date"],
        "team": matches["HomeTeam"],
        "venue": "H",
        "gf": matches["FTHG"], "ga": matches["FTAG"],
        "sf": matches.get("HS"), "sa": matches.get("AS"),
        "stf": matches.get("HST"),
        "xgf": matches.get("home_xg"), "xga": matches.get("away_xg"),
        "poss": matches.get("home_poss"),
        "bcc": matches.get("home_bcc"), "bcca": matches.get("away_bcc"),
        "pts": matches["FTR"].map(lambda r: pts[r][0]),
    })
    away = pd.DataFrame({
        "match_id": matches["match_id"],
        "date": matches["Date"],
        "team": matches["AwayTeam"],
        "venue": "A",
        "gf": matches["FTAG"], "ga": matches["FTHG"],
        "sf": matches.get("AS"), "sa": matches.get("HS"),
        "stf": matches.get("AST"),
        "xgf": matches.get("away_xg"), "xga": matches.get("home_xg"),
        "poss": matches.get("away_poss"),
        "bcc": matches.get("away_bcc"), "bcca": matches.get("home_bcc"),
        "pts": matches["FTR"].map(lambda r: pts[r][1]),
    })
    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values(["team", "date"]).reset_index(drop=True)


def add_rolling(long: pd.DataFrame) -> pd.DataFrame:
    """Rolling pre-match features per team (shifted to exclude current game)."""
    g = long.groupby("team", group_keys=False)

    # Rest days since the team's previous match.
    long["rest_days"] = g["date"].diff().dt.days

    roll_cols = ["gf", "ga", "sf", "sa", "stf", "xgf", "xga", "poss", "bcc", "bcca", "pts"]
    for col in roll_cols:
        long[f"roll_{col}"] = (
            g[col].apply(lambda s: s.shift(1).rolling(ROLL, min_periods=1).mean())
        )

    # Venue-specific form: a team's recent record at HOME (or AWAY) only, so a
    # home side is judged on its home form and the visitor on its away form.
    gv = long.groupby(["team", "venue"], group_keys=False)
    for col in ["gf", "ga", "pts"]:
        long[f"roll_v_{col}"] = (
            gv[col].apply(lambda s: s.shift(1).rolling(ROLL, min_periods=1).mean())
        )

    # Form = points per game over the window; games played so far (experience).
    long["form"] = long["roll_pts"]
    long["played"] = g.cumcount()  # matches already played this history
    return long


def add_h2h(matches: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """Head-to-head: the home team's points-per-game in the last k meetings
    with this opponent (either venue), computed pre-match."""
    from collections import deque, defaultdict

    df = matches.sort_values("Date").reset_index(drop=True)
    hist: dict = defaultdict(deque)
    ppg, ngames = [], []
    for row in df.itertuples(index=False):
        h, a = row.HomeTeam, row.AwayTeam
        dq = hist[frozenset((h, a))]
        pts, cnt = 0, 0
        for past_home, res in dq:
            if res == "D":
                p = 1
            elif past_home == h:          # h hosted that meeting
                p = 3 if res == "H" else 0
            else:                          # h visited that meeting
                p = 3 if res == "A" else 0
            pts += p
            cnt += 1
        ppg.append(pts / cnt if cnt else np.nan)
        ngames.append(cnt)
        dq.append((h, row.FTR))           # record after computing (pre-match)
        while len(dq) > k:
            dq.popleft()
    df["h2h_home_ppg"] = ppg
    df["h2h_n"] = ngames
    return df


def build() -> pd.DataFrame:
    matches = load_matches()
    matches = merge_xg(matches)
    matches = merge_possession(matches)
    matches, _elo_ratings = add_elo(matches)
    matches, _xgelo_ratings = add_xg_elo(matches)
    matches = add_h2h(matches)
    long = add_rolling(to_team_long(matches))

    feat_cols = ["roll_gf", "roll_ga", "roll_sf", "roll_sa", "roll_stf",
                 "roll_xgf", "roll_xga", "roll_poss", "roll_bcc", "roll_bcca",
                 "roll_v_gf", "roll_v_ga", "roll_v_pts", "form", "rest_days", "played"]
    keep = ["match_id"] + feat_cols
    home_feats = long[long["venue"] == "H"][keep].add_prefix("home_").rename(
        columns={"home_match_id": "match_id"})
    away_feats = long[long["venue"] == "A"][keep].add_prefix("away_").rename(
        columns={"away_match_id": "match_id"})

    df = matches.merge(home_feats, on="match_id").merge(away_feats, on="match_id")

    # Difference features (home minus away) often carry the signal cleanly.
    for c in feat_cols:
        df[f"diff_{c}"] = df[f"home_{c}"] - df[f"away_{c}"]

    # Market signal: implied probabilities from Bet365 odds (overround removed).
    if {"B365H", "B365D", "B365A"}.issubset(df.columns):
        inv = df[["B365H", "B365D", "B365A"]].rdiv(1.0)
        s = inv.sum(axis=1)
        df["imp_home"] = inv["B365H"] / s
        df["imp_draw"] = inv["B365D"] / s
        df["imp_away"] = inv["B365A"] / s

    return df


def main() -> None:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} matches with features -> {OUT.relative_to(ROOT)}")
    print(f"Columns: {df.shape[1]}")
    feat_preview = [c for c in df.columns if c.startswith(("diff_", "imp_"))]
    print("Model feature columns:", ", ".join(feat_preview))


if __name__ == "__main__":
    main()
