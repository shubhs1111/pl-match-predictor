"""Assemble a single ui_data.json for the terminal-style dashboard.

Everything here is real: probabilities from predict.py's output, possession from
the Opta file, recent W/D/L form computed from results, the opponent-adjusted
xG-strength edge, and the plain-language reasons from reasons.json. No invented
metrics (no Monte Carlo, Poisson expectancy, sigma scores, or dominance zones).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from elo import add_xg_elo
from features import load_matches, merge_xg

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "data" / "processed" / "predictions_no_odds.csv"
REASONS = ROOT / "reasons.json"
UPCOMING = ROOT / "data" / "raw" / "upcoming_fixtures.csv"
POSS_FILE = ROOT / "_all_events_stats.xlsx"
OUT = ROOT / "ui_data.json"

CODES = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Chelsea": "CHE", "Coventry": "COV", "Crystal Palace": "CRY",
    "Everton": "EVE", "Fulham": "FUL", "Hull": "HUL", "Ipswich": "IPS", "Leeds": "LEE",
    "Liverpool": "LIV", "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW",
    "Nott'm Forest": "NFO", "Sunderland": "SUN", "Tottenham": "TOT",
}
VENUES = {
    "Everton": "Goodison Park, Liverpool", "Fulham": "Craven Cottage, London",
    "Brentford": "Gtech Community Stadium", "Man City": "Etihad Stadium, Manchester",
    "Newcastle": "St. James' Park, Newcastle", "Brighton": "Amex Stadium, Brighton",
    "Bournemouth": "Vitality Stadium, Bournemouth", "Leeds": "Elland Road, Leeds",
    "Nott'm Forest": "City Ground, Nottingham", "Tottenham": "Tottenham Hotspur Stadium",
    "Ipswich": "Portman Road, Ipswich", "Aston Villa": "Villa Park, Birmingham",
    "Coventry": "Coventry Building Society Arena", "Arsenal": "Emirates Stadium, London",
    "Chelsea": "Stamford Bridge, London", "Liverpool": "Anfield, Liverpool",
    "Man United": "Old Trafford, Manchester", "Crystal Palace": "Selhurst Park, London",
    "Hull": "MKM Stadium, Hull", "Sunderland": "Stadium of Light, Sunderland",
}
POSS_MAP = {"Man Utd": "Man United", "Spurs": "Tottenham", "Sheff Utd": "Sheffield United"}
MD_RANGES = {1: "17 – 19 OCT 2026", 2: "23 – 25 OCT 2026", 3: "31 OCT – 02 NOV 2026"}


def recent_form(matches: pd.DataFrame) -> dict:
    """Last five results (newest last) per team, as W/D/L from their view."""
    m = matches.sort_values("Date")
    seq: dict[str, list] = {}
    for row in m.itertuples(index=False):
        h, a = row.HomeTeam, row.AwayTeam
        if row.FTR == "D":
            rh = ra = "D"
        elif row.FTR == "H":
            rh, ra = "W", "L"
        else:
            rh, ra = "L", "W"
        seq.setdefault(h, []).append(rh)
        seq.setdefault(a, []).append(ra)
    return {t: v[-5:] for t, v in seq.items()}


def current_possession() -> dict:
    df = pd.read_excel(POSS_FILE, usecols=["season", "team", "possessionPercentage"])
    df = df[df["season"] == "2026-27"].copy()
    df["team"] = df["team"].replace(POSS_MAP)
    return df.groupby("team")["possessionPercentage"].mean().round(1).to_dict()


def confidence(ph, pd_, pa) -> str:
    top = max(ph, pd_, pa)
    if top >= 0.50:
        return "HIGH CONVICTION"
    if pa > ph:
        return "AWAY FAVORITE"
    return "SOLID LEAN" if top >= 0.42 else "CLOSE CALL"


def main() -> None:
    preds = pd.read_csv(PRED)
    reasons = {r["match"]: r for r in json.loads(REASONS.read_text(encoding="utf-8"))}

    matches = merge_xg(load_matches())
    matches, xgelo = add_xg_elo(matches)
    form = recent_form(matches)
    poss = current_possession()

    matchdays: dict[str, dict] = {}
    for md in sorted(preds["matchday"].unique()):
        rows = preds[preds["matchday"] == md]
        out_matches = []
        for i, r in enumerate(rows.itertuples(index=False), start=1):
            home, away = r.match.split(" vs ")
            ph, pdr, pa = float(r._3), float(r._4), float(r._5)  # P(home),P(draw),P(away)
            xh, xa = xgelo.get(home, 1500), xgelo.get(away, 1500)
            xg_team = home if xh >= xa else away
            rinfo = reasons.get(r.match, {"pro": [], "con": []})
            date = pd.to_datetime(r.date)
            out_matches.append({
                "id": f"m{md}-{i:02d}", "home": home, "away": away,
                "homeCode": CODES.get(home, home[:3].upper()),
                "awayCode": CODES.get(away, away[:3].upper()),
                "date": date.strftime("%a %d %b").upper(),
                "venue": VENUES.get(home, ""),
                "probs": {"home": round(ph, 3), "draw": round(pdr, 3), "away": round(pa, 3)},
                "lean": r.prediction, "leanConfidence": confidence(ph, pdr, pa),
                "homePoss": poss.get(home), "awayPoss": poss.get(away),
                "homeForm": form.get(home, []), "awayForm": form.get(away, []),
                "xgEdgeTeam": xg_team, "xgEdgeCode": CODES.get(xg_team, ""),
                "pro": rinfo["pro"], "con": rinfo["con"],
            })
        matchdays[f"MD{md}"] = {"dateRange": MD_RANGES.get(int(md), ""), "matches": out_matches}

    league_poss = sorted(
        [{"code": CODES.get(t, t[:3].upper()), "team": t, "val": v} for t, v in poss.items()],
        key=lambda d: -d["val"])

    data = {
        "meta": {
            "method": "Gradient-boosting classifier · no betting odds · calibrated",
            "accuracy": 47.3, "baseline": 44.2, "seasons": 6,
            "note": "Possession tested and excluded from the model — no accuracy gain over xG-strength.",
        },
        "leaguePossession": league_poss,
        "matchdays": matchdays,
    }
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"Wrote {OUT.name}: {sum(len(v['matches']) for v in matchdays.values())} matches, "
          f"{len(league_poss)} teams.")


if __name__ == "__main__":
    main()
