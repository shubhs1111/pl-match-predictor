"""Create clear, evidence-led explanations for each match prediction.

The prediction remains the calibrated model's highest-probability outcome. The
card explanation is intentionally simpler: it compares the actual pre-match
data for the two teams and shows which factual signals point to each side.
This avoids presenting a non-linear model interaction as a football fact.

Writes ``reasons.json`` for ``dashboard.html``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from elo import add_elo, add_xg_elo
from features import load_matches, merge_xg, to_team_long
from predict import (CLASSES, FEATS, UPCOMING, calibrated_model, feature_row,
                     h2h_history, overall_snapshot, proba_in_order, venue_snapshot)

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PROC = ROOT / "data" / "processed" / "matches_features.csv"
OUT = ROOT / "reasons.json"
OUT_SCRIPT = ROOT / "reasons.js"

# The single, human-readable comparison representing each data family. The
# related rolling inputs still feed the prediction model; we do not pretend
# that a composite black-box interaction is a simple causal fact.
SIGNALS = {
    "Overall team strength": ("elo_diff", 0.0, 1),
    "Underlying chance quality": ("xgelo_diff", 0.0, 1),
    "Recent results": ("diff_form", 0.0, 1),
    # Fewer shots allowed is better, so this comparison runs in the opposite
    # direction to the home-minus-away feature stored by the model.
    "Recent defensive shot prevention": ("diff_roll_sa", 0.0, -1),
    "Home and away form": ("diff_roll_v_pts", 0.0, 1),
    "Head-to-head record": ("h2h_home_ppg", 1.5, 1),
    "Rest before kickoff": ("diff_rest_days", 0.0, 1),
}


def strength_label(score: float) -> str:
    if score >= 1.0:
        return "Strong edge"
    if score >= 0.5:
        return "Clear edge"
    return "Small edge"


def detail(group: str, home: str, away: str, date: pd.Timestamp, osnap: dict,
           vsnap: dict, elo: dict, xgelo: dict, h2h) -> str:
    """Describe observed values only; never claim they guarantee an outcome."""
    home_form = osnap.get(home, {}).get("form")
    away_form = osnap.get(away, {}).get("form")
    home_venue = vsnap.get((home, "H"), {}).get("roll_v_pts")
    away_venue = vsnap.get((away, "A"), {}).get("roll_v_pts")
    home_date = osnap.get(home, {}).get("last_date")
    away_date = osnap.get(away, {}).get("last_date")
    h2h_ppg, h2h_n = h2h(home, away)

    if group == "Home setting":
        return f"{home} are playing at home."
    if group == "Overall team strength":
        return (f"Overall rating: {home} {elo.get(home, 1500):.0f}; "
                f"{away} {elo.get(away, 1500):.0f}.")
    if group == "Underlying chance quality":
        return (f"Chance-quality rating: {home} {xgelo.get(home, 1500):.0f}; "
                f"{away} {xgelo.get(away, 1500):.0f}.")
    if group == "Recent results":
        return (f"Last five matches: {home} {home_form:.1f} points per game; "
                f"{away} {away_form:.1f}.")
    if group == "Recent defensive shot prevention":
        home_allowed = osnap[home]["roll_sa"]
        away_allowed = osnap[away]["roll_sa"]
        return (f"Last five matches: {home} allowed {home_allowed:.1f} shots per game; "
                f"{away} allowed {away_allowed:.1f}.")
    if group == "Home and away form":
        return (f"Recent venue form: {home} {home_venue:.1f} points per home game; "
                f"{away} {away_venue:.1f} per away game.")
    if group == "Head-to-head record":
        return (f"Across the last {h2h_n} meeting{'s' if h2h_n != 1 else ''}, "
                f"{home} took {h2h_ppg:.1f} points per game.")
    home_rest = (date - home_date).days
    away_rest = (date - away_date).days
    return f"Rest before kickoff: {home} {home_rest} days; {away} {away_rest} days."


def render(factors: list[dict]) -> list[dict]:
    factors.sort(key=lambda factor: -factor["score"])
    return [{
        "factor": factor["factor"],
        "strength": strength_label(factor["score"]),
        "weight": round(float(factor["score"]), 1),
        "detail": factor["detail"],
    } for factor in factors[:4]]


def main() -> None:
    train = pd.read_csv(FEATURES_PROC, dtype={"season": str})
    model = calibrated_model().fit(train[FEATS], train["FTR"])
    scales = train[list(key for key, _, _ in SIGNALS.values())].std().replace(0, np.nan)

    matches = merge_xg(load_matches())
    matches, elo = add_elo(matches)
    matches, xgelo = add_xg_elo(matches)
    long = to_team_long(matches)
    osnap, vsnap = overall_snapshot(long), venue_snapshot(long)
    h2h = h2h_history(matches)

    upcoming = pd.read_csv(UPCOMING)
    upcoming["date"] = pd.to_datetime(upcoming["date"])
    class_index = {outcome: i for i, outcome in enumerate(CLASSES)}
    out = []

    for fixture in upcoming.itertuples(index=False):
        home, away = fixture.HomeTeam, fixture.AwayTeam
        row = feature_row(home, away, fixture.date, osnap, vsnap, elo, xgelo, h2h)
        probabilities = proba_in_order(model, pd.DataFrame([row])[FEATS].astype(float))[0]
        pick_code = CLASSES[int(probabilities.argmax())]
        pick_team = {"H": home, "A": away, "D": "Draw"}[pick_code]

        if pick_code == "H":
            challenger_code, challenger = "A", away
        elif pick_code == "A":
            challenger_code, challenger = "H", home
        else:
            challenger_code = "H" if probabilities[class_index["H"]] >= probabilities[class_index["A"]] else "A"
            challenger = home if challenger_code == "H" else away

        home_signals, away_signals = [], []
        for group, (key, pivot, direction) in SIGNALS.items():
            value = row[key]
            if pd.isna(value) or (group == "Head-to-head record" and not row["h2h_n"]):
                continue
            scale = scales[key]
            if pd.isna(scale):
                continue
            score = direction * (value - pivot) / scale
            if abs(score) < 0.15:  # no meaningful data edge to display
                continue
            factor = {"factor": group, "score": abs(float(score)),
                      "detail": detail(group, home, away, fixture.date, osnap, vsnap,
                                       elo, xgelo, h2h)}
            (home_signals if score > 0 else away_signals).append(factor)

        # Home advantage is a real model input via the rating probabilities.
        # It is context rather than a team-quality claim, so it ranks last.
        home_signals.append({"factor": "Home setting", "score": 0.15,
                             "detail": detail("Home setting", home, away, fixture.date,
                                              osnap, vsnap, elo, xgelo, h2h)})

        by_team = {home: render(home_signals), away: render(away_signals)}
        out.append({
            "matchday": int(fixture.matchday),
            "match": f"{home} vs {away}",
            "pick": pick_team,
            "pick_prob": round(float(probabilities[class_index[pick_code]]), 3),
            "home": home,
            "away": away,
            "challenger": challenger,
            "pro": by_team.get(pick_team, []),
            "con": by_team.get(challenger, []),
        })

    payload = json.dumps(out, indent=1)
    OUT.write_text(payload, encoding="utf-8")
    # A regular script loads from disk when dashboard.html is double-clicked.
    # Unlike fetch(), this is not blocked by the browser's file:// policy.
    OUT_SCRIPT.write_text(f"window.PL_REASONS = {payload};\n", encoding="utf-8")
    print(f"Saved -> {OUT.relative_to(ROOT)} and {OUT_SCRIPT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
