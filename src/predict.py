"""Predict the next three Premier League match days - WITHOUT bookmaker odds.

An honest, data-only predictor. Every input is derived from what has happened
on the pitch: recent form, home/away-specific form, goal and xG output,
opponent-adjusted strength (Elo and an xG-based Elo) and head-to-head history.
No betting-market information is used.

The model is calibrated so its probabilities are trustworthy, and trained on
every completed match we have.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from elo import _expected_home, add_elo, add_xg_elo
from features import ROLL, load_matches, merge_xg, to_team_long

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PROC = ROOT / "data" / "processed" / "matches_features.csv"
UPCOMING = ROOT / "data" / "raw" / "upcoming_fixtures.csv"
OUT = ROOT / "data" / "processed" / "predictions_no_odds.csv"
CLASSES = ["A", "D", "H"]

# The full honest (no-odds) feature set validated in the ablation.
FEATS = [
    "diff_roll_gf", "diff_roll_ga", "diff_roll_sf", "diff_roll_sa",
    "diff_roll_stf", "diff_form", "diff_rest_days", "diff_played",
    "elo_diff", "elo_prob_home",
    "diff_roll_v_gf", "diff_roll_v_ga", "diff_roll_v_pts",
    "xgelo_diff", "xgelo_prob_home",
    "h2h_home_ppg", "h2h_n",
]


def calibrated_model() -> CalibratedClassifierCV:
    base = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, random_state=42)
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def proba_in_order(model, X) -> np.ndarray:
    order = [list(model.classes_).index(c) for c in CLASSES]
    return model.predict_proba(X)[:, order]


def overall_snapshot(long: pd.DataFrame) -> dict:
    snap = {}
    for team, g in long.groupby("team"):
        last = g.tail(ROLL)
        snap[team] = {
            "roll_gf": last["gf"].mean(), "roll_ga": last["ga"].mean(),
            "roll_sf": last["sf"].mean(), "roll_sa": last["sa"].mean(),
            "roll_stf": last["stf"].mean(), "form": last["pts"].mean(),
            "played": len(g), "last_date": g["date"].max(),
        }
    return snap


def venue_snapshot(long: pd.DataFrame) -> dict:
    """Per (team, venue) recent form, e.g. a team's last ROLL home games."""
    snap = {}
    for (team, venue), g in long.groupby(["team", "venue"]):
        last = g.tail(ROLL)
        snap[(team, venue)] = {
            "roll_v_gf": last["gf"].mean(), "roll_v_ga": last["ga"].mean(),
            "roll_v_pts": last["pts"].mean(),
        }
    return snap


def h2h_history(matches: pd.DataFrame, k: int = 6):
    """Final head-to-head state, and a lookup for a new (home, away) pair."""
    hist = defaultdict(deque)
    for row in matches.sort_values("Date").itertuples(index=False):
        hist[frozenset((row.HomeTeam, row.AwayTeam))].append((row.HomeTeam, row.FTR))
        dq = hist[frozenset((row.HomeTeam, row.AwayTeam))]
        while len(dq) > k:
            dq.popleft()

    def lookup(home, away):
        dq = hist.get(frozenset((home, away)), ())
        pts, cnt = 0, 0
        for past_home, res in dq:
            if res == "D":
                p = 1
            elif past_home == home:
                p = 3 if res == "H" else 0
            else:
                p = 3 if res == "A" else 0
            pts += p
            cnt += 1
        return (pts / cnt if cnt else np.nan), cnt

    return lookup


def feature_row(home, away, date, osnap, vsnap, elo, xgelo, h2h) -> dict:
    h, a = osnap.get(home, {}), osnap.get(away, {})
    hv = vsnap.get((home, "H"), {})
    av = vsnap.get((away, "A"), {})
    nan = float("nan")

    def d(key):
        return h.get(key, nan) - a.get(key, nan)

    rh, ra = elo.get(home, 1500.0), elo.get(away, 1500.0)
    xrh, xra = xgelo.get(home, 1500.0), xgelo.get(away, 1500.0)
    ppg, n = h2h(home, away)

    rest_h = (date - h["last_date"]).days if h.get("last_date") is not None else nan
    rest_a = (date - a["last_date"]).days if a.get("last_date") is not None else nan

    return {
        "diff_roll_gf": d("roll_gf"), "diff_roll_ga": d("roll_ga"),
        "diff_roll_sf": d("roll_sf"), "diff_roll_sa": d("roll_sa"),
        "diff_roll_stf": d("roll_stf"), "diff_form": d("form"),
        "diff_rest_days": rest_h - rest_a,
        "diff_played": h.get("played", 0) - a.get("played", 0),
        "elo_diff": rh - ra, "elo_prob_home": _expected_home(rh, ra),
        "diff_roll_v_gf": hv.get("roll_v_gf", nan) - av.get("roll_v_gf", nan),
        "diff_roll_v_ga": hv.get("roll_v_ga", nan) - av.get("roll_v_ga", nan),
        "diff_roll_v_pts": hv.get("roll_v_pts", nan) - av.get("roll_v_pts", nan),
        "xgelo_diff": xrh - xra, "xgelo_prob_home": _expected_home(xrh, xra),
        "h2h_home_ppg": ppg, "h2h_n": n,
    }


def main() -> None:
    train = pd.read_csv(FEATURES_PROC, dtype={"season": str})
    print(f"Training calibrated no-odds model on {len(train)} completed matches...")
    model = calibrated_model().fit(train[FEATS], train["FTR"])

    # Current strengths / form.
    matches = merge_xg(load_matches())
    matches, elo = add_elo(matches)
    matches, xgelo = add_xg_elo(matches)
    long = to_team_long(matches)
    osnap, vsnap = overall_snapshot(long), venue_snapshot(long)
    h2h = h2h_history(matches)

    up = pd.read_csv(UPCOMING)
    up["date"] = pd.to_datetime(up["date"])

    rows, feats = [], []
    for r in up.itertuples(index=False):
        feats.append(feature_row(r.HomeTeam, r.AwayTeam, r.date,
                                 osnap, vsnap, elo, xgelo, h2h))
        rows.append((r.matchday, r.date, r.HomeTeam, r.AwayTeam))

    proba = proba_in_order(model, pd.DataFrame(feats)[FEATS])
    idx = {c: i for i, c in enumerate(CLASSES)}

    out = []
    for (md, date, home, away), p in zip(rows, proba):
        pick = CLASSES[int(p.argmax())]
        label = {"H": home, "D": "Draw", "A": away}[pick]
        out.append({
            "matchday": md, "date": date.strftime("%Y-%m-%d"),
            "match": f"{home} vs {away}",
            "P(home)": round(float(p[idx['H']]), 3),
            "P(draw)": round(float(p[idx['D']]), 3),
            "P(away)": round(float(p[idx['A']]), 3),
            "prediction": label, "confidence": round(float(p.max()), 3),
        })
    res = pd.DataFrame(out)
    res.to_csv(OUT, index=False)

    for md in sorted(res["matchday"].unique()):
        sub = res[res["matchday"] == md]
        span = f"{sub['date'].min()} to {sub['date'].max()}"
        print(f"\n=== Match day {md}  ({span}) ===")
        print(sub[["match", "P(home)", "P(draw)", "P(away)",
                   "prediction", "confidence"]].to_string(index=False))
    print(f"\nSaved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
