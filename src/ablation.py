"""Ablation: how much does each feature group actually add?

Runs the same walk-forward backtest over four feature sets so we can see where
Elo earns its keep - and whether it is redundant once bookmaker odds are
present. This answers the strategy question: do stats features help, or does
the market already contain them?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "matches_features.csv"
SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526"]
CLASSES = ["A", "D", "H"]

STATS = ["diff_roll_gf", "diff_roll_ga", "diff_roll_sf", "diff_roll_sa",
         "diff_roll_stf", "diff_form", "diff_rest_days", "diff_played"]
XG = ["diff_roll_xgf", "diff_roll_xga"]
ELO = ["elo_diff", "elo_prob_home"]
VENUE = ["diff_roll_v_gf", "diff_roll_v_ga", "diff_roll_v_pts"]
XGELO = ["xgelo_diff", "xgelo_prob_home"]
H2H = ["h2h_home_ppg", "h2h_n"]
POSS = ["diff_roll_poss"]
BIGCH = ["diff_roll_bcc", "diff_roll_bcca"]
ODDS = ["imp_home", "imp_draw", "imp_away"]

FULL = STATS + ELO + VENUE + XGELO + H2H

# No-odds models (honest, data-only) vs the odds benchmark.
FEATURE_SETS = {
    "Stats + Elo": STATS + ELO,
    "+ venue + xG-Elo + H2H (full)": FULL,
    "+ possession": FULL + POSS,
    "+ big chances": FULL + BIGCH,
    "+ possession + big chances": FULL + POSS + BIGCH,
    "Odds model (reference)": STATS + ELO + ODDS,
}


def new_model():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, random_state=42,
    )


def walk_forward(df: pd.DataFrame, feats: list[str]):
    ys, ps = [], []
    for i, season in enumerate(SEASONS):
        if i == 0:
            continue
        train = df[df["season"].isin(SEASONS[:i])]
        test = df[df["season"] == season]
        m = new_model()
        m.fit(train[feats], train["FTR"])
        order = [list(m.classes_).index(c) for c in CLASSES]
        ps.append(m.predict_proba(test[feats])[:, order])
        ys.append(test["FTR"].to_numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = np.array(CLASSES)[p.argmax(axis=1)]
    return accuracy_score(y, pred), log_loss(y, p, labels=CLASSES)


def main() -> None:
    df = pd.read_csv(DATA, dtype={"season": str})
    print("=== Feature-set ablation (pooled walk-forward, 1900 matches) ===\n")
    print(f"  {'feature set':22s} {'accuracy':>9s} {'log-loss':>9s}")
    for name, feats in FEATURE_SETS.items():
        acc, ll = walk_forward(df, feats)
        print(f"  {name:22s} {acc:9.3f} {ll:9.3f}")
    print("\n  (bookmaker odds alone: accuracy 0.552, log-loss 0.960)")


if __name__ == "__main__":
    main()
