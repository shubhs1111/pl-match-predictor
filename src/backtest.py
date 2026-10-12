"""Walk-forward backtest across every season.

For each test season we train ONLY on the seasons before it, predict the whole
season, and pool all out-of-sample predictions. This is the honest way to
estimate real-world performance and removes the luck of picking one test year.

Also reports a calibration table: among matches we assign a given home-win
probability, how often does the home team actually win?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "matches_features.csv"

# Completed seasons, in chronological order. The current (partial) season is
# excluded from the backtest.
SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526"]
CLASSES = ["A", "D", "H"]  # lexicographic -> matches sklearn's default ordering

FEATURES = [
    "diff_roll_gf", "diff_roll_ga", "diff_roll_sf", "diff_roll_sa",
    "diff_roll_stf", "diff_form", "diff_rest_days", "diff_played",
    "imp_home", "imp_draw", "imp_away",
    "elo_diff", "elo_prob_home",
]


def new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, random_state=42,
    )


def market_probs(df: pd.DataFrame) -> np.ndarray:
    # Columns ordered to CLASSES = [A, D, H].
    return df[["imp_away", "imp_draw", "imp_home"]].to_numpy()


def metrics(y_true, proba) -> tuple[float, float]:
    pred = np.array(CLASSES)[proba.argmax(axis=1)]
    return accuracy_score(y_true, pred), log_loss(y_true, proba, labels=CLASSES)


def calibration_table(prob_home, y_true, bins=10) -> pd.DataFrame:
    home = (np.asarray(y_true) == "H").astype(int)
    df = pd.DataFrame({"p": prob_home, "home": home})
    df["bucket"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    g = df.groupby("bucket", observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "predicted": g["p"].mean(),
        "actual": g["home"].mean(),
    })


def main() -> None:
    df = pd.read_csv(DATA, dtype={"season": str})

    rows_y, rows_model, rows_market, rows_season = [], [], [], []
    print("=== Per-season walk-forward ===")
    print(f"  {'test season':12s} {'n':>4s} {'model acc':>10s} {'mkt acc':>9s} "
          f"{'model LL':>9s} {'mkt LL':>8s}")

    for i, season in enumerate(SEASONS):
        if i == 0:
            continue  # need at least one prior season to train
        train = df[df["season"].isin(SEASONS[:i])]
        test = df[df["season"] == season]

        model = new_model()
        model.fit(train[FEATURES], train["FTR"])
        order = [list(model.classes_).index(c) for c in CLASSES]
        m_proba = model.predict_proba(test[FEATURES])[:, order]
        k_proba = market_probs(test)

        m_acc, m_ll = metrics(test["FTR"], m_proba)
        k_acc, k_ll = metrics(test["FTR"], k_proba)
        print(f"  {season:12s} {len(test):4d} {m_acc:10.3f} {k_acc:9.3f} "
              f"{m_ll:9.3f} {k_ll:8.3f}")

        rows_y.append(test["FTR"].to_numpy())
        rows_model.append(m_proba)
        rows_market.append(k_proba)
        rows_season.append(np.full(len(test), season))

    y = np.concatenate(rows_y)
    model_proba = np.concatenate(rows_model)
    market_proba = np.concatenate(rows_market)

    print("\n=== Pooled out-of-sample ({} matches) ===".format(len(y)))
    ah = np.tile([0.0, 0.0, 1.0], (len(y), 1))  # always home (A,D,H)
    for name, p in [("Always home", ah), ("Model", model_proba),
                    ("Bookmaker odds", market_proba)]:
        acc, ll = metrics(y, p)
        print(f"  {name:16s} accuracy={acc:6.3f}   log-loss={ll:6.3f}")

    print("\n=== Model calibration (home-win probability) ===")
    print("  bucket of predicted P(home win) -> actual home-win rate")
    ct = calibration_table(model_proba[:, CLASSES.index("H")], y)
    print(ct.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
