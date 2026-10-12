"""Train a gradient-boosting classifier and backtest it on a held-out season.

We respect time: train only on seasons BEFORE the test season, so the backtest
mimics predicting matches we have never seen. We compare the model against two
baselines - always predicting a home win, and the bookmaker odds themselves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "matches_features.csv"

TRAIN_SEASONS = ["2021", "2122", "2223", "2324", "2425"]
TEST_SEASON = "2526"
CLASSES = ["H", "D", "A"]

FEATURES = [
    "diff_roll_gf", "diff_roll_ga", "diff_roll_sf", "diff_roll_sa",
    "diff_roll_stf", "diff_form", "diff_rest_days", "diff_played",
    "imp_home", "imp_draw", "imp_away",
]


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, dtype={"season": str})
    return df


def market_probs(df: pd.DataFrame) -> np.ndarray:
    return df[["imp_home", "imp_draw", "imp_away"]].to_numpy()


def report(name: str, y_true, proba: np.ndarray) -> None:
    pred = np.array(CLASSES)[proba.argmax(axis=1)]
    acc = accuracy_score(y_true, pred)
    ll = log_loss(y_true, proba, labels=CLASSES)
    print(f"  {name:28s} accuracy={acc:6.3f}   log-loss={ll:6.3f}")


def main() -> None:
    df = load()
    train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    test = df[df["season"] == TEST_SEASON].copy()
    print(f"Train: {len(train)} matches ({', '.join(TRAIN_SEASONS)})")
    print(f"Test:  {len(test)} matches ({TEST_SEASON}, held out)\n")

    X_train, y_train = train[FEATURES], train["FTR"]
    X_test, y_test = test[FEATURES], test["FTR"]

    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=4,
        l2_regularization=1.0, random_state=42,
    )
    model.fit(X_train, y_train)
    # Align predict_proba columns to our CLASSES order.
    order = [list(model.classes_).index(c) for c in CLASSES]
    model_proba = model.predict_proba(X_test)[:, order]

    print("=== Backtest on 2025-26 ===")
    # Baseline 1: always home.
    always_home = np.tile([1.0, 0.0, 0.0], (len(test), 1))
    report("Baseline: always home", y_test, always_home)
    # Baseline 2: bookmaker odds.
    report("Baseline: bookmaker odds", y_test, market_probs(test))
    # Our model.
    report("Model: gradient boosting", y_test, model_proba)

    # Which features matter (permutation importance on the test season).
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42,
        scoring="neg_log_loss",
    )
    order_imp = imp.importances_mean.argsort()[::-1]
    print("\n=== Top feature importances ===")
    for i in order_imp[:8]:
        print(f"  {FEATURES[i]:18s} {imp.importances_mean[i]:+.4f}")


if __name__ == "__main__":
    main()
