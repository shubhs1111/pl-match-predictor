"""A football Elo rating system.

Each team carries a rating. Before a match we record each side's current
rating (a pre-match feature), then update both ratings based on the result,
scaled by goal margin. Ratings regress toward the mean between seasons so a
newly promoted/relegated reshuffle doesn't carry stale strength.

Returns, per match, the PRE-match ratings and the Elo-implied home win prob.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE = 1500.0
K = 20.0
HOME_ADV = 65.0          # Elo points added to the home side when computing odds
SEASON_REGRESS = 0.25    # fraction pulled back to BASE at each new season


def _expected_home(r_home: float, r_away: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_away - (r_home + HOME_ADV)) / 400.0))


def _margin_multiplier(goal_diff: int) -> float:
    # Standard FiveThirtyEight-style margin-of-victory multiplier.
    return float(np.log(abs(goal_diff) + 1))


def add_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """Add home_elo, away_elo, elo_diff, elo_prob_home (all pre-match)."""
    df = matches.sort_values("Date").reset_index(drop=True)
    ratings: dict[str, float] = {}
    last_season: dict[str, str] = {}

    home_elo, away_elo, prob_home = [], [], []

    for row in df.itertuples(index=False):
        h, a = row.HomeTeam, row.AwayTeam
        season = row.season
        for team in (h, a):
            if team not in ratings:
                ratings[team] = BASE
            elif last_season.get(team) not in (None, season):
                # New season for this team: regress toward the mean.
                ratings[team] = BASE + (1 - SEASON_REGRESS) * (ratings[team] - BASE)
            last_season[team] = season

        rh, ra = ratings[h], ratings[a]
        exp_h = _expected_home(rh, ra)
        home_elo.append(rh)
        away_elo.append(ra)
        prob_home.append(exp_h)

        # Actual score for the home team.
        if row.FTR == "H":
            s_h = 1.0
        elif row.FTR == "D":
            s_h = 0.5
        else:
            s_h = 0.0

        mult = _margin_multiplier(int(row.FTHG) - int(row.FTAG))
        change = K * mult * (s_h - exp_h)
        ratings[h] = rh + change
        ratings[a] = ra - change

    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["elo_prob_home"] = prob_home
    return df, ratings


def add_xg_elo(matches: pd.DataFrame) -> pd.DataFrame:
    """An Elo rating driven by xG instead of goals.

    The 'result' of each match is the home side's share of total xG, a
    continuous value in [0, 1]. This produces an opponent-adjusted measure of
    underlying strength (attack + defense) that is less noisy than goals and
    not swayed by finishing luck. Returns pre-match ratings per match.
    """
    df = matches.sort_values("Date").reset_index(drop=True)
    ratings: dict[str, float] = {}
    last_season: dict[str, str] = {}
    home_r, away_r, prob_home = [], [], []

    for row in df.itertuples(index=False):
        h, a, season = row.HomeTeam, row.AwayTeam, row.season
        for team in (h, a):
            if team not in ratings:
                ratings[team] = BASE
            elif last_season.get(team) not in (None, season):
                ratings[team] = BASE + (1 - SEASON_REGRESS) * (ratings[team] - BASE)
            last_season[team] = season

        rh, ra = ratings[h], ratings[a]
        exp_h = _expected_home(rh, ra)
        home_r.append(rh)
        away_r.append(ra)
        prob_home.append(exp_h)

        hx, ax = row.home_xg, row.away_xg
        if pd.isna(hx) or pd.isna(ax) or (hx + ax) == 0:
            continue  # no xG for this match; leave ratings unchanged
        s_h = hx / (hx + ax)  # home share of expected goals
        change = K * (s_h - exp_h)
        ratings[h] = rh + change
        ratings[a] = ra - change

    df["xgelo_home"] = home_r
    df["xgelo_away"] = away_r
    df["xgelo_diff"] = df["xgelo_home"] - df["xgelo_away"]
    df["xgelo_prob_home"] = prob_home
    return df, ratings
