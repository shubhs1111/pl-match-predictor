# Premier League Match Predictor

An ML model that predicts the outcome (Home win / Draw / Away win) of upcoming
Premier League matches, aimed at the next ~3 match days.

## Status

- [x] Environment set up (Python 3.12 venv, pandas / numpy / scikit-learn)
- [x] Historical data downloaded (7 seasons, football-data.co.uk)
- [x] Data explored and sanity-checked
- [x] Feature engineering (pre-match features) — `src/features.py`
- [x] Model training + single-season backtest — `src/train.py`
- [x] Walk-forward backtest + calibration check — `src/backtest.py`
- [x] Enrich features (Elo + Understat xG) + ablation — `src/ablation.py`
- [x] Calibrated predictions for the next 3 match days — `src/predict.py`
- [x] Per-match "why" explanations — `src/explain.py` -> `reasons.json`
- [x] Dashboards — `dashboard.html` (simple) and `terminal.html` (quant terminal)

## Dashboards

- **`dashboard.html`** — a clean predictions board (reads `reasons.json`).
- **`terminal.html`** — a "quant terminal" view: deep-dive inspector, fixture
  ribbon, filterable match matrix (reads `ui_data.json`, built by
  `src/build_ui_data.py`).
- **`pl-predictor-terminal.html`** — the terminal view as a single self-contained
  file (data embedded), shareable by just opening it in a browser.

The companion data files are fetched, so open the multi-file dashboards via a
local server (`python -m http.server`) rather than `file://`.

## Data (not included in this repo)

The raw datasets are third-party and are **not** redistributed here. To
reproduce from scratch, obtain them yourself and place them as noted:

- Results & odds: [football-data.co.uk](https://www.football-data.co.uk/englandm.php)
  — fetched automatically by `python src/download_data.py` into `data/raw/`.
- Understat xG (`data/raw/understat_xg.csv`) — per-match xG, scraped from Understat.
- Opta per-match stats (`_all_events_stats.xlsx`) — possession, big chances, etc.
  (sourced from a Kaggle dataset).

Generated model outputs (`reasons.json`, `ui_data.json`,
`data/processed/predictions_no_odds.csv`) are included so the dashboards work.

## Predicting upcoming fixtures (honest, no-odds model)

`python src/predict.py` trains a single calibrated model on every completed
match and predicts the next three match days using ONLY on-pitch data - no
bookmaker odds. Fixtures come from `data/raw/upcoming_fixtures.csv`; output to
`data/processed/predictions_no_odds.csv`.

### Factors used

| Factor | How it's captured |
|---|---|
| Recent form / past matches | Rolling results & points (last 5) |
| Goal stats | Rolling goals for/against |
| Shots | Rolling shots & shots on target |
| Defense + strengths/weaknesses | Opponent-adjusted Elo and **xG-Elo** rating |
| Home field | Elo home advantage + **separate home-form / away-form** |
| Head-to-head matchups | Home team's points-per-game vs this opponent (last 6) |
| Rest / congestion | Days since each side's last match |

**Tested and left out (Opta data, `_all_events_stats.xlsx`, 2009-26):**

| Feature tried | Result | Verdict |
|---|---|---|
| Possession % | 47.3% -> 47.0% | No gain (shown on dashboard as context only) |
| Big chances created/conceded | 47.3% -> 46.8% | No gain |
| Opta xG | — | Untestable: populated only from 2024-25 |

All three measure the same thing the model already has (control + chance
quality), captured by the xG-strength rating, shots and form. They're wired into
the pipeline but excluded from the model. This is the public-data ceiling (~47%);
the remaining gap to the market (~55%) is injuries, line-ups and late team news.

### Honest performance (no odds, walk-forward over 1,900 matches)

~47% accuracy - on par with the odds-based variant and well above the 44%
always-home baseline. Football is high-variance; this is a realistic ceiling
for public-data models.

## Walk-forward backtest results (1,900 out-of-sample matches)

| Approach | Accuracy | Log-loss |
|---|---|---|
| Always predict home | 44.2% | — |
| Our model | 46.5% | 1.238 |
| Bookmaker odds | 55.2% | 0.960 |

Two findings:
1. The bookmaker odds beat our model consistently across all five test seasons.
   (The 2025-26 season we first tested was the market's *weakest* year, which
   had flattered the model.)
2. Our model is **overconfident** — when it predicts a 90%+ home win, the home
   team actually wins only ~72% of the time. Probability calibration is the
   obvious next fix.

## Feature ablation findings (pooled walk-forward)

| Feature set | Accuracy | Log-loss |
|---|---|---|
| Stats only | 45.5% | 1.241 |
| Stats + xG (rolling) | 44.7% | 1.262 |
| Stats + Elo | **46.8%** | **1.228** |
| Stats + Odds | 46.5% | 1.238 |
| Bookmaker odds alone | 55.2% | 0.960 |

- **Elo** (opponent-adjusted, season-spanning) is the best engineered feature.
- **Raw rolling xG** did *not* help — the lesson is that opponent-adjusted
  ratings beat raw rolling averages. To use xG well, build an xG-based strength
  rating (xG-Elo), not a rolling mean.
- Once **bookmaker odds** are present, extra features (Elo, xG) are largely
  redundant — the market already prices them in.
- xG data is sourced from Understat via the built-in browser (see
  `data/raw/understat_xg.csv`).

## Data

Source: [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — free
CSVs, one per season. Columns include full-time results, per-match stats
(shots, shots on target, corners, fouls, cards) and bookmaker odds.

- Training seasons: 2020-21 through 2025-26 (complete)
- Current season: 2026-27 (in progress) — the one we predict

Run `python src/download_data.py` to (re)fetch into `data/raw/`.

### Key facts from exploration (2,280 completed matches)

- Outcome balance: 43.1% home win / 23.6% draw / 33.3% away win
- Home advantage is real: avg 1.56 home goals vs 1.33 away
- Baseline to beat: 43% accuracy (always predict home win)
- Realistic target for a good model: ~53-58% accuracy

## Important modeling note

Per-match stats (shots, cards, corners) describe what happened *during* a match,
so they **cannot** be used directly as inputs to predict that same match — we
won't know them before kickoff. Features must be **pre-match**: rolling form,
rolling average stats over the previous N games, head-to-head history, rest
days, and the bookmaker odds (which are available before the match).

## Layout

```
pl-predictor/
  data/raw/        downloaded season CSVs
  src/
    download_data.py   fetch season CSVs
    explore.py         first-look exploration
  .venv/           virtual environment
```
