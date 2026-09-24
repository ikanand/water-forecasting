# Model card - Dubai day-ahead hourly water demand

| | |
|---|---|
| Registered model | `<catalog>.ml.water_demand_forecast` (Unity Catalog), alias `@champion` serves forecasts |
| Owner | Data science / MLOps team of the water forecasting platform |
| Version of this card | 2026-09-24 (go-live) |
| Evidence | `notebooks/01_eda.ipynb` (EDA), `notebooks/02_modelling.ipynb` (model selection, 12-month backtest) |

## 1. Intended use

* **What it predicts:** city-wide water demand (m3/h) for each of the 24 hours of tomorrow, for site
  `DXB-CITY-TOTAL` (Dubai), issued once a day at 05:00 UTC (09:00 Dubai).
* **Intended users:** network operations, production and storage planning (pumping schedules, reservoir
  levels), and demand management.
* **Out of scope:** intraday re-forecasting, sub-city zones, forecasts beyond 24 h ahead, and extreme events that
  are not in the input data (floods, network incidents). Operators should override the forecast for known
  incidents.

## 2. Inputs (all known at issue time)

| Group | Features |
|---|---|
| Recent demand | lags 48, 72, 168, 336, 504 h; 7-day rolling mean/std (ending 48 h before target); 3-week same-hour mean |
| Weather forecast for D+1 (issued 06:00 on D) | hourly temperature, humidity, precipitation; daily max/mean; cooling degrees at 20 C and at the EDA knot 24.4 C |
| Temperature persistence | forecast mean temperature of D, observed mean temperature of D-1 |
| Calendar | hour, weekday, month, weekend (Sat-Sun), public holiday, Eid, day before/after holiday, Fourier terms |
| Culture and season | Ramadan flag and Ramadan day, school summer break |
| Events | structural / operational / weather event flags from the calendar |
| Trend index | `t_days` (days since 2024-01-01) - used only by growth-aware models to fit and extrapolate the trend |

**No demand lag shorter than 48 h** (enforced in `features.py`): at issue time the newest complete day is D-1.
**Training uses the weather forecast, not observed weather**, so the model learns from the same inputs it gets in
production.

## 3. Model

Each weekly training run fits six candidate families on all history available in the environment and keeps the
best on a 28-day holdout; the winner becomes `@challenger` and replaces `@champion` only if it is at least 1 %
(relative MAPE) better on the holdout days the champion never trained on (at least 7 such days, otherwise the
champion is kept). Judging the champion on data it was trained on would flatter it and block promotions for weeks.

| Candidate | Description |
|---|---|
| `seasonal_naive_168` | same hour last week (sanity baseline) |
| `ridge_fourier` | ridge regression on raw target |
| `lightgbm` | gradient boosting on raw target |
| `lightgbm_trend` | LightGBM on a growth-normalised target |
| `ridge_interactions` | growth-normalised ridge with hour x (day off, Ramadan, cooling degree) interactions |
| `blend_ridge_lgbm` | equal-weight average of the two growth-aware models - **expected champion in prod** |

**Growth normalisation.** Demand grows about 10 % per year. The growth-aware models fit log daily demand
= a + b*t + annual harmonics on the training history, divide the target and level features by exp(a + b*t), and
multiply the prediction back with the trend extrapolated. With less than 18 months of history the slope is set
to 0, because growth cannot be separated from one annual cycle (this protects dev and freshly refreshed envs).

## 4. Performance (12-month rolling-origin backtest, Oct 2025 - Sep 2026, monthly refits)

| Model | MAPE | Bias | Peak-hour MAPE | Daily volume error |
|---|---|---|---|---|
| Previous production (`lightgbm`, raw target) | 3.44 % | -1.34 % | 4.04 % | 2.52 % |
| **`blend_ridge_lgbm` (shipped)** | **2.61 %** | **+0.14 %** | **2.55 %** | **1.82 %** |
| `ridge_interactions` | 2.64 % | +0.12 % | 2.61 % | 1.89 % |
| `lightgbm_trend` | 2.90 % | +0.16 % | 2.66 % | 1.98 % |
| Research only: ensemble LightGBM-trend + SARIMAX + MLP | 2.51 % | -0.11 % | 2.56 % | 1.72 % |
| Seasonal naive (same hour last week) | 5.94 % | -0.04 % | 5.51 % | 4.67 % |

By segment, the shipped `blend_ridge_lgbm`: weekdays 2.45 %, weekends 2.74 %, Ramadan 2.96 %, Eid days 3.76 %
(previous production: 6.19 % on Eid days). The research-only ensemble is a little better still on Eid (3.04 %).

**First live checks (dev, catch-up at go-live):** reconciled daily MAPE 2.09 % (21 Sep), 1.45 % (22 Sep),
4.14 % (23 Sep), using the dev champion trained on 8 months of history.

## 5. Uncertainty

A quantile LightGBM gives P10/P90 bands; raw they are over-confident (62 % coverage for a nominal 80 %).
Online conformal calibration from past errors restores 82 % coverage at +/-5.5 % half-width. **Not yet
published by the pipeline** (follow-up); until then use +/-5.5 % around the point forecast as an 80 % band.

## 6. Known limitations and failure modes

* **First days of Ramadan and Eid days** have the largest errors (6 % on the first Ramadan days; small
  under-forecast on Eid). The intraday profile changes overnight.
* **Unrecorded incidents** (e.g. 2026-01-10, -13 % with no calendar or weather explanation) cannot be forecast;
  an operations-incident feed is the recommended next data source.
* **Floods and similar extreme weather** are too rare to learn (3 days in 2.75 years) - operator override.
* **Short history:** with less than 18 months of data, growth is not extrapolated (slope 0); with about one
  year of history, annual features can anchor to last year's level and under-forecast growth. Environments
  therefore keep at least 2 years (qa, prod).
* **Data:** the source extracts and the weather forecast are realistic simulations; the monitoring job measures
  live accuracy from day one and triggers retraining if daily MAPE exceeds 1.25x the champion's holdout MAPE for
  3 consecutive days.

## 7. Governance

* Every training run is logged to MLflow (parameters, metrics, leaderboard, feature importance, fitted growth).
* Every model version is tagged with family, holdout MAPE, training trigger, data end date and git SHA.
* Every forecast row in `gold.demand_forecasts` records model name, version, family, issue time and run id
  (append-only audit trail).
* Rollback: repoint `@champion` to a previous version; the next forecast uses it.

## 8. Recommended next steps

1. Publish calibrated P10/P90 intervals.
2. Add the MLP as a candidate (ensemble level ~2.5 %); requires PyTorch in the job environment.
3. Business decision on the issue time: a midnight issue with real-time SCADA enables `lag_24` (-4.2 % MAPE).
4. Add an operations-incident feed (planned outages, pressure management) as an event source.
