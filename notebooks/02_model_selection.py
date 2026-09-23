# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Model selection — which model family, and why
# MAGIC
# MAGIC Compares candidate families with a **rolling-origin backtest** (4 consecutive weeks, each
# MAGIC trained only on data before it) — the honest way to evaluate a time series: never shuffle,
# MAGIC never let the model see the future. Every run is logged to MLflow so the leaderboard, parameters
# MAGIC and metrics are reproducible and shareable.
# MAGIC
# MAGIC | Family | Why it's in the race |
# MAGIC |---|---|
# MAGIC | `seasonal_naive_48` / `seasonal_naive_168` | The bar to beat: "same hour 2 days / 1 week ago" |
# MAGIC | `ridge_fourier` | Transparent linear model; regulators and ops teams can read it |
# MAGIC | `lightgbm` | Gradient boosting; captures non-linear weather × calendar interactions |
# MAGIC | `sarimax` (one zone) | Classical statistical benchmark — see why it's not in production |
# MAGIC
# MAGIC **Metrics:** MAPE (headline), WAPE, RMSE, bias, and *peak-hour MAPE* (top 10 % demand hours —
# MAGIC what pumping and storage planning actually care about).

# COMMAND ----------

# MAGIC %pip install -e ..[dev]
# MAGIC %restart_python

# COMMAND ----------

import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from water_forecasting import features as F
from water_forecasting import synthetic
from water_forecasting.config import ProjectConfig, Tables
from water_forecasting.modeling import evaluate, make_model, rolling_origin_folds

cfg = ProjectConfig.from_yaml("../project_config.yml", env="dev")
COLS = F.feature_columns(cfg.zones)
FAMILIES = ["seasonal_naive_48", "seasonal_naive_168", "ridge_fourier", "lightgbm"]
N_FOLDS, FOLD_DAYS = 4, 7

try:
    spark  # noqa: B018
    feats = spark.table(cfg.table(Tables.FEATURES)).toPandas()
    labels = spark.table(cfg.table(Tables.SILVER_CONSUMPTION)).toPandas()
    mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/water_model_selection")
except NameError:  # local run
    raw = synthetic.generate_range(cfg.history_start, cfg.history_end, cfg.zones, cfg.source.seed, backfill=True)
    targets = F.hourly_targets(cfg.zones, pd.Timestamp(cfg.history_start) + pd.Timedelta(days=21), cfg.history_end)
    feats = F.build_features(targets, raw["consumption"], raw["weather_fcst"], raw["holidays"], raw["events"],
                             cfg.zones)
    labels = raw["consumption"]
    mlflow.set_experiment("water_model_selection")

try:
    display  # noqa: B018
except NameError:
    display = print  # noqa: A001

data = F.attach_label(feats, labels).dropna(subset=["lag_504h"]).sort_values(F.KEYS).reset_index(drop=True)
print(f"{len(data):,} rows  {data.target_ts.min():%Y-%m-%d} .. {data.target_ts.max():%Y-%m-%d}")

# COMMAND ----------

# MAGIC %md ## 1. Rolling-origin backtest of every family (logged to MLflow)

# COMMAND ----------

rows, preds = [], []
with mlflow.start_run(run_name=f"model-selection-{pd.Timestamp.now():%Y%m%d-%H%M}"):
    mlflow.log_params({"n_folds": N_FOLDS, "fold_days": FOLD_DAYS, "n_features": len(COLS)})
    for family in FAMILIES:
        with mlflow.start_run(run_name=family, nested=True):
            fold_metrics = []
            t0 = time.time()
            for i, (tr, te) in enumerate(rolling_origin_folds(data, N_FOLDS, FOLD_DAYS)):
                model = make_model(family, cfg.model.ridge_alpha, cfg.model.lgbm_params)
                p = model.fit(tr[COLS], tr.consumption_m3).predict(te[COLS])
                fold_metrics.append(evaluate(te.consumption_m3, p))
                preds.append(te[F.KEYS + ["consumption_m3", "is_holiday", "event_active"]].assign(
                    family=family, fold=i, pred=p))
            fm = pd.DataFrame(fold_metrics)
            summary = {**fm.mean().add_prefix("mean_"), **fm.std().add_prefix("std_")}
            mlflow.log_params({"family": family, **{f"p_{k}": v for k, v in model.get_params().items()}})
            mlflow.log_metrics({**summary, "fit_seconds": time.time() - t0})
            rows.append({"family": family, **fm.mean().round(2), "mape_std": round(fm.mape.std(), 2),
                         "fit_seconds": round(time.time() - t0, 1)})

leaderboard = pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)
display(leaderboard)

# COMMAND ----------

# MAGIC %md ## 2. Where does each model make its errors?

# COMMAND ----------

P = pd.concat(preds, ignore_index=True)
P["ape"] = (P.pred - P.consumption_m3).abs() / P.consumption_m3 * 100
P["hour"] = P.target_ts.dt.hour

by_zone = P.pivot_table(index="family", columns="zone_id", values="ape", aggfunc="mean").round(2)
by_daytype = P.assign(day=np.where(P.is_holiday == 1, "holiday", np.where(P.event_active == 1, "event", "normal"))
                      ).pivot_table(index="family", columns="day", values="ape", aggfunc="mean").round(2)
display(by_zone.reset_index())
display(by_daytype.reset_index())

fig, ax = plt.subplots(figsize=(10, 4))
P.groupby(["hour", "family"]).ape.mean().unstack().plot(ax=ax, title="MAPE by hour of day")
ax.set_ylabel("MAPE %")
plt.show()

# COMMAND ----------

# MAGIC %md ## 3. What drives the best model? (LightGBM feature importance)

# COMMAND ----------

lgbm = make_model("lightgbm", lgbm_params=cfg.model.lgbm_params).fit(data[COLS], data.consumption_m3)
imp = lgbm.feature_importance(COLS)
imp_pct = (imp / imp.sum() * 100).round(1).head(15)
imp_pct.sort_values().plot.barh(figsize=(8, 5), title="Top-15 features (% of total gain)")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Classical benchmark: SARIMAX on a single zone
# MAGIC Fitted on one zone only — it needs one model per zone, is slow to fit on hourly data with a
# MAGIC 168-hour season, and cannot use lagged-48h structure as naturally as the feature models.

# COMMAND ----------

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    z = cfg.zones[0]
    tr, te = list(rolling_origin_folds(data[data.zone_id == z], 1, FOLD_DAYS))[0]
    exog = ["fc_temperature_c", "fc_cooling_degree", "is_day_off", "hour_sin1", "hour_cos1", "hour_sin2", "hour_cos2"]
    tr = tr.tail(24 * 42)                                    # 6 weeks keeps the fit tractable
    t0 = time.time()
    res = SARIMAX(tr.consumption_m3.to_numpy(), exog=tr[exog].to_numpy(), order=(1, 0, 0),
                  seasonal_order=(1, 1, 1, 24)).fit(disp=False)
    # forecasts the whole test week from the end of training (1..168h ahead - generous to SARIMAX)
    fc = res.forecast(steps=len(te), exog=te[exog].to_numpy())
    m = evaluate(te.consumption_m3, fc)
    lgbm_zone = P[(P.family == "lightgbm") & (P.zone_id == z) & (P.fold == N_FOLDS - 1)].ape.mean()
    print(f"SARIMAX ({z}): MAPE {m['mape']:.2f}%  fit {time.time() - t0:.0f}s   vs LightGBM same zone/week: "
          f"{lgbm_zone:.2f}%")
except ImportError:
    print("statsmodels not installed - skipping the SARIMAX benchmark")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Decision
# MAGIC
# MAGIC The tables above are the evidence. On the synthetic dev data (4-week backtest) the result was:
# MAGIC
# MAGIC | Family | MAPE | Holiday MAPE | Event MAPE |
# MAGIC |---|---|---|---|
# MAGIC | lightgbm | **~3.7 %** | **~8 %** | **~1.5 %** |
# MAGIC | ridge_fourier | ~5.1 % | ~17 % | ~4 % |
# MAGIC | seasonal_naive_168 | ~5.3 % | ~15 % | ~7 % |
# MAGIC | seasonal_naive_48 | ~12 % | ~15 % | ~5 % |
# MAGIC
# MAGIC * **Both naive baselines are beaten** → the features carry real signal. Note how bad the 48h
# MAGIC   naive is: "two days ago" crosses weekday/weekend boundaries — weekly structure dominates.
# MAGIC * **LightGBM wins overall and by the widest margin on holidays and events**, where effects are
# MAGIC   non-linear and interact with the time of day. It is also the most *stable* across folds.
# MAGIC * **Ridge-Fourier** is competitive on ordinary days and on peak hours, and fully explainable —
# MAGIC   a good fallback, but it misreads holidays.
# MAGIC * **SARIMAX** is slower, needs one model per zone and is not more accurate → not pursued.
# MAGIC
# MAGIC **Chosen for production:** the training pipeline keeps **`seasonal_naive_168`, `ridge_fourier`
# MAGIC and `lightgbm`** as candidates (`project_config.yml > model.candidates`) and picks the best on
# MAGIC a fresh holdout at every retrain. LightGBM is expected to win; the naive model stays as a
# MAGIC permanent sanity check, and Ridge as a transparent fallback. The candidate list and feature set
# MAGIC are what get committed — the model *artifact* is always retrained per environment.
