# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Exploratory data analysis — hourly water demand
# MAGIC
# MAGIC **Purpose:** understand the data before choosing a model. This notebook is *documentation*:
# MAGIC it is committed to git as the audit trail of *why* the model looks the way it does, but it is
# MAGIC **not** deployed as a job. The production logic lives in `src/water_forecasting/`.
# MAGIC
# MAGIC Runs on the **dev** catalog (a sample of prod, refreshed quarterly). Falls back to generating
# MAGIC synthetic data locally if there is no Spark session.
# MAGIC
# MAGIC | Question | Why it matters for the model |
# MAGIC |---|---|
# MAGIC | What are the daily / weekly cycles? | Hour + day-of-week features, seasonal-naive baseline |
# MAGIC | How strong is the weather effect? | Include weather *forecasts* as features |
# MAGIC | Do holidays and events move demand? | Calendar + event features |
# MAGIC | How far back does autocorrelation reach? | Which lags to use (>= 48h because of day-ahead) |

# COMMAND ----------

# MAGIC %pip install -e ..
# MAGIC %restart_python

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from water_forecasting import synthetic
from water_forecasting.config import ProjectConfig, Tables

CATALOG_ENV = "dev"
cfg = ProjectConfig.from_yaml("../project_config.yml", env=CATALOG_ENV)

try:
    spark  # noqa: B018 - defined on Databricks
    read = lambda t: spark.table(cfg.table(t)).toPandas()  # noqa: E731
    cons, wx_obs, hol, ev = (read(t) for t in (Tables.SILVER_CONSUMPTION, Tables.SILVER_WEATHER_OBS,
                                                Tables.SILVER_HOLIDAYS, Tables.SILVER_EVENTS))
    source = f"Unity Catalog: {cfg.catalog}.silver"
except NameError:
    raw = synthetic.generate_range(cfg.history_start, cfg.history_end, cfg.zones, cfg.source.seed, backfill=True)
    cons, wx_obs, hol, ev = raw["consumption"], raw["weather_obs"], raw["holidays"], raw["events"]
    source = "local synthetic generator"

try:
    display  # noqa: B018
except NameError:
    display = print  # noqa: A001

print(f"Source: {source}")
print(f"{len(cons):,} hourly readings, {cons.zone_id.nunique()} zones, "
      f"{cons.reading_ts.min():%Y-%m-%d} .. {cons.reading_ts.max():%Y-%m-%d}")

# COMMAND ----------

# MAGIC %md ## 1. Data overview and quality

# COMMAND ----------

overview = cons.groupby("zone_id").agg(
    hours=("consumption_m3", "size"),
    mean_m3=("consumption_m3", "mean"),
    p05_m3=("consumption_m3", lambda s: s.quantile(0.05)),
    p95_m3=("consumption_m3", lambda s: s.quantile(0.95)),
    missing_values=("consumption_m3", lambda s: s.isna().sum()),
).round(1)
expected = (cons.reading_ts.max() - cons.reading_ts.min()) / pd.Timedelta(hours=1) + 1
overview["completeness_pct"] = (overview.hours / expected * 100).round(2)
display(overview.reset_index())

# COMMAND ----------

# MAGIC %md ## 2. Seasonality: daily shape, weekly rhythm, annual trend

# COMMAND ----------

df = cons.merge(wx_obs.rename(columns={"obs_ts": "reading_ts"}), on=["zone_id", "reading_ts"], how="left")
df["hour"] = df.reading_ts.dt.hour
df["dow"] = df.reading_ts.dt.dayofweek
df["day_type"] = np.where(df.dow >= 5, "weekend", "weekday")
hol_days = set(pd.to_datetime(hol.holiday_date).dt.normalize())
df["is_holiday"] = df.reading_ts.dt.normalize().isin(hol_days)

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
df.groupby(["hour", "day_type"]).consumption_m3.mean().unstack().plot(ax=axes[0], title="Mean demand by hour")
df.groupby("dow").consumption_m3.mean().plot.bar(ax=axes[1], title="Mean demand by day of week (0=Mon)")
df.set_index("reading_ts").groupby("zone_id").consumption_m3.resample("W").mean().unstack(0).plot(
    ax=axes[2], title="Weekly mean demand per zone")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Finding:** two clear daily peaks (morning, evening), a different shape at weekends (later,
# MAGIC flatter morning), and an annual cycle that follows temperature.
# MAGIC → hour-of-day, day-of-week and Fourier terms are needed; a *same-hour-last-week* baseline
# MAGIC will already be decent.

# COMMAND ----------

# MAGIC %md ## 3. Weather effect

# COMMAND ----------

daily = df.groupby([df.zone_id, df.reading_ts.dt.normalize()]).agg(
    demand=("consumption_m3", "sum"), t_max=("temperature_c", "max"), rain=("precipitation_mm", "sum"))
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for z, g in daily.groupby(level=0):
    axes[0].scatter(g.t_max, g.demand, s=6, alpha=0.5, label=z)
axes[0].set(title="Daily demand vs max temperature", xlabel="max temp (C)", ylabel="m3/day")
axes[0].legend()
daily.assign(rainy=daily.rain > 1).groupby("rainy").demand.mean().plot.bar(ax=axes[1], title="Rainy vs dry days")
plt.tight_layout()
plt.show()

display(daily.reset_index().groupby("zone_id")[["demand", "t_max", "rain"]].corr().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC **Finding:** demand is flat below ~20 °C and rises steeply above it (a *hinge*, not a straight
# MAGIC line) → a `cooling_degree = max(T-20, 0)` feature helps linear models; trees find it alone.
# MAGIC Rain reduces demand (less garden watering).
# MAGIC
# MAGIC ⚠️ At forecast time we only have the **weather forecast**, not the observed weather. Training
# MAGIC therefore uses the forecast issued the day before — otherwise the model would learn from
# MAGIC perfect weather it will never see in production (train/serve skew).

# COMMAND ----------

# MAGIC %md ## 4. Holidays and events

# COMMAND ----------

hd = df[df.dow < 5].groupby(["is_holiday", "hour"]).consumption_m3.mean().unstack(0)
hd.columns = ["normal weekday", "holiday"]
ax = hd.plot(figsize=(9, 4), title="Weekday vs public holiday profile")
plt.show()

if len(ev):
    ev_hours = ev.loc[ev.index.repeat(ev.end_hour - ev.start_hour + 1)].copy()
    ev_hours["reading_ts"] = pd.to_datetime(ev_hours.event_date) + pd.to_timedelta(
        ev_hours.start_hour + ev_hours.groupby(level=0).cumcount(), unit="h")
    flagged = df.merge(ev_hours[["zone_id", "reading_ts", "expected_attendance"]], how="left")
    uplift = flagged.groupby([flagged.expected_attendance.notna(), "hour"]).consumption_m3.mean().unstack(0)
    print("Mean uplift during event hours:",
          f"{(uplift[True] / uplift[False] - 1).dropna().mean() * 100:.1f}%")

# COMMAND ----------

# MAGIC %md ## 5. Autocorrelation — which lags carry signal?

# COMMAND ----------

s = cons[cons.zone_id == cfg.zones[0]].set_index("reading_ts").consumption_m3.asfreq("h")
lags = [1, 24, 48, 72, 168, 336, 504]
acf = pd.Series({lag: s.autocorr(lag) for lag in lags}, name="autocorrelation").round(3)
display(acf.rename_axis("lag_hours").reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC **Finding:** weekly lags (168h, 336h, 504h) carry the most signal, and short lags (1h, 24h)
# MAGIC are strong too — but **we cannot use lags below 48h**. The forecast is issued at 05:00 for all
# MAGIC of tomorrow, so the newest complete day is D-1: the smallest honest lag is **48h**. Using
# MAGIC lag 1h/24h would look great in a notebook and be impossible in production (leakage).
# MAGIC
# MAGIC ### EDA conclusions → feature set (implemented in `src/water_forecasting/features.py`)
# MAGIC * lags 48 / 72 / 168 / 336 / 504 h, 7-day rolling mean & std, 3-week same-hour mean
# MAGIC * weather **forecast** (temp, humidity, rain) + daily max temp, cooling/heating degrees
# MAGIC * hour, day-of-week, month, weekend, holiday, day-before/after holiday, Fourier terms
# MAGIC * event active + attendance
# MAGIC * one-hot zone → **one global model** across zones (more data per model, simpler ops)
