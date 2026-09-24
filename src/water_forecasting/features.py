"""Feature engineering for day-ahead hourly demand.

The one rule that matters: a feature for target hour T may only use information that
exists at *issue time* (05:00 the day before). Concretely:

* consumption lags are >= 48h (the newest complete day at issue time is D-1),
* weather comes from the *forecast* vintage issued the day before - never the observed
  weather - so training sees exactly what serving will see (no train/serve skew),
* calendar (holidays, Ramadan, school breaks, planned events) is known in advance, so
  it is fine as-is.

Single site (city-wide aggregate demand) - one row per hour, no zone dimension.

Pure pandas; the Spark wrapper in lakehouse.py writes the result to the UC feature table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS_H = [48, 72, 168, 336, 504]
EVENT_TYPES = ["structural", "operational", "weather"]
COOLING_KNOT_C = 24.4  # hinge of the demand/temperature response (notebooks/01_eda.ipynb, section 10)
EPOCH = pd.Timestamp("2024-01-01")
TIME_COL = "t_days"  # days since EPOCH; used by growth-aware models to fit/extrapolate the trend

BASE_FEATURES = [
    *[f"lag_{k}h" for k in LAGS_H],
    "roll_mean_7d",
    "roll_std_7d",
    "same_hour_mean_3w",
    "fc_temperature_c",
    "fc_humidity_pct",
    "fc_precipitation_mm",
    "fc_temp_max_day",
    "fc_precip_total_day",
    "fc_cooling_degree",
    "fc_heating_degree",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "is_day_off",
    "day_before_holiday",
    "day_after_holiday",
    "is_ramadan",
    "ramadan_day",
    "is_school_break",
    *[f"event_{et}" for et in EVENT_TYPES],
    "hour_sin1",
    "hour_cos1",
    "hour_sin2",
    "hour_cos2",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    # EDA-driven additions (validated in notebooks/02_modelling.ipynb as feature set "v2")
    "fc_tmean_day",
    "fc_cdd24_day",
    "fc_cdd24_hour",
    "fc_tmean_prev_day",
    "obs_tmean_d2",
    "is_eid",
]
LEVEL_FEATURES = [*[f"lag_{k}h" for k in LAGS_H], "roll_mean_7d", "roll_std_7d", "same_hour_mean_3w"]
KEYS = ["target_ts"]
LABEL = "demand_m3h"


def feature_columns() -> list[str]:
    return BASE_FEATURES


def model_columns() -> list[str]:
    """What every model receives: the features plus the trend index."""
    return BASE_FEATURES + [TIME_COL]


def hourly_targets(first_day, last_day) -> pd.DataFrame:
    hours = pd.date_range(
        pd.Timestamp(first_day).normalize(), pd.Timestamp(last_day).normalize() + pd.Timedelta(hours=23), freq="h"
    )
    return pd.DataFrame({"target_ts": hours})


def _lag_features(targets: pd.DataFrame, consumption: pd.DataFrame, min_lag_h: int) -> pd.DataFrame:
    if min(LAGS_H) < min_lag_h:
        raise ValueError(f"Lag {min(LAGS_H)}h would leak: minimum allowed is {min_lag_h}h")
    s = consumption.set_index("timestamp_local")[LABEL].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    full = pd.date_range(
        min(s.index.min(), targets.target_ts.min() - pd.Timedelta(hours=max(LAGS_H))),
        max(s.index.max(), targets.target_ts.max()),
        freq="h",
    )
    s = s.reindex(full)

    out = pd.DataFrame({"target_ts": full})
    for k in LAGS_H:
        out[f"lag_{k}h"] = s.shift(k).to_numpy()
    base = s.shift(min_lag_h)  # newest value visible at issue time
    out["roll_mean_7d"] = base.rolling(168, min_periods=120).mean().to_numpy()
    out["roll_std_7d"] = base.rolling(168, min_periods=120).std().to_numpy()
    out["same_hour_mean_3w"] = ((s.shift(168) + s.shift(336) + s.shift(504)) / 3).to_numpy()
    return targets.merge(out, on="target_ts", how="left")


def _weather_features(df: pd.DataFrame, weather_fcst: pd.DataFrame) -> pd.DataFrame:
    """Use the forecast vintage issued the day before the target day (never observed weather)."""
    fc = weather_fcst[
        weather_fcst.forecast_issued_local.dt.normalize()
        == weather_fcst.timestamp_local.dt.normalize() - pd.Timedelta(days=1)
    ]
    fc = (
        fc.sort_values("forecast_issued_local")
        .drop_duplicates("timestamp_local", keep="last")
        .rename(
            columns={
                "timestamp_local": "target_ts",
                "forecast_temperature_c": "fc_temperature_c",
                "forecast_relative_humidity_pct": "fc_humidity_pct",
                "forecast_precipitation_mm": "fc_precipitation_mm",
            }
        )
    )
    fc_cols = ["target_ts", "fc_temperature_c", "fc_humidity_pct", "fc_precipitation_mm"]
    df = df.merge(fc[fc_cols], on="target_ts", how="left")
    day = df.target_ts.dt.normalize()
    grp = df.groupby(day)
    df["fc_temp_max_day"] = grp.fc_temperature_c.transform("max")
    df["fc_precip_total_day"] = grp.fc_precipitation_mm.transform("sum", min_count=1)
    df["fc_cooling_degree"] = (df.fc_temperature_c - 20).clip(lower=0)
    df["fc_heating_degree"] = (12 - df.fc_temperature_c).clip(lower=0)

    fc_day_mean = fc.groupby(fc.target_ts.dt.normalize()).fc_temperature_c.mean()
    df["fc_tmean_day"] = day.map(fc_day_mean)
    df["fc_cdd24_day"] = (df.fc_tmean_day - COOLING_KNOT_C).clip(lower=0)
    df["fc_cdd24_hour"] = (df.fc_temperature_c - COOLING_KNOT_C).clip(lower=0)
    # forecast for the day before the target (issued two days before it): known at issue time
    df["fc_tmean_prev_day"] = (day - pd.Timedelta(days=1)).map(fc_day_mean)
    return df


def _observed_weather_features(df: pd.DataFrame, weather_obs: pd.DataFrame) -> pd.DataFrame:
    """Observed temperature of D-1 (the newest complete day at issue time for target day D+1)."""
    obs_day_mean = weather_obs.groupby(weather_obs.timestamp_local.dt.normalize()).temperature_c.mean()
    df["obs_tmean_d2"] = (df.target_ts.dt.normalize() - pd.Timedelta(days=2)).map(obs_day_mean)
    return df


def _calendar_features(df: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    ts = df.target_ts
    day = ts.dt.normalize()
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df.day_of_week >= 5).astype(int)

    join_cols = [
        "date",
        "is_public_holiday",
        "holiday_name",
        "is_ramadan",
        "ramadan_day",
        "is_school_break",
        "event_type",
    ]
    cal = calendar.copy()
    if len(cal):
        cal["date"] = pd.to_datetime(cal.date).dt.normalize()
        cal = cal[join_cols].drop_duplicates("date")
    else:
        cal = pd.DataFrame(columns=join_cols)

    hol = set(cal.loc[cal.is_public_holiday.fillna(False), "date"]) if len(cal) else set()
    df["is_holiday"] = day.isin(hol).astype(int)
    df["is_day_off"] = ((df.is_weekend == 1) | (df.is_holiday == 1)).astype(int)
    df["day_before_holiday"] = (day + pd.Timedelta(days=1)).isin(hol).astype(int)
    df["day_after_holiday"] = (day - pd.Timedelta(days=1)).isin(hol).astype(int)

    df["_day"] = day
    df = df.merge(cal.drop(columns=["is_public_holiday"]), left_on="_day", right_on="date", how="left").drop(
        columns=["_day", "date"]
    )
    df["is_ramadan"] = df.is_ramadan.fillna(False).astype(int)
    df["ramadan_day"] = df.ramadan_day.fillna(0).astype(float)
    df["is_school_break"] = df.is_school_break.fillna(False).astype(int)
    df["is_eid"] = df.holiday_name.fillna("").astype(str).str.contains("Eid").astype(int)
    for et in EVENT_TYPES:
        df[f"event_{et}"] = (df.event_type == et).astype(int)
    df = df.drop(columns=["event_type", "holiday_name"])

    for k in (1, 2):
        df[f"hour_sin{k}"] = np.sin(2 * np.pi * k * df.hour / 24)
        df[f"hour_cos{k}"] = np.cos(2 * np.pi * k * df.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.day_of_week / 7)
    doy = ts.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def build_features(
    targets: pd.DataFrame,
    consumption: pd.DataFrame,
    weather_fcst: pd.DataFrame,
    calendar: pd.DataFrame,
    weather_obs: pd.DataFrame,
    min_lag_h: int = 48,
) -> pd.DataFrame:
    """One row per target_ts with every model column. No label - see attach_label().

    Inputs must reach far enough back: consumption >= 21 days before the first target (lags),
    weather_fcst >= 1 day before it, weather_obs >= 2 days before it.
    """
    df = _lag_features(targets[KEYS].copy(), consumption, min_lag_h)
    df = _weather_features(df, weather_fcst)
    df = _observed_weather_features(df, weather_obs)
    df = _calendar_features(df, calendar)
    df[TIME_COL] = (df.target_ts - EPOCH) / pd.Timedelta(days=1)
    cols = model_columns()
    df[cols] = df[cols].astype(float)
    return df[KEYS + cols].sort_values(KEYS).reset_index(drop=True)


def attach_label(features: pd.DataFrame, consumption: pd.DataFrame) -> pd.DataFrame:
    lab = consumption.rename(columns={"timestamp_local": "target_ts"})[KEYS + [LABEL]].drop_duplicates(
        KEYS, keep="last"
    )
    return features.merge(lab, on=KEYS, how="inner")
