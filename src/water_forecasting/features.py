"""Feature engineering for day-ahead hourly demand.

The one rule that matters: a feature for target hour T may only use information that
exists at *issue time* (05:00 the day before). Concretely:

* consumption lags are >= 48h (the newest complete day at issue time is D-1),
* weather comes from the *forecast* issued at 00:00 the day before - never the observed
  weather - so training sees exactly what serving will see (no train/serve skew),
* calendar and events are known in advance, so they are fine as-is.

Pure pandas; the Spark wrapper in lakehouse.py writes the result to the UC feature table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS_H = [48, 72, 168, 336, 504]

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
    "event_active",
    "event_attendance_k",
    "hour_sin1",
    "hour_cos1",
    "hour_sin2",
    "hour_cos2",
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
]
KEYS = ["zone_id", "target_ts"]
LABEL = "consumption_m3"


def zone_columns(zones: list[str]) -> list[str]:
    return [f"zone_{z}" for z in zones]


def feature_columns(zones: list[str]) -> list[str]:
    return BASE_FEATURES + zone_columns(zones)


def hourly_targets(zones: list[str], first_day, last_day) -> pd.DataFrame:
    hours = pd.date_range(
        pd.Timestamp(first_day).normalize(), pd.Timestamp(last_day).normalize() + pd.Timedelta(hours=23), freq="h"
    )
    return pd.MultiIndex.from_product([zones, hours], names=KEYS).to_frame(index=False)


def _lag_features(targets: pd.DataFrame, consumption: pd.DataFrame, min_lag_h: int) -> pd.DataFrame:
    if min(LAGS_H) < min_lag_h:
        raise ValueError(f"Lag {min(LAGS_H)}h would leak: minimum allowed is {min_lag_h}h")
    wide = consumption.pivot_table(index="reading_ts", columns="zone_id", values=LABEL, aggfunc="last")
    full = pd.date_range(
        min(wide.index.min(), targets.target_ts.min() - pd.Timedelta(hours=max(LAGS_H))),
        max(wide.index.max(), targets.target_ts.max()),
        freq="h",
    )
    wide = wide.reindex(full)

    blocks = {f"lag_{k}h": wide.shift(k) for k in LAGS_H}
    base = wide.shift(min_lag_h)  # newest value visible at issue time
    blocks["roll_mean_7d"] = base.rolling(168, min_periods=120).mean()
    blocks["roll_std_7d"] = base.rolling(168, min_periods=120).std()
    blocks["same_hour_mean_3w"] = (wide.shift(168) + wide.shift(336) + wide.shift(504)) / 3

    long = pd.concat({name: b.stack(future_stack=True) for name, b in blocks.items()}, axis=1)
    long.index.names = ["target_ts", "zone_id"]
    return targets.merge(long.reset_index(), on=KEYS, how="left")


def _weather_features(df: pd.DataFrame, weather_fcst: pd.DataFrame) -> pd.DataFrame:
    fc = weather_fcst[weather_fcst.issued_at == weather_fcst.target_ts.dt.normalize() - pd.Timedelta(days=1)]
    fc = fc.drop_duplicates(KEYS, keep="last").rename(
        columns={
            "temperature_c": "fc_temperature_c",
            "humidity_pct": "fc_humidity_pct",
            "precipitation_mm": "fc_precipitation_mm",
        }
    )
    df = df.merge(fc[KEYS + ["fc_temperature_c", "fc_humidity_pct", "fc_precipitation_mm"]], on=KEYS, how="left")
    day = df.target_ts.dt.normalize()
    grp = df.groupby([df.zone_id, day])
    df["fc_temp_max_day"] = grp.fc_temperature_c.transform("max")
    df["fc_precip_total_day"] = grp.fc_precipitation_mm.transform("sum", min_count=1)
    df["fc_cooling_degree"] = (df.fc_temperature_c - 20).clip(lower=0)
    df["fc_heating_degree"] = (12 - df.fc_temperature_c).clip(lower=0)
    return df


def _calendar_features(df: pd.DataFrame, holidays: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    ts = df.target_ts
    day = ts.dt.normalize()
    hol = set(pd.to_datetime(holidays.holiday_date).dt.normalize()) if len(holidays) else set()
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df.day_of_week >= 5).astype(int)
    df["is_holiday"] = day.isin(hol).astype(int)
    df["is_day_off"] = ((df.is_weekend == 1) | (df.is_holiday == 1)).astype(int)
    df["day_before_holiday"] = (day + pd.Timedelta(days=1)).isin(hol).astype(int)
    df["day_after_holiday"] = (day - pd.Timedelta(days=1)).isin(hol).astype(int)
    for k in (1, 2):
        df[f"hour_sin{k}"] = np.sin(2 * np.pi * k * df.hour / 24)
        df[f"hour_cos{k}"] = np.cos(2 * np.pi * k * df.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.day_of_week / 7)
    doy = ts.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # events -> one row per active (zone, hour)
    if len(events):
        ev = events.assign(event_date=pd.to_datetime(events.event_date).dt.normalize())
        ev = ev.loc[ev.index.repeat(ev.end_hour - ev.start_hour + 1)]
        ev["target_ts"] = ev.event_date + pd.to_timedelta(ev.start_hour + ev.groupby(level=0).cumcount(), unit="h")
        agg = ev.groupby(KEYS).expected_attendance.sum().div(1000).rename("event_attendance_k").reset_index()
        df = df.merge(agg, on=KEYS, how="left")
    else:
        df["event_attendance_k"] = np.nan
    df["event_attendance_k"] = df.event_attendance_k.fillna(0.0)
    df["event_active"] = (df.event_attendance_k > 0).astype(int)
    return df


def build_features(
    targets: pd.DataFrame,
    consumption: pd.DataFrame,
    weather_fcst: pd.DataFrame,
    holidays: pd.DataFrame,
    events: pd.DataFrame,
    zones: list[str],
    min_lag_h: int = 48,
) -> pd.DataFrame:
    """One row per (zone_id, target_ts) with every feature. No label - see attach_label()."""
    df = _lag_features(targets[KEYS].copy(), consumption, min_lag_h)
    df = _weather_features(df, weather_fcst)
    df = _calendar_features(df, holidays, events)
    for z in zones:
        df[f"zone_{z}"] = (df.zone_id == z).astype(int)
    cols = feature_columns(zones)
    df[cols] = df[cols].astype(float)
    return df[KEYS + cols].sort_values(KEYS).reset_index(drop=True)


def attach_label(features: pd.DataFrame, consumption: pd.DataFrame) -> pd.DataFrame:
    lab = consumption.rename(columns={"reading_ts": "target_ts"})[KEYS + [LABEL]].drop_duplicates(KEYS, keep="last")
    return features.merge(lab, on=KEYS, how="inner")
