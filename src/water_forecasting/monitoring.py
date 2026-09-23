"""Reconciliation of forecasts against actuals (pure pandas).

Forecasts and actuals live in separate tables because they arrive at different times;
this joins them once the day has closed. The latest issued forecast per hour is the one scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reconcile(forecasts: pd.DataFrame, actuals: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    day = pd.Timestamp(day).normalize()
    f = forecasts[forecasts.target_ts.dt.normalize() == day]
    f = f.sort_values("issued_at").drop_duplicates(["zone_id", "target_ts"], keep="last")
    a = actuals.rename(columns={"reading_ts": "target_ts", "consumption_m3": "actual_m3"})
    df = f.merge(a[["zone_id", "target_ts", "actual_m3"]], on=["zone_id", "target_ts"], how="inner")
    df["error_m3"] = df.predicted_m3 - df.actual_m3
    df["abs_pct_error"] = np.where(df.actual_m3 > 0, (df.error_m3.abs() / df.actual_m3) * 100, np.nan)
    df["forecast_date"] = day
    return df[
        [
            "forecast_date",
            "zone_id",
            "target_ts",
            "issued_at",
            "model_version",
            "predicted_m3",
            "actual_m3",
            "error_m3",
            "abs_pct_error",
        ]
    ]


def daily_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    """One row per zone plus an ALL row: MAPE, WAPE, bias, peak error."""

    def agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "hours_scored": len(g),
                "mape": g.abs_pct_error.mean(),
                "wape": g.error_m3.abs().sum() / g.actual_m3.sum() * 100,
                "bias_pct": g.error_m3.sum() / g.actual_m3.sum() * 100,
                "max_abs_pct_error": g.abs_pct_error.max(),
            }
        )

    if hourly.empty:
        return pd.DataFrame()
    per_zone = hourly.groupby(["forecast_date", "zone_id"]).apply(agg, include_groups=False).reset_index()
    total = hourly.groupby("forecast_date").apply(agg, include_groups=False).reset_index().assign(zone_id="ALL")
    return pd.concat([per_zone, total], ignore_index=True)
