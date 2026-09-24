"""Data quality between bronze and silver.

Two kinds of outcome:
* row-level problems (nulls, impossible values, duplicates) -> the row goes to quarantine
  with a reason code; the rest of the batch continues. Severity "warn".
* batch-level problems (too many missing hours, no weather forecast for tomorrow)
  -> severity "error": the gate fails and NOTHING is written to silver, so the job
  stops before features/forecast. Better yesterday's forecast than a confidently wrong one.

Pure pandas so every rule is unit-tested in CI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import pandas as pd

from water_forecasting.config import QualityConfig


@dataclass
class CheckResult:
    check_name: str
    table_name: str
    severity: str  # "error" blocks the pipeline, "warn" only reports
    passed: bool
    failed_rows: int
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationOutcome:
    clean: dict[str, pd.DataFrame] = field(default_factory=dict)
    quarantine: list[pd.DataFrame] = field(default_factory=list)
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "error")

    def quarantine_frame(self) -> pd.DataFrame:
        cols = ["table_name", "reason", "record_json"]
        return pd.concat(self.quarantine, ignore_index=True) if self.quarantine else pd.DataFrame(columns=cols)


def _to_quarantine(rows: pd.DataFrame, table: str, reason: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "table_name": table,
            "reason": reason,
            "record_json": [json.dumps(r, default=str) for r in rows.to_dict(orient="records")],
        }
    )


def _split(df: pd.DataFrame, bad: pd.Series, out: ValidationOutcome, table: str, reason: str, severity="warn"):
    n = int(bad.sum())
    if n:
        out.quarantine.append(_to_quarantine(df[bad], table, reason))
    out.results.append(CheckResult(reason, table, severity, n == 0, n, f"{n} row(s) quarantined" if n else "ok"))
    return df[~bad]


def _dedupe(df: pd.DataFrame, keys: list[str], out: ValidationOutcome, table: str) -> pd.DataFrame:
    """Keep the most recently ingested version of each key; quarantine the rest."""
    order = [c for c in ["_ingested_at"] if c in df.columns]
    df = df.sort_values(keys + order)
    dup = df.duplicated(subset=keys, keep="last")
    return _split(df, dup, out, table, "duplicate_key")


def frozen_readings(df: pd.DataFrame, min_hours: int) -> pd.Series:
    """True for readings inside a run of >= min_hours identical consecutive hourly values (stuck meter).
    A stuck value sits inside the valid range, so the range check cannot see it."""
    s = df.sort_values("timestamp_local")
    same = s.demand_m3h.diff().eq(0) & s.timestamp_local.diff().eq(pd.Timedelta(hours=1))
    run = (~same).cumsum()
    run_len = s.groupby(run).demand_m3h.transform("size")
    return (run_len >= min_hours).reindex(df.index)


# ----------------------------------------------------------------- tables ----
def validate_consumption(df, q: QualityConfig, expected_days: pd.DatetimeIndex, out: ValidationOutcome):
    t = "consumption"
    df = _split(df, df[["timestamp_local", "demand_m3h"]].isna().any(axis=1), out, t, "null_value")
    df = _split(
        df,
        (df.demand_m3h < q.demand_min_m3h) | (df.demand_m3h > q.demand_max_m3h),
        out,
        t,
        "out_of_range",
    )
    df = _dedupe(df, ["timestamp_local"], out, t)
    df = _split(df, frozen_readings(df, q.frozen_min_hours), out, t, "frozen_value")

    # completeness on the clean rows - a batch-level, blocking check
    expected = len(expected_days) * 24
    in_window = df.timestamp_local.dt.normalize().isin(expected_days)
    present = df[in_window].drop_duplicates(["timestamp_local"]).shape[0]
    missing = max(expected - present, 0)
    pct = 100 * missing / expected if expected else 0.0
    # An absolute floor keeps a single isolated telemetry gap from blocking a normal one-day batch
    # (24h * 2% = 0.48h, so any single missing hour would otherwise always block). A real outage still
    # blocks: max_missing_hours_abs hours is a small fraction of a multi-day backfill batch.
    allowed_hours = max(q.max_missing_hours_abs, expected * q.max_missing_hours_pct / 100)
    out.results.append(
        CheckResult(
            "hourly_completeness",
            t,
            "error",
            missing <= allowed_hours,
            missing,
            f"{missing}/{expected} hours missing ({pct:.2f}%, limit {q.max_missing_hours_pct}% "
            f"or {q.max_missing_hours_abs}h)",
        )
    )
    out.clean[t] = df


def validate_weather_obs(df, q: QualityConfig, out: ValidationOutcome):
    t = "weather_obs"
    df = _split(df, df[["timestamp_local", "temperature_c"]].isna().any(axis=1), out, t, "null_value")
    df = _split(
        df, (df.temperature_c < q.temperature_min_c) | (df.temperature_c > q.temperature_max_c), out, t, "out_of_range"
    )
    out.clean[t] = _dedupe(df, ["timestamp_local"], out, t)


def validate_weather_fcst(df, q: QualityConfig, target_day: pd.Timestamp | None, out: ValidationOutcome):
    """The forecast vintage issued the day before `target_day` must cover all 24 hours of it."""
    t = "weather_fcst"
    df = _split(
        df,
        df[["forecast_issued_local", "timestamp_local", "forecast_temperature_c"]].isna().any(axis=1),
        out,
        t,
        "null_value",
    )
    df = _split(
        df,
        (df.forecast_temperature_c < q.temperature_min_c) | (df.forecast_temperature_c > q.temperature_max_c),
        out,
        t,
        "out_of_range",
    )
    df = _dedupe(df, ["forecast_issued_local", "timestamp_local"], out, t)
    if target_day is not None:
        target_day = pd.Timestamp(target_day).normalize()
        issued = target_day - pd.Timedelta(days=1)
        cov = df[
            (df.forecast_issued_local.dt.normalize() == issued) & (df.timestamp_local.dt.normalize() == target_day)
        ]
        hours = cov.timestamp_local.nunique()
        out.results.append(
            CheckResult(
                "forecast_horizon_coverage",
                t,
                "error",
                hours >= 24,
                max(24 - hours, 0),
                "ok" if hours >= 24 else f"missing forecast hours for {target_day.date()}: {24 - hours} of 24",
            )
        )
    out.clean[t] = df


def validate_calendar(df: pd.DataFrame, out: ValidationOutcome):
    t = "calendar"
    df = _split(df, df["date"].isna(), out, t, "null_value")
    out.clean[t] = _dedupe(df, ["date"], out, t)


def run_all(
    bronze: dict[str, pd.DataFrame],
    q: QualityConfig,
    consumption_days: pd.DatetimeIndex,
    forecast_target_day: pd.Timestamp | None,
) -> ValidationOutcome:
    out = ValidationOutcome()
    validate_consumption(bronze["consumption"], q, consumption_days, out)
    validate_weather_obs(bronze["weather_obs"], q, out)
    validate_weather_fcst(bronze["weather_fcst"], q, forecast_target_day, out)
    validate_calendar(bronze["calendar"], out)
    return out
