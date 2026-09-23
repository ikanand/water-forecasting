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


# ----------------------------------------------------------------- tables ----
def validate_consumption(df, q: QualityConfig, zones, expected_days: pd.DatetimeIndex, out: ValidationOutcome):
    t = "consumption"
    df = _split(df, df[["zone_id", "reading_ts", "consumption_m3"]].isna().any(axis=1), out, t, "null_value")
    df = _split(df, ~df.zone_id.isin(zones), out, t, "unknown_zone")
    df = _split(
        df,
        (df.consumption_m3 < q.consumption_min_m3) | (df.consumption_m3 > q.consumption_max_m3),
        out,
        t,
        "out_of_range",
    )
    df = _dedupe(df, ["zone_id", "reading_ts"], out, t)

    # completeness on the clean rows - a batch-level, blocking check
    expected = len(zones) * len(expected_days) * 24
    in_window = df.reading_ts.dt.normalize().isin(expected_days)
    present = df[in_window].drop_duplicates(["zone_id", "reading_ts"]).shape[0]
    missing = max(expected - present, 0)
    pct = 100 * missing / expected if expected else 0.0
    out.results.append(
        CheckResult(
            "hourly_completeness",
            t,
            "error",
            pct <= q.max_missing_hours_pct,
            missing,
            f"{missing}/{expected} hours missing ({pct:.2f}%, limit {q.max_missing_hours_pct}%)",
        )
    )
    out.clean[t] = df


def validate_weather_obs(df, q: QualityConfig, zones, out: ValidationOutcome):
    t = "weather_obs"
    df = _split(df, df[["zone_id", "obs_ts", "temperature_c"]].isna().any(axis=1), out, t, "null_value")
    df = _split(
        df, (df.temperature_c < q.temperature_min_c) | (df.temperature_c > q.temperature_max_c), out, t, "out_of_range"
    )
    out.clean[t] = _dedupe(df, ["zone_id", "obs_ts"], out, t)


def validate_weather_fcst(df, q: QualityConfig, zones, target_day: pd.Timestamp | None, out: ValidationOutcome):
    """The forecast issued the day before `target_day` must cover all 24 hours of it, for every zone."""
    t = "weather_fcst"
    df = _split(df, df[["zone_id", "issued_at", "target_ts", "temperature_c"]].isna().any(axis=1), out, t, "null_value")
    df = _split(
        df, (df.temperature_c < q.temperature_min_c) | (df.temperature_c > q.temperature_max_c), out, t, "out_of_range"
    )
    df = _dedupe(df, ["zone_id", "issued_at", "target_ts"], out, t)
    if target_day is not None:
        target_day = pd.Timestamp(target_day).normalize()
        issued = target_day - pd.Timedelta(days=1)
        cov = df[(df.issued_at == issued) & (df.target_ts.dt.normalize() == target_day)]
        per_zone = cov.groupby("zone_id").target_ts.nunique().reindex(zones, fill_value=0)
        short = per_zone[per_zone < 24]
        out.results.append(
            CheckResult(
                "forecast_horizon_coverage",
                t,
                "error",
                short.empty,
                int((24 - short).sum()),
                "ok" if short.empty else f"missing forecast hours for {target_day.date()}: {short.to_dict()}",
            )
        )
    out.clean[t] = df


def validate_calendar(holidays: pd.DataFrame, events: pd.DataFrame, zones, out: ValidationOutcome):
    out.clean["holidays"] = _dedupe(holidays, ["holiday_date"], out, "holidays")
    ev = _split(
        events, (events.end_hour < events.start_hour) | ~events.zone_id.isin(zones), out, "events", "invalid_event"
    )
    out.clean["events"] = _dedupe(ev, ["event_id"], out, "events")


def run_all(
    bronze: dict[str, pd.DataFrame],
    q: QualityConfig,
    zones: list[str],
    consumption_days: pd.DatetimeIndex,
    forecast_target_day: pd.Timestamp | None,
) -> ValidationOutcome:
    out = ValidationOutcome()
    validate_consumption(bronze["consumption"], q, zones, consumption_days, out)
    validate_weather_obs(bronze["weather_obs"], q, zones, out)
    validate_weather_fcst(bronze["weather_fcst"], q, zones, forecast_target_day, out)
    validate_calendar(bronze["holidays"], bronze["events"], zones, out)
    return out
