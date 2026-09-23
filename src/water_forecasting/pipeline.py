"""Pipeline steps called by the job scripts: land -> bronze -> quality gate -> silver -> features
-> forecast -> reconcile/monitor, plus the quarterly prod -> lower-env refresh."""

from __future__ import annotations

import logging
import uuid

import pandas as pd

from water_forecasting import features as F
from water_forecasting import quality, synthetic
from water_forecasting.config import BRONZE_FLOW, ProjectConfig, Tables
from water_forecasting.decisions import needs_retrain
from water_forecasting.lakehouse import (
    append_pdf,
    ensure_feature_table,
    exists,
    merge_pdf,
    read_pdf,
    sim_today,
)
from water_forecasting.monitoring import daily_summary, reconcile

log = logging.getLogger(__name__)

SILVER_TARGETS = {  # validated frame -> (silver table, merge keys)
    "consumption": (Tables.SILVER_CONSUMPTION, ["zone_id", "reading_ts"]),
    "weather_obs": (Tables.SILVER_WEATHER_OBS, ["zone_id", "obs_ts"]),
    "weather_fcst": (Tables.SILVER_WEATHER_FCST, ["zone_id", "issued_at", "target_ts"]),
    "holidays": (Tables.SILVER_HOLIDAYS, ["holiday_date"]),
    "events": (Tables.SILVER_EVENTS, ["event_id"]),
}


# ------------------------------------------------------------ 1. landing ----
def _files_frames(cfg: ProjectConfig, start, end, backfill: bool) -> dict[str, pd.DataFrame]:
    """Replay YOUR generated CSVs (source.mode=files) with the same semantics as the synthetic feed."""
    p = cfg.files_path
    start, end = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    c = pd.read_csv(f"{p}/consumption.csv", parse_dates=["reading_ts"])
    o = pd.read_csv(f"{p}/weather_observations.csv", parse_dates=["obs_ts"])
    f = pd.read_csv(f"{p}/weather_forecasts.csv", parse_dates=["issued_at", "target_ts"])
    h = pd.read_csv(f"{p}/holidays.csv", parse_dates=["holiday_date"])
    e = pd.read_csv(f"{p}/events.csv", parse_dates=["event_date"])
    one = pd.Timedelta(days=1)
    return {
        "consumption": c[(c.reading_ts >= start) & (c.reading_ts < end)],
        "weather_obs": o[(o.obs_ts >= start) & (o.obs_ts < end)],
        "weather_fcst": f[(f.issued_at >= start + one) & (f.issued_at < end + one)],
        "holidays": h if backfill else h.iloc[0:0],
        "events": e if backfill else e.iloc[0:0],
    }


def _last_landed_day(spark, cfg: ProjectConfig) -> pd.Timestamp | None:
    days = []
    for t in (Tables.LANDING_CONSUMPTION, Tables.SILVER_CONSUMPTION):
        if exists(spark, cfg.table(t)):
            col = "reading_ts"
            m = spark.sql(f"SELECT max({col}) m FROM {cfg.table(t)}").first()["m"]
            if m is not None:
                days.append(pd.Timestamp(m).normalize())
    return max(days) if days else None


def land_source(spark, cfg: ProjectConfig, mode: str, fault: str = "none") -> str:
    """Pull from the (mock) source systems into landing. backfill = history; daily = the next day."""
    if mode == "backfill":
        if _last_landed_day(spark, cfg) is not None:
            log.info("Landing already populated - backfill skipped (idempotent).")
            return "skipped"
        start, end, backfill = cfg.history_start, cfg.history_end, True
    else:
        start = end = _last_landed_day(spark, cfg) + pd.Timedelta(days=1)
        backfill = False

    if cfg.source.mode == "synthetic":
        frames = synthetic.generate_range(
            start, end, cfg.zones, cfg.source.seed, cfg.source.inject_anomaly_rate, fault, backfill
        )
    else:
        frames = _files_frames(cfg, start, end, backfill)
    if frames["consumption"].empty:
        raise RuntimeError(f"Source returned no consumption for {start}..{end} (end of replay data?)")

    batch_id = f"{pd.Timestamp(end):%Y%m%d}-{uuid.uuid4().hex[:8]}"
    for key, (landing, _bronze) in BRONZE_FLOW.items():
        df = frames[key].assign(_batch_id=batch_id, _landed_at=pd.Timestamp.now(tz="UTC").tz_localize(None))
        append_pdf(spark, df, cfg.table(landing))
    log.info("Landed batch %s for %s..%s (%s consumption rows)", batch_id, start, end, len(frames["consumption"]))
    return batch_id


# ------------------------------------------------------------- 2. bronze ----
def ingest_bronze(spark, cfg: ProjectConfig) -> None:
    """landing -> bronze, append-only, idempotent on _batch_id. One _ingested_at for the whole run."""
    ts = spark.sql("SELECT current_timestamp() AS ts").first()["ts"]
    for key, (landing, bronze) in BRONZE_FLOW.items():
        src, dst = cfg.table(landing), cfg.table(bronze)
        if not exists(spark, src):
            continue
        if not exists(spark, dst):
            spark.sql(f"CREATE TABLE {dst} AS SELECT *, TIMESTAMP'{ts}' AS _ingested_at FROM {src}")
        else:
            spark.sql(
                f"INSERT INTO {dst} SELECT *, TIMESTAMP'{ts}' AS _ingested_at FROM {src} s "
                f"WHERE NOT EXISTS (SELECT 1 FROM {dst} b WHERE b._batch_id = s._batch_id)"
            )
        log.info("bronze %s up to date", key)


# ------------------------------------------------------- 3. quality gate ----
def quality_gate(spark, cfg: ProjectConfig, run_id: str, allow_no_new_data: bool = False) -> bool:
    """Validate everything in bronze newer than silver's watermark. Writes silver ONLY if the gate passes.

    allow_no_new_data=True is used by the (re-runnable) setup job; the daily job treats "no new
    meter data" as a blocking freshness failure - a silent source outage must not go unnoticed.
    """
    silver_c = cfg.table(Tables.SILVER_CONSUMPTION)
    wm = None
    if exists(spark, silver_c):
        wm = spark.sql(f"SELECT max(_ingested_at) m FROM {silver_c}").first()["m"]
    where = f"_ingested_at > TIMESTAMP'{wm}'" if wm else None
    bronze = {k: read_pdf(spark, cfg.table(b), where) for k, (_l, b) in BRONZE_FLOW.items()}

    cons = bronze["consumption"]
    if cons.empty and allow_no_new_data and wm is not None:
        log.info("No new bronze data and silver already populated - nothing to validate.")
        return True
    if cons.empty:
        results = [quality.CheckResult("freshness", "consumption", "error", False, 0, "no new meter data arrived")]
        outcome = quality.ValidationOutcome(results=results)
    else:
        days = pd.date_range(cons.reading_ts.min().normalize(), cons.reading_ts.max().normalize(), freq="D")
        target_day = days.max() + pd.Timedelta(days=2)  # tomorrow, seen from the next 05:00 run
        outcome = quality.run_all(bronze, cfg.quality, cfg.zones, days, target_day)

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    res = pd.DataFrame([r.as_dict() for r in outcome.results]).assign(run_id=run_id, env=cfg.env, checked_at=now)
    append_pdf(spark, res, cfg.table(Tables.DQ_RESULTS))
    append_pdf(spark, outcome.quarantine_frame().assign(run_id=run_id, detected_at=now), cfg.table(Tables.QUARANTINE))
    for r in outcome.results:
        (log.info if r.passed else log.warning)("[%s] %s.%s: %s", r.severity, r.table_name, r.check_name, r.detail)

    if not outcome.passed:
        log.error("QUALITY GATE FAILED - silver not updated; features and forecast will not run.")
        return False
    for key, (table, keys) in SILVER_TARGETS.items():
        merge_pdf(spark, outcome.clean[key], cfg.table(table), keys)
    log.info("Quality gate passed - silver updated.")
    return True


# ----------------------------------------------------------- 4. features ----
def build_features(spark, cfg: ProjectConfig, mode: str) -> None:
    today = sim_today(spark, cfg)
    target_day = today + pd.Timedelta(days=1)
    s = lambda t: cfg.table(t)  # noqa: E731
    if mode == "backfill":
        first = pd.Timestamp(spark.sql(f"SELECT min(reading_ts) m FROM {s(Tables.SILVER_CONSUMPTION)}").first()["m"])
        first = first.normalize() + pd.Timedelta(days=21)  # 3 weeks of warm-up for the longest lag
    else:
        first = today - pd.Timedelta(days=2)  # recompute recent days + tomorrow only
    lookback = first - pd.Timedelta(days=30)

    cons = read_pdf(spark, s(Tables.SILVER_CONSUMPTION), f"reading_ts >= '{lookback}'")
    fcst = read_pdf(spark, s(Tables.SILVER_WEATHER_FCST), f"target_ts >= '{first}'")
    hol, ev = read_pdf(spark, s(Tables.SILVER_HOLIDAYS)), read_pdf(spark, s(Tables.SILVER_EVENTS))
    targets = F.hourly_targets(cfg.zones, first, target_day)
    feats = F.build_features(targets, cons, fcst, hol, ev, cfg.zones, cfg.forecast.min_lag_hours)
    feats["feature_computed_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)

    table = ensure_feature_table(spark, cfg)
    if mode == "backfill":
        spark.sql(f"DELETE FROM {table}")
    merge_pdf(spark, feats, table, F.KEYS)
    log.info("Features written for %s..%s (%s rows) into %s", first.date(), target_day.date(), len(feats), table)


# ----------------------------------------------------------- 5. forecast ----
def forecast(spark, cfg: ProjectConfig, run_id: str) -> None:
    from water_forecasting.registry import load_alias

    today = sim_today(spark, cfg)
    target_day = today + pd.Timedelta(days=1)
    X = read_pdf(spark, cfg.table(Tables.FEATURES), f"to_date(target_ts) = '{target_day.date()}'")
    expected = len(cfg.zones) * cfg.forecast.horizon_hours
    if len(X) != expected:
        raise RuntimeError(f"Expected {expected} feature rows for {target_day.date()}, found {len(X)}")

    model, version = load_alias(cfg, "champion")
    if model is None:
        raise RuntimeError(f"No @champion for {cfg.registered_model_name} - run the training job first.")
    out = X[F.KEYS].copy()
    out["predicted_m3"] = model.predict(X[F.feature_columns(cfg.zones)]).round(3)
    out = out.assign(
        forecast_run_id=run_id,
        issued_at=today + pd.Timedelta(hours=cfg.forecast.issue_hour_utc),
        model_name=cfg.registered_model_name,
        model_version=str(version.version),
        env=cfg.env,
        created_at=pd.Timestamp.now(tz="UTC").tz_localize(None),
    )
    append_pdf(spark, out, cfg.table(Tables.FORECASTS))  # append-only: the audit trail of every claim
    log.info(
        "Forecast for %s written with model v%s (total %.0f m3)",
        target_day.date(),
        version.version,
        out.predicted_m3.sum(),
    )


# ------------------------------------------------------------ 6. monitor ----
def monitor(spark, cfg: ProjectConfig) -> tuple[bool, str]:
    """Nightly reconciliation: yesterday's forecasts vs the actuals that have now landed."""
    from water_forecasting.registry import get_alias_version

    day = sim_today(spark, cfg) - pd.Timedelta(days=1)
    f_tbl = cfg.table(Tables.FORECASTS)
    if not exists(spark, f_tbl):
        return False, "no forecasts yet"
    fc = read_pdf(spark, f_tbl, f"to_date(target_ts) = '{day.date()}'")
    act = read_pdf(spark, cfg.table(Tables.SILVER_CONSUMPTION), f"to_date(reading_ts) = '{day.date()}'")
    hourly = reconcile(fc, act, day)
    if hourly.empty:
        log.info("Nothing to reconcile for %s (no forecast was issued for it).", day.date())
    else:
        merge_pdf(spark, hourly, cfg.table(Tables.FORECAST_ACCURACY), ["zone_id", "target_ts"])
        merge_pdf(spark, daily_summary(hourly), cfg.table(Tables.DAILY_ACCURACY), ["forecast_date", "zone_id"])
        log.info("Reconciled %s: MAPE %.2f%%", day.date(), hourly.abs_pct_error.mean())

    d_tbl = cfg.table(Tables.DAILY_ACCURACY)
    recent = (
        []
        if not exists(spark, d_tbl)
        else (read_pdf(spark, d_tbl, "zone_id = 'ALL'").sort_values("forecast_date").mape.tolist())
    )
    champ = get_alias_version(cfg, "champion")
    baseline = float(champ.tags["holdout_mape"]) if champ and "holdout_mape" in champ.tags else None
    return needs_retrain(recent, baseline, cfg.decisions.degradation_factor, cfg.decisions.degradation_days)


# -------------------------------------------------- 7. prod -> lower env ----
def refresh_from_prod(spark, cfg: ProjectConfig) -> None:
    """Quarterly: copy a contiguous recent window of prod SILVER into dev/qa.

    Contiguous (not random rows) because lag features need an unbroken hourly series; long enough
    (qa: >1 year) to cover a full seasonal cycle incl. holidays and events. Data is zone-level
    aggregate demand - no customer PII - so no masking is needed; add it here if that changes.
    """
    if cfg.env == "prod":
        log.info("Refresh is a no-op in prod.")
        return
    src_cat, days = cfg.refresh_source_catalog, cfg.refresh_lookback_days or 180
    src_c = cfg.table(Tables.SILVER_CONSUMPTION, catalog=src_cat)
    latest = pd.Timestamp(spark.sql(f"SELECT max(reading_ts) m FROM {src_c}").first()["m"]).normalize()
    start = latest - pd.Timedelta(days=days)
    windowed = {
        Tables.SILVER_CONSUMPTION: "reading_ts",
        Tables.SILVER_WEATHER_OBS: "obs_ts",
        Tables.SILVER_WEATHER_FCST: "target_ts",
    }
    for t in (*windowed, Tables.SILVER_HOLIDAYS, Tables.SILVER_EVENTS):
        where = f"WHERE {windowed[t]} >= '{start}'" if t in windowed else ""
        spark.sql(f"CREATE OR REPLACE TABLE {cfg.table(t)} AS SELECT * FROM {cfg.table(t, catalog=src_cat)} {where}")
        log.info("Refreshed %s from %s", cfg.table(t), src_cat)
    # landing/bronze are reset so the daily replay continues from the refreshed silver
    for _k, (landing, bronze) in BRONZE_FLOW.items():
        for t in (landing, bronze):
            spark.sql(f"DROP TABLE IF EXISTS {cfg.table(t)}")
    log.info("Lower env %s now mirrors prod silver %s..%s", cfg.env, start.date(), latest.date())
