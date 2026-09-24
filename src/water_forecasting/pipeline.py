"""Pipeline steps called by the job scripts: land -> bronze -> quality gate -> silver -> features
-> forecast -> reconcile/monitor, plus the quarterly prod -> lower-env refresh.

Clock: "today" is the local calendar day (config forecast.timezone). Only complete days are landed.
Self-healing: land_source lands the WHOLE gap up to yesterday in one call (not just the next single
day), and forecast()/monitor() loop over every day still owed a forecast or a reconciliation. A run
that was missed does not leave the environment permanently behind - the next successful run catches
up completely and backfills the forecast/accuracy history for the days in between."""

from __future__ import annotations

import logging
import uuid

import pandas as pd

from water_forecasting import clock, quality
from water_forecasting import features as F
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
ONE_DAY = pd.Timedelta(days=1)

SILVER_TARGETS = {  # validated frame -> (silver table, merge keys)
    "consumption": (Tables.SILVER_CONSUMPTION, ["timestamp_local"]),
    "weather_obs": (Tables.SILVER_WEATHER_OBS, ["timestamp_local"]),
    "weather_fcst": (Tables.SILVER_WEATHER_FCST, ["forecast_issued_local", "timestamp_local"]),
    "calendar": (Tables.SILVER_CALENDAR, ["date"]),
}


# ------------------------------------------------------------ 1. landing ----
def _apply_fault(frames: dict[str, pd.DataFrame], fault: str | None) -> None:
    """Demo switch: break the just-pulled batch on purpose to show the quality gate blocking."""
    if not fault or fault == "none":
        return
    if fault == "missing_forecast":
        frames["weather_fcst"] = frames["weather_fcst"].iloc[0:0]
    elif fault == "meter_outage":
        c = frames["consumption"]
        if not c.empty:
            last_day = c.timestamp_local.dt.normalize().max()
            gap = (c.timestamp_local >= last_day + pd.Timedelta(hours=6)) & (
                c.timestamp_local < last_day + pd.Timedelta(hours=14)
            )
            frames["consumption"] = c[~gap]
    else:
        raise ValueError(f"Unknown fault {fault!r}; use none | missing_forecast | meter_outage")


def _warehouse_frames(spark, cfg: ProjectConfig, start, end, now_local: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Pull a window from the source-system catalog (mlops_dev.bronze by default) - the closest
    thing this demo has to a real SCADA/weather/calendar feed. Same real Dubai data feeds every env,
    sliced by each env's own history window (see project_config.yml > environments)."""
    w = cfg.source.warehouse
    start, end = pd.Timestamp(start), pd.Timestamp(end) + ONE_DAY

    def pull(table: str, ts_cols: list[str], where: str) -> pd.DataFrame:
        df = spark.table(f"{w.catalog}.{w.schema_}.{table}").where(where).toPandas()
        for c in ts_cols:
            df[c] = pd.to_datetime(df[c])
        return df

    window = f"timestamp_local >= '{start}' AND timestamp_local < '{end}'"
    return {
        "consumption": pull(w.consumption_table, ["timestamp_local", "timestamp_utc"], window),
        "weather_obs": pull(w.weather_obs_table, ["timestamp_local", "timestamp_utc"], window),
        # +2 days: the quality gate checks forecast coverage for "tomorrow, as seen from the first daily
        # run after this land" (end + 2 days). Never take a vintage issued after the current time.
        "weather_fcst": pull(
            w.weather_fcst_table,
            ["forecast_issued_local", "timestamp_local"],
            f"timestamp_local >= '{start}' AND timestamp_local < '{end + 2 * ONE_DAY}' "
            f"AND forecast_issued_local <= '{now_local}'",
        ),
        # +2 days too: calendar facts (holidays, Ramadan, events) are known well in advance in reality,
        # and the target day's row must be in silver before build_features runs for it (see quality_gate).
        "calendar": pull(
            w.calendar_table, ["date"], f"date >= '{start.date()}' AND date < '{(end + 2 * ONE_DAY).date()}'"
        ),
    }


def _files_frames(cfg: ProjectConfig, start, end, now_local: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Replay YOUR generated CSVs (source.mode=files) with the same column structure as mode=warehouse."""
    p = cfg.files_path
    start, end = pd.Timestamp(start), pd.Timestamp(end) + ONE_DAY
    c = pd.read_csv(f"{p}/consumption.csv", parse_dates=["timestamp_local", "timestamp_utc"])
    o = pd.read_csv(f"{p}/weather_observations.csv", parse_dates=["timestamp_local", "timestamp_utc"])
    f = pd.read_csv(f"{p}/weather_forecasts.csv", parse_dates=["forecast_issued_local", "timestamp_local"])
    cal = pd.read_csv(f"{p}/calendar.csv", parse_dates=["date"])
    fcst_end = end + 2 * ONE_DAY  # see the matching comment in _warehouse_frames
    return {
        "consumption": c[(c.timestamp_local >= start) & (c.timestamp_local < end)],
        "weather_obs": o[(o.timestamp_local >= start) & (o.timestamp_local < end)],
        "weather_fcst": f[
            (f.timestamp_local >= start) & (f.timestamp_local < fcst_end) & (f.forecast_issued_local <= now_local)
        ],
        "calendar": cal[(cal.date >= start) & (cal.date < fcst_end)],
    }


def _last_landed_day(spark, cfg: ProjectConfig) -> pd.Timestamp | None:
    days = []
    for t in (Tables.LANDING_CONSUMPTION, Tables.SILVER_CONSUMPTION):
        if exists(spark, cfg.table(t)):
            m = spark.sql(f"SELECT max(timestamp_local) m FROM {cfg.table(t)}").first()["m"]
            if m is not None:
                days.append(pd.Timestamp(m).normalize())
    return max(days) if days else None


def land_source(spark, cfg: ProjectConfig, mode: str, fault: str = "none") -> str:
    """Pull complete days from the source system into landing.

    backfill: the configured history window (capped at yesterday); skipped if anything is landed.
    daily:    every complete day after the newest landed one, through yesterday, in one batch - or
              nothing if already up to date. Landing the whole gap (not just the next single day) is
              what makes the pipeline self-healing after a missed run.
    """
    now_local = clock.local_now(cfg.forecast.timezone)
    today = now_local.normalize()
    if mode == "backfill":
        if _last_landed_day(spark, cfg) is not None:
            log.info("Landing already populated - backfill skipped (idempotent).")
            return "skipped"
        start, end = pd.Timestamp(cfg.history_start), clock.backfill_end(cfg.history_end, today)
    else:
        day = clock.next_day_to_land(_last_landed_day(spark, cfg), today)
        if day is None:
            log.info("Source up to date: every complete day before %s is already landed.", today.date())
            return "up_to_date"
        start, end = day, today - ONE_DAY

    if cfg.source.mode == "warehouse":
        frames = _warehouse_frames(spark, cfg, start, end, now_local)
    else:
        frames = _files_frames(cfg, start, end, now_local)
    _apply_fault(frames, fault)
    if frames["consumption"].empty:
        raise RuntimeError(f"Source returned no consumption for {start}..{end}")

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
            # BY NAME: matches columns by name, not position - a landing schema change (e.g. a new
            # calendar column) does not silently shift data into the wrong bronze column.
            spark.sql(
                f"INSERT INTO {dst} BY NAME SELECT *, TIMESTAMP'{ts}' AS _ingested_at FROM {src} s "
                f"WHERE NOT EXISTS (SELECT 1 FROM {dst} b WHERE b._batch_id = s._batch_id)"
            )
        log.info("bronze %s up to date", key)


# ------------------------------------------------------- 3. quality gate ----
def _record(spark, cfg: ProjectConfig, run_id: str, outcome: quality.ValidationOutcome) -> None:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    res = pd.DataFrame([r.as_dict() for r in outcome.results]).assign(run_id=run_id, env=cfg.env, checked_at=now)
    append_pdf(spark, res, cfg.table(Tables.DQ_RESULTS))
    append_pdf(spark, outcome.quarantine_frame().assign(run_id=run_id, detected_at=now), cfg.table(Tables.QUARANTINE))
    for r in outcome.results:
        (log.info if r.passed else log.warning)(
            "%s %s.%s (%s): %s", "PASS" if r.passed else "FAIL", r.table_name, r.check_name, r.severity, r.detail
        )


def quality_gate(spark, cfg: ProjectConfig, run_id: str, allow_no_new_data: bool = False) -> bool:
    """Validate everything in bronze newer than silver's watermark. Writes silver ONLY if the gate passes.

    No new data is fine when silver already holds yesterday (a same-day re-run) - and always for the
    re-runnable setup job. Otherwise it is a blocking freshness failure: a silent source outage must
    not go unnoticed.
    """
    silver_c = cfg.table(Tables.SILVER_CONSUMPTION)
    wm, latest = None, None
    if exists(spark, silver_c):
        row = spark.sql(f"SELECT max(_ingested_at) wm, max(timestamp_local) latest FROM {silver_c}").first()
        wm, latest = row["wm"], row["latest"]
    where = f"_ingested_at > TIMESTAMP'{wm}'" if wm else None
    bronze = {k: read_pdf(spark, cfg.table(b), where) for k, (_l, b) in BRONZE_FLOW.items()}

    cons = bronze["consumption"]
    if cons.empty:
        today = clock.local_today(cfg.forecast.timezone)
        fresh = clock.is_fresh(latest, today)
        ok = fresh or (allow_no_new_data and wm is not None)
        detail = f"no new batch; newest complete day in silver is {pd.Timestamp(latest).date() if latest else None}"
        outcome = quality.ValidationOutcome(
            results=[quality.CheckResult("freshness", "consumption", "error", ok, 0, detail)]
        )
        _record(spark, cfg, run_id, outcome)
        if ok:
            log.info("Silver already up to date - nothing new to validate.")
            return True
        log.error("QUALITY GATE FAILED - actuals are stale; features and forecast will not run.")
        return False

    days = pd.date_range(cons.timestamp_local.min().normalize(), cons.timestamp_local.max().normalize(), freq="D")
    target_day = days.max() + 2 * ONE_DAY  # tomorrow, seen from the next issue
    outcome = quality.run_all(bronze, cfg.quality, days, target_day)
    _record(spark, cfg, run_id, outcome)

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
    target_day = today + ONE_DAY
    s = lambda t: cfg.table(t)  # noqa: E731
    if mode == "backfill":
        first = pd.Timestamp(
            spark.sql(f"SELECT min(timestamp_local) m FROM {s(Tables.SILVER_CONSUMPTION)}").first()["m"]
        )
        first = first.normalize() + pd.Timedelta(days=21)  # 3 weeks of warm-up for the longest lag
    else:
        # Recompute recent days (late corrections) + tomorrow, and reach back further if land_source
        # just caught up several days at once - every day since the last forecast needs one too.
        first = today - 2 * ONE_DAY
        if exists(spark, s(Tables.FORECASTS)):
            last_forecast = spark.sql(f"SELECT max(to_date(target_ts)) m FROM {s(Tables.FORECASTS)}").first()["m"]
            if last_forecast is not None:
                first = min(first, pd.Timestamp(last_forecast) + ONE_DAY)
    lookback = first - pd.Timedelta(days=30)

    cons = read_pdf(spark, s(Tables.SILVER_CONSUMPTION), f"timestamp_local >= '{lookback}'")
    fcst = read_pdf(spark, s(Tables.SILVER_WEATHER_FCST), f"timestamp_local >= '{first - ONE_DAY}'")
    obs = read_pdf(spark, s(Tables.SILVER_WEATHER_OBS), f"timestamp_local >= '{first - 2 * ONE_DAY}'")
    cal = read_pdf(spark, s(Tables.SILVER_CALENDAR))
    targets = F.hourly_targets(first, target_day)
    feats = F.build_features(targets, cons, fcst, cal, obs, cfg.forecast.min_lag_hours)
    feats["feature_computed_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)

    table = ensure_feature_table(spark, cfg)
    if mode == "backfill":
        spark.sql(f"DELETE FROM {table}")
    merge_pdf(spark, feats, table, F.KEYS)
    log.info("Features written for %s..%s (%s rows) into %s", first.date(), target_day.date(), len(feats), table)


# ----------------------------------------------------------- 5. forecast ----
def forecast(spark, cfg: ProjectConfig, run_id: str) -> None:
    """Issue a forecast for every target day that does not have one yet, up to tomorrow. Normally
    that is exactly one day; after land_source catches up several days at once, it is several -
    each with the issued_at timestamp it would have had if issued on its own day."""
    from water_forecasting.registry import load_alias

    today = sim_today(spark, cfg)
    final_target = today + ONE_DAY
    first_target = final_target
    if exists(spark, cfg.table(Tables.FORECASTS)):
        last_forecast = spark.sql(f"SELECT max(to_date(target_ts)) m FROM {cfg.table(Tables.FORECASTS)}").first()["m"]
        if last_forecast is not None:
            first_target = min(final_target, pd.Timestamp(last_forecast) + ONE_DAY)

    model, version = load_alias(cfg, "champion")
    if model is None:
        raise RuntimeError(f"No @champion for {cfg.registered_model_name} - run the training job first.")

    # One read, one prediction batch, one write for the whole range - not N Spark round-trips.
    X = read_pdf(
        spark,
        cfg.table(Tables.FEATURES),
        f"to_date(target_ts) >= '{first_target.date()}' AND to_date(target_ts) <= '{final_target.date()}'",
    )
    n_days = (final_target - first_target).days + 1
    expected = cfg.forecast.horizon_hours * n_days
    if len(X) != expected:
        raise RuntimeError(
            f"Expected {expected} feature rows for {first_target.date()}..{final_target.date()}, found {len(X)}"
        )
    day = X.target_ts.dt.normalize()
    out = X[F.KEYS].copy()
    out["predicted_m3"] = model.predict(X[F.model_columns()]).round(3)
    out = out.assign(
        forecast_run_id=run_id,
        issued_at=(day - ONE_DAY) + pd.Timedelta(hours=cfg.forecast.issue_hour_utc),
        model_name=cfg.registered_model_name,
        model_version=str(version.version),
        model_family=version.tags.get("family", "unknown"),
        env=cfg.env,
        created_at=pd.Timestamp.now(tz="UTC").tz_localize(None),
    )
    append_pdf(spark, out, cfg.table(Tables.FORECASTS))  # append-only: the audit trail of every claim
    for d, g in out.groupby(day):
        log.info(
            "Forecast for %s written with model v%s (total %.0f m3)", d.date(), version.version, g.predicted_m3.sum()
        )


# ------------------------------------------------------------ 6. monitor ----
def monitor(spark, cfg: ProjectConfig) -> tuple[bool, str]:
    """Reconcile every day whose forecast has not been scored yet, up to yesterday (the newest day
    with complete actuals). Normally that is one day; after a catch-up it can be several."""
    from water_forecasting.registry import get_alias_version

    last_complete = sim_today(spark, cfg) - ONE_DAY
    f_tbl, d_tbl = cfg.table(Tables.FORECASTS), cfg.table(Tables.DAILY_ACCURACY)
    if not exists(spark, f_tbl):
        return False, "no forecasts yet"
    if exists(spark, d_tbl):
        last_reconciled = spark.sql(f"SELECT max(forecast_date) m FROM {d_tbl}").first()["m"]
        first_day = min(last_complete, pd.Timestamp(last_reconciled) + ONE_DAY) if last_reconciled else last_complete
    else:
        # First-ever monitoring run: no accuracy history to resume from, so start at the earliest
        # forecasted day rather than just yesterday - otherwise a catch-up's backlog of forecasts
        # would silently never be reconciled.
        earliest = spark.sql(f"SELECT min(to_date(target_ts)) m FROM {f_tbl}").first()["m"]
        first_day = min(last_complete, pd.Timestamp(earliest)) if earliest else last_complete

    if first_day <= last_complete:
        # One read and one MERGE for the whole backlog - a MERGE per day gets slower as the target
        # table grows, so a wide catch-up must not do N of them.
        fc = read_pdf(
            spark,
            f_tbl,
            f"to_date(target_ts) >= '{first_day.date()}' AND to_date(target_ts) <= '{last_complete.date()}'",
        )
        lo, hi = first_day.date(), last_complete.date()
        act = read_pdf(
            spark,
            cfg.table(Tables.SILVER_CONSUMPTION),
            f"to_date(timestamp_local) >= '{lo}' AND to_date(timestamp_local) <= '{hi}'",
        )
        per_day = [reconcile(fc, act, day) for day in pd.date_range(first_day, last_complete)]
        per_day = [h for h in per_day if not h.empty]
        if per_day:
            hourly = pd.concat(per_day, ignore_index=True)
            merge_pdf(spark, hourly, cfg.table(Tables.FORECAST_ACCURACY), ["target_ts"])
            merge_pdf(spark, daily_summary(hourly), cfg.table(Tables.DAILY_ACCURACY), ["forecast_date"])
            for day, g in hourly.groupby("forecast_date"):
                log.info("Reconciled %s: MAPE %.2f%%", pd.Timestamp(day).date(), g.abs_pct_error.mean())
        else:
            log.info(
                "Nothing to reconcile for %s..%s (no forecast was issued for it).",
                first_day.date(),
                last_complete.date(),
            )
    recent = [] if not exists(spark, d_tbl) else read_pdf(spark, d_tbl).sort_values("forecast_date").mape.tolist()
    champ = get_alias_version(cfg, "champion")
    baseline = float(champ.tags["holdout_mape"]) if champ and "holdout_mape" in champ.tags else None
    return needs_retrain(recent, baseline, cfg.decisions.degradation_factor, cfg.decisions.degradation_days)


# -------------------------------------------------- 7. prod -> lower env ----
def refresh_from_prod(spark, cfg: ProjectConfig) -> None:
    """Quarterly: copy a contiguous recent window of prod SILVER into dev/qa.

    Contiguous (not random rows) because lag features need an unbroken hourly series; long enough
    (qa: >1 year) to cover a full seasonal cycle incl. holidays and events. Data is city-wide
    aggregate demand - no customer PII - so no masking is needed; add it here if that changes.
    """
    if cfg.env == "prod":
        log.info("Refresh is a no-op in prod.")
        return
    src_cat, days = cfg.refresh_source_catalog, cfg.refresh_lookback_days or 180
    src_c = cfg.table(Tables.SILVER_CONSUMPTION, catalog=src_cat)
    latest = pd.Timestamp(spark.sql(f"SELECT max(timestamp_local) m FROM {src_c}").first()["m"]).normalize()
    start = latest - pd.Timedelta(days=days)
    windowed = {
        Tables.SILVER_CONSUMPTION: "timestamp_local",
        Tables.SILVER_WEATHER_OBS: "timestamp_local",
        Tables.SILVER_WEATHER_FCST: "timestamp_local",
    }
    for t in (*windowed, Tables.SILVER_CALENDAR):
        where = f"WHERE {windowed[t]} >= '{start}'" if t in windowed else ""
        spark.sql(f"CREATE OR REPLACE TABLE {cfg.table(t)} AS SELECT * FROM {cfg.table(t, catalog=src_cat)} {where}")
        log.info("Refreshed %s from %s", cfg.table(t), src_cat)
    # landing/bronze are reset so the daily pipeline continues from the refreshed silver
    for _k, (landing, bronze) in BRONZE_FLOW.items():
        for t in (landing, bronze):
            spark.sql(f"DROP TABLE IF EXISTS {cfg.table(t)}")
    log.info("Lower env %s now mirrors prod silver %s..%s", cfg.env, start.date(), latest.date())
