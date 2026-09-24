import pandas as pd

from water_forecasting import quality

DAYS = pd.date_range("2025-03-01", "2025-05-10")
TARGET = pd.Timestamp("2025-05-12")


def _run(bronze, cfg):
    return quality.run_all(bronze, cfg.quality, DAYS, TARGET)


def _result(out, name, table):
    return next(r for r in out.results if r.check_name == name and r.table_name == table)


def test_clean_data_passes(bronze, cfg):
    out = _run(bronze, cfg)
    assert out.passed
    assert out.quarantine_frame().empty


def test_bad_rows_are_quarantined_not_blocking(bronze, cfg):
    c = bronze["consumption"].copy()
    c.loc[c.index[:3], "demand_m3h"] = -5.0
    bronze["consumption"] = pd.concat([c, c.iloc[10:12]])  # + 2 duplicates
    out = _run(bronze, cfg)
    assert out.passed
    reasons = out.quarantine_frame().reason.value_counts()
    assert reasons["out_of_range"] == 3 and reasons["duplicate_key"] == 2
    assert not out.clean["consumption"].duplicated(["timestamp_local"]).any()


def test_missing_hours_block_the_pipeline(bronze, cfg):
    c = bronze["consumption"]
    bronze["consumption"] = c.iloc[: int(len(c) * 0.95)]  # lose 5% > 2% limit
    out = _run(bronze, cfg)
    assert not out.passed
    assert not _result(out, "hourly_completeness", "consumption").passed


def test_missing_weather_forecast_for_tomorrow_blocks(bronze, cfg):
    f = bronze["weather_fcst"]
    issued = TARGET - pd.Timedelta(days=1)
    bronze["weather_fcst"] = f[f.forecast_issued_local.dt.normalize() != issued.normalize()]
    out = _run(bronze, cfg)
    assert not out.passed
    assert "24 of 24" in _result(out, "forecast_horizon_coverage", "weather_fcst").detail


def test_stuck_meter_is_quarantined_not_blocking(bronze, cfg):
    c = bronze["consumption"].copy()
    c.loc[c.index[100:104], "demand_m3h"] = 79737.3  # 4 identical consecutive hourly readings
    bronze["consumption"] = c
    out = _run(bronze, cfg)
    assert out.passed
    frozen = out.quarantine_frame().query("reason == 'frozen_value'")
    assert len(frozen) == 4
    assert not out.clean["consumption"].demand_m3h.eq(79737.3).any()


def test_two_identical_readings_are_not_a_stuck_meter(bronze, cfg):
    c = bronze["consumption"].copy()
    c.loc[c.index[100:102], "demand_m3h"] = 79737.3
    bronze["consumption"] = c
    out = _run(bronze, cfg)
    assert out.quarantine_frame().query("reason == 'frozen_value'").empty


def test_latest_ingested_duplicate_wins(bronze, cfg):
    c = bronze["consumption"]
    newer = c.head(1).assign(demand_m3h=999.0, _ingested_at=pd.Timestamp("2026-02-01"))
    bronze["consumption"] = pd.concat([c, newer])
    out = _run(bronze, cfg)
    row = out.clean["consumption"].set_index("timestamp_local").loc[newer.iloc[0]["timestamp_local"]]
    assert row.demand_m3h == 999.0
