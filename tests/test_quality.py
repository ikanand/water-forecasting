import pandas as pd

from tests.conftest import ZONES
from water_forecasting import quality

DAYS = pd.date_range("2025-03-01", "2025-05-10")
TARGET = pd.Timestamp("2025-05-12")


def _run(bronze, cfg):
    return quality.run_all(bronze, cfg.quality, ZONES, DAYS, TARGET)


def _result(out, name, table):
    return next(r for r in out.results if r.check_name == name and r.table_name == table)


def test_clean_data_passes(bronze, cfg):
    out = _run(bronze, cfg)
    assert out.passed
    assert out.quarantine_frame().empty


def test_bad_rows_are_quarantined_not_blocking(bronze, cfg):
    c = bronze["consumption"].copy()
    c.loc[c.index[:3], "consumption_m3"] = -5.0
    bronze["consumption"] = pd.concat([c, c.iloc[10:12]])  # + 2 duplicates
    out = _run(bronze, cfg)
    assert out.passed
    reasons = out.quarantine_frame().reason.value_counts()
    assert reasons["out_of_range"] == 3 and reasons["duplicate_key"] == 2
    assert not out.clean["consumption"].duplicated(["zone_id", "reading_ts"]).any()


def test_missing_hours_block_the_pipeline(bronze, cfg):
    c = bronze["consumption"]
    bronze["consumption"] = c.iloc[: int(len(c) * 0.95)]  # lose 5% > 2% limit
    out = _run(bronze, cfg)
    assert not out.passed
    assert not _result(out, "hourly_completeness", "consumption").passed


def test_missing_weather_forecast_for_tomorrow_blocks(bronze, cfg):
    f = bronze["weather_fcst"]
    bronze["weather_fcst"] = f[~((f.zone_id == "north") & (f.issued_at == TARGET - pd.Timedelta(days=1)))]
    out = _run(bronze, cfg)
    assert not out.passed
    assert "north" in _result(out, "forecast_horizon_coverage", "weather_fcst").detail


def test_latest_ingested_duplicate_wins(bronze, cfg):
    c = bronze["consumption"]
    newer = c.head(1).assign(consumption_m3=999.0, _ingested_at=pd.Timestamp("2026-02-01"))
    bronze["consumption"] = pd.concat([c, newer])
    out = _run(bronze, cfg)
    row = (
        out.clean["consumption"]
        .set_index(["zone_id", "reading_ts"])
        .loc[tuple(newer.iloc[0][["zone_id", "reading_ts"]])]
    )
    assert row.consumption_m3 == 999.0
