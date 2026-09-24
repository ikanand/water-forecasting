import pandas as pd
import pytest

from water_forecasting.monitoring import daily_summary, reconcile

DAY = pd.Timestamp("2025-05-01")


def _frames():
    ts = pd.date_range(DAY, periods=24, freq="h")
    old = pd.DataFrame(
        {
            "target_ts": ts,
            "issued_at": DAY - pd.Timedelta(hours=20),
            "model_version": "1",
            "predicted_m3": 50.0,
        }
    )
    new = old.assign(issued_at=DAY - pd.Timedelta(hours=19), predicted_m3=110.0, model_version="2")
    act = pd.DataFrame({"timestamp_local": ts, "demand_m3h": 100.0})
    return pd.concat([old, new]), act


def test_latest_issued_forecast_is_scored():
    fc, act = _frames()
    h = reconcile(fc, act, DAY)
    assert len(h) == 24
    assert (h.model_version == "2").all()
    assert h.abs_pct_error.mean() == pytest.approx(10.0)


def test_daily_summary_has_one_row_per_day():
    fc, act = _frames()
    s = daily_summary(reconcile(fc, act, DAY))
    assert len(s) == 1
    assert s.iloc[0].bias_pct == pytest.approx(10.0)
