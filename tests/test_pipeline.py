import pandas as pd
import pytest

from water_forecasting import pipeline as P


class _FilesCfg:
    """cfg.files_path resolves to a Unity Catalog volume path - _files_frames only needs that one
    attribute, so a tiny stand-in avoids reconstructing a UC volume tree under tmp_path."""

    def __init__(self, files_path):
        self.files_path = str(files_path)


@pytest.fixture
def files_cfg(raw, tmp_path):
    p = tmp_path / "raw_files"
    p.mkdir(parents=True)
    raw["consumption"].to_csv(p / "consumption.csv", index=False)
    raw["weather_obs"].to_csv(p / "weather_observations.csv", index=False)
    raw["weather_fcst"].to_csv(p / "weather_forecasts.csv", index=False)
    raw["calendar"].to_csv(p / "calendar.csv", index=False)
    return _FilesCfg(p)


def test_calendar_is_landed_as_far_ahead_as_the_weather_forecast(files_cfg):
    """The target day's calendar row must already be in silver when build_features runs for it - the
    same reason the weather forecast is pulled 2 days past `end` (see the matching comment in pipeline.py)."""
    now = pd.Timestamp("2025-04-21 09:00")
    end = pd.Timestamp("2025-04-20")
    frames = P._files_frames(files_cfg, end, end, now)
    assert frames["calendar"].date.max() >= end + 2 * P.ONE_DAY
    assert frames["weather_fcst"].timestamp_local.max().normalize() >= end + 2 * P.ONE_DAY


def test_weather_forecast_never_uses_a_vintage_from_the_future(files_cfg):
    now = pd.Timestamp("2025-04-21 05:00")  # before the 06:00 vintage is issued
    frames = P._files_frames(files_cfg, "2025-04-20", "2025-04-20", now)
    assert (frames["weather_fcst"].forecast_issued_local <= now).all()
