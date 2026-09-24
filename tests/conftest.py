from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from water_forecasting.config import ProjectConfig

ROOT = Path(__file__).resolve().parents[1]
SITE = "DXB-CITY-TOTAL"
LOCATION = "Dubai (25.2048N, 55.2708E)"

# Fixed, known-in-advance calendar facts used by the tests below.
HOLIDAY_DATE = pd.Timestamp("2025-04-21")
EVENT_DATE = pd.Timestamp("2025-04-23")
RAMADAN_DATES = pd.date_range("2025-04-25", "2025-04-27")


def _make_raw(start: str, end: str, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Small deterministic fixture set matching the real (mlops_dev) Dubai-data schema.

    Pure pandas so tests stay fast and Spark-free; not a general-purpose generator.
    """
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    days = pd.date_range(start, end, freq="D")
    hod = hours.hour.to_numpy()

    demand = 100_000 * (1 + 0.3 * np.sin(2 * np.pi * (hod - 8) / 24)) + rng.normal(0, 1000, len(hours))
    consumption = pd.DataFrame(
        {
            "timestamp_local": hours,
            "timestamp_utc": hours - pd.Timedelta(hours=4),
            "site_id": SITE,
            "demand_m3h": demand.round(1),
            "source_system": "SCADA-HIST",
            "quality_note": None,
        }
    )

    temp = 20 + 8 * np.sin(2 * np.pi * (hod - 9) / 24) + rng.normal(0, 0.5, len(hours))
    weather_obs = pd.DataFrame(
        {
            "location": LOCATION,
            "timestamp_local": hours,
            "timestamp_utc": hours - pd.Timedelta(hours=4),
            "temperature_c": temp.round(2),
            "apparent_temperature_c": (temp - 1).round(2),
            "relative_humidity_pct": np.clip(60 + rng.normal(0, 5, len(hours)), 15, 100).round(1),
            "precipitation_mm": 0.0,
            "wind_speed_kmh": np.clip(15 + rng.normal(0, 3, len(hours)), 0, None).round(1),
            "solar_radiation_wm2": np.clip(400 * np.sin(2 * np.pi * (hod - 6) / 24), 0, None).round(1),
            "weather_source": "simulated_climatology_dubai",
        }
    )

    # Forecasts issued daily at 06:00 local, 41h horizon - matches the real feed. Issued one day
    # before `start` and one day after `end` too, so "tomorrow" of the last actual day is covered.
    issue_days = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=1), pd.Timestamp(end) + pd.Timedelta(days=1))
    fcst_rows = []
    for d in issue_days:
        issued = d + pd.Timedelta(hours=6)
        targets = pd.date_range(d + pd.Timedelta(days=1), periods=41, freq="h")
        th = targets.hour.to_numpy()
        t_temp = 20 + 8 * np.sin(2 * np.pi * (th - 9) / 24) + rng.normal(0, 1.0, len(targets))
        fcst_rows.append(
            pd.DataFrame(
                {
                    "forecast_issued_local": issued,
                    "timestamp_local": targets,
                    "forecast_temperature_c": t_temp.round(2),
                    "forecast_relative_humidity_pct": np.clip(60 + rng.normal(0, 5, len(targets)), 15, 100).round(1),
                    "forecast_precipitation_mm": 0.0,
                    "forecast_source": "simulated_day_ahead_nwp",
                }
            )
        )
    weather_fcst = pd.concat(fcst_rows, ignore_index=True)

    is_ramadan = days.isin(RAMADAN_DATES)
    calendar = pd.DataFrame(
        {
            "date": days,
            "holiday_name": np.where(days == HOLIDAY_DATE, "Eid al-Fitr", None),
            "is_public_holiday": days == HOLIDAY_DATE,
            "day_of_week": days.day_name(),
            "is_weekend": days.dayofweek >= 5,
            "is_ramadan": is_ramadan,
            "ramadan_day": np.where(is_ramadan, (days - RAMADAN_DATES.min()).days + 1, 0),
            "is_school_break": is_ramadan,
            "event_name": np.where(days == EVENT_DATE, "Test Event", None),
            "event_type": np.where(days == EVENT_DATE, "structural", None),
            "event_note": np.where(days == EVENT_DATE, "unit test event", None),
        }
    )

    return {"consumption": consumption, "weather_obs": weather_obs, "weather_fcst": weather_fcst, "calendar": calendar}


@pytest.fixture(scope="session")
def cfg() -> ProjectConfig:
    return ProjectConfig.from_yaml(ROOT / "project_config.yml", env="dev")


@pytest.fixture(scope="session")
def raw() -> dict[str, pd.DataFrame]:
    """~10 weeks of clean fixture data matching the real Dubai-data schema."""
    return _make_raw("2025-03-01", "2025-05-10", seed=7)


@pytest.fixture
def bronze(raw):
    return {k: v.assign(_ingested_at=pd.Timestamp("2026-01-01")) for k, v in raw.items()}
