"""Mock source systems for the demo: water meters (SCADA), weather observations,
a weather-forecast API, a holiday calendar and an events calendar.

Everything is *deterministic per day*: generating 2 years at once or one day at a
time gives identical numbers. That is what lets the daily job "replay" the next day
as if new data had just arrived from the source systems.

Pure pandas/numpy - no Spark - so it also runs locally and in unit tests.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

ZONE_PROFILES: dict[str, dict[str, float]] = {
    "north": {"base_m3": 1200.0, "temp_offset": -2.5},
    "central": {"base_m3": 2000.0, "temp_offset": 0.0},
    "south": {"base_m3": 1500.0, "temp_offset": 3.0},
}
_DEFAULT_ZONE = {"base_m3": 1000.0, "temp_offset": 0.0}

# Hourly demand shape (relative), morning + evening peaks. Normalised to mean 1.
_WEEKDAY = np.array(
    [
        0.45,
        0.38,
        0.35,
        0.34,
        0.38,
        0.60,
        1.05,
        1.55,
        1.60,
        1.30,
        1.10,
        1.05,
        1.08,
        1.02,
        0.98,
        0.98,
        1.05,
        1.20,
        1.40,
        1.45,
        1.30,
        1.05,
        0.80,
        0.60,
    ]
)
_WEEKEND = np.array(
    [
        0.55,
        0.45,
        0.40,
        0.38,
        0.38,
        0.42,
        0.55,
        0.80,
        1.15,
        1.45,
        1.55,
        1.45,
        1.35,
        1.20,
        1.10,
        1.05,
        1.08,
        1.20,
        1.35,
        1.35,
        1.25,
        1.05,
        0.85,
        0.68,
    ]
)
WEEKDAY_PROFILE = _WEEKDAY / _WEEKDAY.mean()
WEEKEND_PROFILE = _WEEKEND / _WEEKEND.mean()
_DOW_FACTOR = {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00, 4: 1.01, 5: 0.97, 6: 0.94}

HOLIDAYS_MMDD = [
    (1, 1, "New Year's Day"),
    (1, 6, "Epiphany"),
    (5, 1, "Labour Day"),
    (8, 15, "Assumption"),
    (10, 12, "National Day"),
    (11, 1, "All Saints"),
    (12, 6, "Constitution Day"),
    (12, 8, "Immaculate Conception"),
    (12, 25, "Christmas Day"),
]
EVENT_TYPES = ["concert", "football_match", "festival", "marathon"]
EPOCH = pd.Timestamp("2023-01-01")
FORECAST_HORIZON_H = 48  # the weather API publishes 48 hourly steps at 00:00 every day
EVENTS_KNOWN_AHEAD_DAYS = 60


def _rng(seed: int, *parts) -> np.random.Generator:
    key = "|".join(str(p) for p in (seed, *parts)).encode()
    return np.random.default_rng(zlib.crc32(key))


def _zone(zone: str) -> dict[str, float]:
    return ZONE_PROFILES.get(zone, _DEFAULT_ZONE)


# ------------------------------------------------------------------ weather ----
def true_weather_day(zone: str, day: pd.Timestamp, seed: int) -> pd.DataFrame:
    """The 'real' weather for one zone-day (24 rows). Observations and forecasts derive from it."""
    day = pd.Timestamp(day).normalize()
    rng = _rng(seed, "wx", zone, day.date())
    hours = np.arange(24)
    doy = day.dayofyear
    seasonal = 14.0 - 9.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)  # ~5C Jan, ~23C Jul
    summerness = (1 - np.cos(2 * np.pi * (doy - 15) / 365.25)) / 2  # 0 winter .. 1 summer
    diurnal = 5.0 * np.sin(2 * np.pi * (hours - 9) / 24)  # peak mid-afternoon
    temp = seasonal + _zone(zone)["temp_offset"] + diurnal + rng.normal(0, 3.0) + rng.normal(0, 0.6, 24)

    precip = np.zeros(24)
    if rng.random() < 0.30 - 0.15 * summerness:
        start, dur = int(rng.integers(0, 20)), int(rng.integers(2, 8))
        precip[start : start + dur] = rng.gamma(1.5, 1.2, size=len(precip[start : start + dur]))
    humidity = np.clip(65 + 20 * (precip > 0) - 1.0 * (temp - 15) + rng.normal(0, 4, 24), 15, 100)
    return pd.DataFrame(
        {
            "zone_id": zone,
            "ts": day + pd.to_timedelta(hours, unit="h"),
            "temperature_c": temp.round(2),
            "humidity_pct": humidity.round(1),
            "precipitation_mm": precip.round(2),
        }
    )


def weather_observations(zones: list[str], day: pd.Timestamp, seed: int) -> pd.DataFrame:
    out = []
    for z in zones:
        t = true_weather_day(z, day, seed)
        noise = _rng(seed, "obs", z, pd.Timestamp(day).date()).normal(0, 0.2, len(t))
        t["temperature_c"] = (t["temperature_c"] + noise).round(2)
        out.append(t.rename(columns={"ts": "obs_ts"}))
    return pd.concat(out, ignore_index=True)


def weather_forecasts(zones: list[str], issue_day: pd.Timestamp, seed: int) -> pd.DataFrame:
    """Forecast issued at issue_day 00:00 for the next 48 hours (lead 1..48h). Error grows with lead."""
    issue_day = pd.Timestamp(issue_day).normalize()
    out = []
    for z in zones:
        truth = pd.concat([true_weather_day(z, issue_day + pd.Timedelta(days=d), seed) for d in range(3)])
        targets = issue_day + pd.to_timedelta(np.arange(1, FORECAST_HORIZON_H + 1), unit="h")
        t = truth.set_index("ts").reindex(targets).rename_axis("ts").reset_index()
        lead = np.arange(1, FORECAST_HORIZON_H + 1)
        rng = _rng(seed, "fcst", z, issue_day.date())
        bias = rng.normal(0, 0.8)  # a whole run can be off
        t["temperature_c"] = (t["temperature_c"] + bias + rng.normal(0, 0.5 + 0.03 * lead)).round(2)
        missed = (t["precipitation_mm"] > 0) & (rng.random(len(t)) < 0.15)
        t["precipitation_mm"] = np.where(missed, 0.0, t["precipitation_mm"] * rng.lognormal(0, 0.3, len(t))).round(2)
        t["humidity_pct"] = np.clip(t["humidity_pct"] + rng.normal(0, 3, len(t)), 15, 100).round(1)
        t = t.rename(columns={"ts": "target_ts"})
        t["issued_at"] = issue_day
        t["lead_hours"] = lead
        t["zone_id"] = z
        out.append(
            t[["zone_id", "issued_at", "target_ts", "lead_hours", "temperature_c", "humidity_pct", "precipitation_mm"]]
        )
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------- calendar ----
def holidays(years: list[int]) -> pd.DataFrame:
    rows = [(pd.Timestamp(y, m, d), name, "national") for y in years for m, d, name in HOLIDAYS_MMDD]
    return pd.DataFrame(rows, columns=["holiday_date", "holiday_name", "scope"])


def events(zones: list[str], days: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rows = []
    for day in days:
        for z in zones:
            rng = _rng(seed, "evt", z, day.date())
            if rng.random() < 0.035:
                etype = EVENT_TYPES[int(rng.integers(0, len(EVENT_TYPES)))]
                start = 8 if etype == "marathon" else int(rng.integers(17, 21))
                end = min(start + int(rng.integers(3, 6)), 23)
                rows.append((f"{z}-{day:%Y%m%d}", z, day, start, end, etype, int(rng.integers(5, 61)) * 1000))
    return pd.DataFrame(
        rows,
        columns=["event_id", "zone_id", "event_date", "start_hour", "end_hour", "event_type", "expected_attendance"],
    )


# -------------------------------------------------------------- consumption ----
def consumption_day(
    zone: str, day: pd.Timestamp, seed: int, holiday_dates: set, day_events: pd.DataFrame
) -> pd.DataFrame:
    day = pd.Timestamp(day).normalize()
    wx = true_weather_day(zone, day, seed)
    rng = _rng(seed, "cons", zone, day.date())
    is_holiday = day in holiday_dates
    profile = WEEKEND_PROFILE if (day.dayofweek >= 5 or is_holiday) else WEEKDAY_PROFILE
    level = _zone(zone)["base_m3"] * (1 + 0.00004 * (day - EPOCH).days)  # slow population growth
    dow = 0.95 if is_holiday else _DOW_FACTOR[day.dayofweek]
    t = wx["temperature_c"].to_numpy()
    temp_eff = 1 + 0.022 * np.clip(t - 20, 0, None) - 0.005 * np.clip(12 - t, 0, None)
    rain_eff = 1 - 0.05 * (wx["precipitation_mm"].to_numpy() > 0.3)  # less garden watering
    event_eff = np.ones(24)
    todays = day_events[(day_events["zone_id"] == zone) & (day_events["event_date"] == day)]
    for _, e in todays.iterrows():
        event_eff[int(e.start_hour) : int(e.end_hour) + 1] += e.expected_attendance / 600_000
    value = level * profile * dow * temp_eff * rain_eff * event_eff * (1 + rng.normal(0, 0.025, 24))
    return pd.DataFrame(
        {"zone_id": zone, "reading_ts": wx["ts"], "consumption_m3": value.round(3), "meter_status": "OK"}
    )


# ------------------------------------------------------------------- public ----
def _inject_anomalies(df: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """Realistic dirt: duplicate readings, negative values, missing hours."""
    if rate <= 0 or df.empty:
        return df
    r = rng.random(len(df))
    dupes = df[r < rate].assign(consumption_m3=lambda d: d.consumption_m3 * 1.01)
    df = df.copy()
    df.loc[(r >= rate) & (r < 2 * rate), "consumption_m3"] *= -1
    df = df[~((r >= 2 * rate) & (r < 3 * rate))]
    return pd.concat([df, dupes], ignore_index=True)


def _apply_fault(out: dict[str, pd.DataFrame], fault: str | None, zones: list[str]) -> None:
    """Demo switch: break the data on purpose to show the quality gate blocking the pipeline."""
    if not fault or fault == "none":
        return
    z = zones[0]
    if fault == "missing_forecast":
        out["weather_fcst"] = out["weather_fcst"][out["weather_fcst"].zone_id != z]
    elif fault == "meter_outage":
        c = out["consumption"]
        last_day = c.reading_ts.dt.normalize().max()
        gap = (
            (c.zone_id == z)
            & (c.reading_ts >= last_day + pd.Timedelta(hours=6))
            & (c.reading_ts < last_day + pd.Timedelta(hours=14))
        )
        out["consumption"] = c[~gap]
    else:
        raise ValueError(f"Unknown fault {fault!r}; use none | missing_forecast | meter_outage")


def generate_range(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    zones: list[str],
    seed: int = 42,
    anomaly_rate: float = 0.0,
    fault: str | None = None,
    backfill: bool = False,
) -> dict[str, pd.DataFrame]:
    """What the source systems publish for days [start, end] (inclusive).

    Emitting day X means: meter readings + weather observations *of* X, and the weather
    forecast issued at X+1 00:00 (the one the 05:00 forecast run on X+1 will use).
    Events become known 60 days ahead; holidays a couple of years ahead.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    days = pd.date_range(start, end, freq="D")
    hol_years = (
        list(range(start.year, end.year + 3))
        if backfill
        else ([end.year + 2] if (end.month, end.day) == (1, 1) else [])
    )
    hol_all = holidays(list(range(start.year - 1, end.year + 3)))
    hol_dates = set(hol_all.holiday_date)
    evt_days = pd.date_range(
        start if backfill else end + pd.Timedelta(days=EVENTS_KNOWN_AHEAD_DAYS),
        end + pd.Timedelta(days=EVENTS_KNOWN_AHEAD_DAYS),
        freq="D",
    )
    evt_needed = events(zones, days, seed)  # to shape consumption
    rng = _rng(seed, "anomaly", start.date(), end.date())
    out = {
        "consumption": _inject_anomalies(
            pd.concat(
                [consumption_day(z, d, seed, hol_dates, evt_needed) for d in days for z in zones], ignore_index=True
            ),
            anomaly_rate,
            rng,
        ),
        "weather_obs": pd.concat([weather_observations(zones, d, seed) for d in days], ignore_index=True),
        "weather_fcst": pd.concat(
            [weather_forecasts(zones, d + pd.Timedelta(days=1), seed) for d in days], ignore_index=True
        ),
        "holidays": holidays(hol_years) if hol_years else holidays([]),
        "events": events(zones, evt_days, seed),
    }
    _apply_fault(out, fault, zones)
    return out
