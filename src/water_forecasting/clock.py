"""Wall-clock rules that make the daily pipeline behave like a real operation (pure functions).

* Only COMPLETE days are ever landed: a day is complete once the local calendar has moved past it.
* A lagging environment (fresh bootstrap, missed runs, lower env after a refresh) catches up one day
  per run - each run issues the forecast that would have been issued on that day.
* Once caught up, a re-run on the same day is a no-op for landing and passes the freshness check.
"""

from __future__ import annotations

import pandas as pd

ONE_DAY = pd.Timedelta(days=1)


def local_now(tz: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Current local wall-clock time as a naive timestamp (the source data is naive local time)."""
    ts = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(tz).tz_localize(None)


def local_today(tz: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
    return local_now(tz, now).normalize()


def next_day_to_land(last_landed: pd.Timestamp | None, today: pd.Timestamp) -> pd.Timestamp | None:
    """The next complete day to pull from the source, or None when everything complete is landed."""
    if last_landed is None:
        raise ValueError("Nothing landed yet - run the setup job (backfill) first.")
    nxt = pd.Timestamp(last_landed).normalize() + ONE_DAY
    return nxt if nxt < today else None


def backfill_end(history_end: str, today: pd.Timestamp) -> pd.Timestamp:
    """History never extends into today or the future, whatever the config says."""
    return min(pd.Timestamp(history_end).normalize(), today - ONE_DAY)


def is_fresh(latest_actual_day: pd.Timestamp | None, today: pd.Timestamp) -> bool:
    """Actuals are fresh when yesterday (the newest day that can be complete) is present."""
    return latest_actual_day is not None and pd.Timestamp(latest_actual_day).normalize() >= today - ONE_DAY
