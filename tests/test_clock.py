import pandas as pd
import pytest

from water_forecasting import clock

TODAY = pd.Timestamp("2026-09-23")


def test_local_today_uses_dubai_calendar():
    # 21:30 UTC on the 22nd is already 01:30 on the 23rd in Dubai (UTC+4)
    assert clock.local_today("Asia/Dubai", pd.Timestamp("2026-09-22 21:30", tz="UTC")) == TODAY
    assert clock.local_now("Asia/Dubai", pd.Timestamp("2026-09-22 21:30", tz="UTC")).hour == 1


def test_catch_up_lands_one_complete_day_per_run():
    assert clock.next_day_to_land(pd.Timestamp("2026-09-19"), TODAY) == pd.Timestamp("2026-09-20")
    assert clock.next_day_to_land(pd.Timestamp("2026-09-21 23:00"), TODAY) == pd.Timestamp("2026-09-22")


def test_today_is_never_landed():
    assert clock.next_day_to_land(pd.Timestamp("2026-09-22"), TODAY) is None
    assert clock.next_day_to_land(pd.Timestamp("2026-09-23"), TODAY) is None


def test_daily_run_before_setup_fails_loudly():
    with pytest.raises(ValueError, match="setup"):
        clock.next_day_to_land(None, TODAY)


def test_backfill_never_reaches_today():
    assert clock.backfill_end("2026-09-19", TODAY) == pd.Timestamp("2026-09-19")
    assert clock.backfill_end("2030-01-01", TODAY) == pd.Timestamp("2026-09-22")


def test_freshness_is_measured_against_the_wall_clock():
    assert clock.is_fresh(pd.Timestamp("2026-09-22 23:00"), TODAY)
    assert not clock.is_fresh(pd.Timestamp("2026-09-21 23:00"), TODAY)
    assert not clock.is_fresh(None, TODAY)
