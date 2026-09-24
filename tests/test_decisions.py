import pandas as pd

from water_forecasting.decisions import fair_comparison_window, needs_retrain, should_promote

HOLDOUT = pd.Series(pd.date_range("2026-08-27", "2026-09-23 23:00", freq="h"))


def test_champion_is_judged_only_on_hours_it_never_saw():
    mask, note = fair_comparison_window(HOLDOUT, "2026-09-10 23:00", min_days=7)
    assert mask is not None and HOLDOUT[mask].min() == pd.Timestamp("2026-09-11")
    assert "13 holdout days" in note


def test_too_few_unseen_days_keeps_the_champion():
    mask, note = fair_comparison_window(HOLDOUT, "2026-09-19 23:00", min_days=7)
    assert mask is None and "keep champion" in note


def test_untagged_champion_uses_the_full_holdout():
    mask, _ = fair_comparison_window(HOLDOUT, None)
    assert mask.all()


def test_first_model_is_promoted_if_acceptable():
    assert should_promote(5.0, None)[0]
    assert not should_promote(20.0, None, max_acceptable_mape=15.0)[0]


def test_promote_only_on_meaningful_improvement():
    assert should_promote(4.90, 5.00, min_improvement_pct=1.0)[0]  # 2% better
    assert not should_promote(4.97, 5.00, min_improvement_pct=1.0)[0]  # 0.6% better
    assert not should_promote(5.20, 5.00)[0]  # worse


def test_retrain_needs_consecutive_bad_days():
    assert needs_retrain([4, 7, 7, 7], baseline_mape=5, degradation_factor=1.25, degradation_days=3)[0]
    assert not needs_retrain([7, 7, 4, 7], baseline_mape=5)[0]  # one good day resets
    assert not needs_retrain([7, 7], baseline_mape=5)[0]  # not enough history
    assert needs_retrain([4, 4, 4], baseline_mape=None)[0]  # no baseline -> retrain
