from water_forecasting.decisions import needs_retrain, should_promote


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
