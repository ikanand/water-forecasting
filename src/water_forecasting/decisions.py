"""The automated MLOps decisions, as small pure functions with unit tests.

* fair_comparison_window - which holdout hours can champion and challenger be compared on?
* should_promote         - does the newly trained challenger replace the champion?
* needs_retrain          - has live accuracy degraded enough to retrain now?
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def fair_comparison_window(
    target_ts: pd.Series, champion_trained_through: str | pd.Timestamp | None, min_days: int = 7
) -> tuple[pd.Series | None, str]:
    """Mask of holdout hours the champion has NOT seen in its own training - or None if fewer than
    `min_days` such days exist. Comparing on hours the champion was trained on flatters it (in-sample
    error) and would block every promotion for weeks after each promotion."""
    if champion_trained_through is None or pd.isna(pd.Timestamp(champion_trained_through)):
        return pd.Series(True, index=target_ts.index), "champion has no trained_through tag - full holdout"
    mask = target_ts > pd.Timestamp(champion_trained_through)
    days = target_ts[mask].dt.normalize().nunique()
    if days < min_days:
        return None, f"only {days} holdout day(s) after the champion's training end (< {min_days}) - keep champion"
    return mask, f"compared on {days} holdout days unseen by the champion"


def should_promote(
    challenger_mape: float,
    champion_mape: float | None,
    min_improvement_pct: float = 1.0,
    max_acceptable_mape: float = 15.0,
) -> tuple[bool, str]:
    if challenger_mape > max_acceptable_mape:
        return False, f"challenger MAPE {challenger_mape:.2f}% exceeds ceiling {max_acceptable_mape}%"
    if champion_mape is None:
        return True, "no champion yet - first acceptable model becomes champion"
    improvement = (champion_mape - challenger_mape) / champion_mape * 100
    if improvement >= min_improvement_pct:
        return True, f"challenger better by {improvement:.2f}% (>= {min_improvement_pct}%)"
    return False, f"improvement {improvement:.2f}% < required {min_improvement_pct}% - keep champion"


def needs_retrain(
    recent_daily_mape: Sequence[float],
    baseline_mape: float | None,
    degradation_factor: float = 1.25,
    degradation_days: int = 3,
) -> tuple[bool, str]:
    """True when the last `degradation_days` days are ALL worse than factor x baseline.

    Requiring consecutive days avoids retraining on a single odd day (a storm, a burst main).
    """
    if baseline_mape is None:
        return True, "no baseline MAPE on the champion - retrain"
    if len(recent_daily_mape) < degradation_days:
        return False, f"only {len(recent_daily_mape)} day(s) of accuracy history"
    threshold = baseline_mape * degradation_factor
    last = list(recent_daily_mape)[-degradation_days:]
    if all(m > threshold for m in last):
        return True, f"last {degradation_days} days MAPE {[round(m, 2) for m in last]} > {threshold:.2f}%"
    return False, f"accuracy within tolerance (threshold {threshold:.2f}%)"
