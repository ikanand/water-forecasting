"""Reconcile yesterday's forecast vs actuals, update accuracy tables, decide whether to retrain."""

import logging

from water_forecasting.lakehouse import get_spark
from water_forecasting.pipeline import monitor
from water_forecasting.runtime import load_config, parse_args, set_task_value

args = parse_args()
cfg = load_config(args)
retrain, reason = monitor(get_spark(), cfg)
logging.getLogger("monitor").info("needs_retrain=%s: %s", retrain, reason)
set_task_value("needs_retrain", "true" if retrain else "false")
