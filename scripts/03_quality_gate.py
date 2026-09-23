"""bronze -> checks -> silver (only if the gate passes). Sets task value dq_passed for the condition task."""

from water_forecasting.lakehouse import get_spark
from water_forecasting.pipeline import quality_gate
from water_forecasting.runtime import load_config, parse_args, set_task_value

args = parse_args([("--allow_no_new_data", {"action": "store_true"})])
cfg = load_config(args)
passed = quality_gate(get_spark(), cfg, str(args.job_run_id), args.allow_no_new_data)
set_task_value("dq_passed", "true" if passed else "false")
