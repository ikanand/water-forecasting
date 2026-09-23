"""Train all candidate families, log to MLflow, register winner as @challenger, promote if better."""

from water_forecasting.lakehouse import get_spark
from water_forecasting.runtime import load_config, parse_args, set_task_value
from water_forecasting.training import train

args = parse_args([("--trigger", {"default": "scheduled"})])
cfg = load_config(args)
result = train(get_spark(), cfg, cfg.experiment_path(args.root_path), args.trigger, args.git_sha, str(args.job_run_id))
set_task_value("promoted", str(result["promoted"]).lower())
