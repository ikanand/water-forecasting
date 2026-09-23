"""Score tomorrow's 24 hours per zone with @champion and append to gold.demand_forecasts."""

from water_forecasting.lakehouse import get_spark
from water_forecasting.pipeline import forecast
from water_forecasting.runtime import load_config, parse_args

args = parse_args()
cfg = load_config(args)
forecast(get_spark(), cfg, str(args.job_run_id))
